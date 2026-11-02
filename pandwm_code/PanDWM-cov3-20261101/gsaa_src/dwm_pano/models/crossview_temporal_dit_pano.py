from typing import Optional, Union
import einops
import diffusers
import torch
import math

import dwm.models.adapters
from dwm.models.crossview_temporal import VTSelfAttentionBlock, AlphaBlender, Mixer


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(
            self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(torch.nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0,
                 ray_octaves=4, ray_start_octave=0,
                 cond_proj_dim=72, in_channels=1536):
        super().__init__()
        self.pos_encoding = PositionalEncoding(
            num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

        self.proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)

    def forward(self, pos, rays):
        """Encode a ray field.

        Args:
            pos: either one origin per view ``[B, 3]`` (broadcast over the
                grid -- the legacy behaviour) or a per-pixel field
                ``[B, H, W, 3]``. The per-pixel form is what carries Plucker
                moments; see `get_pano_plucker_moments`.
            rays: unit directions ``[B, H, W, 3]``.

        Channel count is identical either way (`pos_octaves` * 3 * 2), so
        `self.proj` stays shape-compatible with existing checkpoints.
        """
        batchsize, height, width, dims = rays.shape
        if pos.dim() == 4:
            pos_enc = self.pos_encoding(pos.flatten(1, 2))
            pos_enc = pos_enc.view(batchsize, height, width, pos_enc.shape[-1])
        else:
            pos_enc = self.pos_encoding(pos.unsqueeze(1))
            pos_enc = pos_enc.view(batchsize, 1, 1, pos_enc.shape[-1])
            pos_enc = pos_enc.repeat(1, height, width, 1)

        rays = rays.flatten(1, 2)
        ray_enc = self.ray_encoding(rays)
        ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
        x = torch.cat((pos_enc, ray_enc), -1)

        return self.proj(x)


def get_rays(
        camera_intrinsics: torch.tensor,
        camera_transforms: torch.tensor,
        target_size: Union[int, tuple[int]],):
    ''' get rays
        Args:
            camera_transforms: (B*T*V, 4, 4), cam2world
            intrinsics: (B*T*V, 3, 3)
        Returns:
            rays_o: [B*T*V, 3]
            rays_d: [B*T*V, H, W, 3]
    '''
    device = camera_transforms.device
    dtype = camera_transforms.dtype
    camera_transforms = camera_transforms.to(dtype=torch.float32)
    camera_intrinsics = camera_intrinsics.to(dtype=torch.float32)

    H, W = (target_size, target_size) if isinstance(
        target_size, int) else target_size
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs])  # (H*W, 3)
    directions = torch.inverse(
        camera_intrinsics) @ points_coord.unsqueeze(0)  # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions  # (batch_size, 3, H*W)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.transpose(1, 2).view(-1, H, W, 3).to(dtype=dtype)

    rays_o = camera_transforms[:, :3, 3].to(dtype=dtype)  # (batch_size, 3)

    return rays_o, rays_d


# -----------------------------------------------------------------------------
# Analytic equirectangular rays for the stitched-pano halves.
#
# `get_rays` above assumes a pinhole camera (`inverse(K) @ [i, j, 1]`).  The
# pano path feeds it `_make_virtual_pano_K`, whose own docstring states the
# values are "not physically pinhole-correct" and were written for the
# *implicit* conditioning branch (which only embeds them as an opaque
# "pano-half signature").  Under `perspective_modeling_type == "explicit"`
# those numbers are instead consumed as real geometry, which is unsound: a
# pinhole model cannot represent a half's 201.6 deg of azimuth at all.
# Measured consequences of the pinhole path (token grid 16x144):
#   * horizontal span  89.8 deg  (should be 201.6)
#   * vertical span     88.2 deg  (should be  33.3)
#   * mean ray elevation +54.0 deg -- every ray points at the sky
#   * the two halves disagree by 16.1 deg (L/R seam) and 12.7 deg (wrap seam)
#     about the direction of the SAME physical azimuth
#
# The constants and the angle<->pixel algebra below are the exact inverse of
# `dwm_pano.datasets.pano_projection.project_ego_point_to_pano`, which is the
# authoritative forward projection used to *render* the panorama.  Keeping the
# two mutually inverse is what makes the ray map agree with pixel content;
# `test_pano_equirect_rays_invert_projection` pins that round trip.
# -----------------------------------------------------------------------------
_PANO_SRC_W = 9600
_PANO_SRC_H = 600
_PANO_H_PRE_CROP = 900
_PANO_CROP_TOP = 150
_PANO_VERT_FOV_HALF_RAD = math.pi * 50.0 / 180.0 / 2.0
_PANO_HALF_RANGES = ((0, 5376), (4224, 9600))


def get_pano_equirect_rays(
        view_count: int,
        target_size: tuple[int],
        device=None,
        dtype=torch.float32):
    """Ray origins/directions for stitched-pano halves, in nuScenes ego frame.

    Args:
        view_count: number of pano halves (2 for this project's V=2 layout).
        target_size: ``(H, W)`` of the token grid the rays are consumed at.

    Returns:
        ``(rays_o, rays_d)`` shaped ``[view_count, 3]`` and
        ``[view_count, H, W, 3]``, matching `get_rays`' contract.

    The panorama is rendered about the ego origin, so every half shares
    ``rays_o = 0``; the directional field is what carries the geometry.
    """
    if view_count != len(_PANO_HALF_RANGES):
        raise ValueError(
            "pano equirect rays expect {} halves, got view_count={}".format(
                len(_PANO_HALF_RANGES), view_count))

    height, width = target_size
    # Token-grid centre of each cell, as a fraction of the half's extent.
    u = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    v = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height

    # Row -> elevation.  Inverse of
    #   ty_pre = -ang_y / VERT_FOV_HALF_RAD * (H_PRE/2) + H_PRE/2 + 0.5
    #   ty     = ty_pre - CROP_TOP
    ty = v * _PANO_SRC_H
    ty_pre = ty + _PANO_CROP_TOP
    ang_y = -(ty_pre - 0.5 - _PANO_H_PRE_CROP / 2.0) \
        / (_PANO_H_PRE_CROP / 2.0) * _PANO_VERT_FOV_HALF_RAD

    rays_d = []
    for half_start, half_end in _PANO_HALF_RANGES:
        # Column -> absolute pano column -> azimuth.  Inverse of
        #   tx = ang_x / (2*pi) * PANO_W + PANO_W/2 + 0.5
        tx = half_start + u * float(half_end - half_start)
        ang_x = (tx - 0.5 - _PANO_SRC_W / 2.0) / _PANO_SRC_W * 2.0 * math.pi

        cos_y = torch.cos(ang_y).unsqueeze(-1)          # [H, 1]
        sin_x = torch.sin(ang_x).unsqueeze(0)           # [1, W]
        cos_x = torch.cos(ang_x).unsqueeze(0)           # [1, W]
        # pano frame (px, py, pz), then undo the ego->pano axis remap
        # (px = -y_ego, py = x_ego, pz = z_ego).
        x_ego = cos_y * cos_x
        y_ego = -cos_y * sin_x
        z_ego = torch.sin(ang_y).unsqueeze(-1).expand(height, width)
        d = torch.stack((x_ego, y_ego, z_ego), dim=-1)
        rays_d.append(d / d.norm(dim=-1, keepdim=True))

    rays_d = torch.stack(rays_d).to(dtype=dtype)
    rays_o = torch.zeros(view_count, 3, device=device, dtype=dtype)
    return rays_o, rays_d


def get_pano_column_camera_index(view_count: int, width: int):
    """Which of the six physical cameras each token column was stitched from.

    Delegates the Voronoi ownership to
    `dwm_pano.datasets.pano_projection.pano_column_camera_indices` -- the same
    authoritative table the hard-seam stitcher used -- rather than restating
    the boundaries here. Returned indices follow that module's ``CAM_ORDER``,
    which is also the packing order of the dataset's
    ``pano_camera_translations``.

    Returns:
        LongTensor ``[view_count, width]``.
    """
    from dwm_pano.datasets.pano_projection import pano_column_camera_indices

    columns = (torch.arange(width, dtype=torch.float64) + 0.5) / width
    per_view = []
    for half_start, half_end in _PANO_HALF_RANGES[:view_count]:
        tx = half_start + columns.numpy() * float(half_end - half_start)
        per_view.append(torch.from_numpy(pano_column_camera_indices(tx)))
    return torch.stack(per_view).long()


def get_pano_plucker_moments(rays_d, camera_translations, column_camera_index):
    """Plucker moments ``m = o x d`` with a per-column ray origin.

    Why per-column: a stitched pano half is assembled from three different
    physical cameras mounted at different points on the vehicle, but the
    virtual pano extrinsics place a single camera at the ego origin.  That
    makes the usual single-origin Plucker moment identically zero (measured:
    max |m| = 0.000e+00), i.e. three dead channels.  Feeding each column its
    real source-camera origin is what makes the moment carry information --
    and the information it carries ("this column was observed from a camera
    over there") changes precisely at the seams, where the source camera
    switches.

    Args:
        rays_d: ``[B*T*V, H, W, 3]`` unit directions.
        camera_translations: ``[B, T, 6, 3]`` real camera positions, in the
            same frame as `rays_d` and ordered per ``CAM_ORDER``.
        column_camera_index: ``[V, W]`` from `get_pano_column_camera_index`.

    Returns:
        ``[B*T*V, H, W, 3]`` moments.
    """
    view_count, width = column_camera_index.shape
    batch_size, sequence_length = camera_translations.shape[:2]
    if rays_d.shape[0] != batch_size * sequence_length * view_count:
        raise ValueError(
            "rays_d batch {} does not match B*T*V = {}*{}*{}".format(
                rays_d.shape[0], batch_size, sequence_length, view_count))

    index = column_camera_index.to(camera_translations.device)
    # [B, T, 6, 3] -> gather per (view, column) -> [B, T, V, W, 3]
    origins = camera_translations[:, :, index, :]
    # (b, t, v) flattens v-fastest, matching how the caller tiles rays.
    origins = origins.flatten(0, 2).unsqueeze(1)            # [B*T*V, 1, W, 3]
    origins = origins.expand(-1, rays_d.shape[1], -1, -1)   # broadcast over H
    return torch.cross(origins.to(rays_d.dtype), rays_d, dim=-1)


class DiTCrossviewTemporalConditionModel(diffusers.SD3Transformer2DModel):
    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        projection_class_embeddings_input_dim: int = None,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = False,
        enable_temporal: bool = False,
        crossview_attention_type: str = None,
        temporal_attention_type: str = None,
        merge_factor: float = 2, merge_strategy="learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "",
        pano_equirect_rays: bool = False,
        pano_plucker_rays: bool = False,
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mask_module=None,
        **kwargs
    ):
        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs
        )
        self.crossview_gradient_checkpointing = crossview_gradient_checkpointing
        self.temporal_gradient_checkpointing = temporal_gradient_checkpointing
        self.disable_view_emb_on_temporal_module = disable_view_emb_on_temporal_module

        # image condition adapter
        if condition_image_adapter_config is not None:
            self.condition_image_adapter = \
                dwm.models.adapters.ImageAdapter(
                    **condition_image_adapter_config
                )
        else:
            self.condition_image_adapter = None

        # views and frames index embedding
        inner_dim = attention_head_dim * num_attention_heads
        self.index_proj = diffusers.models.embeddings.Timesteps(
            inner_dim, True, 0)

        self.perspective_modeling_type = perspective_modeling_type
        # Default False: the pinhole path stays bit-exact for every existing
        # checkpoint. Enable only together with a finetune/eval that expects
        # the corrected ray field (see get_pano_equirect_rays).
        self.pano_equirect_rays = pano_equirect_rays
        # Plucker moments are only meaningful on a correct direction field:
        # m = o x d reuses d, so a wrong d yields a wrong m. Refuse the
        # combination rather than silently producing garbage.
        if pano_plucker_rays and not pano_equirect_rays:
            raise ValueError(
                "pano_plucker_rays requires pano_equirect_rays: the moment "
                "m = o x d is computed from the same directions, so it is "
                "only meaningful once the ray field itself is correct.")
        self.pano_plucker_rays = pano_plucker_rays
        self._pano_column_camera_index = None
        if perspective_modeling_type == "explicit":
            # Explicit Perspective Modeling
            self.rayencoder = RayEncoder(
                cond_proj_dim=72, in_channels=self.inner_dim)   
        elif perspective_modeling_type == "implicit":
            # Implicit Perspective Modeling
            self.view_cam_proj = diffusers.models.embeddings.Timesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.view_embedding = diffusers.models.embeddings.TimestepEmbedding(
                in_channels=projection_class_embeddings_input_dim, 
                time_embed_dim=self.inner_dim)  

        self.enable_crossview = enable_crossview
        self.crossview_attention_type = crossview_attention_type
        self.crossview_block_layers = crossview_block_layers
        if enable_crossview:
            self.view_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

            self.crossview_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(crossview_block_layers))
            ])

            self.view_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

        self.enable_temporal = enable_temporal
        self.temporal_attention_type = temporal_attention_type
        self.temporal_block_layers = temporal_block_layers
        if enable_temporal:
            self.time_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

            self.temporal_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(temporal_block_layers))
            ])

            self.time_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

            # §12.3 causal AR retrofit: install a FlexAttention-aware
            # processor on every temporal block so that callers can pass a
            # `BlockMask` (block-causal pattern from LongLive) via the
            # `temporal_attention_mask` argument. The processor is a strict
            # superset of AttnProcessor2_0 — it falls back to SDPA when no
            # BlockMask is passed, so it is safe even for non-causal training.
            from dwm_pano.models.flex_block_causal import \
                BlockCausalFlexAttnProcessor
            for _temporal_block in self.temporal_transformer_blocks:
                _temporal_block.attn1.set_processor(
                    BlockCausalFlexAttnProcessor())

        # Depth Net
        self.depth_net = None   # TODO: support joint training of image and lidar

        # Mask Reconstruction
        self.mask_module = mask_module

    def forward_crossview_block_and_mix_result(
        self, crossview_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        view_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, height: int, disable_crossview: torch.BoolTensor,
        crossview_attention_mask, crossview_attention_index
    ):
        crossview_hidden_states = hidden_states + \
            view_emb      # [b*T*V, h*w, c]
        if self.crossview_attention_type == "fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) hw c -> bt v hw c", v=view_count)
            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).flatten(0, 1)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, crossview_attention_index)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t) (v n) hw c -> (b v) (t n hw) c",
                v=view_count, n=3, t=sequence_length)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b v) (t n hw) c -> (b t v) n hw c",
                v=view_count, t=sequence_length, n=3)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, :, :]

        elif self.crossview_attention_type == "adj_fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) hw c -> b t v hw c",
                v=view_count, t=sequence_length)

            adj_frame_index = torch.cat(
                (torch.arange(view_count).unsqueeze(0),
                 torch.arange(view_count).unsqueeze(0)))
            adj_frame_index[0, 1:] -= 1
            adj_frame_index = adj_frame_index.t().flatten()
            adj_frame_index = adj_frame_index.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1).expand(batch_size, -1, view_count,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).to(
                                             crossview_attention_index)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, adj_frame_index)

            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length*2, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1])
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 2, crossview_attention_index)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "b (t l) (v n) hw c -> (b t v) (l n hw) c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) (l n hw) c -> (b t v) l n hw c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, 1, :, :]

        elif self.crossview_attention_type == "full":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> bt (h v w) c",
                v=view_count, w=width)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "bt (h v w) c -> (bt v) (h w) c",
                v=view_count, w=width)

        elif self.crossview_attention_type == "rowwise":
            if crossview_attention_mask is not None:
                crossview_attention_mask = crossview_attention_mask\
                    .repeat_interleave(width, 2)\
                    .repeat_interleave(width, 1)\
                    .repeat_interleave(sequence_length*height, 0)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> (bt h) (v w) c",
                w=width, v=view_count)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt h) (v w) c -> (bt v) (h w) c",
                bt=batch_size*sequence_length, v=view_count)

        else:
            raise f"Not support {self.crossview_attention_type}"

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            crossview_hidden_states.view(
                batch_size, sequence_length * view_count,
                *crossview_hidden_states.shape[1:]),
            image_only_indicator=disable_crossview).flatten(0, 1)

    def forward_temporal_block_and_mix_result(
        self, temporal_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        sequence_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, disable_temporal: torch.BoolTensor,
        temporal_attention_mask: torch.Tensor = None,
    ):
        # `temporal_attention_mask`: caller passes a base mask of shape
        # (T_seq, T_seq) where T_seq is the post-rearrange sequence length
        # (T for "pointwise", T*W for "rowwise", T*HW for "full"). We expand
        # it to (post_batch, T_seq, T_seq) here to match diffusers' Attention
        # processor view expectation. See §12.3 of causal_ar_pivot.md.
        temporal_hidden_states = hidden_states + sequence_emb
        if self.temporal_attention_type == "full":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v) (t hw) c",
                b=batch_size, t=sequence_length)
            mask_expanded = self._expand_temporal_mask(
                temporal_attention_mask, post_batch=temporal_hidden_states.shape[0])
            temporal_hidden_states = temporal_block(
                temporal_hidden_states,
                self_attention_mask=mask_expanded)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v) (t hw) c -> (b t v) hw c",
                b=batch_size, t=sequence_length)
        elif self.temporal_attention_type == "rowwise":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) (h w) c -> (b v h) (t w) c",
                b=batch_size, v=view_count, w=width)
            mask_expanded = self._expand_temporal_mask(
                temporal_attention_mask, post_batch=temporal_hidden_states.shape[0])
            temporal_hidden_states = temporal_block(
                temporal_hidden_states,
                self_attention_mask=mask_expanded)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v h) (t w) c -> (b t v) (h w) c",
                b=batch_size, v=view_count, w=width)
        else:
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v hw) t c",
                b=batch_size, t=sequence_length)
            mask_expanded = self._expand_temporal_mask(
                temporal_attention_mask, post_batch=temporal_hidden_states.shape[0])
            temporal_hidden_states = temporal_block(
                temporal_hidden_states,
                self_attention_mask=mask_expanded)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v hw) t c -> (b t v) hw c",
                b=batch_size, v=view_count, t=sequence_length)

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            temporal_hidden_states.view(
                batch_size, sequence_length * view_count,
                *temporal_hidden_states.shape[1:]),
            image_only_indicator=disable_temporal).flatten(0, 1)

    @staticmethod
    def _expand_temporal_mask(mask, post_batch):
        """Expand a base mask of shape (T_seq, T_seq) to (post_batch, T_seq, T_seq)
        as required by diffusers Attention processor's prepare_attention_mask +
        view path. mask=None passes through unchanged.

        BlockMask (FlexAttention) is also passed through unchanged because it
        is constructed with B=H=None and broadcasts automatically; see
        `dwm_pano/models/flex_block_causal.py`.
        """
        if mask is None:
            return None
        from torch.nn.attention.flex_attention import BlockMask
        if isinstance(mask, BlockMask):
            return mask
        if mask.dim() == 2:
            return mask.unsqueeze(0).expand(post_batch, -1, -1)
        return mask

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        frustum_bev_residuals: torch.Tensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_image_tensor: torch.Tensor = None,
        disable_crossview: torch.BoolTensor = None,
        disable_temporal: torch.BoolTensor = None,
        crossview_attention_mask: torch.Tensor = None,
        crossview_attention_index: torch.Tensor = None,
        temporal_attention_mask: torch.Tensor = None,
        camera_intrinsics: torch.Tensor = None,
        camera_transforms: torch.Tensor = None,
        camera_intrinsics_norm: torch.Tensor = None,
        camera2referego: torch.Tensor = None,
        pano_camera_translations: torch.Tensor = None,
        added_time_ids: torch.Tensor = None,
        noise: torch.Tensor = None,
        return_dict: bool = False
    ):
        """`temporal_attention_mask`: optional additive mask passed through to
        `forward_temporal_block_and_mix_result` -> `temporal_block.forward(...)`
        -> diffusers Attention. Used for causal AR retrofit (§12.3 of
        causal_ar_pivot.md). Shape depends on `temporal_attention_type`:
          - "pointwise" (default in our DFoT config): (T, T) lower-triangular
          - "rowwise": (T*W, T*W) block-causal
          - "full":    (T*HW, T*HW) block-causal
        Caller is responsible for the right shape; pipeline construction logic
        in `dwm_pano/pipelines/ctsd_pano.py` handles this. mask=None preserves
        the original bidirectional behavior bit-exactly.
        """
        should_add_dim = len(sample.shape) < 6
        if should_add_dim:
            sample = sample.unsqueeze(2)
            timestep = timestep.unsqueeze(2)
            if condition_image_tensor is not None:
                condition_image_tensor = condition_image_tensor.unsqueeze(2)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(2)
            if disable_temporal is not None:
                disable_temporal = disable_temporal.unsqueeze(2)
            if pooled_projections is not None:
                pooled_projections = pooled_projections.unsqueeze(2)

        hidden_states = sample
        batch_size, sequence_length, view_count, _, height, width = \
            hidden_states.shape

        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        self.view_count = view_count
        self.width = width

        hidden_states = hidden_states.flatten(0, 2)     # [b, 16, 32, 56]
        pooled_projections = pooled_projections.flatten(0, 2)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 2)


        hidden_states = self.pos_embed(hidden_states)   # [b, 448, 1536]    
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states)    # [b, 154, 1536]
        if self.mask_module is not None and noise is not None:
            y_t = encoder_hidden_states
            y_t = einops.rearrange(
                encoder_hidden_states, 
                "(b t v) s c -> b v t s c", t=sequence_length, v=view_count)
            y_lens = [y_t.shape[1]] * y_t.shape[0]

        temb = self.time_text_embed(timestep.flatten(), pooled_projections)

        view_cam_emb = 0
        if self.perspective_modeling_type == "implicit":
            view_emb = self.view_cam_proj(added_time_ids.flatten())\
                .to(dtype=hidden_states.dtype)
            view_cam_emb = self.view_embedding(
                view_emb.view(batch_size * sequence_length * view_count, -1))\
                .unsqueeze(1)
        elif self.perspective_modeling_type == "explicit":
            camera_intrinsics_norm = camera_intrinsics_norm.clone()
            camera_intrinsics_norm[..., 0, 0] = \
                camera_intrinsics_norm[..., 0, 0] * width
            camera_intrinsics_norm[..., 1, 1] = \
                camera_intrinsics_norm[..., 1, 1] * height
            camera_intrinsics_norm[..., 0, 2] = \
                camera_intrinsics_norm[..., 0, 2] * width
            camera_intrinsics_norm[..., 1, 2] = \
                camera_intrinsics_norm[..., 1, 2] * height

            if self.pano_equirect_rays:
                # Pano halves are equirectangular, not pinhole: derive the ray
                # field analytically and ignore the pseudo K / virtual
                # extrinsics entirely. Those virtual extrinsics are static per
                # half (pure yaw, zero translation), so nothing time-varying is
                # lost by bypassing them.
                rays_o, rays_d = get_pano_equirect_rays(
                    view_count, (height, width),
                    device=hidden_states.device, dtype=hidden_states.dtype)
                # (b, t, v) flattens v-fastest, so tiling matches that order.
                repeat_count = batch_size * sequence_length
                rays_o = rays_o.repeat(repeat_count, 1)
                rays_d = rays_d.repeat(repeat_count, 1, 1, 1)
                if self.pano_plucker_rays:
                    if pano_camera_translations is None:
                        raise ValueError(
                            "pano_plucker_rays needs pano_camera_translations "
                            "(the real six-camera positions); the pipeline "
                            "must pass it through. Refusing to fall back to "
                            "the ego origin, which would make every moment "
                            "identically zero.")
                    if self._pano_column_camera_index is None or \
                            self._pano_column_camera_index.shape != \
                            (view_count, width):
                        self._pano_column_camera_index = \
                            get_pano_column_camera_index(view_count, width)
                    # Per-pixel moments replace the (constant) origin encoding;
                    # channel count is unchanged, so `proj` stays compatible.
                    rays_o = get_pano_plucker_moments(
                        rays_d, pano_camera_translations,
                        self._pano_column_camera_index)
            else:
                rays_o, rays_d = get_rays(
                    camera_intrinsics_norm.flatten(0, 2),
                    camera2referego.flatten(0, 2),
                    (height, width)
                )
            raymap = self.rayencoder(rays_o, rays_d)
            view_cam_emb = raymap.flatten(1, 2)

        condition_residuals = None if \
            self.condition_image_adapter is None or \
            condition_image_tensor is None else \
            self.condition_image_adapter(condition_image_tensor)

        if self.mask_module is not None and noise is not None:
            hidden_states = einops.rearrange(hidden_states, "(b t v) hw c -> (b v) t hw c", 
                t=sequence_length, v=view_count)
            noise = einops.rearrange(noise, "b t v c h w -> (b v) c t h w")
            hidden_states, mask_metas, condition_residuals = self.mask_module.random_masking(
                hidden_states, noise, height, width, timestep, condition_residuals=condition_residuals)
            hidden_states = einops.rearrange(hidden_states, "(b v) t hw c -> (b t v) hw c", 
                t=sequence_length, v=view_count)
            ori_width = width
            width = int(width*(1-self.mask_module.mask_ratio))

        for i, block in enumerate(self.transformer_blocks):
            if self.mask_module is not None and noise is not None and\
                self.mask_module.is_first_decoder_layer(
                    i, len(self.transformer_blocks)):
                hidden_states = einops.rearrange(hidden_states, 
                    "(b t v) hw c -> (b v t) hw c", t=sequence_length, v=view_count)
                temb_v_first = einops.rearrange(temb, 
                    "(b t v) c -> (b v t) c", t=sequence_length, v=view_count)
                hidden_states = self.mask_module.mask_reconstruction(
                    hidden_states, mask_metas, ori_shape=(batch_size*view_count, 
                    sequence_length, height, ori_width), y_t=y_t, y_lens=y_lens, 
                    temb=temb_v_first)
                hidden_states = einops.rearrange(hidden_states, 
                    "(b v t) hw c -> (b t v) hw c", t=sequence_length, v=view_count)
                width = int(width/(1-self.mask_module.mask_ratio))

            if condition_residuals is not None and len(condition_residuals) > 0:
                hidden_states = hidden_states + \
                    condition_residuals.pop(0).flatten(0, 2)\
                    .flatten(2).permute(0, 2, 1)    

            # text-spatio
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                encoder_hidden_states, hidden_states = \
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        use_reentrant=False
                    )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb
                )

            if encoder_hidden_states is not None:
                self.encoder_hidden_states_var = torch.var(encoder_hidden_states).item()

            # temporal
            if self.enable_temporal and i in self.temporal_block_layers:
                sequence_emb = torch\
                    .arange(sequence_length, device=hidden_states.device)\
                    .unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, view_count)
                sequence_emb = self.index_proj(sequence_emb.flatten())\
                    .to(dtype=hidden_states.dtype)
                sequence_emb = self.time_pos_embeds[
                    self.temporal_block_layers.index(i)](sequence_emb).unsqueeze(1)

                if self.enable_crossview and not self.disable_view_emb_on_temporal_module:
                    sequence_emb = sequence_emb + view_cam_emb

                if self.training and self.temporal_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_temporal_block_and_mix_result,
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states, sequence_emb,
                        batch_size, sequence_length, view_count, width,
                        disable_temporal, temporal_attention_mask,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_temporal_block_and_mix_result(
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states, sequence_emb,
                        batch_size, sequence_length, view_count, width,
                        disable_temporal,
                        temporal_attention_mask=temporal_attention_mask)

            # cross-view
            if self.enable_crossview and i in self.crossview_block_layers:
                view_emb = torch\
                    .arange(view_count, device=hidden_states.device)\
                    .unsqueeze(0).unsqueeze(0)\
                    .repeat(batch_size, sequence_length, 1)
                view_emb = self.index_proj(view_emb.flatten())\
                    .to(dtype=hidden_states.dtype)
                view_emb = self.view_pos_embeds[
                    self.crossview_block_layers.index(i)](view_emb).unsqueeze(1)

                view_emb = view_emb + view_cam_emb

                if self.training and self.crossview_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_crossview_block_and_mix_result,
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb, batch_size,
                        sequence_length, view_count, width, height, disable_crossview,
                        crossview_attention_mask,
                        crossview_attention_index,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_crossview_block_and_mix_result(
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(
                            i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb,
                        batch_size, sequence_length, view_count, width, height,
                        disable_crossview, crossview_attention_mask,
                        crossview_attention_index)

        # debug code
        self.hidden_states_var = hidden_states.var().item()
        if self.enable_temporal:
            self.temporal_embedding_var = sequence_emb.var().item()

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                batch_size, sequence_length, view_count,
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        result = [output,]
        if should_add_dim:
            output = output.squeeze(2)
        if return_dict:
            return {
                "noise_pred": output,
            }
        return result, _, _
