"""Scene construction for the v3 GS renders and the lateral-shift evaluation (2026-10-19, report section 40).

The v2 review scene (export_gs_v2_clips.py: frame t's Gaussians with the depth-edge filter and the max-scale cap,
plus the static Gaussians of nearby frames as 4D fill) with three new switches, all off by default so that the
default scene is exactly v2's:

* ownership (B1, after Instant NuRec's frustum-ownership pruning): a fill Gaussian from nearby frame s is kept only
  if no other fill frame sees it from clearly closer range, D_s <= min_j D_j + own_tol_m + own_tol_rel * min_j D_j,
  where D_j is the distance to frame j's rig centre if the Gaussian lies inside frame j's +/-16.7 deg band and
  within static_range of it, else infinity.  With own_include_current frame t also competes (off by default:
  frame t is composited on top anyway, and its own disocclusion holes need the fill).
* da3_relabel (C1): every frame's Gaussians are moved along their exact rays to the Depth Anything 3 range
  (dwm_pano_4d/da3_depth.py, windows (t - gap, t, t + gap) of VAE-decoded frames, da3_scale = ground by
  default, section 40.1); pixels DA3 calls sky or without a DA3 range below far_cut keep the decoder geometry.
* sky_layer (B2): the Gaussians of pixels DA3 calls sky are dropped; renderers then take the sky colour from the
  frame's own decoded pano along the ray direction (sky is at infinity, so a pure translation does not move it).
* box_reading (section 44): "legacy" reads gt_boxes as (x, y, z, dx, dy, dz, yaw) (export_gs_4d_demo.boxes_in_ego);
  the 12 Hz pkl stores mmdet3d v0.x boxes (x, y, z, w, l, h, -yaw - pi/2), which "mmdet3d_v0x" reads correctly.  The
  two agree only for boxes aligned with the LiDAR axes; a diagonal box is mirrored and covers about half the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

from dwm_pano.export_gs_4d_demo import boxes_in_ego, ego_to_world, inside_boxes, transform_gaussians
from dwm_pano.export_gs_v2_clips import edge_mask
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD

RIG_HEIGHT_M = 1.6


@dataclass
class SceneOptions:
    edge_rel_h: float = 0.1
    edge_rel_v: float = 0.3
    edge_dilate: int = 1
    max_scale: float = 1.0
    window: tuple = (-20, -15, -10, -5, 5, 10)
    static_range: float = 40.0
    static_min_opacity: float = 0.3
    box_margin: float = 0.3
    ownership: bool = False
    own_include_current: bool = False
    own_tol_m: float = 0.5
    own_tol_rel: float = 0.1
    da3_relabel: bool = False
    sky_layer: bool = False
    far_cut: float = 99.0
    da3_gap: int = 4
    da3_scale: str = "ground"                                           # section 40.1
    da3_mode: str = "ray"                                               # "scale" = C1b, section 40.3
    # 2026-10-25 (round 2b): "global" = one DA3/decoder median range ratio per frame, the decoder keeps its local
    # depth; "blend" = per pixel, decoder range below blend_near_m, DA3 range beyond blend_far_m, linear in between
    # (weight from the DA3 range); both move Gaussians like "scale" (rescale_gaussians)
    blend_near_m: float = 10.0
    blend_far_m: float = 30.0
    drop_bottom_rows: int = 0                                           # section 41.6: C2's band-edge line
    box_reading: str = "legacy"                                         # section 44: "legacy" | "mmdet3d_v0x"
    min_opacity: float = 0.0                                            # 2026-10-24 round 2 (E): prune faint Gaussians
    # 2026-10-26 (GS_CLIPDEPTH_PLAN_20261026.md): da3_window "clip" = one DA3 pass over all frames of the clip with
    # one ground scale (V1; da3_chunk N > 0 = chunks of N frames overlapping by N // 2, chained by their overlap and
    # each frame read from the chunk it is most central in); da3_fuse = static depth fusion over fuse_offsets with
    # the relative occlusion tolerance fuse_tol_rel (V2, dwm_pano/da3_fusion.py); "frame" = the windows above
    da3_window: str = "frame"
    da3_chunk: int = 0
    da3_fuse: bool = False
    fuse_offsets: tuple = (-8, -6, -4, -2, 2, 4, 6, 8)
    fuse_tol_rel: float = 0.1
    # 2026-10-26 (GS_CLIPDEPTH_PLAN_20261026.md section 7): da3_mode "lowfreq" = the decoder range times
    # exp(low-passed log(DA3 / decoder)), lf_filter "gauss" (lf_sigma_px) or "guided" (lf_radius_px, lf_eps; guide =
    # log decoder range), dwm_pano/da3_lowfreq.py; Gaussians moved like "scale"
    lf_filter: str = "gauss"
    lf_sigma_px: float = 32.0
    lf_radius_px: int = 32
    lf_eps: float = 0.01
    # 2026-10-30 (GS_FT3_PLAN_20261030.md): da3_mode "bounded" = da3_depth.bounded_range, DA3 with the decoder's
    # residual bounded to a factor exp(+-bound_c) (the geometry pipeline_relabel.py trains with); moved like "scale"
    bound_c: float = 0.0
    # 2026-10-30 (GS_FT3_PLAN_20261030.md): put every centre back on its pixel ray (origin + direction * decoder
    # range) before any relabel, dropping the decoder's xyz offset (pipeline_relabel.py no_offset)
    no_offset: bool = False
    # 2026-10-30 (section 6): > 0 = keep the offset's across-ray component bounded to this many decoder pixels
    # (da3_depth.plane_offsets, pipeline_relabel.py plane_offset_px)
    plane_offset_px: float = 0.0
    # 2026-10-31 (PanDWM-d2-refiner-20261021/FUSE_PLAN_20261031.md): colour every Gaussian with its pixel of the
    # SD3.5-VAE-decoded halves (the decoder grid is the halves' pixel grid) instead of the decoder's rgb head
    vae_colour: bool = False
    # 2026-11-01 (PanDWM-d2-refiner-20261021/CONSIST_PLAN_20261101.md section 5.2): coverage of the sky and of far
    # content.  sky_dome_m > 0: the Gaussians of the pixels DA3 calls sky (prob >= 0.3, mask dilated by sky_dome_dilate
    # px so silhouettes against the sky stay covered) become an opaque dome at that radius along their rays (EVolSplat
    # / WonderWorld style background layer; isotropic scale = radius * pixel pitch * sky_dome_scale), exempt from the
    # edge and scale filters; far_scale_exempt_m > 0: max_scale only caps Gaussians closer than that range
    sky_dome_m: float = 0.0
    sky_dome_scale: float = 1.0
    sky_dome_dilate: int = 2
    far_scale_exempt_m: float = 0.0
    # 2026-11-01 (CONSIST_PLAN_20261101.md section 5.3): seam_bridge = K > 0 fills the parallax wedge at every camera
    # seam of the panorama.  Pixels on the two sides of a seam are rays from DIFFERENT camera centres, so the scene
    # between the last ray of camera A and the first ray of camera B is in no pixel; seen from any other camera it
    # renders as an empty vertical stripe (the main hole source of the single-frame recon, alpha < 0.02 on 12 % of the
    # band).  K Gaussians per row are placed on the segment between the two kept seam Gaussians, colour and opacity
    # interpolated, isotropic scale seam_bridge_scale * segment length / (K + 1); segments longer than
    # seam_bridge_max_rel * range are left open (depth discontinuities)
    seam_bridge: int = 0
    seam_bridge_scale: float = 0.8
    seam_bridge_max_rel: float = 0.5
    # the endpoints are the nearest KEPT Gaussians within seam_bridge_search columns on each side (the columns right
    # at a seam are usually removed by the depth-edge filter, coverage_diag 787349)
    seam_bridge_search: int = 1
    # "current": the bridges join frame t's Gaussians (depth-sorted with them; GS-Dev shift eval 787462: they cut in
    # front of visible content, 0 m FVD 101 -> 138); "under": they are kept apart (SceneBuilder.under(t)) and the
    # renderers composite them UNDER the whole frame (current over fill over under), so they only show through holes
    seam_bridge_layer: str = "current"
    # 2026-11-01 (CONSIST_PLAN_20261101.md 5.11): edge_extend = N > 0 adds, in the under layer, the background of every
    # horizontal depth edge (range ratio > 1 + edge_extend_tol between neighbouring columns) continued N decoder
    # pixels behind the foreground: copies of the background Gaussian on the rays of the foreground pixels, at the
    # background range (3D Photo Inpainting / WonderWorld style layered background); a copy stops where the pixel is
    # no longer foreground.  They only show where a camera move uncovers the background.
    edge_extend: int = 0
    edge_extend_tol: float = 0.2
    edge_extend_search: int = 6
    # 2026-11-01 (CONSIST_PLAN_20261101.md 5.12): sky_dome_layer "under" keeps the sky pixels' own Gaussians and puts
    # the dome in the under layer (the "replace" dome moved misclassified pixels to 300 m: AbsRel 0.30 -> 0.44);
    # band_extend = N > 0 continues the top and the lowest kept rows of the band N rows beyond it, in the under layer:
    # above, a vertical surface at the source's horizontal distance; below, the ground plane when the source lies on
    # the ground (z < band_ground_z in ego), else again a vertical surface
    sky_dome_layer: str = "replace"
    band_extend: int = 0
    band_ground_z: float = 0.5
    notes: dict = field(default_factory=dict)


def boxes_in_ego_v0x(info):
    """gt_boxes as the 12 Hz pkl stores them, mmdet3d v0.x (x, y, z, w, l, h, yaw') with yaw' = -yaw - pi/2 and z the
    box centre: centres (N, 3), rotations (N, 3, 3) whose first axis is the heading, half sizes (l, w, h) / 2, in ego
    (same layout as export_gs_4d_demo.boxes_in_ego, for its inside_boxes)."""
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    boxes = boxes.reshape(boxes.shape[0], -1)[:, :7]
    if boxes.shape[1] != 7:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
    q = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    centres = boxes[:, :3] @ q.T + t
    heading = -boxes[:, 6] - np.pi / 2
    cos, sin = np.cos(heading), np.sin(heading)
    rz = np.zeros((boxes.shape[0], 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = cos, -sin, sin, cos, 1
    half = np.stack([boxes[:, 4], boxes[:, 3], boxes[:, 5]], axis=1) / 2.0
    return centres, q[None] @ rz, half


class SceneBuilder:
    def __init__(self, pipe, batch, infos, latents, opts: SceneOptions, device, da3=None):
        self.pipe, self.batch, self.infos, self.latents, self.opts, self.device = \
            pipe, batch, infos, latents, opts, device
        self.da3 = da3
        if (opts.da3_relabel or opts.sky_layer) and da3 is None:
            raise ValueError("da3_relabel / sky_layer need a Da3Ranger")
        pipe._prepare_nuscenes6_supervision(batch)
        self.world = ego_to_world(infos)                               # (T, 4, 4) to frame 0's ego, numpy
        self.frames = latents.shape[0]
        self.content = pipe.pixel_content_ray_valid.reshape(-1)
        self.grid_shape = tuple(pipe.pixel_content_ray_valid.shape)     # (2, H, W)
        self._decoded, self._static, self._canvas, self._da3 = {}, {}, {}, {}
        self.colours = {}
        # 2026-10-31 (FUSE_PLAN_20261031.md): optional {t: (2, H, W) bool} over the decoder grid; decode(t) keeps
        # only those pixels' Gaussians (a completion scene contributes where the base scene is missing)
        self.pixel_keep = None
        # 2026-10-31 (FUSE_PLAN_20261031.md section 5.8): optional {t: (2, H, W)} range factors applied after the
        # relabel (rescale_gaussians about the pixel origins); _ranges[t] keeps the ranges before that factor
        self.range_scale = None
        self._ranges = {}
        self._under = {}
        self.stats = {"da3_scale": [], "da3_ground_over_metric": [], "sky_fraction": [],
                      "da3_use_fraction": [], "fill_kept_fraction": []}

    # ------------------------------------------------------------------
    def canvas(self, t):
        """Frame t's VAE-decoded pano as the (3, H, 4114) unpadded stitch canvas in [0, 1]."""
        if t not in self._canvas:
            lat = self.latents[t:t + 1].to(self.device, dtype=self.pipe.latent_directions.dtype)
            halves = self.pipe.decode_half_latents(lat).float()          # (1, 2, 3, H, W)
            self._canvas[t] = self.pipe._unpadded_stitch_canvas(halves)[0].clamp(0, 1)
            if len(self._canvas) > 3 * len(self.opts.window) + 8:     # never evict the frame just decoded
                self._canvas.pop(min(k for k in self._canvas if k != t))
        return self._canvas[t]

    def frame_boxes(self, s):
        if self.opts.box_reading not in ("legacy", "mmdet3d_v0x"):
            raise ValueError(f"box_reading {self.opts.box_reading!r}")
        return (boxes_in_ego if self.opts.box_reading == "legacy" else boxes_in_ego_v0x)(self.infos[s])

    def _clip_da3_crops(self):
        """Clip-level DA3 (V1): unscaled crop depth / sky (F * 6, h, w), crop intrinsics (6, 3, 3), and the chunking
        actually used.  One pass when da3_chunk is 0 or covers the clip."""
        frames, n = self.frames, self.opts.da3_chunk
        canvases = torch.stack([self.canvas(f) for f in range(frames)])
        e2w = torch.as_tensor(self.world, dtype=torch.float64)
        if n <= 0 or n >= frames:
            res = self.da3.frame_ranges(self.pipe, canvases, e2w)
            return res["depth_crop"], res["sky_crop"], res["K_crop"][:6], [(0, frames)]
        step = max(1, n // 2)
        starts = list(range(0, frames - n + 1, step))
        if starts[-1] + n < frames:
            starts.append(frames - n)
        depth, sky, owner, prev = [None] * frames, [None] * frames, [None] * frames, None
        owned = self.da3.owned_masks(self.pipe)
        for a in starts:
            res = self.da3.frame_ranges(self.pipe, canvases[a:a + n], e2w[a:a + n])
            d = res["depth_crop"].view(n, 6, *res["depth_crop"].shape[-2:])
            s = res["sky_crop"].view_as(d)
            if prev is not None:                                    # chain to the previous chunk on the overlap
                pa, pd = prev
                ov = [f for f in range(a, min(pa + n, a + n))]
                ref = pd[[f - pa for f in ov]]
                cur = d[[f - a for f in ov]]
                ok = owned[None] & (s[[f - a for f in ov]] < 0.3) & (ref > 1e-3) & (cur > 1e-3)
                d = d * (ref[ok] / cur[ok]).median()
            for f in range(a, a + n):
                centrality = abs(f - (a + (n - 1) / 2.0))
                if owner[f] is None or centrality < owner[f]:
                    owner[f], depth[f], sky[f] = centrality, d[f - a], s[f - a]
            prev = (a, d)
        return (torch.stack(depth).flatten(0, 1), torch.stack(sky).flatten(0, 1), res["K_crop"][:6],
                [(a, a + n) for a in starts])

    def _clip_da3(self):
        """Fill self._da3 for every frame from one clip-level DA3 solution (V1), fused when da3_fuse (V2)."""
        if self.opts.da3_scale != "ground":
            raise ValueError("da3_window clip supports da3_scale ground only")
        pipe, frames = self.pipe, self.frames
        depth, sky, K, chunks = self._clip_da3_crops()
        rig = self.da3.rig(pipe)
        h, w = depth.shape[-2:]
        scale = self.da3.ground_scale(rig, K.repeat(frames, 1, 1), depth, sky < 0.3, frames)
        depth_m = (depth * scale).view(frames, 6, h, w)
        self.stats["da3_scale"].append(scale)
        self.stats.setdefault("da3_clip_chunks", []).append(len(chunks))
        if self.opts.da3_fuse:
            from dwm_pano.da3_fusion import ClipGeometry, fuse_static_depth
            geo = ClipGeometry(depth_m, sky.view(frames, 6, h, w), K, self.da3.owned_masks(pipe),
                               rig["cam2ego"].to(depth.device), torch.as_tensor(self.world, device=depth.device),
                               [self.frame_boxes(s) for s in range(frames)], self.opts.far_cut, self.opts.box_margin)
            depth_m, fstats = fuse_static_depth(geo, self.opts.fuse_offsets, self.opts.fuse_tol_rel)
            for k, v in fstats.items():
                self.stats.setdefault(k, []).append(v)
            del geo
        ranges = self.da3.ranges_from_crops(pipe, depth_m.flatten(0, 1), frames)
        sky_grid = self.da3.to_decoder_grid(pipe, sky, frames)
        for f in range(frames):
            self._da3[f] = {"range": ranges[f], "sky": sky_grid[f]}

    def da3_maps(self, t):
        if self.opts.da3_window == "clip":
            if not self._da3:
                self._clip_da3()
            return self._da3[t]
        if self.opts.da3_window != "frame":
            raise ValueError(f"da3_window {self.opts.da3_window!r}")
        if t not in self._da3:
            gap = self.opts.da3_gap
            window = sorted({max(0, t - gap), t, min(self.frames - 1, t + gap)})
            canvases = torch.stack([self.canvas(f) for f in window])
            e2w = torch.as_tensor(self.world[window], dtype=torch.float64)
            res = self.da3.frame_ranges(self.pipe, canvases, e2w)
            i = window.index(t)
            self._da3[t] = {"range": res[f"range_{self.opts.da3_scale}"][i], "sky": res["sky"][i]}
            self.stats["da3_scale"].append(res[f"scale_{self.opts.da3_scale}"])
            self.stats["da3_ground_over_metric"].append(res["ground_over_metric"])
        return self._da3[t]

    def decode(self, t):
        """Frame t's Gaussians (filtered, in frame t's ego) and its flat keep mask over the decoder grid."""
        if t in self._decoded:
            return self._decoded[t]
        pipe, opts = self.pipe, self.opts
        sel = torch.tensor([t], dtype=torch.long)
        lo, po = pipe._frame_ray_origins(self.batch, sel)
        lat = self.latents[t:t + 1].to(self.device, dtype=pipe.latent_directions.dtype)
        out = pipe.decoder_wrapper(lat, pipe.latent_directions.unsqueeze(0), pipe.pixel_directions.unsqueeze(0),
                                   None, previous_latents=None, temporal_context_mask=0.0,
                                   latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)
        g, rng = out[0][0].float(), out[1][0].float()
        if opts.vae_colour:
            halves = pipe.decode_half_latents(lat)[0].float()              # (2, 3, H, W), the decoder grid
            if tuple(halves.shape[-2:]) != tuple(rng.shape[-2:]):
                raise RuntimeError(f"VAE halves {tuple(halves.shape)} vs decoder grid {tuple(rng.shape)}")
            g = torch.cat([g[:, 0:3], halves.permute(0, 2, 3, 1).reshape(-1, 3).to(g.device), g[:, 6:]], dim=-1)
        if opts.no_offset or opts.plane_offset_px > 0:
            ray_o = po.reshape(-1, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(g[:, 0:3])
            ray_d = pipe.pixel_directions.reshape(-1, 3).to(g.device, torch.float32)
            if opts.no_offset:
                g = torch.cat([ray_o + ray_d * rng.reshape(-1, 1), g[:, 3:]], dim=-1)
            else:
                from dwm_pano_4d.da3_depth import plane_offsets
                g = plane_offsets(g, ray_o, ray_d, rng, opts.plane_offset_px)
        if t not in self.colours:                                      # v2's --bg-fill sky / ground colours
            rgb = g[:, 3:6].reshape(*rng.shape, 3)
            self.colours[t] = (rgb[:, :8].reshape(-1, 3).mean(0), rgb[:, -8:].reshape(-1, 3).mean(0))
        keep = self.content.to(g.device).clone()
        if self.pixel_keep is not None:
            keep &= self.pixel_keep[t].to(g.device).reshape(-1)
        if opts.drop_bottom_rows > 0:          # the lowest decoder rows of both halves (the band's lower edge)
            rows = torch.arange(self.grid_shape[1], device=keep.device)
            bottom = rows >= self.grid_shape[1] - opts.drop_bottom_rows
            keep &= ~bottom[None, :, None].expand(self.grid_shape).reshape(-1)
        if opts.da3_relabel or opts.sky_layer:
            maps = self.da3_maps(t)
            sky = maps["sky"].float() >= 0.3
            self.stats["sky_fraction"].append(float(sky.reshape(-1)[self.content].float().mean()))
            if opts.da3_relabel:
                r = maps["range"].float()
                valid = self.content.view(self.grid_shape) & (r > 0.5) & (r < opts.far_cut) & ~sky
                dirs = pipe.pixel_directions.reshape(-1, 3).float()
                origins = po.reshape(-1, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(dirs)
                if opts.da3_mode in ("scale", "global", "blend", "lowfreq", "bounded"):
                    from dwm_pano_4d.da3_depth import rescale_gaussians
                    target = r
                    if opts.da3_mode == "global":
                        both = valid & (rng > 0.5)
                        ratio = (r[both] / rng[both]).median() if bool(both.any()) else rng.new_tensor(1.0)
                        target = rng * ratio
                        valid = self.content.view(self.grid_shape) & (rng > 0)
                        self.stats.setdefault("global_ratio", []).append(float(ratio))
                    elif opts.da3_mode == "blend":
                        w = ((r - opts.blend_near_m) / max(opts.blend_far_m - opts.blend_near_m, 1e-3)).clamp(0, 1)
                        target = w * r + (1 - w) * rng
                    elif opts.da3_mode == "bounded":
                        from dwm_pano_4d.da3_depth import bounded_range
                        target = bounded_range(rng, r, opts.bound_c)
                    elif opts.da3_mode == "lowfreq":
                        from dwm_pano.da3_lowfreq import lowfreq_log_ratio
                        both = valid & (rng > 0.5)
                        log_rng = rng.clamp_min(1e-3).log()
                        rho = r.clamp_min(1e-3).log() - log_rng
                        rho_s, lf = lowfreq_log_ratio(rho, both, log_rng, opts.lf_filter, opts.lf_sigma_px,
                                                      opts.lf_radius_px, opts.lf_eps)
                        target = rng * rho_s.exp()
                        valid = self.content.view(self.grid_shape) & (rng > 0)
                        lf["lf_mean_abs_log_corr"] = float(rho_s[valid].abs().mean())
                        lf["lf_mean_abs_log_resid"] = float((rho - rho_s)[both].abs().mean()) if bool(both.any()) \
                            else 0.0
                        for k, v in lf.items():
                            self.stats.setdefault(k, []).append(v)
                    g, rng = rescale_gaussians(g, rng, target, valid, origins)
                else:
                    from dwm_pano_4d.dac_depth import relabel_gaussians
                    g, rng = relabel_gaussians(g, rng, r, valid, origins, dirs)
                self.stats["da3_use_fraction"].append(float(valid.reshape(-1)[self.content].float().mean()))
            if opts.sky_layer:
                keep &= ~sky.reshape(-1)
        self._ranges[t] = rng
        if self.range_scale is not None and t in self.range_scale:
            from dwm_pano_4d.da3_depth import rescale_gaussians
            org = po.reshape(-1, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(g[:, 0:3])
            ok = self.content.to(g.device).view(self.grid_shape) & (rng > 0)
            g, rng = rescale_gaussians(g, rng, rng * self.range_scale[t].to(rng.device, rng.dtype), ok, org)
        dome = None
        if opts.sky_dome_m > 0:
            if not (opts.da3_relabel or opts.sky_layer):
                raise ValueError("sky_dome_m needs the DA3 sky map (da3_relabel)")
            sky_grid = (self.da3_maps(t)["sky"].float() >= 0.3).float().to(g.device)
            if opts.sky_dome_dilate > 0:
                k = 2 * opts.sky_dome_dilate + 1
                sky_grid = F.max_pool2d(sky_grid[:, None], k, stride=1, padding=opts.sky_dome_dilate)[:, 0]
            in_dome = (sky_grid > 0.5).reshape(-1) & self.content.to(g.device)
            sky_core = (self.da3_maps(t)["sky"].float() >= 0.3).reshape(-1).to(g.device)
            dirs = pipe.pixel_directions.reshape(-1, 3).to(g.device, torch.float32)
            dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            org = po.reshape(-1, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(dirs)
            dome = g[in_dome].clone()
            dome[:, 0:3] = org[in_dome] + dirs[in_dome] * opts.sky_dome_m
            dome[:, 6] = 1.0
            dome[:, 7:11] = dome.new_tensor([1.0, 0.0, 0.0, 0.0])
            dome[:, 11:14] = opts.sky_dome_m * self._pixel_pitch() * opts.sky_dome_scale
            if opts.sky_dome_layer == "replace":
                keep &= ~sky_core
            elif opts.sky_dome_layer != "under":
                raise ValueError(f"sky_dome_layer {opts.sky_dome_layer!r}")
            self.stats.setdefault("dome_fraction", []).append(float(in_dome[self.content.to(g.device)].float().mean()))
        if opts.edge_rel_h > 0 or opts.edge_rel_v > 0:
            keep &= ~edge_mask(rng, opts.edge_rel_h, opts.edge_rel_v, opts.edge_dilate).to(g.device)
        bridge = self._seam_bridge(g, keep, rng) if opts.seam_bridge > 0 else None
        ext = self._edge_extend(g, keep, rng, po) if opts.edge_extend > 0 else None      # always under
        band = self._band_extend(g, keep, rng, po) if opts.band_extend > 0 else None
        g, rng_kept = g[keep], rng.reshape(-1)[keep.to(rng.device)].to(g.device)
        small = g[:, 11:14].amax(dim=-1) <= opts.max_scale
        if opts.far_scale_exempt_m > 0:
            small |= rng_kept >= opts.far_scale_exempt_m
        g = g[small]
        under = [x for x in (ext, band) if x is not None]
        if dome is not None and opts.sky_dome_layer == "under":
            under.append(dome)
            dome = None
        if dome is not None:
            g = torch.cat([g, dome])
        if bridge is not None:
            if opts.seam_bridge_layer == "under":
                under.append(bridge)
            elif opts.seam_bridge_layer == "current":
                g = torch.cat([g, bridge])
            else:
                raise ValueError(f"seam_bridge_layer {opts.seam_bridge_layer!r}")
        if under:
            self._under[t] = torch.cat(under)
        if opts.min_opacity > 0.0:
            g = g[g[:, 6] >= opts.min_opacity]
        self._decoded[t] = g
        if len(self._decoded) > len(opts.window) + 4:
            old = min(self._decoded)
            self._decoded.pop(old)
            self._ranges.pop(old, None)
            self._under.pop(old, None)
        return g

    def static_world(self, s):
        """Static Gaussians of frame s (outside the padded boxes, near, opaque) in frame 0's ego."""
        if s not in self._static:
            opts = self.opts
            g = self.decode(s)
            if opts.box_reading not in ("legacy", "mmdet3d_v0x"):
                raise ValueError(f"box_reading {opts.box_reading!r}")
            c, r, h = (boxes_in_ego if opts.box_reading == "legacy" else boxes_in_ego_v0x)(self.infos[s])
            dynamic = inside_boxes(g[:, 0:3], c, r, h, opts.box_margin)
            keep = (~dynamic & (torch.linalg.norm(g[:, 0:2], dim=-1) <= opts.static_range)
                    & (g[:, 6] >= opts.static_min_opacity))
            self._static[s] = transform_gaussians(g[keep], self.world[s])
        return self._static[s]

    def _band_distance(self, means_world, j):
        """Distance from frame j's rig centre, infinity outside its +/-16.7 deg band or static range."""
        m = torch.as_tensor(self.world[j], device=means_world.device, dtype=means_world.dtype)
        local = (means_world - m[:3, 3]) @ m[:3, :3]                 # world -> frame j ego
        horizontal = torch.linalg.norm(local[:, 0:2], dim=-1)
        elevation = torch.atan2(local[:, 2] - RIG_HEIGHT_M, horizontal.clamp_min(1e-3))
        distance = torch.linalg.norm(local - local.new_tensor([0.0, 0.0, RIG_HEIGHT_M]), dim=-1)
        inside = (elevation.abs() <= PANO_LAT_HALF_RAD) & (horizontal <= self.opts.static_range)
        return torch.where(inside, distance, torch.full_like(distance, math.inf))

    def fill(self, t):
        """4D fill: static Gaussians of the window frames, in frame t's ego (ownership-pruned if enabled)."""
        opts = self.opts
        frames = [t + o for o in opts.window if 0 <= t + o < self.frames]
        for s in list(self._static):
            if s < t + min(opts.window):
                del self._static[s]
        if not frames:
            return self.decode(t)[:0]
        parts = [self.static_world(s) for s in frames]
        if opts.ownership:
            competitors = frames + ([t] if opts.own_include_current else [])
            kept, total = [], 0
            for s, g in zip(frames, parts):
                total += g.shape[0]
                if not g.shape[0]:
                    kept.append(g)
                    continue
                d = torch.stack([self._band_distance(g[:, 0:3], j) for j in competitors])
                d_own = d[competitors.index(s)]
                best = d.min(dim=0).values
                ok = torch.isfinite(d_own) & (d_own <= best + opts.own_tol_m + opts.own_tol_rel * best)
                kept.append(g[ok])
            self.stats["fill_kept_fraction"].append(sum(k.shape[0] for k in kept) / max(1, total))
            parts = kept
        to_t = torch.linalg.inv(torch.as_tensor(self.world[t], device=self.device, dtype=torch.float32))
        # 2026-10-30 (GS_NEARFIX_PLAN_20261030.md, mag_cull.py): each fill Gaussian's source rig centre in frame t's
        # ego, row-aligned with the returned tensor (read by eval_gs_shift --mag-cull; unused otherwise)
        rig = torch.tensor([0.0, 0.0, RIG_HEIGHT_M, 1.0], device=self.device, dtype=torch.float32)
        centres = [(to_t @ torch.as_tensor(self.world[s], device=self.device, dtype=torch.float32) @ rig)[:3]
                   .expand(g.shape[0], 3) for s, g in zip(frames, parts)]
        self.fill_sources = {t: torch.cat(centres)}
        return transform_gaussians(torch.cat(parts).float(), to_t)

    def under(self, t):
        """Frame t's under-layer Gaussians (seam bridges with seam_bridge_layer "under"; empty otherwise)."""
        g = self.decode(t)
        return self._under.get(t, g[:0])

    def _band_extend(self, g, keep, rng, po):
        """Under-layer continuation of the band beyond its top and bottom rows (SceneOptions.band_extend)."""
        opts, (v, h, w) = self.opts, self.grid_shape
        grid = g.reshape(v, h, w, -1)
        kgrid = keep.reshape(v, h, w).to(g.device)
        r = rng.reshape(v, h, w).to(g.device).float()
        dirs = self.pipe.pixel_directions.to(g.device, torch.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        org = po.reshape(v, h, w, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(dirs)
        el_all = torch.asin(dirs[..., 2].clamp(-1, 1))
        pitch = float((el_all[:, :-1] - el_all[:, 1:]).abs().median())          # rows are ordered by elevation
        top_down = bool((el_all[:, 0] > el_all[:, -1]).float().mean() > 0.5)    # row 0 is the top row
        rows = {"top": 0 if top_down else h - 1}
        low = h - 1 - opts.drop_bottom_rows if top_down else opts.drop_bottom_rows
        rows["bottom"] = low
        out = []
        for side, row in rows.items():
            ok = kgrid[:, row]                                                   # (v, w)
            vi, ci = torch.nonzero(ok, as_tuple=True)
            if not vi.numel():
                continue
            src = grid[vi, row, ci]
            d0 = dirs[vi, row, ci]
            o = org[vi, row, ci]
            r0 = r[vi, row, ci]
            el0 = torch.asin(d0[:, 2].clamp(-1, 1))
            az = torch.atan2(d0[:, 1], d0[:, 0])
            horiz = r0 * torch.cos(el0)
            ground = (src[:, 2] < opts.band_ground_z) if side == "bottom" else torch.zeros_like(r0, dtype=torch.bool)
            sign = 1.0 if side == "top" else -1.0
            for k in range(1, opts.band_extend + 1):
                el = el0 + sign * k * pitch
                d = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
                rk = horiz / torch.cos(el).clamp_min(1e-3)
                if side == "bottom":
                    hit = o[:, 2] / (-d[:, 2]).clamp_min(1e-3)                   # ground plane z = 0
                    rk = torch.where(ground & (d[:, 2] < -1e-3), hit, rk)
                cp = src.clone()
                cp[:, 0:3] = o + d * rk[:, None]
                cp[:, 11:14] = src[:, 11:14] * (rk / r0.clamp_min(1e-3))[:, None]
                out.append(cp)
        band = torch.cat(out) if out else g[:0]
        self.stats.setdefault("band_extend_gaussians", []).append(int(band.shape[0]))
        return band

    def _edge_extend(self, g, keep, rng, po):
        """Under-layer copies of the background across horizontal depth edges (SceneOptions.edge_extend).

        For an edge between columns c (foreground) and c + 1 (background, range ratio > 1 + tol) the source is the
        first KEPT pixel among c + 1 ... c + edge_extend_search on the background side whose range is still above the
        foreground's (the pixels right at an edge are usually removed by the depth-edge filter: edge_extend 787527 with
        the adjacent pixel as source added nothing).  Copies are placed on the rays of the columns from the source
        back towards (and past) the edge, up to edge_extend columns beyond it, at the source's range; a copy is made
        where the column is not kept or is still in front of the background."""
        opts, (v, h, w) = self.opts, self.grid_shape
        grid = g.reshape(v, h, w, -1)
        kgrid = keep.reshape(v, h, w).to(g.device)
        r = rng.reshape(v, h, w).to(g.device).float()
        dirs = self.pipe.pixel_directions.to(g.device, torch.float32)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        org = po.reshape(v, h, w, 3).to(g.device, torch.float32) if po is not None else torch.zeros_like(dirs)
        thr = 1.0 + opts.edge_extend_tol
        m = max(1, opts.edge_extend_search)
        out = []
        for away in (1, -1):                                  # background on the right of the edge / on the left
            if away == 1:
                fg_r, bg_r = r[..., :-1], r[..., 1:]          # c foreground, c + 1 background
                fg_cols = torch.arange(0, w - 1, device=g.device)
            else:
                fg_r, bg_r = r[..., 1:], r[..., :-1]
                fg_cols = torch.arange(1, w, device=g.device)
            edge = (bg_r > thr * fg_r) & (fg_r > 0.5)
            vi, hi, ci = torch.nonzero(edge, as_tuple=True)
            if not vi.numel():
                continue
            fc = fg_cols[ci]
            rf = r[vi, hi, fc]
            src = torch.full_like(fc, -1)
            for k in range(1, m + 1):                          # first kept background pixel moving away from the edge
                cc = fc + away * k
                ok = (cc >= 0) & (cc < w) & (src < 0)
                cc = cc.clamp(0, w - 1)
                ok &= kgrid[vi, hi, cc] & (r[vi, hi, cc] > thr * rf)
                src = torch.where(ok, cc, src)
            have = src >= 0
            vi, hi, fc, src = vi[have], hi[have], fc[have], src[have]
            if not vi.numel():
                continue
            rb = r[vi, hi, src]
            base = grid[vi, hi, src]
            gap = (src - fc).abs()                            # columns between the source and the edge, inclusive
            for k in range(1, int(gap.max()) + opts.edge_extend):
                tc = src - away * k
                inside = (tc >= 0) & (tc < w) & (k < gap + opts.edge_extend)
                tcc = tc.clamp(0, w - 1)
                need = inside & (~kgrid[vi, hi, tcc] | (r[vi, hi, tcc] * thr < rb))
                if not bool(need.any()):
                    continue
                sel = need.nonzero().flatten()
                cp = base[sel].clone()
                cp[:, 0:3] = org[vi[sel], hi[sel], tcc[sel]] + dirs[vi[sel], hi[sel], tcc[sel]] * rb[sel, None]
                out.append(cp)
        ext = torch.cat(out) if out else g[:0]
        self.stats.setdefault("edge_extend_gaussians", []).append(int(ext.shape[0]))
        return ext

    def _seam_bridge(self, g, keep, rng):
        """Bridge Gaussians across the camera seams of both halves (see SceneOptions.seam_bridge)."""
        opts, (v, h, w) = self.opts, self.grid_shape
        grid = g.reshape(v, h, w, -1)
        kgrid = keep.reshape(v, h, w).to(g.device)
        rgrid = rng.reshape(v, h, w).to(g.device)
        owner = self.pipe.pixel_camera_indices.to(g.device)            # (2, W) camera of each column
        k = opts.seam_bridge
        frac = torch.arange(1, k + 1, device=g.device, dtype=g.dtype) / (k + 1)
        out = []
        rows = torch.arange(h, device=g.device)
        n = max(1, opts.seam_bridge_search)

        def nearest(half, cols):
            """Per row, the first kept column in the ordered candidate list cols (or -1)."""
            cols = [c for c in cols if 0 <= c < w]
            if not cols:
                return torch.full((h,), -1, device=g.device, dtype=torch.long)
            cand = torch.tensor(cols, device=g.device)
            ok = kgrid[half][:, cand]                                           # (H, n)
            first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((h,), -1, device=g.device))
            return torch.where(first >= 0, cand[first.clamp_min(0)], first)

        for half in range(v):
            for c in torch.nonzero(owner[half, 1:] != owner[half, :-1]).flatten().tolist():
                ca = nearest(half, [c - i for i in range(n)])
                cb = nearest(half, [c + 1 + i for i in range(n)])
                ok = (ca >= 0) & (cb >= 0)
                ca, cb = ca.clamp_min(0), cb.clamp_min(0)
                a, b = grid[half, rows, ca], grid[half, rows, cb]               # (H, 14) each
                seg = b[:, 0:3] - a[:, 0:3]
                length = seg.norm(dim=-1)
                ok &= length <= opts.seam_bridge_max_rel * torch.minimum(rgrid[half, rows, ca], rgrid[half, rows, cb])
                if not bool(ok.any()):
                    continue
                a, b, seg, length = a[ok], b[ok], seg[ok], length[ok]
                f = frac[None, :, None]
                br = a[:, None, :].expand(-1, k, -1).clone()
                br[..., 0:3] = a[:, None, 0:3] + f * seg[:, None]
                br[..., 3:6] = (1 - f) * a[:, None, 3:6] + f * b[:, None, 3:6]
                br[..., 6] = torch.minimum(a[:, 6], b[:, 6])[:, None].expand(-1, k)
                br[..., 7:11] = br.new_tensor([1.0, 0.0, 0.0, 0.0])
                size = torch.maximum(opts.seam_bridge_scale * length / (k + 1),
                                     0.5 * (a[:, 11:14].amax(-1) + b[:, 11:14].amax(-1)))
                br[..., 11:14] = size[:, None, None].expand(-1, k, 3)
                out.append(br.reshape(-1, g.shape[-1]))
        bridge = torch.cat(out) if out else g[:0]
        self.stats.setdefault("seam_bridge_gaussians", []).append(int(bridge.shape[0]))
        return bridge

    def _pixel_pitch(self):
        """Median angle (rad) between neighbouring decoder rays, the larger of the two grid directions."""
        if not hasattr(self, "_pitch"):
            d = self.pipe.pixel_directions.float()
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            h = torch.acos((d[:, :, 1:] * d[:, :, :-1]).sum(-1).clamp(-1, 1)).median()
            v = torch.acos((d[:, 1:] * d[:, :-1]).sum(-1).clamp(-1, 1)).median()
            self._pitch = float(torch.maximum(h, v))
        return self._pitch

    def summary(self):
        return {k: (float(np.mean(v)) if v else None) for k, v in self.stats.items()}


def composite(front_rgb, front_alpha, back_rgb, back_alpha):
    """Front over back for (N, 3, H, W) / (N, H, W) renders."""
    return front_rgb + (1.0 - front_alpha[:, None]) * back_rgb, front_alpha + (1.0 - front_alpha) * back_alpha


def composite_depth(front_depth, front_alpha, back_depth, back_alpha):
    """Expected depth of front over back ((N, H, W) each; gsplat's ED is already normalised by alpha)."""
    wf, wb = front_alpha, (1.0 - front_alpha) * back_alpha
    return (wf * front_depth + wb * back_depth) / (wf + wb).clamp_min(1e-6)

def background_row_fill(rgb, alpha, depth, thr=0.5, max_gap=0, smooth=5):
    """Depth-image-based-rendering hole fill (B3): every uncovered pixel (alpha < thr) takes, along its row, the
    colour of whichever nearest covered pixel (left or right) is farther away, because a disocclusion shows
    background.  Rows with no covered pixel on either side stay uncovered.  The filled pixels are then box-blurred
    vertically over `smooth` rows (filled pixels only) to break the horizontal streaks.  Returns (rgb', alpha')
    with alpha' = 1 on filled pixels; remaining holes are left for pushpull_fill.  rgb (3, H, W), alpha / depth
    (H, W).  max_gap > 0 fills only holes up to that many columns wide."""
    covered = alpha.float() >= thr
    h, w = covered.shape
    cols = torch.arange(w, device=rgb.device).expand(h, w)
    left = torch.where(covered, cols, torch.full_like(cols, -1)).cummax(dim=1).values
    right = torch.where(covered, cols, torch.full_like(cols, w)).flip(1).cummin(dim=1).values.flip(1)
    has_l, has_r = left >= 0, right < w
    li, ri = left.clamp(0, w - 1), right.clamp(0, w - 1)
    d = depth.float()
    dl, dr = d.gather(1, li), d.gather(1, ri)
    use_left = has_l & (~has_r | (dl >= dr))
    src = torch.where(use_left, li, ri)
    hole = ~covered & (has_l | has_r)
    if max_gap > 0:
        hole &= (torch.where(has_r, ri, torch.full_like(ri, w)) - torch.where(has_l, li, torch.full_like(li, -1))) \
            <= max_gap + 1
    filled = rgb.gather(2, src[None].expand(3, h, w))
    out = torch.where(hole[None], filled, rgb)
    if smooth > 1 and bool(hole.any()):
        k = torch.ones(1, 1, smooth, 1, device=rgb.device) / smooth
        hm = hole.float()[None, None]
        num = torch.nn.functional.conv2d((out * hole[None].float())[:, None], k, padding=(smooth // 2, 0))[:, 0]
        den = torch.nn.functional.conv2d(hm, k, padding=(smooth // 2, 0))[0, 0]
        blurred = num / den.clamp_min(1e-6)
        out = torch.where(hole[None] & (den[None] > 0), blurred, out)
    return out, torch.where(hole, torch.ones_like(alpha.float()), alpha.float())
