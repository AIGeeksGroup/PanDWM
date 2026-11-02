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
        if opts.edge_rel_h > 0 or opts.edge_rel_v > 0:
            keep &= ~edge_mask(rng, opts.edge_rel_h, opts.edge_rel_v, opts.edge_dilate).to(g.device)
        g = g[keep]
        g = g[g[:, 11:14].amax(dim=-1) <= opts.max_scale]
        if opts.min_opacity > 0.0:
            g = g[g[:, 6] >= opts.min_opacity]
        self._decoded[t] = g
        if len(self._decoded) > len(opts.window) + 4:
            self._decoded.pop(min(self._decoded))
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
