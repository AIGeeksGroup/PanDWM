"""Depth Anything 3 metric range for the pano band (2026-10-19, report section 40).

DAC (report section 37) ran a single-image panorama model on the 33 deg band and failed (sky at 2-6 m, near road
25-30 % too close, blurry).  Here the six nuScenes cameras are cut back out of the stitched pano as pinhole crops and
several frames are reconstructed together with their known intrinsics, rig extrinsics and ego poses.

* Code: github ByteDance-Seed/Depth-Anything-3 @ 3d835ec1 (Apache-2.0), used as a library from its src/ (only the
  network: torch, einops, omegaconf, addict).  Weights: DA3-LARGE-1.1 (any-view, pose conditioned, CC BY-NC 4.0)
  and DA3METRIC-LARGE (single-image metric depth + sky, Apache-2.0).
* Crops: the pipeline's own hard-seam grid (calibrated_stitch.build_owned_pano_sampling_grid) maps every canvas
  pixel to the source-camera pixel it shows.  Inverting it (scattered linear interpolation, nearest outside the
  owned patch) gives for camera k a pinhole image on camera k's own optical centre and orientation, scaled to one
  common crop size (DA3 batches need equal sizes) with the intrinsics adjusted.  Crop pixels outside the owned
  patch are edge-stretched and their depth is never read back.
* Model: the recipe of depth_anything_3.model.da3.NestedDepthAnything3Net with the Large any-view model in place of
  the Giant one (the published nested checkpoint, 6.76 GB, was not downloaded): any-view depth least-squares scaled
  to the metric model's depth on confident non-sky pixels.
* Scale: "pose" (primary) makes the predicted camera centres match the known ones (Umeyama), what
  DepthAnything3.inference(align_to_input_ext_scale=True) does when poses are given; "metric" keeps the metric
  model's scale.  The pose / metric ratio is returned as a diagnostic.
* Read back: every decoder pixel samples its owner camera's z-depth at the source pixel it shows and turns it into
  the Euclidean range along its exact content ray (z / z-cosine = z * pipe.half_range_correction).
* Sky: the metric model's sky probability (sky where >= 0.3, DA3's threshold), read back the same way.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dwm_pano.calibrated_stitch import build_owned_pano_sampling_grid, half_canvas_pano_axes
from dwm_pano.datasets.pano_projection import PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE
from dwm_pano.gs_geometry import PANO_LAT_HALF_RAD
from dwm_pano.pipelines.ctsd_pano_gs_decoder import PANO_CANVAS_OVERLAP_COLUMNS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SKY_THRESHOLD = 0.3


def load_da3_net(repo_src, weights_dir, device):
    """One DepthAnything3Net from a Hugging Face folder (config.json + model.safetensors).

    The official loader (PyTorchModelHubMixin.from_pretrained) is not strict.  DA3-LARGE-1.1 ships without the
    norm parameters of three auxiliary ray heads (head.scratch.output_conv2_aux.{1,2,3}.2), which only feed the
    ray output that _process_camera_estimation discards (poses come from cam_dec); any other missing or
    unexpected key is an error."""
    import json
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object
    with open(os.path.join(weights_dir, "config.json"), "r") as handle:
        spec = json.load(handle)
    net = create_object(OmegaConf.create(spec["config"]))
    state = load_file(os.path.join(weights_dir, "model.safetensors"))
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    allowed = [k for k in missing if k.startswith("head.scratch.output_conv2_aux.")]
    if unexpected or len(allowed) != len(missing):
        raise RuntimeError(f"DA3 load {weights_dir}: missing {sorted(set(missing) - set(allowed))[:8]}, "
                           f"unexpected {unexpected[:8]}")
    if missing:
        print(f"[da3] {os.path.basename(weights_dir)}: {len(missing)} unused aux-ray-head keys not in the "
              f"checkpoint (as with the official non-strict loader)", flush=True)
    return net.to(device).eval(), spec.get("model_name", os.path.basename(weights_dir))


def umeyama_scale(src, dst):
    """Scale s of the similarity dst ~ s R src + t (Umeyama 1991); src, dst (N, 3) float64."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2] = -1.0
    var_s = (xs ** 2).sum() / len(src)
    return float((d * sign).sum() / max(var_s, 1e-12))


def as_4x4(m):
    if m.shape[-2:] == (4, 4):
        return m
    bottom = torch.zeros(*m.shape[:-2], 1, 4, dtype=m.dtype, device=m.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat([m, bottom], dim=-2)


class Da3Ranger:
    def __init__(self, repo_src, anyview_dir, metric_dir, device, crop_hw=(392, 784)):
        if crop_hw[0] % 14 or crop_hw[1] % 14:
            raise ValueError("DA3 crops must be multiples of the 14 px patch")
        self.anyview, self.anyview_name = load_da3_net(repo_src, anyview_dir, device)
        self.metric, self.metric_name = load_da3_net(repo_src, metric_dir, device)
        self.device = device
        self.crop_h, self.crop_w = crop_hw
        self._rigs = {}
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    def rig(self, pipe):
        """Crop grids / intrinsics and decoder read-back grids for the pipeline's active sensor rig (cached)."""
        key = pipe._active_calibration_signature
        if key in self._rigs:
            return self._rigs[key]
        from scipy.interpolate import griddata
        dev = self.device
        cam2ego = pipe.nuscenes_camera_to_ego.float()
        k_render = pipe.nuscenes_render_intrinsics.float()
        hr, wr = int(pipe.nuscenes_render_height), int(pipe.nuscenes_render_width)
        projection = pipe.training_config.get("stitch_projection_model", "onebev_legacy")
        hh, hw = pipe.training_config["half_height"], pipe.training_config["half_width"]

        def owned(rows, columns):
            grid, owner, valid = build_owned_pano_sampling_grid(
                cam2ego, k_render, source_size=(hr, wr), pano_rows=rows, pano_columns=columns,
                projection_model=projection, seam_erosion_px=0)
            return (grid[..., 0] + 1.0) * (wr / 2.0), (grid[..., 1] + 1.0) * (hr / 2.0), owner, valid

        rows_c, cols_c = half_canvas_pano_axes(half_size=(hh, hw), right_trim_px=PANO_CANVAS_OVERLAP_COLUMNS,
                                               device=dev, dtype=torch.float32)
        px, py, owner_c, valid_c = owned(rows_c, cols_c)
        canvas_h, canvas_w = owner_c.shape
        rr, cc = torch.meshgrid(torch.arange(canvas_h, device=dev, dtype=torch.float32),
                                torch.arange(canvas_w, device=dev, dtype=torch.float32), indexing="ij")
        crops = []
        for k in range(6):
            m = (owner_c == k) & valid_c
            x, y, r, c = px[m].double().cpu().numpy(), py[m].double().cpu().numpy(), \
                rr[m].double().cpu().numpy(), cc[m].double().cpu().numpy()
            wraps = c.max() - c.min() > canvas_w / 2          # the camera across the canvas wrap (CAM_BACK)
            if wraps:
                c = np.where(c < canvas_w / 2, c + canvas_w, c)
            x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
            s = min(self.crop_w / (x1 - x0), self.crop_h / (y1 - y0))
            ox, oy = (self.crop_w - s * (x1 - x0)) / 2.0, (self.crop_h - s * (y1 - y0)) / 2.0
            u = (np.arange(self.crop_w) + 0.5 - ox) / s + x0
            v = (np.arange(self.crop_h) + 0.5 - oy) / s + y0
            qu, qv = np.meshgrid(u, v)
            query = np.stack([qu.ravel(), qv.ravel()], axis=-1)
            points, values = np.stack([x, y], axis=-1), np.stack([c, r], axis=-1)
            lin = griddata(points, values, query, method="linear")
            k_crop = k_render[k].clone().double().cpu()
            k_crop[0, 0] *= s
            k_crop[1, 1] *= s
            k_crop[0, 2] = (k_crop[0, 2] - x0) * s + ox
            k_crop[1, 2] = (k_crop[1, 2] - y0) * s + oy
            # Outside camera k's owned patch: the canvas pixel seen in the same direction (ideal band mapping), i.e.
            # the neighbour camera's content with a small parallax error, instead of an edge stretch.
            pix = np.stack([np.arange(self.crop_w)[None, :].repeat(self.crop_h, 0) + 0.5,
                            np.arange(self.crop_h)[:, None].repeat(self.crop_w, 1) + 0.5,
                            np.ones((self.crop_h, self.crop_w))], axis=-1)
            d = pix @ np.linalg.inv(k_crop.numpy()).T @ cam2ego[k, :3, :3].double().cpu().numpy().T
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            col = (np.arctan2(-d[..., 1], d[..., 0]) + np.pi) / (2.0 * np.pi) * canvas_w - 0.5
            row = (PANO_LAT_HALF_RAD - np.arcsin(np.clip(d[..., 2], -1.0, 1.0))) / (2.0 * PANO_LAT_HALF_RAD) \
                * canvas_h - 0.5
            if wraps:
                col = np.where(col < canvas_w / 2, col + canvas_w, col)
            ideal = np.stack([col, np.clip(row, 0.0, canvas_h - 1.0)], axis=-1).reshape(-1, 2)
            cr = np.where(np.isnan(lin), ideal, lin).reshape(self.crop_h, self.crop_w, 2)
            owned_k = ~np.isnan(lin[:, 0]).reshape(self.crop_h, self.crop_w)   # inside camera k's owned patch hull
            crops.append({"cr": torch.from_numpy(cr).float().to(dev), "K": k_crop.float().to(dev),
                          "x0": x0, "y0": y0, "s": s, "ox": ox, "oy": oy,
                          "owned": torch.from_numpy(owned_k).to(dev)})
        pad = int(max(float(cp["cr"][..., 0].max()) for cp in crops) - canvas_w + 2)
        pad = max(pad, 1)
        padded_w = canvas_w + pad
        for cp in crops:                                       # grid_sample coordinates on the wrap-padded canvas
            gx = (cp["cr"][..., 0] + 0.5) / padded_w * 2.0 - 1.0
            gy = (cp["cr"][..., 1] + 0.5) / canvas_h * 2.0 - 1.0
            cp["grid"] = torch.stack([gx, gy], dim=-1)[None]

        halves = []
        for column_range in (PANO_HALF_LEFT_RANGE, PANO_HALF_RIGHT_RANGE):
            rows, columns = pipe._half_pano_axes(column_range, hh, hw, dev, torch.float32)
            xh, yh, owner_h, valid_h = owned(rows, columns)
            grid = torch.zeros(hh, hw, 2, device=dev)
            for k, cp in enumerate(crops):
                sel = owner_h == k
                u = (xh - cp["x0"]) * cp["s"] + cp["ox"]
                v = (yh - cp["y0"]) * cp["s"] + cp["oy"]
                g = torch.stack([u / self.crop_w * 2.0 - 1.0, v / self.crop_h * 2.0 - 1.0], dim=-1)
                grid = torch.where(sel[..., None], g, grid)
            halves.append({"owner": owner_h, "valid": valid_h, "grid": grid})
        rig = {"crops": crops, "halves": halves, "pad": pad, "canvas_hw": (canvas_h, canvas_w),
               "cam2ego": cam2ego.double()}
        self._rigs[key] = rig
        return rig

    # ------------------------------------------------------------------
    def crop_images(self, pipe, canvases):
        """(F, 3, Hc, Wc) unpadded stitch canvases in [0, 1] -> (F * 6, 3, crop_h, crop_w), frame-major."""
        rig = self.rig(pipe)
        padded = torch.cat([canvases, canvases[..., :rig["pad"]]], dim=-1).float()
        out = []
        for f in range(canvases.shape[0]):
            for cp in rig["crops"]:
                out.append(F.grid_sample(padded[f:f + 1], cp["grid"], mode="bilinear", padding_mode="border",
                                         align_corners=False)[0])
        return torch.stack(out).clamp(0.0, 1.0)

    def to_decoder_grid(self, pipe, per_view, frame_count):
        """(F * 6, crop_h, crop_w) values -> (F, 2, H, W) on the decoder halves (each pixel reads its owner)."""
        rig = self.rig(pipe)
        out = []
        for f in range(frame_count):
            views = per_view[f * 6:(f + 1) * 6][:, None].float()
            halves = []
            for half in rig["halves"]:
                acc = torch.zeros(half["owner"].shape, device=self.device)
                for k in range(6):
                    sampled = F.grid_sample(views[k:k + 1], half["grid"][None], mode="bilinear",
                                            padding_mode="border", align_corners=False)[0, 0]
                    acc = torch.where(half["owner"] == k, sampled, acc)
                halves.append(acc)
            out.append(torch.stack(halves))
        return torch.stack(out)

    @torch.no_grad()
    def frame_ranges(self, pipe, canvases, ego_to_world):
        """canvases (F, 3, Hc, Wc) in [0, 1], ego_to_world (F, 4, 4) -> dict of (F, 2, H, W) decoder-grid maps:
        range_pose (primary), range_metric, sky (probability), conf; plus scalars scale_pose / scale_metric."""
        rig = self.rig(pipe)
        frame_count = canvases.shape[0]
        images = self.crop_images(pipe, canvases)
        intrinsics = torch.stack([cp["K"] for _ in range(frame_count) for cp in rig["crops"]])
        e2w = ego_to_world.double().to(self.device)
        ref_inv = torch.linalg.inv(e2w[0])
        c2w = torch.stack([ref_inv @ e2w[f] @ rig["cam2ego"][k].to(self.device)
                           for f in range(frame_count) for k in range(6)])
        w2c = torch.linalg.inv(c2w)
        # DepthAnything3._normalize_extrinsics: first view as reference, translations / median camera distance
        norm = w2c @ torch.linalg.inv(w2c[:1])
        centres = torch.linalg.inv(norm)[:, :3, 3]
        median = torch.clamp(torch.median(centres.norm(dim=-1)), min=1e-1)
        norm[:, :3, 3] = norm[:, :3, 3] / median
        x = ((images - self.mean) / self.std)[None]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = self.anyview(x, norm[None].float(), intrinsics[None].float(), export_feat_layers=[],
                               infer_gs=False, use_ray_pose=False, ref_view_strategy="saddle_balanced")
            metric_out = self.metric(x)
        depth = out.depth[0].float()
        conf = out.depth_conf[0].float()
        focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
        metric = metric_out.depth[0].float() * (focal[:, None, None] / 300.0)
        sky = metric_out.sky[0].float()
        non_sky = sky < SKY_THRESHOLD
        median_conf = torch.quantile(conf[non_sky][:100000], 0.5) if non_sky.any() else conf.median()
        align = (conf >= median_conf) & non_sky & (depth > 1e-3) & (metric > 1e-2)
        scale_metric = float((metric[align] * depth[align]).sum() / (depth[align] ** 2).sum().clamp_min(1e-12))
        pred_c2w = torch.linalg.inv(as_4x4(out.extrinsics[0].double()))
        scale_pose = umeyama_scale(pred_c2w[:, :3, 3].cpu().numpy(), c2w[:, :3, 3].cpu().numpy())
        scale_ground = self.ground_scale(rig, intrinsics, depth, non_sky, frame_count)
        correction = pipe.half_range_correction.to(self.device).float()
        out_maps = {"sky": self.to_decoder_grid(pipe, sky, frame_count),
                    "conf": self.to_decoder_grid(pipe, conf, frame_count),
                    "scale_pose": scale_pose, "scale_metric": scale_metric, "scale_ground": scale_ground,
                    "pose_over_metric": scale_pose / max(scale_metric, 1e-12),
                    "ground_over_metric": scale_ground / max(scale_metric, 1e-12), "crops": images}
        for name, scale in (("pose", scale_pose), ("metric", scale_metric), ("ground", scale_ground)):
            out_maps[f"range_{name}"] = self.to_decoder_grid(pipe, depth * scale, frame_count) * correction[None]
        # 2026-10-26 (clip-level DA3, GS_CLIPDEPTH_PLAN_20261026.md): the crop-space maps behind the ranges above,
        # frame-major (F * 6, crop_h, crop_w); depth_crop is the unscaled any-view depth (multiply by a scale_*)
        out_maps.update(depth_crop=depth, sky_crop=sky, K_crop=intrinsics)
        return out_maps

    def ranges_from_crops(self, pipe, depth_crop, frame_count):
        """(F * 6, crop_h, crop_w) metric camera z -> (F, 2, H, W) decoder ranges, as frame_ranges does."""
        correction = pipe.half_range_correction.to(self.device).float()
        return self.to_decoder_grid(pipe, depth_crop, frame_count) * correction[None]

    def owned_masks(self, pipe):
        """(6, crop_h, crop_w) bool: crop pixels inside each camera's owned patch (the pixels read back)."""
        return torch.stack([cp["owned"] for cp in self.rig(pipe)["crops"]])

    def ground_scale(self, rig, intrinsics, depth, non_sky, frame_count, bottom=0.25, min_dip_deg=5.0):
        """Scale that puts the near road on the ego ground plane (z = 0, the nuScenes ego origin is on the
        ground): median over the bottom quarter of every crop, rays at least 5 deg below the horizon, non-sky,
        of -camera_height / (vertical component of the camera-to-point vector)."""
        h, w = depth.shape[-2:]
        u = torch.arange(w, device=self.device, dtype=torch.float32) + 0.5
        v = torch.arange(h, device=self.device, dtype=torch.float32) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
        values = []
        for i in range(depth.shape[0]):
            k = i % 6
            rot = rig["cam2ego"][k, :3, :3].float().to(self.device)
            height = float(rig["cam2ego"][k, 2, 3])
            ray = torch.einsum("ij,hwj->hwi", torch.linalg.inv(intrinsics[i].float()), pix)     # z = 1 rays
            vertical = torch.einsum("j,hwj->hw", rot[2], ray)                                   # ego z per unit z
            dip = vertical / ray.norm(dim=-1)
            sel = (vv >= (1.0 - bottom) * h) & (dip < -math.sin(math.radians(min_dip_deg))) & non_sky[i]
            if sel.any():
                values.append((-height / (vertical[sel] * depth[i][sel]).clamp(max=-1e-6)))
        if not values:
            return float("nan")
        return float(torch.cat(values).median())


def rescale_gaussians(g, pred_range, new_range, valid, origins):
    """C1b (report section 40.3): scale each valid pixel's Gaussian about its ray origin by s = new / pred range,
    mean o + s (m - o) and scales * s, so its footprint seen from that camera is unchanged (the original view
    renders as before up to the compositing order) while its distance follows the new range.  g (N, 14) in
    decoder order; pred_range / new_range / valid (V, H, W); origins (N, 3).  Returns (g', range')."""
    use = valid.reshape(-1)
    ratio = (new_range.reshape(-1).float() / pred_range.reshape(-1).float().clamp_min(1e-3)).clamp(0.25, 4.0)
    out = g.clone()
    s = ratio[use, None].to(g.dtype)
    out[use, 0:3] = origins[use].to(g.dtype) + s * (g[use, 0:3] - origins[use].to(g.dtype))
    out[use, 11:14] = g[use, 11:14] * s
    return out, torch.where(valid, new_range.float(), pred_range.float())


PIXEL_PITCH_RAD = 2.0 * math.pi / 2304          # decoder grid column pitch (2304 columns per 360 deg)


def plane_offsets(g, origins, dirs, ranges, px):
    """2026-10-30 (GS_FT3_PLAN_20261030.md section 6): centres on their pixel rays, origin + dir * range, plus only
    the component of the decoder's xyz offset across the ray, softly bounded to px decoder pixels at that range
    (norm n -> m tanh(n / m), m = px * PIXEL_PITCH_RAD * range; StereoGaussians-style in-plane offset).  The depth
    along the ray is the range itself.  g (N, 14) decoder order, origins / dirs (N, 3), ranges (N,) or any shape
    with N elements.  Used in training (pipeline_relabel.py plane_offset_px) and at test time (gs_scene_v3)."""
    r = ranges.reshape(-1, 1).to(g.dtype)
    unit = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    on_ray = origins + dirs * r
    off = g[:, 0:3] - on_ray
    perp = off - (off * unit).sum(-1, keepdim=True) * unit
    n = perp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    m = px * PIXEL_PITCH_RAD * r.clamp_min(1e-3)
    return torch.cat([on_ray + perp * (m * torch.tanh(n / m) / n), g[:, 3:]], dim=-1)


def bounded_range(pred_range, da3_range, bound):
    """2026-10-30 (GS_FT3_PLAN_20261030.md): DA3 range with a bounded decoder residual,
    da3 * exp(bound * tanh(log(pred / da3) / bound)): the decoder may move a Gaussian at most a factor exp(+-bound)
    off DA3 and is DA3 itself when it agrees.  bound 0 = DA3 (the "scale" relabel).  Differentiable in pred_range;
    used both in training (pipeline_relabel.py) and at test time (gs_scene_v3 da3_mode "bounded")."""
    if bound <= 0.0:
        return da3_range
    log_da3 = da3_range.clamp_min(1e-3).log()
    resid = pred_range.clamp_min(1e-3).log() - log_da3
    return (log_da3 + bound * torch.tanh(resid / bound)).exp()

