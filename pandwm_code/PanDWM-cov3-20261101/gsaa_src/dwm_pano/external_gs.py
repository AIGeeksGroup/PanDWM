"""External feed-forward Gaussian models on our generated panoramas (2026-10-27, GS_ANCHOR_ZEROSHOT_PLAN_20261026.md).

For frame t of a clip, the decoded panorama (SceneBuilder.canvas) is cut into the six pinhole crops of
dwm_pano_4d.da3_depth.Da3Ranger (camera k's own optical centre and orientation), resized to long side `long_side`
(multiples of 14, intrinsics scaled), and reconstructed by
  "infinisplat"  InfiniSplat v2 (zju3dv/InfiniSplat @ be49948e, provided cameras): Gaussian centres on the frozen DA3
                 depth; the raw per-pixel Gaussians are taken before its voxel merge, restricted to each crop's owned
                 patch (the pixels our panorama shows from that camera), then merged with its own merge();
  "da3gs"        the Gaussian head of DA3-GIANT-1.1 (pose-conditioned, infer_gs=True), per-pixel Gaussians restricted
                 to the owned patches (no merge).
Both live in the method's own normalised frame.  They are moved into frame t's ego with a rotation from the six camera
orientations (polar factor of sum_i R_gt_i R_m_i^T) and a scale = median over trusted owned pixels of our clip-level
DA3 metric z-depth (clip_da3_cache.py, GS_CLIPDEPTH_PLAN V1) over the method's own z-depth of its per-pixel centres;
each camera's Gaussians are re-anchored at that camera's true centre, X = s R (X_m - C_m,i) + C_gt,i (2026-10-28 smoke:
with six nearly co-centred cameras the methods' camera baselines are ill-conditioned, a single similarity left the
centres 0.85 m off on average).  scale_mode "camera" (2026-10-28 diagnostic: InfiniSplat's per-camera depth scales
differ by up to 1.6x on our crops, within-camera spread 5-40 %) uses each camera's own median ratio s_i instead of s.  Output: (N, 14) Gaussians in our decoder layout (means, rgb = 0.5 +
C0 * SH DC, opacity, quaternion wxyz, scales) plus alignment stats.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


def light_ranger(device, crop_hw=(392, 784)):
    """A Da3Ranger without networks: rig(), crop_images(), owned_masks() only (no depth_anything_3 import)."""
    from dwm_pano_4d.da3_depth import IMAGENET_MEAN, IMAGENET_STD, Da3Ranger
    r = Da3Ranger.__new__(Da3Ranger)
    r.device, r.crop_h, r.crop_w, r._rigs = device, crop_hw[0], crop_hw[1], {}
    r.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    r.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return r


def rot_to_quat_wxyz(R):
    m = R.double()
    t = m.trace()
    if t > 0:
        s = (t + 1.0).sqrt() * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(torch.argmax(torch.diagonal(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = (1.0 + m[i, i] - m[j, j] - m[k, k]).sqrt() * 2
        q = [None] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in q])
    return (q / q.norm()).float()


def quat_mul(a, b):
    """Hamilton product of wxyz quaternions, a (4,) with b (N, 4)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def similarity(c2w_gt, c2w_m, z_ours, z_m, valid):
    """(s, R, t) mapping method coordinates to ours, plus diagnostics."""
    M = (c2w_gt[:, :3, :3].double() @ c2w_m[:, :3, :3].double().transpose(1, 2)).sum(0)
    u, _, vt = torch.linalg.svd(M)
    d = torch.ones(3, dtype=M.dtype, device=M.device)
    d[2] = torch.sign(torch.det(u @ vt))
    R = (u @ torch.diag(d) @ vt).float()
    ratio = (z_ours[valid] / z_m[valid]).float()
    if ratio.numel() < 1000:
        raise RuntimeError(f"only {ratio.numel()} trusted pixels for the scale")
    s = float(ratio.median())
    cg, cm = c2w_gt[:, :3, 3].float(), c2w_m[:, :3, 3].float()
    t = cg.mean(0) - s * (R @ cm.mean(0))
    resid = (s * cm @ R.T + t - cg).norm(dim=-1)                  # what one similarity would leave (diagnostic)
    per_cam = torch.stack([(z_ours[i][valid[i]] / z_m[i][valid[i]]).median() if bool(valid[i].any())
                           else z_ours.new_tensor(float("nan")) for i in range(z_ours.shape[0])])
    per_cam = per_cam[torch.isfinite(per_cam)]
    rot_err = torch.stack([torch.arccos(((R @ c2w_m[i, :3, :3].float() @ c2w_gt[i, :3, :3].float().T).trace() - 1)
                                        .div(2).clamp(-1, 1)) for i in range(c2w_gt.shape[0])])
    return s, R, t, {"scale": s, "scale_iqr_rel": float((ratio.quantile(0.75) - ratio.quantile(0.25)) / ratio.median()),
                     "centre_resid_m": float(resid.mean()), "rot_err_deg": float(torch.rad2deg(rot_err).max()),
                     "scale_cam_spread_rel": float((per_cam.max() - per_cam.min()) / per_cam.median()),
                     "trusted_pixels": int(ratio.numel())}


class ExternalGaussians:
    def __init__(self, method, weights, device, cache_dir, long_side=504, da3_src="", is_mode="provided",
                 scale_mode="frame"):
        self.method, self.device, self.cache_dir, self.is_mode = method, device, cache_dir, is_mode
        if scale_mode not in ("frame", "camera"):
            raise ValueError(f"scale_mode {scale_mode!r}")
        self.scale_mode = scale_mode
        if is_mode not in ("provided", "panorama"):
            raise ValueError(f"is_mode {is_mode!r}")
        self.ranger = light_ranger(device)
        f = long_side / self.ranger.crop_w
        self.hw = (max(14, round(self.ranger.crop_h * f / 14) * 14), max(14, round(self.ranger.crop_w * f / 14) * 14))
        self._cache_clip, self._cache = None, None
        self.stats = []
        if method == "infinisplat":
            from infinisplat.pipeline import Pipeline
            self.pipeline = Pipeline.from_pretrained(weights, device=str(device))
            self.model = self.pipeline.model
        elif method == "da3gs":
            from dwm_pano_4d.da3_depth import load_da3_net
            import json
            self.net, _ = load_da3_net(da3_src, weights, device)
            cfg = json.loads(open(f"{weights}/config.json").read())["config"]
            self.pred_color = bool(cfg.get("gs_adapter", {}).get("pred_color", False))
            if self.net.gs_head is None:
                raise RuntimeError(f"{weights} has no Gaussian head")
            # Only the SH DC term is used (rotation invariant): skip the higher-band SH rotation, which needs e3nn.
            import depth_anything_3.model.gs_adapter as gs_adapter_module
            gs_adapter_module.rotate_sh = lambda sh, rotations, *a, **k: sh
        else:
            raise ValueError(f"external method {method!r}")

    # ------------------------------------------------------------------
    def clip_depth(self, clip_index, t):
        if self._cache_clip != clip_index:
            z = np.load(f"{self.cache_dir}/clip{clip_index:03d}.npz")
            if tuple(z["depth"].shape[-2:]) != self.hw:
                raise ValueError(f"cache {z['depth'].shape} vs crops {self.hw}")
            self._cache = (torch.from_numpy(z["depth"].astype(np.float32)), torch.from_numpy(z["sky"]))
            self._cache_clip = clip_index
        return self._cache[0][t].to(self.device), self._cache[1][t].to(self.device)

    def inputs(self, scene, t):
        pipe, r = scene.pipe, self.ranger
        crops = r.crop_images(pipe, scene.canvas(t)[None])                                # (6, 3, 392, 784)
        h, w = self.hw
        images = F.interpolate(crops, size=(h, w), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)
        K = torch.stack([cp["K"] for cp in r.rig(pipe)["crops"]]).clone().float()
        K[:, 0] *= w / r.crop_w
        K[:, 1] *= h / r.crop_h
        owned = F.interpolate(r.owned_masks(pipe).float()[:, None], size=(h, w), mode="nearest")[:, 0] > 0.5
        c2w = r.rig(pipe)["cam2ego"].float().to(self.device)
        return images, K, owned, c2w

    @torch.no_grad()
    def __call__(self, scene, t, clip_index):
        images, K, owned, c2w = self.inputs(scene, t)
        h, w = self.hw
        if self.method == "infinisplat":
            raw, c2w_m = self._infinisplat_raw(images, K, c2w)
        else:
            raw, c2w_m = self._da3gs_raw(images, K, c2w)
        n = 6 * h * w
        if raw["means"].shape[0] != n:
            raise RuntimeError(f"{raw['means'].shape[0]} raw Gaussians, expected {n}")
        pts = raw["means"].view(6, h, w, 3).float()
        z_m = torch.einsum("vhwj,vj->vhw", pts - c2w_m[:, None, None, :3, 3], c2w_m[:, :3, 2])
        z_ours, sky = self.clip_depth(clip_index, t)
        valid = owned & (sky == 0) & (z_ours > 0.5) & (z_ours < 99.0) & (z_m > 1e-4) & torch.isfinite(z_m)
        s, R, _, st = similarity(c2w, c2w_m, z_ours, z_m, valid)
        cam = torch.arange(6, device=self.device).view(6, 1, 1).expand(6, h, w).reshape(-1)
        s_cam = torch.full((6,), s, device=self.device)
        if self.scale_mode == "camera":
            for i in range(6):
                if int(valid[i].sum()) >= 1000:
                    s_cam[i] = (z_ours[i][valid[i]] / z_m[i][valid[i]]).median()
            st["scale_cam_min"], st["scale_cam_max"] = float(s_cam.min()), float(s_cam.max())
        sp = s_cam[cam][:, None]
        anchored = (raw["means"].float() - c2w_m[cam, :3, 3]) @ R.T * sp + c2w[cam, :3, 3]  # per-camera re-anchoring
        keep = owned.reshape(-1)
        kept = {k: v[keep] for k, v in raw.items() if torch.is_tensor(v) and v.shape[0] == n}
        kept["means"] = anchored[keep]
        q = quat_mul(rot_to_quat_wxyz(R).to(self.device), F.normalize(kept["quats"].float(), dim=-1))
        kept["quats"], kept["scales"] = q, kept["scales"].float() * sp[keep]
        if self.method == "infinisplat":                           # its own voxel merge, now in our metric ego frame
            from infinisplat.gaussians import merge
            kept.pop("base_rgb", None)
            kept, _ = merge(kept)
        sh = kept["sh"].float()
        dc = sh[:, :, 0] if sh.shape[1] == 3 else sh[:, 0, :]                            # (N, 3)
        rgb = dc.clamp(0, 1) if (self.method == "da3gs" and self.pred_color) else (0.5 + C0 * dc).clamp(0, 1)
        g = torch.cat([kept["means"].float(), rgb, kept["opacities"].float().reshape(-1, 1),
                       F.normalize(kept["quats"].float(), dim=-1), kept["scales"].float()], dim=-1)
        g = g[torch.isfinite(g).all(-1)]
        st.update(gaussians=int(g.shape[0]), owned_fraction=float(owned.float().mean()))
        self.last = {"z_ours": z_ours, "z_m": z_m, "valid": valid, "scale": s}
        self.stats.append(st)
        return g

    def _infinisplat_raw(self, images, K, c2w):
        from infinisplat.geometry import depth_to_world_coords_points
        m = self.model
        x, cw, k = images[None], c2w[None].clone(), K[None].to(self.device)
        kwargs = {}
        if self.is_mode == "panorama":                  # its co-centred mode: all crops at the rig's mean centre
            cw[..., :3, 3] = cw[0, :, :3, 3].mean(0)
            kwargs["panorama_shared_center"] = True
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens, start, depth, poses, Kp, _ = m._da3(x, cw, k, **kwargs)            # as InfiniSplat.forward
            parts = []
            for i in range(x.shape[1]):
                feats = m.gs_feature_head.forward_chunk(tokens, x, start, i, i + 1)
                parts.append(m.gaussian_head(feats, x[:, i:i + 1], depth[:, i:i + 1].clamp_min(1e-6),
                                             poses[:, i:i + 1], Kp[:, i:i + 1]))
            raw = {key: torch.cat([p[key] for p in parts], 0) for key in parts[0]}
        with torch.autocast(device_type="cuda", enabled=False):
            points, _, _ = depth_to_world_coords_points(depth[0].float(), poses[0].float(), Kp[0].float())
            raw["means"] = points.reshape(-1, 3)
        return raw, poses[0].float()

    def _da3gs_raw(self, images, K, c2w):
        from dwm_pano_4d.da3_depth import as_4x4
        w2c = torch.linalg.inv(c2w.double())
        norm = w2c @ torch.linalg.inv(w2c[:1])                     # DepthAnything3._normalize_extrinsics
        centres = torch.linalg.inv(norm)[:, :3, 3]
        norm[:, :3, 3] = norm[:, :3, 3] / torch.clamp(torch.median(centres.norm(dim=-1)), min=1e-1)
        x = ((images - self.ranger.mean) / self.ranger.std)[None]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = self.net(x, norm[None].float(), K[None].to(self.device).float(), export_feat_layers=[],
                           infer_gs=True, use_ray_pose=False, ref_view_strategy="saddle_balanced")
        gs = out.gaussians
        pred = as_4x4(out.extrinsics[0].double())
        from depth_anything_3.utils.pose_align import batch_align_poses_umeyama
        try:                                                        # exactly GaussianAdapter step 1.2
            _, _, scale = batch_align_poses_umeyama(norm[None].float(), pred[None].float())
        except Exception:
            scale = torch.ones(1, device=self.device)
        scale = torch.clamp(scale, min=1 / 3.0, max=3.0)
        c2w_m = torch.linalg.inv(pred).float()
        c2w_m[:, :3, 3] = c2w_m[:, :3, 3] * scale.float().reshape(())                 # as GaussianAdapter step 1.2
        op = gs.opacities[0]
        raw = {"means": gs.means[0].float(), "scales": gs.scales[0].float(), "quats": gs.rotations[0].float(),
               "sh": gs.harmonics[0].float(), "opacities": (op[..., 0, 0] if op.ndim == 3 else op).float()}
        return raw, c2w_m

    def summary(self):
        if not self.stats:
            return {}
        keys = self.stats[0].keys()
        return {f"ext_{k}": float(np.mean([s[k] for s in self.stats])) for k in keys}
