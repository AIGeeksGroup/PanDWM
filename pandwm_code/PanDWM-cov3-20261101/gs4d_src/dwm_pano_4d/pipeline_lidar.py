"""4D GS decoder with a LiDAR depth term on top of the DA3 pipeline (2026-10-24, round-2 fine-tune R3,
PanDWM-gs-ft2-20261024/GS_FT2_PLAN_20261024.md).

For every own frame of a step, the frame's Gaussians are rendered at the six real cameras of that frame, each at its
own exposure pose relative to the frame's camera-time ego (c2w = inv(E_ref) @ E_cam @ S_cam; the Gaussians live in
the camera rig, see dataset.py pose_source), at gs4d_lidar.render_hw with the calibrated K scaled by
calibrated_stitch.prepare_gsplat_intrinsics.  The LIDAR_TOP sample_data nearest in time to the frame (index built by
build_lidar_index.py, moved with that sweep's own ego pose) is projected into the same cameras; at every point with
min_z < z <= max_z inside the image whose rendered alpha > min_alpha (detached), the term is
    weight_log_lidar * mean |log d_render - log z|          (expected depth, nearest pixel)
added to the frame's own loss through PanoGaussianDecoder4DDepthPipeline._extra_own_loss.  A frame missing from the
index raises unless allow_missing.  The report gains own_log_lidar_l1 and lidar_points.  weight 0 = the parent.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dwm_pano_4d.pipeline_da3 import PanoGaussianDecoder4DDa3Pipeline

DEFAULT_LIDAR = {"index": "", "nusc_root": "", "weight_log_lidar": 0.0, "render_hw": [224, 400], "min_z": 0.5,
                 "max_z": 80.0, "min_alpha": 0.5, "allow_missing": False}


class PanoGaussianDecoder4DLidarPipeline(PanoGaussianDecoder4DDa3Pipeline):
    def __init__(self, *args, gs4d_lidar: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gs4d_lidar = dict(DEFAULT_LIDAR)
        self.gs4d_lidar.update(gs4d_lidar or {})
        unknown = set(self.gs4d_lidar) - set(DEFAULT_LIDAR)
        if unknown:
            raise ValueError(f"unknown gs4d_lidar options: {sorted(unknown)}")
        self._lidar_index = None

    def _index(self):
        if self._lidar_index is None:
            z = np.load(self.gs4d_lidar["index"], allow_pickle=False)
            self._lidar_index = {
                "row": {int(t): i for i, t in enumerate(z["timestamps"].tolist())},
                "paths": [str(x) for x in z["paths"].tolist()], "path_index": z["path_index"],
                "sweep_pose": z["sweep_pose"], "cam_global": z["cam_global"], "K": z["K"], "e_ref": z["e_ref"]}
        return self._lidar_index

    def _extra_own_loss(self, batch, frame, gaussians):
        cfg = self.gs4d_lidar
        if cfg["weight_log_lidar"] <= 0.0:
            return None, {}
        from dwm_pano.calibrated_stitch import prepare_gsplat_intrinsics
        index = self._index()
        ts = int(batch["frame_timestamps_us"][0, frame])
        row = index["row"].get(ts)
        if row is None:
            if cfg["allow_missing"]:
                return None, {"lidar_missing": 1.0}
            raise KeyError(f"frame {ts} not in the LiDAR index {cfg['index']}")
        h, w = (int(v) for v in cfg["render_hw"])
        cam_global = index["cam_global"][row].astype(np.float64)                     # (6, 4, 4)
        e_ref = index["e_ref"][row].astype(np.float64)
        sweep_pose = index["sweep_pose"][row].astype(np.float64)
        K = index["K"][row].astype(np.float64)                                      # (6, 3, 3) at 1600x900
        c2w = torch.as_tensor(np.linalg.inv(e_ref)[None] @ cam_global, device=self.device, dtype=torch.float32)
        Ks = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(k), (900, 1600), (h, w)) for k in K])
        _, depth, alpha = self.render_targets(gaussians, target_c2w=c2w, target_intrinsics=Ks.float().to(self.device),
                                              width=w, height=h)
        path = os.path.join(cfg["nusc_root"], index["paths"][int(index["path_index"][row])])
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        cams, us, vs, zs = [], [], [], []
        Kn = Ks.double().cpu().numpy()
        for c in range(6):
            l2c = np.linalg.inv(cam_global[c]) @ sweep_pose
            p = pts @ l2c[:3, :3].T + l2c[:3, 3]
            z = p[:, 2]
            ok = (z > cfg["min_z"]) & (z <= cfg["max_z"])
            p, z = p[ok], z[ok]
            u = np.round(Kn[c, 0, 0] * p[:, 0] / z + Kn[c, 0, 2]).astype(np.int64)
            v = np.round(Kn[c, 1, 1] * p[:, 1] / z + Kn[c, 1, 2]).astype(np.int64)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            cams.append(np.full(int(inside.sum()), c))
            us.append(u[inside])
            vs.append(v[inside])
            zs.append(z[inside])
        cam_i = torch.as_tensor(np.concatenate(cams), device=self.device)
        u_i = torch.as_tensor(np.concatenate(us), device=self.device)
        v_i = torch.as_tensor(np.concatenate(vs), device=self.device)
        z_t = torch.as_tensor(np.concatenate(zs), device=self.device, dtype=torch.float32)
        d = depth[cam_i, v_i, u_i]
        keep = (alpha[cam_i, v_i, u_i].detach() > cfg["min_alpha"]) & (d.detach() > 1e-3)
        n = int(keep.sum())
        if n == 0:
            return None, {"lidar_points": 0.0}
        loss = (d[keep].clamp_min(1e-3).log() - z_t[keep].log()).abs().mean()
        return cfg["weight_log_lidar"] * loss, {"own_log_lidar_l1": float(loss.detach()), "lidar_points": float(n)}
