"""LiDAR index for pipeline_lidar.py (2026-10-24, round-2 fine-tune, CPU only).

For every info of a 12 Hz pkl: the LIDAR_TOP sample_data nearest in time (same log, keyframe or sweep; the pkl's own
lidar_path can be 250 ms older than the cameras, report section 44), that sweep's own pose (ego @ calibration), the six
cameras' global poses (E_cam @ S_cam, CAM_ORDER), their 1600x900 intrinsics and the camera-time ego pose
(recon.targets.camera_time_pose).  Output npz: timestamps (N,) int64, paths (P,) str (relative to the nuScenes root),
path_index (N,) int32, sweep_pose (N, 4, 4) f8, cam_global (N, 6, 4, 4) f8, K (N, 6, 3, 3) f4, e_ref (N, 4, 4) f8,
dt_ms (N,) f4.

    python -m dwm_pano_4d.build_lidar_index --data-pkl <pkl> --meta-dir <v1.0-trainval> --out lidar_index.npz
"""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from dwm_pano.recon.lidar_sync import LIDAR_RE, pose
from dwm_pano.recon.targets import CAM_ORDER, camera_time_pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--meta-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    infos = pickle.load(open(args.data_pkl, "rb"))["infos"]
    logs = {LIDAR_RE.search(str(i["lidar_path"]).replace("\\", "/")).group(1) for i in infos}
    print(f"[lidar-index] {len(infos)} infos, {len(logs)} logs; loading sample_data", flush=True)
    sweeps = defaultdict(list)
    for r in json.loads((args.meta_dir / "sample_data.json").read_text()):
        m = LIDAR_RE.search(r["filename"])
        if m and m.group(1) in logs:
            sweeps[m.group(1)].append((int(r["timestamp"]), r["filename"], r["ego_pose_token"],
                                       r["calibrated_sensor_token"]))
    for v in sweeps.values():
        v.sort()
    need = {e for v in sweeps.values() for _, _, e, _ in v}
    ego = {r["token"]: pose(r["translation"], r["rotation"])
           for r in json.loads((args.meta_dir / "ego_pose.json").read_text()) if r["token"] in need}
    calib = {r["token"]: pose(r["translation"], r["rotation"])
             for r in json.loads((args.meta_dir / "calibrated_sensor.json").read_text())}
    stamps = {log: [r[0] for r in v] for log, v in sweeps.items()}
    n = len(infos)
    ts_all = np.zeros(n, dtype=np.int64)
    path_index = np.zeros(n, dtype=np.int32)
    sweep_pose = np.zeros((n, 4, 4))
    cam_global = np.zeros((n, 6, 4, 4))
    K = np.zeros((n, 6, 3, 3), dtype=np.float32)
    e_ref = np.zeros((n, 4, 4))
    dt = np.zeros(n, dtype=np.float32)
    paths, path_id = [], {}
    for i, info in enumerate(infos):
        t = int(info["timestamp"])
        log = LIDAR_RE.search(str(info["lidar_path"]).replace("\\", "/")).group(1)
        rows, st = sweeps[log], stamps[log]
        j = bisect.bisect_left(st, t)
        j = min((q for q in (j - 1, j) if 0 <= q < len(rows)), key=lambda q: abs(st[q] - t))
        sts, fname, e_tok, c_tok = rows[j]
        if fname not in path_id:
            path_id[fname] = len(paths)
            paths.append(fname)
        ts_all[i], path_index[i], dt[i] = t, path_id[fname], (sts - t) / 1e3
        sweep_pose[i] = ego[e_tok] @ calib[c_tok]
        for c, name in enumerate(CAM_ORDER):
            cam = info["cams"][name]
            cam_global[i, c] = pose(cam["ego2global_translation"], cam["ego2global_rotation"]) @ \
                pose(cam["sensor2ego_translation"], cam["sensor2ego_rotation"])
            K[i, c] = np.asarray(cam["camera_intrinsics"], dtype=np.float32)
        e_ref[i] = camera_time_pose(info)
    if len(set(ts_all.tolist())) != n:
        raise RuntimeError("duplicate info timestamps")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, timestamps=ts_all, paths=np.asarray(paths), path_index=path_index, sweep_pose=sweep_pose,
             cam_global=cam_global, K=K, e_ref=e_ref, dt_ms=dt)
    print(f"[lidar-index] wrote {args.out}: {n} frames, {len(paths)} sweeps, |dt| ms median "
          f"{np.median(np.abs(dt)):.1f} p99 {np.quantile(np.abs(dt), 0.99):.1f} max {np.abs(dt).max():.1f}", flush=True)


if __name__ == "__main__":
    main()
