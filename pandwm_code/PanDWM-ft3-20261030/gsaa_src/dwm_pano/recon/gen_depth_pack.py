"""LiDAR depth pack for Table 5 (Gaussians on generated Test-S scenes, 2026-10-23, RECON_PLAN_20261022.md).

For every Test-S clip (tests213_manifest_20261018.json, the frozen eval enumeration rebuilt as in targets.py) and every
clip frame k in --frames (default 3, 11, 19, 27 = generated frames 0, 8, 16, 24), the generated scene of frame k is
compared with the RECORDED LiDAR of frame k (static structure follows the conditioning poses and map).  Per frame:
  * the six real cameras: K at 1600x900 and c2w in frame k's camera-time ego (targets.camera_time_pose), the frame
    the Gaussians of frame k live in;
  * the LIDAR_TOP sample_data nearest in time to the info (lidar_sync.py logic), moved with its own ego pose into each
    camera (camera at its own recorded ego pose); points with 0.5 < z <= 80 m that land inside the 1600x900 image;
  * static = outside every dynamic 3D box (DYNAMIC_CLASSES, mmdet3d v0.x reading, 0.3 m margin) after moving the
    points into the info's LiDAR frame (lidar_to_box).
Output <out>/clip<test_index:03d>.npz: frames (F,), c2w (F, 6, 4, 4) f4, K (F, 6, 3, 3) f4, counts (F, 6) i4, and the
concatenated points uv16 (N, 2) uint16 = round(16 * native pixel coords), z (N,) f4, static (N,) bool.

    python -m dwm_pano.recon.gen_depth_pack --manifest tests213_manifest_20261018.json --data-pkl <val pkl>
        --meta-dir <v1.0-trainval> --nusc-root <root> --out <dir>
"""

from __future__ import annotations

import argparse
import bisect
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from dwm_pano.datasets.nuscenes_pano import _enumerate_pano_clips, _group_infos_by_scene, _load_token_to_scene
from dwm_pano.recon.lidar_sync import LIDAR_RE, pose
from dwm_pano.recon.targets import BAD_TIMESTAMPS_US, CAM_ORDER, DYNAMIC_CLASSES, camera_time_pose, raw_path

BOX_MARGIN = 0.3
MAX_DEPTH = 80.0


def inside_boxes_v0x(points, boxes, margin):
    inside = np.zeros(points.shape[0], dtype=bool)
    for x, y, z, w, l, h, yaw_v0x in boxes:
        heading = -yaw_v0x - np.pi / 2
        c, s = np.cos(heading), np.sin(heading)
        rel = points - np.array([x, y, z])
        along = rel[:, 0] * c + rel[:, 1] * s
        across = -rel[:, 0] * s + rel[:, 1] * c
        inside |= (np.abs(along) <= l / 2 + margin) & (np.abs(across) <= w / 2 + margin) & \
            (np.abs(rel[:, 2]) <= h / 2 + margin)
    return inside


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--meta-dir", required=True, type=Path)
    p.add_argument("--nusc-root", required=True, type=Path)
    p.add_argument("--frames", default="3,11,19,27")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    frames = [int(x) for x in args.frames.split(",")]
    manifest = json.loads(args.manifest.read_text())
    pkl = pickle.load(open(args.data_pkl, "rb"))
    scenes = _group_infos_by_scene(pkl["infos"], token2scene=_load_token_to_scene(str(args.meta_dir)), strict=True)
    pairs = list(_enumerate_pano_clips(scenes, 35, manifest["dataset_fps_stride_tuples"],
                                       bad_timestamps_us=BAD_TIMESTAMPS_US))
    if len(pairs) != manifest["dataset_len"]:
        raise RuntimeError(f"enumeration gives {len(pairs)} clips, manifest says {manifest['dataset_len']}")
    jobs = []
    for row in manifest["clips"]:
        clip, audit = pairs[int(row["dataset_index"])]
        if audit["scene_token"] != row["scene_token"] or int(clip[0]["timestamp"]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"test_index {row['test_index']}: dataset index is not the manifest clip")
        jobs.append((int(row["test_index"]), [clip[k] for k in frames]))
    logs = {LIDAR_RE.search(raw_path(i["lidar_path"], args.nusc_root)).group(1) for _, infos in jobs for i in infos}
    print(f"[depth-pack] {len(jobs)} clips x {len(frames)} frames, {len(logs)} logs; loading sample_data", flush=True)
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
    args.out.mkdir(parents=True, exist_ok=True)
    dts, npts = [], []
    for test_index, infos in jobs:
        c2w_all, k_all, counts, uv_all, z_all, st_all = [], [], [], [], [], []
        for info in infos:
            e_ref = camera_time_pose(info)
            cam_global, c2w, ks = [], [], []
            for name in CAM_ORDER:
                c = info["cams"][name]
                g = pose(c["ego2global_translation"], c["ego2global_rotation"]) @ \
                    pose(c["sensor2ego_translation"], c["sensor2ego_rotation"])
                cam_global.append(g)
                c2w.append(np.linalg.inv(e_ref) @ g)
                ks.append(np.asarray(c["camera_intrinsics"], dtype=np.float64))
            log = LIDAR_RE.search(raw_path(info["lidar_path"], args.nusc_root)).group(1)
            rows = sweeps[log]
            stamps = [r[0] for r in rows]
            i = bisect.bisect_left(stamps, int(info["timestamp"]))
            j = min((q for q in (i - 1, i) if 0 <= q < len(rows)), key=lambda q: abs(stamps[q] - int(info["timestamp"])))
            ts, fname, e_tok, c_tok = rows[j]
            dts.append(abs(ts - int(info["timestamp"])) / 1e3)
            sweep_pose = ego[e_tok] @ calib[c_tok]
            info_pose = pose(info["ego2global_translation"], info["ego2global_rotation"]) @ \
                pose(info["lidar2ego_translation"], info["lidar2ego_rotation"])
            pts = np.fromfile(args.nusc_root / fname, dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
            to_box = np.linalg.inv(info_pose) @ sweep_pose
            raw = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
            boxes = raw.reshape(raw.shape[0], -1)[:, :7] if raw.size else np.zeros((0, 7))
            keep = [q for q, n in enumerate(info.get("gt_names", [])) if str(n) in DYNAMIC_CLASSES]
            static = ~inside_boxes_v0x(pts @ to_box[:3, :3].T + to_box[:3, 3], boxes[keep], BOX_MARGIN)
            frame_counts = []
            for g, k in zip(cam_global, ks):
                l2c = np.linalg.inv(g) @ sweep_pose
                cam = pts @ l2c[:3, :3].T + l2c[:3, 3]
                z = cam[:, 2]
                u = k[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + k[0, 2]
                v = k[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + k[1, 2]
                ok = (z > 0.5) & (z <= MAX_DEPTH) & (u >= -0.5) & (u < 1599.5) & (v >= -0.5) & (v < 899.5)
                uv_all.append(np.stack([np.round((u[ok] + 0.5) * 16), np.round((v[ok] + 0.5) * 16)], 1).astype(np.uint16))
                z_all.append(z[ok].astype(np.float32))
                st_all.append(static[ok])
                frame_counts.append(int(ok.sum()))
            c2w_all.append(np.stack(c2w))
            k_all.append(np.stack(ks))
            counts.append(frame_counts)
            npts.append(sum(frame_counts))
        np.savez_compressed(args.out / f"clip{test_index:03d}.npz", frames=np.asarray(frames, dtype=np.int32),
                            c2w=np.stack(c2w_all).astype(np.float32), K=np.stack(k_all).astype(np.float32),
                            counts=np.asarray(counts, dtype=np.int32), uv16=np.concatenate(uv_all),
                            z=np.concatenate(z_all), static=np.concatenate(st_all))
    print(f"[depth-pack] wrote {len(jobs)} clips; |dt| ms median {np.median(dts):.1f} max {max(dts):.1f}; "
          f"points per frame median {int(np.median(npts))}", flush=True)
    (args.out / "README.json").write_text(json.dumps({
        "note": "gen_depth_pack.py: uv16 = round(16 * (pixel + 0.5)) at 1600x900 (pixel centres at integers), so "
                "pixel = uv16 / 16 - 0.5; static = outside dynamic boxes (mmdet3d v0.x, 0.3 m margin); c2w in frame "
                "k's camera-time ego", "frames": frames, "manifest": str(args.manifest)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
