"""Time-synchronised LiDAR for the tab:gs_recon depth metrics (2026-10-22, RECON_PLAN_20261022.md).

The 12 Hz interpolated infos point most non-keyframe targets at a LiDAR sweep 10-250 ms older than the info, while
targets.py moved the points with the info's own ego pose.  This picks, for every target, the LIDAR_TOP sample_data
(keyframe or sweep, same log) closest in time to the info timestamp and moves its points with that sweep's own ego
pose and calibration:
    lidar_to_cam[c] = inv(E_cam S_cam) @ E_sweep @ L2E_sweep            (camera at its own timestamp)
    lidar_to_box    = inv(E_info @ L2E_info) @ E_sweep @ L2E_sweep      (the info's LiDAR frame, where gt_boxes live)
and writes them as targets[...]["lidar_sync"].  It also reports how far the info's LiDAR pose (E_info @ L2E_info,
rebuilt from targets.json) is from the pkl sweep's and the nearest sweep's recorded poses, and how far the camera
poses of targets.json are from the cameras' recorded ego poses.

    python -m dwm_pano.recon.lidar_sync --targets targets.json --meta-dir <v1.0-trainval> --nusc-root <root>
        --out targets_sync.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

LIDAR_RE = re.compile(r"(n\d{3}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}[+-]\d{4})__LIDAR_TOP__(\d+)\.pcd\.bin$")


def pose(translation, rotation_wxyz):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def rel_name(path):
    """samples/... or sweeps/... part of a raw nuScenes path (the sample_data filename)."""
    return re.sub(r"^.*?/((samples|sweeps)/)", r"\1", path)


def describe(name, values):
    v = np.asarray(values, dtype=np.float64)
    print(f"[lidar-sync] {name}: mean {v.mean():.4f} median {np.median(v):.4f} p90 {np.quantile(v, 0.9):.4f} "
          f"max {v.max():.4f} (n {v.size})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--meta-dir", required=True, type=Path)
    p.add_argument("--nusc-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    data = json.loads(args.targets.read_text())
    logs = {LIDAR_RE.search(t["lidar"]["path"]).group(1) for s in data["samples"] for t in s["targets"].values()}
    cam_files = {rel_name(c["image"]) for s in data["samples"] for t in s["targets"].values() for c in t["cams"]}
    print(f"[lidar-sync] {len(logs)} logs, {len(cam_files)} camera images; loading sample_data", flush=True)
    sweeps, cam_ego = defaultdict(list), {}
    for r in json.loads((args.meta_dir / "sample_data.json").read_text()):
        m = LIDAR_RE.search(r["filename"])
        if m and m.group(1) in logs:
            sweeps[m.group(1)].append((int(r["timestamp"]), r["filename"], r["ego_pose_token"],
                                       r["calibrated_sensor_token"]))
        elif r["filename"] in cam_files:
            cam_ego[r["filename"]] = r["ego_pose_token"]
    for v in sweeps.values():
        v.sort()
    need = {e for v in sweeps.values() for _, _, e, _ in v} | set(cam_ego.values())
    print(f"[lidar-sync] {sum(map(len, sweeps.values()))} LIDAR_TOP records, {len(cam_ego)} camera records; "
          f"loading ego_pose", flush=True)
    ego = {r["token"]: pose(r["translation"], r["rotation"])
           for r in json.loads((args.meta_dir / "ego_pose.json").read_text()) if r["token"] in need}
    calib = {r["token"]: pose(r["translation"], r["rotation"])
             for r in json.loads((args.meta_dir / "calibrated_sensor.json").read_text())}
    by_file = {f: (ts, e, c) for v in sweeps.values() for ts, f, e, c in v}
    stats = defaultdict(list)
    for s in data["samples"]:
        e_input = np.asarray(s["input"]["ego2global"])
        for t in s["targets"].values():
            m = LIDAR_RE.search(t["lidar"]["path"])
            rows = sweeps[m.group(1)]
            stamps = [r[0] for r in rows]
            i = bisect.bisect_left(stamps, t["timestamp"])
            j = min((k for k in (i - 1, i) if 0 <= k < len(rows)), key=lambda k: abs(stamps[k] - t["timestamp"]))
            ts, fname, e_tok, c_tok = rows[j]
            sweep_pose = ego[e_tok] @ calib[c_tok]                                   # E_sweep @ L2E_sweep
            cam_pose = [e_input @ np.asarray(c["c2w"]) for c in t["cams"]]           # E_cam @ S_cam (global)
            info_pose = cam_pose[0] @ np.asarray(t["lidar"]["lidar_to_cam"][0])     # E_info @ L2E_info
            t["lidar_sync"] = {"path": str(args.nusc_root / fname), "timestamp": ts,
                               "dt_ms": (ts - t["timestamp"]) / 1e3,
                               "lidar_to_cam": [(np.linalg.inv(cp) @ sweep_pose).tolist() for cp in cam_pose],
                               "lidar_to_box": (np.linalg.inv(info_pose) @ sweep_pose).tolist()}
            keyframe = "/samples/" in t["lidar"]["path"]
            tag = "key" if keyframe else "interp"
            stats[f"{tag} |dt| pkl sweep ms"].append(abs(int(m.group(2)) - t["timestamp"]) / 1e3)
            stats[f"{tag} |dt| nearest sweep ms"].append(abs(ts - t["timestamp"]) / 1e3)
            old = by_file[rel_name(t["lidar"]["path"])]
            old_pose = ego[old[1]] @ calib[old[2]]
            stats[f"{tag} info pose vs pkl sweep pose m"].append(
                float(np.linalg.norm(old_pose[:3, 3] - info_pose[:3, 3])))
            stats[f"{tag} info pose vs nearest sweep pose m"].append(
                float(np.linalg.norm(sweep_pose[:3, 3] - info_pose[:3, 3])))
            for c, cp in zip(t["cams"], cam_pose):
                rec = ego[cam_ego[rel_name(c["image"])]]
                s_cam = np.asarray(c["sensor2ego"])
                stats[f"{tag} camera pose vs recorded m"].append(
                    float(np.linalg.norm((rec @ s_cam)[:3, 3] - cp[:3, 3])))
    for k in sorted(stats):
        describe(k, stats[k])
    data["lidar_sync_note"] = "lidar_sync.py: nearest LIDAR_TOP sample_data by time, its own ego pose and calibration"
    args.out.write_text(json.dumps(data) + "\n")
    print(f"[lidar-sync] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
