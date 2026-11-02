"""Mine real lateral ground truth inside the nuScenes train split (2026-10-26, GS_IMPROVEMENT_SURVEY_20261026.md
section 4, step 2 preparation; after Lilja et al. CVPR'24 / MTGS / EUVS).

For every info of a 12 Hz train pkl (the anchor), frames of OTHER passes recorded at the same map location whose
camera-time ego pose (recon.targets.camera_time_pose), expressed in the anchor's camera-time ego frame, lies within
|dx| <= max_dx longitudinally, min_dy <= |dy| <= max_dy laterally, |dz| <= max_dz and |dyaw| <= max_dyaw deg.  The
infos carry no scene token: a partner is another pass if it comes from another log (log name from lidar_path) or from
the same log at least min_gap_s apart; scenes (for the coverage counts) are the keyframes' sample -> scene tokens,
interpolated frames taking the scene of the nearest keyframe of their log.  Per (anchor, partner scene) only the
partner frame with the smallest |dx| is kept, so each other drive past the same spot counts once.  Near pairs
(near_dy <= |dy| < min_dy) are counted separately.  Conditions: night from info timeofday / description, rain from
the description; capture date from the log name.

Output: <out>.npz (anchor, partner (info indices), dx, dy, dz, dyaw_deg, dt_days, anchor_night, partner_night,
anchor_rain, partner_rain) and <out>.json (summary: pair counts by |dy| bin, anchor coverage, scenes, locations,
conditions, time gaps).  CPU only.

    python -m dwm_pano_4d.mine_lateral_pairs --data-pkl <train pkl> --meta-dir <v1.0-trainval> --out <prefix>
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import re

import numpy as np
from scipy.spatial import cKDTree

from dwm_pano.recon.lidar_sync import LIDAR_RE
from dwm_pano.recon.targets import camera_time_pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--meta-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-dx", type=float, default=2.0)
    p.add_argument("--min-dy", type=float, default=1.0)
    p.add_argument("--max-dy", type=float, default=5.0)
    p.add_argument("--near-dy", type=float, default=0.25)
    p.add_argument("--max-dz", type=float, default=1.0)
    p.add_argument("--max-dyaw", type=float, default=15.0)
    p.add_argument("--min-gap-s", type=float, default=60.0)
    args = p.parse_args()
    infos = pickle.load(open(args.data_pkl, "rb"))["infos"]
    meta = lambda name: json.loads((args.meta_dir / f"{name}.json").read_text())
    sample_scene = {r["token"]: r["scene_token"] for r in meta("sample")}
    n = len(infos)
    print(f"[mine] {n} infos; keys of the first: {sorted(infos[0].keys())[:30]}", flush=True)
    log_of = [LIDAR_RE.search(str(i["lidar_path"]).replace("\\", "/")).group(1) for i in infos]
    ts = np.array([int(i["timestamp"]) for i in infos], dtype=np.int64)
    scene_of = [sample_scene.get(i.get("token", "")) for i in infos]
    keyed = defaultdict(list)                                   # log -> [(timestamp, scene)] of the keyframes
    for k, s in enumerate(scene_of):
        if s is not None:
            keyed[log_of[k]].append((ts[k], s))
    for log in keyed:
        keyed[log].sort()
    for k, s in enumerate(scene_of):
        if s is None:
            rows_k = keyed[log_of[k]]
            j = int(np.argmin([abs(t - ts[k]) for t, _ in rows_k]))
            scene_of[k] = rows_k[j][1]
    keyframe_fraction = float(np.mean([sample_scene.get(i.get("token", "")) is not None for i in infos]))
    loc = np.array([str(i["location"]) for i in infos])
    desc = [str(i.get("description", "")).lower() for i in infos]
    tod = [str(i.get("timeofday", "")).lower() for i in infos]
    night_k = np.array([("night" in d) or ("night" in t) for d, t in zip(desc, tod)])
    rain_k = np.array(["rain" in d for d in desc])
    log_id = {l: i for i, l in enumerate(sorted(set(log_of)))}
    lid = np.array([log_id[l] for l in log_of])
    log_date = {l: datetime.strptime(re.search(r"(\d{4}-\d{2}-\d{2})", l).group(1), "%Y-%m-%d") for l in log_id}
    poses = np.stack([camera_time_pose(info) for info in infos])                      # (N, 4, 4) ego -> global
    xy = poses[:, :2, 3]
    yaw = np.arctan2(poses[:, 1, 0], poses[:, 0, 0])
    scene_id = {s: i for i, s in enumerate(sorted(set(scene_of)))}
    sid = np.array([scene_id[s] for s in scene_of])
    radius = math.hypot(args.max_dx, args.max_dy) + 0.01
    rows = defaultdict(list)
    near_pairs = 0
    for location in sorted(set(loc.tolist())):
        idx = np.nonzero(loc == location)[0]
        tree = cKDTree(xy[idx])
        neigh = tree.query_ball_point(xy[idx], r=radius)
        for a_local, cand in enumerate(neigh):
            a = idx[a_local]
            c = idx[np.asarray(cand, dtype=np.int64)]
            c = c[(lid[c] != lid[a]) | (np.abs(ts[c] - ts[a]) >= args.min_gap_s * 1e6)]
            if c.size == 0:
                continue
            dyaw = np.degrees(np.angle(np.exp(1j * (yaw[c] - yaw[a]))))
            c, dyaw = c[np.abs(dyaw) <= args.max_dyaw], dyaw[np.abs(dyaw) <= args.max_dyaw]
            if c.size == 0:
                continue
            rel = np.einsum("ij,nj->ni", np.linalg.inv(poses[a])[:3, :3], poses[c, :3, 3] - poses[a, :3, 3])
            dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
            ok = (np.abs(dx) <= args.max_dx) & (np.abs(dz) <= args.max_dz) & (np.abs(dy) <= args.max_dy)
            near_mask = ok & (np.abs(dy) >= args.near_dy) & (np.abs(dy) < args.min_dy)
            ok &= np.abs(dy) >= args.min_dy
            near_pairs += len(set(sid[c[near_mask]].tolist()))
            best = {}
            for j in np.nonzero(ok)[0]:
                s = sid[c[j]]
                if s not in best or abs(dx[j]) < abs(dx[best[s]]):
                    best[s] = j
            for j in best.values():
                b = c[j]
                rows["anchor"].append(a)
                rows["partner"].append(b)
                rows["dx"].append(dx[j]); rows["dy"].append(dy[j]); rows["dz"].append(dz[j])
                rows["dyaw_deg"].append(dyaw[j])
                rows["dt_days"].append(abs((log_date[log_of[b]] - log_date[log_of[a]]).days))
                rows["anchor_night"].append(night_k[a]); rows["partner_night"].append(night_k[b])
                rows["anchor_rain"].append(rain_k[a]); rows["partner_rain"].append(rain_k[b])
        print(f"[mine] {location}: {idx.size} frames, {len(rows['anchor'])} pairs so far", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **out)
    m = len(out["anchor"])
    ady = np.abs(out["dy"]) if m else np.zeros(0)
    same_cond = (out["anchor_night"] == out["partner_night"]) & (out["anchor_rain"] == out["partner_rain"]) if m \
        else np.zeros(0, dtype=bool)
    anchors = set(out["anchor"].tolist()) if m else set()
    anchor_scenes = {scene_of[a] for a in anchors}
    per_anchor = Counter(out["anchor"].tolist()) if m else Counter()
    summary = {
        "data_pkl": str(args.data_pkl), "criteria": {k: getattr(args, k) for k in
                                                     ("max_dx", "min_dy", "max_dy", "near_dy", "max_dz", "max_dyaw")},
        "frames": n, "scenes": len(scene_id), "logs": len(log_id), "keyframe_fraction": keyframe_fraction,
        "pairs": m, "pairs_same_log": int((lid[out["anchor"]] == lid[out["partner"]]).sum()) if m else 0,
        "pairs_by_abs_dy": {f"{lo}-{lo + 1}m": int(((ady >= lo) & (ady < lo + 1)).sum()) for lo in (1, 2, 3, 4)},
        "anchor_frames_with_pair": len(anchors), "anchor_frame_fraction": len(anchors) / n,
        "anchor_scenes_with_pair": len(anchor_scenes), "anchor_scene_fraction": len(anchor_scenes) / len(scene_id),
        "partners_per_anchor_median": float(np.median(list(per_anchor.values()))) if per_anchor else 0.0,
        "same_condition_fraction": float(same_cond.mean()) if m else 0.0,
        "same_condition_pairs": int(same_cond.sum()) if m else 0,
        "anchor_frames_with_same_condition_pair": len(set(out["anchor"][same_cond].tolist())) if m else 0,
        "dt_days_median": float(np.median(out["dt_days"])) if m else None,
        "dt_days_le_7_fraction": float((out["dt_days"] <= 7).mean()) if m else None,
        "near_pairs_0.25_1m": int(near_pairs),
        "by_location": {l: {"frames": int((loc == l).sum()),
                            "anchor_frames_with_pair": int(sum(1 for a in anchors if loc[a] == l))}
                        for l in sorted(set(loc.tolist()))},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print("MINE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
