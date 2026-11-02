"""Cross-log pose consistency of the mined lateral pairs (2026-10-26, mine_lateral_pairs.py output), CPU only.

The pairs are only usable as ground truth if the global poses of different logs agree.  For a random sample of pairs
the two frames' LiDAR scans (the pkl's lidar_path, moved to global with that frame's own pkl pose ego2global @
lidar2ego, i.e. the LiDAR-time pose the scan belongs to) are compared: for the partner's points within 30 m of the
anchor and 0.3-3 m above the anchor's ground, outside both frames' padded boxes, the distance to the nearest anchor
point.  Reported per pair: median and 90th percentile NN distance, and the fraction under 0.2 m.  Control: the same
statistic for the anchor against a frame of its own log about 1 s later (same drive, same calibration).

    python -m dwm_pano_4d.check_lateral_pairs --data-pkl <pkl> --pairs <pairs_train.npz> --nusc-root <root> --out <json>
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import cKDTree

from dwm_pano.recon.targets import raw_path


def pose(t, q):
    m = np.eye(4)
    m[:3, :3] = Quaternion(q).rotation_matrix
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def scan_global(info, root):
    pts = np.fromfile(raw_path(info["lidar_path"], root), dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    boxes = boxes.reshape(boxes.shape[0], -1)[:, :7] if boxes.size else np.zeros((0, 7))
    keep = np.ones(len(pts), dtype=bool)
    for b in boxes:                                     # mmdet3d v0.x (x, y, z, w, l, h, yaw'), lidar frame
        heading = -b[6] - np.pi / 2
        c, s = np.cos(heading), np.sin(heading)
        d = pts - b[:3]
        lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        keep &= ~((np.abs(lx) <= b[4] / 2 + 0.5) & (np.abs(ly) <= b[3] / 2 + 0.5) & (np.abs(d[:, 2]) <= b[5] / 2 + 0.5))
    m = pose(info["ego2global_translation"], info["ego2global_rotation"]) @ \
        pose(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    return pts[keep] @ m[:3, :3].T + m[:3, 3], m


def icp(src, tree, dst, iters=30, keep=0.7, max_d=2.0):
    """Trimmed point-to-point ICP of src onto dst (both near their common origin): (R, t, final median NN)."""
    R, t = np.eye(3), np.zeros(3)
    d = None
    for _ in range(iters):
        cur = src @ R.T + t
        d, j = tree.query(cur, k=1)
        ok = (d <= min(max_d, np.quantile(d, keep)))
        if ok.sum() < 100:
            break
        a, b = cur[ok], dst[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - ma).T @ (b - mb))
        dr = (u @ vt).T
        if np.linalg.det(dr) < 0:
            vt[-1] *= -1
            dr = (u @ vt).T
        R, t = dr @ R, dr @ t + (mb - dr @ ma)
    return R, t, float(np.median(d)) if d is not None else None


def nn_stats(anchor, partner, a_pose, zmin=0.3, zmax=4.0):
    """Before / after trimmed ICP of the partner scan onto the anchor scan (points within 40 m of the anchor, 0.3-4 m
    above its ground): the ICP correction (translation norm, yaw) is the pose disagreement."""
    centre, ground = a_pose[:3, 3], a_pose[2, 3] - 1.84               # LIDAR_TOP is ~1.84 m above the ground
    def near(x):
        return x[(np.linalg.norm(x[:, :2] - centre[:2], axis=1) <= 40.0) & (x[:, 2] >= ground + zmin)
                 & (x[:, 2] <= ground + zmax)] - centre
    src, dst = near(partner), near(anchor)
    if len(src) < 500 or len(dst) < 500:
        return None
    tree = cKDTree(dst)
    d0, _ = tree.query(src, k=1)
    R, t, med = icp(src, tree, dst)
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {"median": float(np.median(d0)), "median_after": med, "icp_t": float(np.linalg.norm(t)),
            "icp_txy": float(np.linalg.norm(t[:2])), "icp_tz": float(abs(t[2])), "icp_yaw_deg": abs(yaw),
            "points": int(len(src))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--nusc-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keyframes-only", action="store_true", help="pairs whose anchor and partner are both keyframes")
    args = p.parse_args()
    infos = pickle.load(open(args.data_pkl, "rb"))["infos"]
    z = np.load(args.pairs)
    rng = np.random.default_rng(args.seed)
    pool = np.arange(len(z["anchor"]))
    if args.keyframes_only:
        key = np.array([bool(i.get("is_key_frame", False)) for i in infos])
        pool = pool[key[z["anchor"]] & key[z["partner"]]]
        print(f"[check] keyframe pairs: {pool.size}", flush=True)
    pick = rng.choice(pool, size=min(args.count, pool.size), replace=False)
    ts = np.array([int(i["timestamp"]) for i in infos], dtype=np.int64)
    log = np.array([str(i["lidar_path"]).replace("\\", "/").rsplit("/", 1)[-1].split("__")[0] for i in infos])
    rows = []
    for k in pick:
        a, b = int(z["anchor"][k]), int(z["partner"][k])
        try:
            pa, ma = scan_global(infos[a], args.nusc_root)
            pb, _ = scan_global(infos[b], args.nusc_root)
        except FileNotFoundError as exc:
            rows.append({"pair": int(k), "error": str(exc)})
            continue
        row = {"pair": int(k), "dy": float(z["dy"][k]), "dt_days": int(z["dt_days"][k]),
               "cross": nn_stats(pa, pb, ma), "cross_high": nn_stats(pa, pb, ma, 2.0, 6.0)}
        later = np.nonzero((log == log[a]) & (ts > ts[a] + 0.9e6) & (ts < ts[a] + 1.3e6))[0]
        if args.keyframes_only:
            later = later[[bool(infos[int(q)].get("is_key_frame", False)) for q in later]] if later.size else later
        if later.size:
            pc, _ = scan_global(infos[int(later[0])], args.nusc_root)
            row["control_1s"] = nn_stats(pa, pc, ma)
            row["control_1s_high"] = nn_stats(pa, pc, ma, 2.0, 6.0)
        rows.append(row)
    def agg(key):
        vals = [r[key] for r in rows if r.get(key)]
        if not vals:
            return None
        fields = ("median", "median_after", "icp_t", "icp_txy", "icp_tz", "icp_yaw_deg")
        return {f: float(np.median([v[f] for v in vals])) for f in fields} | \
            {f + "_p90": float(np.quantile([v[f] for v in vals], 0.9)) for f in ("icp_txy", "icp_yaw_deg")} | \
            {"pairs": len(vals), "icp_txy_over_0.3_fraction": float(np.mean([v["icp_txy"] > 0.3 for v in vals])),
             "icp_txy_over_0.5_fraction": float(np.mean([v["icp_txy"] > 0.5 for v in vals]))}
    out = {"sampled": len(rows), "errors": sum(1 for r in rows if "error" in r), "keyframes_only": args.keyframes_only,
           "cross_log": agg("cross"), "control_same_log_1s": agg("control_1s"),
           "cross_log_high_2_6m": agg("cross_high"), "control_same_log_1s_high_2_6m": agg("control_1s_high"),
           "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("CHECK_SUMMARY " + json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
