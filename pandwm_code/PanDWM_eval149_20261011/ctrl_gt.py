"""Controllability ground truth per Val-150 clip (2026-10-21; protocol CONTROLLABILITY_PLAN_20261021.md, frozen).

For every clip, clip frame 3..34 and camera (cam_band_feats.CAM_ORDER), and for each timing source
  band     the 12 Hz info of the clip frame (LIDAR_TOP-synced): our band, recband, MagicDrive-V2
  opendwm  the camera sweep the OpenDWM loader picked for that camera (rec448, UniMLVG), by its sample_data token
           in opendwm_val_items_official.json (the image filename is asserted as well)
the image is scored against the annotations of the info that holds that camera sweep:
  * boxes: pkl gt_boxes (LiDAR frame, (w, l, h), ccw yaw) -> camera with cams[c].sensor2lidar (includes the ego motion
    between LiDAR and camera time) -> native K -> 256x448 edge coords. Hull = projection of the box clipped at
    z = 0.5 m. Required vehicle (car/truck/bus) / pedestrian GT and ignore regions by the rules of section 5;
  * road: per pixel ray (native K, cams[c].sensor2ego, cams[c].ego2global) hits the world plane z = 0; GT road = hit
    inside the map-expansion drivable_area raster (exterior minus holes, 0.05 m); region = protocol mask & hit within
    50 m & not inside the hull of any annotated box within 60 m. near = region pixels within 20 m.
Output: <out>/<timing>/clipNNN.npz (NNN = Val-150 test_index), resumable (existing files are kept).
  python3 ctrl_gt.py --base-config cd_eval_frozen149_hpc3.json --out-dir GT --timings band,opendwm
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion

import cam_band_feats as cbf
import ctrl_common as cc
from dwm_pano.calibrated_stitch import camera_records_to_arrays

CAMS = cbf.CAM_ORDER


def rig_of(info):
    c2e, K, size = camera_records_to_arrays(info["cams"], camera_order=CAMS)
    return [np.asarray(c2e, np.float64), np.asarray(K, np.float64), np.asarray(size, np.float64)]


def same_rig(a, b):
    return (np.abs(a[0][:, :3, :3] - b[0][:, :3, :3]).max() < 1e-3 and np.abs(a[0][:, :3, 3] - b[0][:, :3, 3]).max() < 1e-2
            and np.abs(a[1] - b[1]).max() < 0.5 and np.array_equal(a[2], b[2]))


class DrivableMap:
    """drivable_area polygons of one map-expansion JSON (world frame), rasterised on demand."""

    def __init__(self, path):
        raw = json.load(open(path))
        nodes = {n["token"]: (n["x"], n["y"]) for n in raw["node"]}
        polys = {p["token"]: p for p in raw["polygon"]}
        self.items = []
        for rec in raw["drivable_area"]:
            for pt in rec["polygon_tokens"]:
                p = polys[pt]
                ext = np.array([nodes[t] for t in p["exterior_node_tokens"]], np.float64)
                holes = [np.array([nodes[t] for t in h["node_tokens"]], np.float64) for h in p.get("holes", [])]
                holes = [h for h in holes if len(h) >= 3]
                if len(ext) >= 3:
                    self.items.append((ext, holes, ext.min(0), ext.max(0)))

    def raster(self, x0, y0, x1, y1, res=cc.RASTER_RES):
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        R = np.zeros((ny, nx), np.uint8)
        for ext, holes, lo, hi in self.items:
            if hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1:
                continue
            c0, c1 = max(0, int((lo[0] - x0) / res) - 1), min(nx, int(math.ceil((hi[0] - x0) / res)) + 1)
            r0, r1 = max(0, int((lo[1] - y0) / res) - 1), min(ny, int(math.ceil((hi[1] - y0) / res)) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            tmp = np.zeros((r1 - r0, c1 - c0), np.uint8)
            org = np.array([x0 + c0 * res, y0 + r0 * res])

            def px(a):  # world -> sub-window pixel-centre coords, 2 fractional bits
                return np.round(((a - org) / res - 0.5) * 4).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(tmp, [px(ext)], 1, lineType=cv2.LINE_8, shift=2)
            if holes:
                cv2.fillPoly(tmp, [px(h) for h in holes], 0, lineType=cv2.LINE_8, shift=2)
            R[r0:r1, c0:c1] |= tmp
        return R


_RAYS = {}


def rays(K):
    """Unit-free camera rays [H*W, 3] at the 256x448 pixel centres (native index coords), cached per K."""
    key = tuple(np.round(K, 4).ravel())
    if key not in _RAYS:
        jj, ii = np.meshgrid(np.arange(cc.W), np.arange(cc.H))
        u = (jj + 0.5) * cc.NW / cc.W - 0.5
        v = (ii + 0.5) * cc.NH / cc.H - 0.5
        uv1 = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 0)
        _RAYS[key] = (np.linalg.inv(K) @ uv1).T
    return _RAYS[key]


def to448(p, K):
    """camera-frame points [n, 3] (z > 0) -> 256x448 edge coords [n, 2]."""
    u = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 1] * p[:, 1] / p[:, 2] + K[0, 2]
    v = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return np.stack([(u + 0.5) * cc.W / cc.NW, (v + 0.5) * cc.H / cc.NH], 1)


def clipped_hull_points(cor):
    """Corners [8, 3] (camera frame) -> projected points of the box clipped at z = NEAR (in front part), or None."""
    z = cor[:, 2]
    pts = [cor[z >= cc.NEAR]]
    for a, b in cc.EDGES:
        za, zb = z[a] - cc.NEAR, z[b] - cc.NEAR
        if za * zb < 0:
            pts.append((cor[a] + (cc.NEAR - z[a]) / (z[b] - z[a]) * (cor[b] - cor[a]))[None])
    p = np.concatenate(pts, 0)
    return p if len(p) else None


def fill_hull(img, pts448):
    q = np.clip(np.round((pts448 - 0.5) * 16), -2e6, 2e6).astype(np.int32)
    hull = cv2.convexHull(q.reshape(-1, 1, 2))
    cv2.fillConvexPoly(img, hull, 1, lineType=cv2.LINE_8, shift=4)


def pix_window(box):
    """pixel index window [i0, i1) x [j0, j1) of the pixel centres inside an edge-coord box, clipped to the image."""
    x0, y0, x1, y1 = box
    j0, j1 = max(0, math.ceil(x0 - 0.5)), min(cc.W, math.ceil(x1 - 0.5))
    i0, i1 = max(0, math.ceil(y0 - 0.5)), min(cc.H, math.ceil(y1 - 0.5))
    return i0, i1, j0, j1


def mask_clip(box, mask_c):
    """(box clipped to the bounding box of the mask pixels inside it, number of such pixels)."""
    i0, i1, j0, j1 = pix_window(box)
    if i1 <= i0 or j1 <= j0:
        return None, 0
    sub = mask_c[i0:i1, j0:j1]
    n = int(sub.sum())
    if n == 0:
        return None, 0
    rows = np.flatnonzero(sub.any(1))
    cols = np.flatnonzero(sub.any(0))
    return np.array([max(box[0], j0 + cols[0]), max(box[1], i0 + rows[0]),
                     min(box[2], j0 + cols[-1] + 1), min(box[3], i0 + rows[-1] + 1)], np.float64), n


def sweep_gt(info, cam, mask_c, raster, raster_org):
    """GT of one camera sweep: box records and the road GT / region / near bits [256, 448]."""
    rec = info["cams"][cam]
    K = np.asarray(rec["camera_intrinsics"], np.float64)
    S = np.asarray(rec["sensor2lidar_rotation"], np.float64)
    T = np.asarray(rec["sensor2lidar_translation"], np.float64)
    boxes = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7)
    names = [str(n) for n in info["gt_names"]]
    n = len(names)
    npts = np.asarray(info["num_lidar_pts"], np.int64).reshape(-1) + np.asarray(info["num_radar_pts"], np.int64).reshape(-1)
    vis = np.asarray(info["visibility"]).astype(np.int64).reshape(-1)
    ids = [str(x) for x in info.get("gt_box_ids", [""] * n)]
    # corners in the LiDAR frame (pano_via_stitch._ego_corners_from_boxes without the lidar->ego step)
    cor = cc.TEMPLATE[None] * boxes[:, None, 3:6]
    cy, sy = np.cos(boxes[:, 6])[:, None], np.sin(boxes[:, 6])[:, None]
    x, y = cor[..., 0].copy(), cor[..., 1].copy()
    cor[..., 0], cor[..., 1] = x * cy - y * sy, x * sy + y * cy
    cor += boxes[:, None, :3]
    cor_c = (cor - T) @ S                      # X_cam = S^T (X_lidar - T)
    cen_c = (boxes[:, :3] - T) @ S
    Rl2e = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    cen_e = boxes[:, :3] @ Rl2e.T + np.asarray(info["lidar2ego_translation"], np.float64)
    dist = np.hypot(cen_e[:, 0], cen_e[:, 1])
    hull = [None] * n
    rfull = np.full((n, 4), np.nan)
    for b in range(n):
        p = clipped_hull_points(cor_c[b])
        if p is not None:
            q = to448(p, K)
            hull[b] = q
            rfull[b] = [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
    allfront = (cor_c[..., 2] >= cc.NEAR).all(1)
    # occlusion by nearer boxes (painter order by centre depth) and the hull union for the road region
    occ = np.zeros(n)
    cum = np.zeros((cc.H, cc.W), np.uint8)
    hull60 = np.zeros((cc.H, cc.W), np.uint8)
    for b in np.argsort(cen_c[:, 2], kind="mergesort"):
        if hull[b] is None:
            continue
        i0, i1, j0, j1 = pix_window(rfull[b])
        occ[b] = float(cum[i0:i1, j0:j1].mean()) if (i1 > i0 and j1 > j0) else 1.0
        fill_hull(cum, hull[b])
        if dist[b] <= cc.HULL_RANGE:
            fill_hull(hull60, hull[b])
    recs = []
    for b in range(n):
        if hull[b] is None:
            continue
        name = names[b]
        roles = []
        for cls, names_c, rng, ign in ((cc.VEH, cc.VEH_NAMES, cc.VEH_RANGE, cc.veh_ignore_name),
                                       (cc.PED, cc.PED_NAMES, cc.PED_RANGE, cc.ped_ignore_name)):
            if name in names_c:
                roles.append((cls, rng))
            elif ign(name):
                roles.append((cls, None))
        if not roles:
            continue
        x0, y0, x1, y1 = rfull[b]
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        mbox, nin = mask_clip(rfull[b], mask_c)
        vmask = nin / area
        for cls, rng in roles:
            if mbox is None:
                continue
            height = mbox[3] - mbox[1]
            req = (rng is not None and allfront[b] and dist[b] <= rng and npts[b] >= 1 and vis[b] >= cc.VIS_MIN
                   and vmask >= cc.VMASK_MIN and occ[b] <= cc.OCC_MAX and height >= cc.MIN_H)
            recs.append((cls, cc.REQ if req else cc.IGN, *mbox, vis[b], dist[b], cen_c[b, 2], vmask, occ[b],
                         y1 - y0, int(allfront[b]), npts[b], name, ids[b]))
    # road
    R_s2e = Quaternion(rec["sensor2ego_rotation"]).rotation_matrix
    t_s2e = np.asarray(rec["sensor2ego_translation"], np.float64)
    R_e2g = Quaternion(rec["ego2global_rotation"]).rotation_matrix
    t_e2g = np.asarray(rec["ego2global_translation"], np.float64)
    R_wc, o = R_e2g @ R_s2e, R_e2g @ t_s2e + t_e2g
    d = rays(K) @ R_wc.T
    dz = d[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = np.where(dz < 0, -o[2] / np.where(dz < 0, dz, -1.0), 0.0)
    px, py = o[0] + s * d[:, 0], o[1] + s * d[:, 1]
    hd = np.where(dz < 0, s * np.hypot(d[:, 0], d[:, 1]), np.inf)
    ground = (dz < 0) & (hd <= cc.ROAD_RANGE)
    col = np.floor((np.where(ground, px, raster_org[0]) - raster_org[0]) / cc.RASTER_RES).astype(np.int64)
    row = np.floor((np.where(ground, py, raster_org[1]) - raster_org[1]) / cc.RASTER_RES).astype(np.int64)
    inside = (col >= 0) & (col < raster.shape[1]) & (row >= 0) & (row < raster.shape[0])
    if (ground & ~inside).any():
        raise RuntimeError("ground hit outside the clip raster; enlarge RASTER_MARGIN")
    road = ground & inside & (raster[np.clip(row, 0, raster.shape[0] - 1), np.clip(col, 0, raster.shape[1] - 1)] > 0)
    road, ground, hd = road.reshape(cc.H, cc.W), ground.reshape(cc.H, cc.W), hd.reshape(cc.H, cc.W)
    region = mask_c & ground & (hull60 == 0)
    near = region & (hd <= cc.NEAR_ROAD)
    return recs, road, region, near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pkl", default="/path/to/pandwm_root/pandwm_data/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl")
    ap.add_argument("--items", default="/path/to/pandwm_root/pandwm_runs/baselines_20261019/opendwm_val_items_official.json")
    ap.add_argument("--map", default="val150_scene_map.json")
    ap.add_argument("--maps-dir", default="/path/to/pandwm_root/pandwm_data/nuscenes_maps_expansion")
    ap.add_argument("--timings", default="band,opendwm")
    ap.add_argument("--positions", default="")
    a = ap.parse_args()
    timings = a.timings.split(",")
    out = Path(a.out_dir)
    for t in timings:
        (out / t).mkdir(parents=True, exist_ok=True)
    config, dataset, collate, rows = cbf.load_val150(a.base_config)
    base = dataset.base_dataset
    if a.positions:
        keep = {int(p) for p in a.positions.split(",")}
        rows = [r for r in rows if int(r["test_index"]) in keep]
    infos = pickle.load(open(a.pkl, "rb"))["infos"]
    tok = {}
    for i, inf in enumerate(infos):
        for c, r in inf["cams"].items():
            if r["sample_data_token"] in tok:
                raise RuntimeError(f"duplicate camera token {r['sample_data_token']}")
            tok[r["sample_data_token"]] = (i, c)
    items = {int(x["index"]): x for x in json.load(open(a.items))}
    odwm_of = {int(v["test_index"]): int(k) for k, v in json.load(open(a.map)).items()}
    geometry = cbf.LegacyInverseOwnGeometry()
    maps = {}
    t0 = time.time()
    for n_done, row in enumerate(rows):
        pos = int(row["test_index"])
        todo = [t for t in timings if not (out / t / f"clip{pos:03d}.npz").exists()]
        if not todo:
            continue
        window = base.clips[int(row["relaxed_index"])]
        if int(window[0]["timestamp"]) != int(row["anchor_timestamp_us"]) or len(window) != cbf.FRAMES:
            raise RuntimeError(f"clip {pos}: dataset window is not the manifest clip")
        rig = rig_of(window[0])
        for inf in window:
            if not same_rig(rig, rig_of(inf)):
                raise RuntimeError(f"clip {pos}: calibration changes within the clip")
        _, mask = cc.eval_mask(geometry, *rig)
        rig_key = hashlib.md5((np.round(rig[0], 3) + 0.0).tobytes() + (np.round(rig[1], 1) + 0.0).tobytes()).hexdigest()[:8]
        loc = window[0]["location"]
        if loc not in maps:
            maps[loc] = DrivableMap(os.path.join(a.maps_dir, f"{loc}.json"))
        for timing in todo:
            sweeps = []  # [32][6] (info, cam)
            for t in range(cc.FIRST, cc.LAST + 1):
                fr = []
                for cam in CAMS:
                    if timing == "band":
                        fr.append((window[t], cam))
                    else:
                        it = items[odwm_of[pos]]
                        if it["scene_token"] != row["scene_token"]:
                            raise RuntimeError(f"clip {pos}: OpenDWM item is another scene")
                        r = it["frames"][t][cam]
                        if r["token"] not in tok:
                            raise RuntimeError(f"clip {pos} t {t} {cam}: OpenDWM sweep token not in the pkl")
                        i, c = tok[r["token"]]
                        if c != cam or os.path.basename(infos[i]["cams"][cam]["data_path"]) != os.path.basename(r["filename"]):
                            raise RuntimeError(f"clip {pos} t {t} {cam}: token / image mismatch")
                        fr.append((infos[i], cam))
                sweeps.append(fr)
            for fr in sweeps:
                for inf, _ in fr:
                    if not same_rig(rig, rig_of(inf)):
                        raise RuntimeError(f"clip {pos}: sweep calibration differs from the clip rig")
            # world raster around every camera position of this timing
            cams_xy = []
            for fr in sweeps:
                for inf, cam in fr:
                    r = inf["cams"][cam]
                    cams_xy.append(np.asarray(r["ego2global_translation"][:2], np.float64))
            cams_xy = np.array(cams_xy)
            lo, hi = cams_xy.min(0) - cc.RASTER_MARGIN, cams_xy.max(0) + cc.RASTER_MARGIN
            raster = maps[loc].raster(lo[0], lo[1], hi[0], hi[1])
            cache = {}
            road = np.zeros((cc.NFR, 6, cc.H, cc.W), bool)
            region = np.zeros_like(road)
            near = np.zeros_like(road)
            recs, sweep_tok, img_name, info_ts, cam_ts = [], [], [], [], []
            for k, fr in enumerate(sweeps):
                for ci, (inf, cam) in enumerate(fr):
                    r = inf["cams"][cam]
                    key = r["sample_data_token"]
                    if key not in cache:
                        cache[key] = sweep_gt(inf, cam, mask[ci], raster, lo)
                    rr, road[k, ci], region[k, ci], near[k, ci] = cache[key]
                    recs += [(k, ci) + x for x in rr]
                    sweep_tok.append(key)
                    img_name.append(os.path.basename(r["data_path"]))
                    info_ts.append(int(inf["timestamp"]))
                    cam_ts.append(float(r["timestamp"]))
            cols = list(zip(*recs)) if recs else [[]] * 18
            f32 = lambda v: np.asarray(v, np.float32)
            tmp = out / timing / f"clip{pos:03d}.tmp.npz"
            np.savez_compressed(
                tmp, test_index=pos, scene_token=row["scene_token"], rig_key=rig_key, location=loc,
                mask=cc.pack(mask), road=cc.pack(road), region=cc.pack(region), near=cc.pack(near),
                sweep_token=np.array(sweep_tok).reshape(cc.NFR, 6), image_name=np.array(img_name).reshape(cc.NFR, 6),
                info_ts=np.array(info_ts, np.int64).reshape(cc.NFR, 6), cam_ts=np.array(cam_ts).reshape(cc.NFR, 6),
                b_t=np.asarray(cols[0], np.int16), b_cam=np.asarray(cols[1], np.int8), b_cls=np.asarray(cols[2], np.int8),
                b_role=np.asarray(cols[3], np.int8), b_box=np.stack([f32(cols[4]), f32(cols[5]), f32(cols[6]), f32(cols[7])], 1)
                if recs else np.zeros((0, 4), np.float32),
                b_vis=np.asarray(cols[8], np.int8), b_dist=f32(cols[9]), b_depth=f32(cols[10]), b_vmask=f32(cols[11]),
                b_occ=f32(cols[12]), b_hfull=f32(cols[13]), b_allfront=np.asarray(cols[14], np.int8),
                b_npts=np.asarray(cols[15], np.int32), b_name=np.asarray(cols[16], dtype="U40"),
                b_inst=np.asarray(cols[17], dtype="U40"),
                protocol=json.dumps({"timing": timing, "plan": "CONTROLLABILITY_PLAN_20261021.md (frozen 2026-10-21)",
                                     "mask": "onebev_inverse_own & ~top52", "near_plane_m": cc.NEAR}))
            os.replace(tmp, out / timing / f"clip{pos:03d}.npz")
            nreq = sum(1 for x in recs if x[3] == cc.REQ)
            print(f"[gt] clip{pos:03d} {timing}: {len(cache)} sweeps, {len(recs)} box records ({nreq} required), "
                  f"road {road.sum()} / region {region.sum()} px", flush=True)
        if (n_done + 1) % 10 == 0:
            print(f"[gt] {n_done + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print("CTRL_GT_DONE", flush=True)


if __name__ == "__main__":
    main()
