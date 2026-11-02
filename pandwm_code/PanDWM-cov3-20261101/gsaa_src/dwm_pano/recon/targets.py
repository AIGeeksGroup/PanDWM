"""Targets of the Gaussian reconstruction table (tab:gs_recon) on Test-S (2026-10-22,
PanDWM-gs-recon-20261022/RECON_PLAN_20261022.md).

For every Test-S clip (tests213_manifest_20261018.json; dataset_index into the frozen eval enumeration: val pkl,
35 frames, fps_stride_tuples [[10, 5.0]], the dataset's bad_timestamps_us) the input frame is clip frame 20; the 4D
history frames are clip frames 0, 5, 10, 15; the targets are
    next   clip frame 21,
    half   clip frame 25 (0.5 s after the input),
    back   the first 12 Hz scene frame behind the input whose path length (ego x-y, frame to frame) is >= --distance,
    fwd    the same ahead of the input.
For every target and each of the six cameras (bridge CAM_ORDER) the file gives the recorded image (raw nuScenes path),
its 3x3 intrinsics at 1600x900, and the camera-to-ego transform in the INPUT frame's ego system,
    c2w = inv(E_input) @ E_cam(target camera timestamp) @ S_cam,
with S = sensor2ego, E_cam the camera's own recorded ego pose and E_input the input frame's camera-time ego pose, the
mean of its six cameras' ego poses (camera_time_pose).  The info-level ego2global of the 12 Hz pkl is the ego pose of
the info's LiDAR sweep, which for interpolated frames lies up to 250 ms before the cameras (section 44); the scene's
Gaussians live in the camera rig, so the camera-time pose is their frame ("ego2global_pkl" keeps the pkl pose).
The LiDAR sweep of the target info has lidar_to_cam[c] = inv(E_cam S_cam) @ E_info @ L2E (E_info = that sweep's own
pose, so the transform is exact); the dynamic 3D boxes are given in the info's LiDAR frame as the pkl stores them,
mmdet3d v0.x (x, y, z, w, l, h, -yaw - pi/2), for the static / all depth split.  "input" carries the same camera
entries for the input frame itself (the six recorded images the external methods take as input) and its lidar2ego.

    python -m dwm_pano.recon.targets --manifest tests213_manifest_20261018.json --data-pkl <val pkl>
        --meta-dir <v1.0-trainval> --nusc-root <raw nuScenes root> --out targets.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from dwm_pano.datasets.nuscenes_pano import _enumerate_pano_clips, _group_infos_by_scene, _load_token_to_scene

CAM_ORDER = ("CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT")
INPUT_FRAME = 20
HISTORY_FRAMES = (0, 5, 10, 15)
DYNAMIC_CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle")
BAD_TIMESTAMPS_US = [1531884295049111]


def pose(translation, rotation_wxyz):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    m[:3, 3] = np.asarray(translation, dtype=np.float64)
    return m


def camera_time_pose(info, as_quaternion=False):
    """Mean ego pose of the info's six cameras (each at its own timestamp): translations averaged, rotations by the
    sign-aligned, normalised quaternion mean.  4x4, or (translation, wxyz quaternion) with as_quaternion."""
    cams = [info["cams"][name] for name in CAM_ORDER]
    t = np.mean([np.asarray(c["ego2global_translation"], dtype=np.float64) for c in cams], axis=0)
    qs = [np.asarray(c["ego2global_rotation"], dtype=np.float64) for c in cams]
    qs = [q if np.dot(q, qs[0]) >= 0 else -q for q in qs]
    q = np.mean(qs, axis=0)
    q = q / np.linalg.norm(q)
    return (t, q) if as_quaternion else pose(t, q)


def camera_time_infos(infos):
    """Copies of the infos whose ego2global is the camera-time pose (camera_time_pose) and whose lidar2ego is adjusted
    so that ego2global @ lidar2ego (where the LiDAR points and gt_boxes live) is unchanged."""
    from pyquaternion import Quaternion
    out = []
    for info in infos:
        t_ref, q_ref = camera_time_pose(info, as_quaternion=True)
        e_ref, q_ref = pose(t_ref, q_ref), Quaternion(q_ref)
        q_pkl, q_l2e = Quaternion(info["ego2global_rotation"]), Quaternion(info["lidar2ego_rotation"])
        lidar_in_ref = np.linalg.inv(e_ref) @ pose(info["ego2global_translation"], info["ego2global_rotation"]) \
            @ pose(info["lidar2ego_translation"], info["lidar2ego_rotation"])
        new = dict(info)
        new["ego2global_translation"] = e_ref[:3, 3].tolist()
        new["ego2global_rotation"] = q_ref.elements.tolist()
        new["lidar2ego_translation"] = lidar_in_ref[:3, 3].tolist()
        new["lidar2ego_rotation"] = (q_ref.inverse * q_pkl * q_l2e).normalised.elements.tolist()
        out.append(new)
    return out


def raw_path(data_path, nusc_root):
    rel = str(data_path).replace("\\", "/")
    marker = "nuscenes/"
    if marker not in rel:
        raise ValueError(f"unexpected data_path {data_path!r}")
    return str(Path(nusc_root) / rel.split(marker, 1)[1])


def walk(scene, index, distance, step):
    """First index from `index` in direction `step` (+1 / -1) whose path length reaches `distance`, else None."""
    pos = [np.asarray(info["ego2global_translation"], dtype=np.float64)[:2] for info in scene]
    travelled, i = 0.0, index
    while 0 <= i + step < len(scene):
        travelled += float(np.linalg.norm(pos[i + step] - pos[i]))
        i += step
        if travelled >= distance:
            return i, travelled
    return None, travelled


def target_entry(info, e_input_inv, nusc_root):
    cams = []
    e_info = pose(info["ego2global_translation"], info["ego2global_rotation"])
    l2e = pose(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    lidar_to_cam = []
    for name in CAM_ORDER:
        c = info["cams"][name]
        e_cam = pose(c["ego2global_translation"], c["ego2global_rotation"])
        s_cam = pose(c["sensor2ego_translation"], c["sensor2ego_rotation"])
        cams.append({"name": name, "image": raw_path(c["data_path"], nusc_root), "timestamp": int(c["timestamp"]),
                     "K": np.asarray(c["camera_intrinsics"], dtype=np.float64).tolist(), "size": [1600, 900],
                     "c2w": (e_input_inv @ e_cam @ s_cam).tolist(), "sensor2ego": s_cam.tolist()})
        lidar_to_cam.append((np.linalg.inv(e_cam @ s_cam) @ e_info @ l2e).tolist())
    raw_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float64)
    boxes = raw_boxes.reshape(raw_boxes.shape[0], -1)[:, :7] if raw_boxes.size else np.zeros((0, 7))
    names = [str(n) for n in info.get("gt_names", [])]
    keep = [i for i, n in enumerate(names) if n in DYNAMIC_CLASSES]
    return {"timestamp": int(info["timestamp"]), "token": info["token"], "cams": cams,
            "lidar": {"path": raw_path(info["lidar_path"], nusc_root), "lidar_to_cam": lidar_to_cam,
                      "dynamic_boxes": boxes[keep, :7].tolist() if keep else [],
                      "dynamic_names": [names[i] for i in keep]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--meta-dir", required=True)
    p.add_argument("--nusc-root", required=True)
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    pkl = pickle.load(open(args.data_pkl, "rb"))
    scenes = _group_infos_by_scene(pkl["infos"], token2scene=_load_token_to_scene(args.meta_dir), strict=True)
    pairs = list(_enumerate_pano_clips(scenes, 35, manifest["dataset_fps_stride_tuples"],
                                       bad_timestamps_us=BAD_TIMESTAMPS_US))
    if len(pairs) != manifest["dataset_len"]:
        raise RuntimeError(f"enumeration gives {len(pairs)} clips, manifest says {manifest['dataset_len']}")
    samples, missing = [], {"back": 0, "fwd": 0}
    for row in manifest["clips"]:
        clip, audit = pairs[int(row["dataset_index"])]
        if audit["scene_token"] != row["scene_token"] or int(clip[0]["timestamp"]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"test_index {row['test_index']}: dataset index is not the manifest clip")
        scene = scenes[row["scene_token"]]
        stamps = [int(i["timestamp"]) for i in scene]
        t_info = clip[INPUT_FRAME]
        t_index = stamps.index(int(t_info["timestamp"]))
        e_input = camera_time_pose(t_info)
        e_input_inv = np.linalg.inv(e_input)
        targets = {"next": clip[INPUT_FRAME + 1], "half": clip[INPUT_FRAME + 5]}
        travelled = {}
        for name, step in (("back", -1), ("fwd", 1)):
            j, dist = walk(scene, t_index, args.distance, step)
            travelled[name] = dist
            if j is None:
                missing[name] += 1
            else:
                targets[name] = scene[j]
        history = [int(clip[i]["timestamp"]) for i in HISTORY_FRAMES]
        samples.append({
            "test_index": int(row["test_index"]), "scene_token": row["scene_token"],
            "anchor_timestamp_us": int(row["anchor_timestamp_us"]), "dataset_index": int(row["dataset_index"]),
            "input": dict({"clip_frame": INPUT_FRAME, "timestamp": int(t_info["timestamp"]), "token": t_info["token"],
                           "ego2global": e_input.tolist(), "ego2global_source": "camera_time_pose",
                           "ego2global_pkl": pose(t_info["ego2global_translation"],
                                                  t_info["ego2global_rotation"]).tolist(),
                           "lidar2ego": pose(t_info["lidar2ego_translation"], t_info["lidar2ego_rotation"]).tolist()},
                          cams=target_entry(t_info, e_input_inv, args.nusc_root)["cams"]),
            "history_clip_frames": list(HISTORY_FRAMES), "history_timestamps": history,
            "targets": {name: dict(target_entry(info, e_input_inv, args.nusc_root),
                                   in_history=int(info["timestamp"]) in history,
                                   path_m=(travelled.get(name)))
                        for name, info in targets.items()}})
    out = {"note": "tab:gs_recon targets on Test-S (dwm_pano/recon/targets.py)", "distance_m": args.distance,
           "input_frame": INPUT_FRAME, "history_frames": list(HISTORY_FRAMES), "cam_order": list(CAM_ORDER),
           "manifest": str(args.manifest), "count": len(samples), "missing": missing, "samples": samples}
    args.out.write_text(json.dumps(out) + "\n")
    print(f"[recon-targets] {len(samples)} samples, missing {missing}; "
          f"history overlaps {sum(t.get('in_history', False) for s in samples for t in s['targets'].values())}")


if __name__ == "__main__":
    main()
