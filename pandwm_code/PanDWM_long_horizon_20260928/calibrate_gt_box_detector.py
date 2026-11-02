"""Calibrate a COCO detector and box projection on GT panorama RGB first.

No generated-frame box IoU is computed here.  The detector's view-domain and
projection errors must be quantified before such a metric can be used.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.transforms.functional import to_tensor

import dwm.common
from dwm_pano.datasets.instance_tracks import frame_instance_half_boxes
from dwm_pano.datasets.nuscenes_pano import (
    _slice_full_pano_to_halves, _slice_full_pano_to_halves_nearest)
from dwm_pano.datasets.pano_via_stitch import BBOX_COLORS


GT_TO_COCO = {
    "car": 3, "truck": 8, "bus": 6, "trailer": 8,
    "construction_vehicle": 8, "motorcycle": 4, "bicycle": 2,
    "pedestrian": 1,
}


def iou(a, b):
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-9))


def count_matches(gt, pred, threshold):
    candidates = sorted(
        ((iou(g["box"], p["box"]), gi, pi)
         for gi, g in enumerate(gt) for pi, p in enumerate(pred)
         if g["category"] == p["category"]), reverse=True)
    used_g, used_p = set(), set()
    for score, gi, pi in candidates:
        if score < threshold:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    return len(used_g)


def rendered_box_calibration(info, bbox_dir, gt, pred, width, height):
    path = Path(bbox_dir) / f"{int(info['timestamp'])}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    halves = _slice_full_pano_to_halves_nearest(
        Image.open(path).convert("RGB"), width, height)
    result = [[], []]
    line_metrics = [{}, {}]
    for view, half in enumerate(halves):
        image = np.asarray(half)
        counts = Counter(g["name"] for g in gt[view])
        for g in gt[view]:
            name = g["name"]
            if counts[name] != 1 or name not in BBOX_COLORS:
                continue
            color = np.asarray(BBOX_COLORS[name])
            mask = np.all(np.abs(image.astype(np.int16) - color) <= 2, axis=-1)
            ys, xs = np.nonzero(mask)
            if len(xs) < 10:
                continue
            rendered = np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            result[view].append(iou(g["box"], rendered))
        condition_mask = np.zeros((height, width), dtype=bool)
        covered_mask = np.zeros((height, width), dtype=bool)
        evidenced = 0
        for category in sorted(set(GT_TO_COCO.values())):
            class_mask = np.zeros((height, width), dtype=bool)
            for name, coco in GT_TO_COCO.items():
                if coco == category and name in BBOX_COLORS:
                    color = np.asarray(BBOX_COLORS[name])
                    class_mask |= np.all(
                        np.abs(image.astype(np.int16) - color) <= 2,
                        axis=-1)
            condition_mask |= class_mask
            class_covered = np.zeros((height, width), dtype=bool)
            for detection in pred[view]:
                if detection["category"] != category:
                    continue
                x0, y0, x1, y1 = np.asarray(detection["box"]).round().astype(int)
                x0, x1 = np.clip((x0, x1), 0, width)
                y0, y1 = np.clip((y0, y1), 0, height)
                if x1 <= x0 or y1 <= y0:
                    continue
                if int(class_mask[y0:y1, x0:x1].sum()) >= 10:
                    evidenced += 1
                class_covered[y0:y1, x0:x1] = True
            covered_mask |= class_mask & class_covered
        line_metrics[view] = {
            "rendered_condition_line_pixels": int(condition_mask.sum()),
            "rendered_condition_line_pixels_covered_by_detector": int(
                covered_mask.sum()),
            "detections_with_same_class_line_evidence": evidenced,
        }
    return result, line_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-limit", type=int, default=8)
    parser.add_argument("--frames", type=int, nargs="+", default=(18, 66, 130, 194))
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.base_config.read_text())
    ds = config["validation_dataset"]["base_dataset"]
    ds["sequence_length"] = 195
    ds["fps_stride_tuples"] = [[10, 20]]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    raw = dataset.base_dataset
    split = json.loads(args.scene_manifest.read_text())
    scenes = split["dev"][:args.scene_limit]
    if len(scenes) != args.scene_limit:
        raise ValueError("insufficient dev scenes")
    indices = {item["scene_token"]: i for i, item in enumerate(raw.clip_audit)}
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights).cuda().eval()
    model.transform.min_size = (256,)
    model.transform.max_size = 2304
    width, height = raw.half_size
    if (width, height) != (2304, 256):
        raise ValueError("calibration expects model's 2304x256 half panoramas")
    records = []
    for scene in scenes:
        index = indices[scene["scene_token"]]
        for frame in args.frames:
            info = raw.clips[index][frame]
            ts = int(info["timestamp"])
            full = Image.open(Path(raw.rgb_dir) / f"{ts}.jpg").convert("RGB")
            halves = _slice_full_pano_to_halves(full, width, height)
            outputs = model([to_tensor(image).cuda() for image in halves])
            ids, boxes, valid = frame_instance_half_boxes(
                info, half_size=(width, height), min_visibility=2)
            names = dict(zip(info["gt_box_ids"], info["gt_names"]))
            per_view_gt = [[], []]
            per_view_pred = [[], []]
            for token, per_box, per_valid in zip(ids, boxes, valid):
                name = names.get(token, "")
                if name not in GT_TO_COCO:
                    continue
                for view in range(2):
                    if per_valid[view]:
                        xyxy = per_box[view] * [width, height, width, height]
                        per_view_gt[view].append({"box": xyxy,
                                                  "category": GT_TO_COCO[name],
                                                  "name": name})
            for view, out in enumerate(outputs):
                for box, label, score in zip(
                        out["boxes"].cpu().numpy(),
                        out["labels"].cpu().numpy(),
                        out["scores"].cpu().numpy()):
                    if score < args.score_threshold:
                        break
                    if int(label) in set(GT_TO_COCO.values()):
                        per_view_pred[view].append({"box": box,
                                                    "category": int(label),
                                                    "score": float(score)})
            projection_ious, line_metrics = rendered_box_calibration(
                info, raw.bbox_dir, per_view_gt, per_view_pred,
                width, height)
            for view in range(2):
                g, p = per_view_gt[view], per_view_pred[view]
                record = {
                    "scene_token": scene["scene_token"], "frame": frame,
                    "timestamp_us": ts, "view": view,
                    "gt_count": len(g), "detection_count": len(p),
                    "match_iou_0p3": count_matches(g, p, 0.3),
                    "match_iou_0p5": count_matches(g, p, 0.5),
                    "projection_iou_single_class": projection_ious[view],
                    **line_metrics[view],
                }
                records.append(record)
                if frame == args.frames[0]:
                    canvas = halves[view].copy()
                    draw = ImageDraw.Draw(canvas)
                    for item in g:
                        draw.rectangle(item["box"].tolist(), outline="lime", width=2)
                    for item in p:
                        draw.rectangle(item["box"].tolist(), outline="red", width=2)
                    canvas.save(args.output /
                                f"gt_scene{scene['index']}_view{view}.png")
            print(f"[gt-calibration] scene={scene['index']} frame={frame}",
                  flush=True)
    n_gt = sum(r["gt_count"] for r in records)
    n_pred = sum(r["detection_count"] for r in records)
    projection = [x for r in records for x in r["projection_iou_single_class"]]
    n_line = sum(r["rendered_condition_line_pixels"] for r in records)
    n_covered = sum(r["rendered_condition_line_pixels_covered_by_detector"]
                    for r in records)
    n_evidenced = sum(r["detections_with_same_class_line_evidence"]
                      for r in records)
    summary = {
        "detector": "torchvision Faster R-CNN ResNet50 FPN v2 COCO DEFAULT",
        "domain": "GT resized panorama halves, no generated frames",
        "score_threshold": args.score_threshold,
        "n_gt_boxes": n_gt, "n_detections": n_pred,
        "recall_iou_0p3": sum(r["match_iou_0p3"] for r in records) / max(n_gt, 1),
        "recall_iou_0p5": sum(r["match_iou_0p5"] for r in records) / max(n_gt, 1),
        "precision_iou_0p5": sum(r["match_iou_0p5"] for r in records) / max(n_pred, 1),
        "single_class_projection_count": len(projection),
        "single_class_projection_iou_median": (
            float(np.median(projection)) if projection else None),
        "rendered_condition_line_coverage_by_detector": n_covered / max(n_line, 1),
        "detector_same_class_line_evidence_rate": n_evidenced / max(n_pred, 1),
        "records": records,
        "caveat": "detector errors, annotation errors and geometric projection errors are not fully separable; map segmentation remains uncalibrated",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
