"""Controllability predictions (2026-10-21; protocol CONTROLLABILITY_PLAN_20261021.md, frozen).

Every arm becomes the uint8 [32, 6, 3, 256, 448] camera views of the camera-space protocol exactly as in
cam_band_feats.py (latents: decode + stitch + LegacyInverseOwnGeometry resample; recband: recorded canvas resampled;
views: six-camera run, resized to 256x448 if needed), then x = u8 / 255 with x = 0.5 outside the protocol mask
(onebev_inverse_own and native rows 0..51). On x:
  frcnn    torchvision fasterrcnn_resnet50_fpn_v2 COCO_V1, defaults (min 800 / max 1333, score 0.05, NMS 0.5, 100 dets)
  dfine    D-FINE-X Objects365+COCO (robustness only): 640x640 bilinear, post_process_object_detection(0.05), top 100
  m2f_cs   Mask2Former Swin-L Cityscapes semantic: 2x bilinear to 512x896, ImageNet mean/std, HF
           post_process_semantic_segmentation(target 256x448); road = Cityscapes label "road"
  m2f_mv   Mask2Former Swin-L Mapillary Vistas semantic (robustness only), same input; road = MV road-surface classes
Detections kept: COCO person / car / bus / truck (torchvision ids 1 / 3 / 6 / 8), all scores >= 0.05.
Output <out>/<arm>__<det>+<seg>/clipNNN.npz (NNN = Val-150 test_index): det_img (t * 6 + camera), det_label, det_score,
det_box (xyxy, 256x448 edge coords), road (packed bits [32, 6, 14336]), mask (packed [6, 14336]); resumable.
--unmasked NAME: that arm's networks see the unmasked views (diagnostic G5; scoring still uses the mask).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

import cam_band_feats as cbf
import ctrl_common as cc
from long_feats import I3D
from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance

# Mapillary Vistas v1.2 road-surface classes (robustness segmenter; fixed before any result was seen)
MV_ROAD = ("road", "crosswalk - plain", "lane marking - crosswalk", "lane marking - general", "manhole", "pothole",
           "service lane", "bike lane")


class FRCNN:
    tag = "frcnn"

    def __init__(self, device, model_dir):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        self.m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval().to(device)

    @torch.no_grad()
    def __call__(self, x, bs=24):
        out = []
        for k in range(0, len(x), bs):
            for o in self.m(list(x[k:k + bs])):
                lab = o["labels"].cpu().numpy()
                keep = np.isin(lab, list(cc.COCO_CLASS))
                out.append((o["boxes"].cpu().numpy()[keep], o["scores"].cpu().numpy()[keep], lab[keep]))
        return out


class DFINE:
    tag = "dfine"

    def __init__(self, device, model_dir):
        from transformers import DFineForObjectDetection, RTDetrImageProcessor
        path = os.path.join(model_dir, "dfine-xlarge-obj2coco")
        self.m = DFineForObjectDetection.from_pretrained(path).eval().to(device)
        self.proc = RTDetrImageProcessor.from_pretrained(path)
        self.lab = {int(k): cc.COCO_NAME_TO_ID[v] for k, v in self.m.config.id2label.items() if v in cc.COCO_NAME_TO_ID}
        if sorted(self.lab.values()) != [1, 3, 6, 8]:
            raise RuntimeError(f"D-FINE labels {self.m.config.id2label}")
        size = self.proc.size
        self.hw = (size["height"], size["width"])

    @torch.no_grad()
    def __call__(self, x, bs=32):
        out = []
        for k in range(0, len(x), bs):
            xb = F.interpolate(x[k:k + bs], size=self.hw, mode="bilinear", align_corners=False)
            o = self.m(pixel_values=xb)
            res = self.proc.post_process_object_detection(o, threshold=cc.DET_MIN_SCORE,
                                                          target_sizes=[(cc.H, cc.W)] * len(xb))
            for r in res:
                s, lab, b = r["scores"].cpu().numpy(), r["labels"].cpu().numpy(), r["boxes"].cpu().numpy()
                order = np.argsort(-s, kind="mergesort")[:cc.MAX_DETS]
                s, lab, b = s[order], lab[order], b[order]
                keep = np.isin(lab, list(self.lab))
                out.append((b[keep], s[keep], np.array([self.lab[int(v)] for v in lab[keep]], np.int64)))
        return out


class M2F:
    def __init__(self, device, model_dir, repo, road_names):
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        path = os.path.join(model_dir, repo)
        self.m = Mask2FormerForUniversalSegmentation.from_pretrained(path).eval().to(device)
        self.proc = Mask2FormerImageProcessor.from_pretrained(path)
        self.mean = torch.tensor(self.proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.proc.image_std, device=device).view(1, 3, 1, 1)
        names = {int(k): v.lower() for k, v in self.m.config.id2label.items()}
        ids = [k for k, v in names.items() if v in road_names]
        if len(ids) != len(road_names):
            raise RuntimeError(f"road classes {road_names} not all in {names}")
        self.road = torch.tensor(ids, device=device)
        print(f"[m2f] {repo}: road ids {ids} = {[names[i] for i in ids]}", flush=True)

    @torch.no_grad()
    def __call__(self, x, bs=16):
        out = []
        for k in range(0, len(x), bs):
            xb = F.interpolate(x[k:k + bs], size=(2 * cc.H, 2 * cc.W), mode="bilinear", align_corners=False)
            o = self.m(pixel_values=(xb - self.mean) / self.std)
            seg = self.proc.post_process_semantic_segmentation(o, target_sizes=[(cc.H, cc.W)] * len(xb))
            out += [torch.isin(s, self.road).cpu().numpy() for s in seg]
        return out


def make_models(names, device, model_dir):
    det = seg = None
    for n in names:
        if n == "frcnn":
            det = FRCNN(device, model_dir)
        elif n == "dfine":
            det = DFINE(device, model_dir)
        elif n == "m2f_cs":
            seg = M2F(device, model_dir, "mask2former-swin-large-cityscapes-semantic", ("road",))
            seg.tag = "m2f_cs"
        elif n == "m2f_mv":
            seg = M2F(device, model_dir, "mask2former-swin-large-mapillary-vistas-semantic", MV_ROAD)
            seg.tag = "m2f_mv"
        else:
            raise ValueError(n)
    if det is None or seg is None:
        raise ValueError("--models needs one detector and one segmenter")
    return det, seg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=latents:R1[+R2] | NAME=recband | NAME=views:RUN")
    ap.add_argument("--map", default="val150_scene_map.json")
    ap.add_argument("--blend", type=int, default=128)
    ap.add_argument("--source-ref", action="append", default=[], help="NAME=pano feature npz (checked every --check-every clips)")
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument("--positions", default="")
    ap.add_argument("--models", default="frcnn,m2f_cs")
    ap.add_argument("--model-dir", default="/path/to/pandwm_root/models_2026/ctrl_eval")
    ap.add_argument("--unmasked", action="append", default=[])
    a = ap.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    print(f"CTRL_PRED_START models={a.models} arms={a.arm} torch {torch.__version__} tf32 conv "
          f"{torch.backends.cudnn.allow_tf32} matmul {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    det, seg = make_models(a.models.split(","), device, a.model_dir)
    tag = f"{det.tag}+{seg.tag}"
    arms = {}
    for spec in a.arm:
        name, kind = spec.split("=", 1)
        kind, _, arg = kind.partition(":")
        if kind not in ("latents", "recband", "views"):
            raise ValueError(spec)
        d = Path(a.out_dir) / f"{name}__{tag}"
        d.mkdir(parents=True, exist_ok=True)
        arms[name] = {"kind": kind, "arg": arg, "dir": d}
    refs = dict(s.split("=", 1) for s in a.source_ref)
    config, dataset, collate, rows = cbf.load_val150(a.base_config)
    if a.positions:
        keep = {int(p) for p in a.positions.split(",")}
        rows = [r for r in rows if int(r["test_index"]) in keep]
    by_test = {int(v["test_index"]): (int(k), v["scene_token"]) for k, v in json.load(open(a.map)).items()}
    decode = cbf.make_decoder(config, device) if any(v["kind"] == "latents" for v in arms.values()) else None
    fid = fvd = None
    if refs:
        fid = FrechetInceptionDistance(normalize=True).to(device)
        fvd = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(device)
    geometry = cbf.LegacyInverseOwnGeometry()
    t_start = time.time()
    times = {"views": 0.0, "det": 0.0, "seg": 0.0}
    n_img = 0
    for n, row in enumerate(rows):
        pos = int(row["test_index"])
        todo = [k for k, v in arms.items() if not (v["dir"] / f"clip{pos:03d}.npz").exists()]
        if not todo:
            continue
        batch = collate([dataset[int(row["relaxed_index"])]])
        if int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"clip {pos}: dataset clip is not the manifest clip")
        rig = cbf.batch_rig(batch)
        geo, mask = cc.eval_mask(geometry, *rig)
        geo = dict(geo, valid=mask)
        for name in todo:
            arm = arms[name]
            t0 = time.time()
            if arm["kind"] == "views":
                ci, token = by_test[pos]
                if token != row["scene_token"]:
                    raise RuntimeError(f"map: clip {ci} is scene {token}, Val-150 {pos} is {row['scene_token']}")
                u8, seed, src = cbf.run_views(arm["arg"], ci, row, rig, device)
            else:
                if arm["kind"] == "recband":
                    canvas, seed, src = cbf.recorded_canvas(batch, device, a.blend), -1, "recorded canvas"
                else:
                    canvas, seed, src = cbf.ours_canvas(arm["arg"].split("+"), row, decode, a.blend)
                if name in refs and pos % a.check_every == 0:
                    cbf.pano_source_check(name, canvas, refs[name], pos, fid, fvd, True)
                u8 = cbf.canvas_to_cams(canvas, geo)
            x = u8.float() / 255.0 if name in a.unmasked else cbf.masked_float(u8, mask)
            x = x.flatten(0, 1)  # [192, 3, H, W], image index = t * 6 + camera
            torch.cuda.synchronize()
            t1 = time.time()
            dets = det(x)
            torch.cuda.synchronize()
            t2 = time.time()
            road = np.stack(seg(x)).reshape(cc.NFR, 6, cc.H, cc.W)
            torch.cuda.synchronize()
            t3 = time.time()
            times["views"] += t1 - t0
            times["det"] += t2 - t1
            times["seg"] += t3 - t2
            n_img += len(x)
            img = np.concatenate([np.full(len(d[1]), i, np.int16) for i, d in enumerate(dets)])
            tmp = arm["dir"] / f"clip{pos:03d}.tmp.npz"
            np.savez_compressed(
                tmp, test_index=pos, scene_token=row["scene_token"], seed=seed, src=src,
                det_img=img, det_label=np.concatenate([d[2] for d in dets]).astype(np.int16),
                det_score=np.concatenate([d[1] for d in dets]).astype(np.float32),
                det_box=np.concatenate([d[0].reshape(-1, 4) for d in dets]).astype(np.float32),
                road=cc.pack(road), mask=cc.pack(mask), unmasked_input=name in a.unmasked,
                protocol=json.dumps({"models": tag, "plan": "CONTROLLABILITY_PLAN_20261021.md (frozen 2026-10-21)",
                                     "mask": "onebev_inverse_own & ~top52", "blend": a.blend}))
            os.replace(tmp, arm["dir"] / f"clip{pos:03d}.npz")
        if (n + 1) % 5 == 0 or n + 1 == len(rows) or n < 2:
            el = time.time() - t_start
            print(f"[ctrl_pred] {n + 1}/{len(rows)} {el:.0f}s; per image: views {1e3 * times['views'] / max(n_img, 1):.1f} ms, "
                  f"{det.tag} {1e3 * times['det'] / max(n_img, 1):.1f} ms, {seg.tag} {1e3 * times['seg'] / max(n_img, 1):.1f} ms "
                  f"({n_img} images)", flush=True)
    print("CTRL_PRED_DONE", flush=True)


if __name__ == "__main__":
    main()
