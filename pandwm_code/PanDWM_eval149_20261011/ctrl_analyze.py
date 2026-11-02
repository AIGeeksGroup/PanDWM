"""Controllability metrics (2026-10-21; protocol CONTROLLABILITY_PLAN_20261021.md sections 6-8, frozen).

Per image (clip frame 3..34 x camera) and class (vehicle = COCO car / bus / truck, pedestrian = COCO person):
detections -> vehicle class-agnostic NMS 0.5 -> centre inside the protocol mask -> top 100 by score; COCOeval matching
at IoU 0.5 (required GT first; ignore regions match by IoA >= 0.5 like iscrowd and may absorb several detections);
unmatched detections lower than 16 px are ignored (size range on both sides). Recall = TP(score >= 0.5) / required GT;
AP50 = COCO 101-point interpolated precision, pooled over all images of the arm. Obj. recall / Obj. AP = mean of the two
classes. Road IoU = sum TP / sum (TP + FP + FN) over the region pixels. Intervals: point +- 1.96 SE of a delete-one-scene
jackknife (Val-150: one clip per scene); seed groups: mean of the seed values (SD ddof=1), jackknife of the seed mean.

  python3 ctrl_analyze.py --gt-dir GT --pred-dir PRED --tag frcnn+m2f_cs --out analysis.json \
     --arm 'ours=P_s42+P_s1000+P_s2026@band' --arm 'unimlvg_P=unimlvg_P_s42@opendwm' --arm 'rec448=rec448@opendwm' \
     --arm 'recband=recband@band' --pair ours:unimlvg_P --pair ours:recband --pair unimlvg_P:rec448 \
     --floor 'floor_rec448=rec448@opendwm' --floor 'floor_recband=recband@band' [--subset common] [--diag rec448]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ctrl_common as cc

NPOS = 150


def load_dir(d):
    out = {}
    for f in sorted(Path(d).glob("clip*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=False)
        out[int(z["test_index"])] = {k: z[k] for k in z.files}
    return out


def prep_dets(p, img, mask_c):
    """Detections of one image -> {cls: (boxes, scores)} after mapping, vehicle NMS, mask-centre filter, top 100."""
    sel = p["_by_img"].get(img)
    out = {}
    for cls in (cc.VEH, cc.PED):
        if sel is None:
            out[cls] = (np.zeros((0, 4)), np.zeros(0))
            continue
        lab = p["det_label"][sel]
        k = sel[np.array([cc.COCO_CLASS.get(int(v), -1) == cls for v in lab], bool)]
        b, s = p["det_box"][k].astype(np.float64), p["det_score"][k].astype(np.float64)
        if cls == cc.VEH and len(b):
            keep = cc.nms(b, s, cc.IOU_T)
            b, s = b[keep], s[keep]
        if len(b):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            j, i = np.floor(cx).astype(int), np.floor(cy).astype(int)
            ok = (i >= 0) & (i < cc.H) & (j >= 0) & (j < cc.W)
            ok[ok] = mask_c[i[ok], j[ok]]
            b, s = b[ok], s[ok]
        order = np.argsort(-s, kind="mergesort")[:cc.MAX_DETS]
        out[cls] = (b[order], s[order])
    return out


def coco_match(db, ds, gb, gign, size_rule=True):
    """COCOeval.evaluateImg at one IoU threshold: (tp, ignored, matched gt index) per detection (score-sorted)."""
    nd, ng = len(db), len(gb)
    tp = np.zeros(nd, bool)
    ig = np.zeros(nd, bool)
    mi = np.full(nd, -1)
    if ng:
        og = np.argsort(gign.astype(int), kind="mergesort")
        gb, gign = gb[og], gign[og]
        ious = cc.box_iou(db, gb, crowd=gign)
        gtm = np.zeros(ng, bool)
        for d in range(nd):
            iou, m = min(cc.IOU_T, 1 - 1e-10), -1
            for g in range(ng):
                if gtm[g] and not gign[g]:
                    continue
                if m > -1 and not gign[m] and gign[g]:
                    break
                if ious[d, g] < iou:
                    continue
                iou, m = ious[d, g], g
            if m == -1:
                continue
            gtm[m] = True
            mi[d] = og[m]
            ig[d] = gign[m]
            tp[d] = not gign[m]
    if size_rule and nd:
        small = (db[:, 3] - db[:, 1]) < cc.MIN_H
        ig |= (mi < 0) & small
    return tp, ig, mi


def evaluate(pred, gt, positions, gt_of=None, use=None, size_rule=True, diag=False, coco_export=None):
    """Per-clip sufficient statistics of one prediction set against one GT set."""
    P = len(positions)
    det = {c: {"clip": [], "score": [], "tp": []} for c in (cc.VEH, cc.PED)}
    npos = {c: np.zeros(P, np.int64) for c in (cc.VEH, cc.PED)}
    npos_cam = {c: np.zeros((P, 6), np.int64) for c in (cc.VEH, cc.PED)}
    road = np.zeros((P, 6, 3, 3), np.int64)  # [clip, cam, (all, near, far), (tp, fp, fn)]
    region_px = np.zeros((P, 6), np.int64)
    pairs = []
    for k, pos in enumerate(positions):
        p, g = pred[pos], gt[(gt_of or {}).get(pos, pos)]
        if not np.array_equal(p["mask"], g["mask"]):
            raise RuntimeError(f"clip {pos}: prediction mask differs from the GT mask")
        u = np.ones((cc.NFR, 6), bool) if use is None else use[pos]
        pr, gr, rg, nr = p["road"], g["road"], g["region"], g["near"]
        region_px[k] = (cc.popcount(rg) * u).sum(0)
        for z, reg in enumerate((rg, rg & nr, rg & ~nr)):
            for w, bits in enumerate((pr & gr & reg, pr & ~gr & reg, ~pr & gr & reg)):
                cnt = cc.popcount(bits) * u
                road[k, :, z, w] = cnt.sum(0)
        mask = cc.unpack(g["mask"])
        if "_by_img" not in p:
            by = {}
            for i, v in enumerate(p["det_img"]):
                by.setdefault(int(v), []).append(i)
            p["_by_img"] = {kk: np.array(v) for kk, v in by.items()}
        gidx = {}
        for i in range(len(g["b_t"])):
            gidx.setdefault((int(g["b_t"][i]), int(g["b_cam"][i]), int(g["b_cls"][i])), []).append(i)
        for t in range(cc.NFR):
            for c in range(6):
                if not u[t, c]:
                    continue
                dets = prep_dets(p, t * 6 + c, mask[c])
                for cls in (cc.VEH, cc.PED):
                    db, ds = dets[cls]
                    gi = np.array(gidx.get((t, c, cls), []), np.int64)
                    gb = g["b_box"][gi].astype(np.float64).reshape(-1, 4)
                    gign = (g["b_role"][gi] == cc.IGN) if len(gi) else np.zeros(0, bool)
                    tp, ig, mi = coco_match(db, ds, gb, gign, size_rule)
                    keep = ~ig
                    det[cls]["clip"].append(np.full(int(keep.sum()), k))
                    det[cls]["score"].append(ds[keep])
                    det[cls]["tp"].append(tp[keep])
                    nreq = int((~gign).sum())
                    npos[cls][k] += nreq
                    npos_cam[cls][k, c] += nreq
                    if diag:
                        for d in np.flatnonzero(tp & (ds >= cc.SCORE_T)):
                            gbx = gb[mi[d]]
                            pairs.append((cls, c, float(cc.box_iou(db[d:d + 1], gbx[None])[0, 0]),
                                          (db[d, 0] + db[d, 2] - gbx[0] - gbx[2]) / 2, (db[d, 1] + db[d, 3] - gbx[1] - gbx[3]) / 2))
                    if coco_export is not None:
                        coco_export.append((k, t, c, cls, db, ds, gb, gign))
    for cls in det:
        for key in det[cls]:
            det[cls][key] = np.concatenate(det[cls][key]) if det[cls][key] else np.zeros(0)
        det[cls]["tp"] = det[cls]["tp"].astype(bool)
        det[cls]["clip"] = det[cls]["clip"].astype(np.int64)
        order = np.argsort(-det[cls]["score"], kind="mergesort")
        for key in ("clip", "score", "tp"):
            det[cls][key] = det[cls][key][order]
    return {"det": det, "npos": npos, "npos_cam": npos_cam, "road": road, "region_px": region_px, "pairs": pairs}


def ap101(tp, npos):
    """COCO 101-point AP of score-sorted, non-ignored detections."""
    if npos <= 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(~tp)
    rc = tpc / npos
    pr = tpc / (tpc + fpc + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    inds = np.searchsorted(rc, np.linspace(0.0, 1.0, 101), side="left")
    q = np.where(inds < len(pr), pr[np.minimum(inds, len(pr) - 1)], 0.0)
    return float(q.mean())


def metrics(st, drop=None):
    """All metrics of one statistics set; drop = clip position to leave out (jackknife)."""
    m = {}
    for cls, nm in zip((cc.VEH, cc.PED), cc.CLASS_NAMES):
        d = st["det"][cls]
        keep = np.ones(len(d["tp"]), bool) if drop is None else d["clip"] != drop
        tp, sc = d["tp"][keep], d["score"][keep]
        npos = st["npos"][cls].sum() - (0 if drop is None else st["npos"][cls][drop])
        hi = sc >= cc.SCORE_T
        m[f"recall_{nm}"] = float(tp[hi].sum() / npos) if npos else float("nan")
        m[f"precision_{nm}"] = float(tp[hi].sum() / max(hi.sum(), 1))
        m[f"ap50_{nm}"] = ap101(tp, npos)
    m["obj_recall"] = (m["recall_vehicle"] + m["recall_pedestrian"]) / 2
    m["obj_ap50"] = (m["ap50_vehicle"] + m["ap50_pedestrian"]) / 2
    r = st["road"].sum(1)  # [clip, zone, 3]
    if drop is not None:
        r = np.delete(r, drop, 0)
    r = r.sum(0)
    for z, nm in enumerate(("road_iou", "road_iou_near", "road_iou_far")):
        m[nm] = float(r[z, 0] / max(r[z].sum(), 1))
    return m


def per_camera(st):
    out = {}
    for c in range(6):
        r = st["road"][:, c, 0].sum(0)
        out[c] = {"road_iou": float(r[0] / max(r.sum(), 1)),
                  "n_vehicle": int(st["npos_cam"][cc.VEH][:, c].sum()), "n_pedestrian": int(st["npos_cam"][cc.PED][:, c].sum()),
                  "region_px": int(st["region_px"][:, c].sum())}
    return out


def jack(values_full, values_rep):
    """point +- 1.96 SE (delete-one-scene jackknife)."""
    rep = np.asarray(values_rep, np.float64)
    n = len(rep)
    se = float(np.sqrt((n - 1) / n * ((rep - rep.mean()) ** 2).sum()))
    return {"value": float(values_full), "se": se, "ci95": [float(values_full) - 1.96 * se, float(values_full) + 1.96 * se]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--tag", default="frcnn+m2f_cs")
    ap.add_argument("--arm", action="append", default=[], help="NAME=PRED[+PRED...]@TIMING")
    ap.add_argument("--pair", action="append", default=[], help="A:B (difference A - B)")
    ap.add_argument("--floor", action="append", default=[], help="NAME=PRED@TIMING: GT of the next scene with the same rig")
    ap.add_argument("--subset", default="", help="common = only images whose camera sweep is the same in both timings")
    ap.add_argument("--no-size-rule", action="store_true")
    ap.add_argument("--diag", action="append", default=[], help="arm names for matched-pair diagnostics (G3)")
    ap.add_argument("--coco-export", default="", help="ARM:DIR: write COCO json of that arm (first seed) for the parity check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gts = {}

    def gt_of(timing):
        if timing not in gts:
            gts[timing] = load_dir(Path(a.gt_dir) / timing)
        return gts[timing]
    use = None
    if a.subset == "common":
        b, o = gt_of("band"), gt_of("opendwm")
        use = {p: b[p]["sweep_token"] == o[p]["sweep_token"] for p in b}
    preds = {}

    def pred_of(name):
        if name not in preds:
            preds[name] = load_dir(Path(a.pred_dir) / f"{name}__{a.tag}")
        return preds[name]
    out = {"tag": a.tag, "subset": a.subset or "all", "size_rule": not a.no_size_rule, "arms": {}, "pairs": {}, "floors": {},
           "diag": {}, "protocol": "CONTROLLABILITY_PLAN_20261021.md (frozen 2026-10-21)"}
    reps = {}
    export = a.coco_export.split(":", 1) if a.coco_export else None
    specs = [(s, False) for s in a.arm] + [(s, True) for s in a.floor]
    for spec, is_floor in specs:
        name, rest = spec.split("=", 1)
        names, timing = rest.split("@")
        g = gt_of(timing)
        per_seed, rep_seed, seeds = [], [], []
        for pn in names.split("+"):
            p = pred_of(pn)
            positions = sorted(p)
            if positions != sorted(g) or len(positions) != NPOS:
                raise RuntimeError(f"{pn}: {len(positions)} prediction clips vs {len(g)} GT clips")
            remap = None
            if is_floor:
                rigs = {}
                for q in positions:
                    rigs.setdefault(str(g[q]["rig_key"]), []).append(q)
                remap = {}
                for lst in rigs.values():
                    for i, q in enumerate(lst):
                        remap[q] = lst[(i + 1) % len(lst)]
            ex = [] if (export and export[0] == name and pn == names.split("+")[0]) else None
            st = evaluate(p, g, positions, gt_of=remap, use=use, size_rule=not a.no_size_rule,
                          diag=name in a.diag, coco_export=ex)
            full = metrics(st)
            rep = [metrics(st, drop=k) for k in range(len(positions))]
            per_seed.append(full)
            rep_seed.append(rep)
            seeds.append(int(p[positions[0]]["seed"]))
            if name in a.diag and pn == names.split("+")[0]:
                pr = np.array(st["pairs"], np.float64).reshape(-1, 5)
                dg = {}
                for cls, nm in zip((cc.VEH, cc.PED), cc.CLASS_NAMES):
                    for c in range(6):
                        s = pr[(pr[:, 0] == cls) & (pr[:, 1] == c)]
                        dg[f"{nm}_cam{c}"] = {"n": int(len(s)), "median_iou": float(np.median(s[:, 2])) if len(s) else None,
                                              "median_dx": float(np.median(s[:, 3])) if len(s) else None,
                                              "median_dy": float(np.median(s[:, 4])) if len(s) else None}
                out["diag"][name] = {"pairs": dg, "per_camera": {str(k): v for k, v in per_camera(st).items()}}
            if ex is not None:
                export_coco(ex, Path(export[1]), full)
            print(f"[analyze] {name} {pn}: " + ", ".join(f"{k} {v:.4f}" for k, v in full.items()), flush=True)
        keys = per_seed[0].keys()
        value = {k: float(np.mean([s[k] for s in per_seed])) for k in keys}
        rep = {k: [float(np.mean([r[i][k] for r in rep_seed])) for i in range(NPOS)] for k in keys}
        reps[name] = (value, rep)
        row = {"preds": names.split("+"), "timing": timing, "seeds": seeds, "per_seed": per_seed}
        for k in keys:
            row[k] = jack(value[k], rep[k])
            if len(per_seed) > 1:
                row[k]["sd_seeds"] = float(np.std([s[k] for s in per_seed], ddof=1))
        row["n_required"] = {nm: int(st["npos"][cls].sum()) for cls, nm in zip((cc.VEH, cc.PED), cc.CLASS_NAMES)}
        row["region_px"] = int(st["region_px"].sum())
        (out["floors"] if is_floor else out["arms"])[name] = row
    for pspec in a.pair:
        x, y = pspec.split(":")
        vx, rx = reps[x]
        vy, ry = reps[y]
        out["pairs"][pspec] = {k: jack(vx[k] - vy[k], np.array(rx[k]) - np.array(ry[k])) for k in vx}
    Path(a.out).write_text(json.dumps(out, indent=1))
    show = ("obj_recall", "recall_vehicle", "recall_pedestrian", "obj_ap50", "road_iou")
    print("\n" + " | ".join(["arm"] + list(show)))
    for sect in ("arms", "floors", "pairs"):
        for name, row in out[sect].items():
            cells = []
            for k in show:
                r = row[k]
                cells.append(f"{100 * r['value']:.1f} [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]"
                             + (f" sd {100 * r['sd_seeds']:.2f}" if "sd_seeds" in r else ""))
            print(" | ".join([f"{sect}:{name}"] + cells))
    print(f"WROTE {a.out}")


def export_coco(ex, d, full):
    """COCO json (GT with ignore regions as iscrowd, processed detections) for the pycocotools parity check (G6)."""
    d.mkdir(parents=True, exist_ok=True)
    images, anns, dets = [], [], []
    img_id = {}
    for k, t, c, cls, db, ds, gb, gign in ex:
        key = (k, t, c)
        if key not in img_id:
            img_id[key] = len(img_id) + 1
            images.append({"id": img_id[key], "width": cc.W, "height": cc.H})
        for b, ig in zip(gb, gign):
            anns.append({"id": len(anns) + 1, "image_id": img_id[key], "category_id": cls + 1, "iscrowd": int(ig),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        for b, s in zip(db, ds):
            dets.append({"image_id": img_id[key], "category_id": cls + 1, "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "vehicle"}, {"id": 2, "name": "pedestrian"}]},
              open(d / "gt.json", "w"))
    json.dump(dets, open(d / "dets.json", "w"))
    json.dump({"ap50_vehicle": full["ap50_vehicle"], "ap50_pedestrian": full["ap50_pedestrian"]}, open(d / "ours.json", "w"))
    print(f"[export] {len(images)} images, {len(anns)} gt, {len(dets)} dets -> {d}", flush=True)


if __name__ == "__main__":
    main()
