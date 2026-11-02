"""Table 5 scorer for external methods (2026-10-23): the metrics of eval_gs_shift.py on precomputed panoramas.

Our row is scored inside dwm_pano.eval_gs_shift (renders, FID / FVD states, LiDAR depth).  An external method (DiST-4D)
writes, per Test-S clip, its stitched panoramas and depth maps, and this script scores them with the SAME instruments:
  <pano-dir>/clip<test_index:03d>/dy<d>.pt   uint8 (32, 3, 256, 4114): the 32 generated frames (clip frames 3..34)
                                             stitched into the V=1 4114 x 256 panorama exactly like the video table
                                             (stitch_views.py, onebev_legacy), for d in 0, +1, -1, +2, -2, +4, -4
                                             (file names dy0, dy1, dy-1, dy2, dy-2, dy4, dy-4);
  <pano-dir>/clip<test_index:03d>/depth.pt   float16 (F, 6, H, W) metric camera z of the 0 m frames listed in the depth
                                             pack (gen_depth_pack.py, clip frames 3, 11, 19, 27), cameras in CAM_ORDER,
                                             each map the full 1600x900 frame resized to (H, W); NaN where the method
                                             has no prediction (e.g. rows it crops).
Arms "<d>" pool +d and -d like eval_gs_shift (both renders count as fakes, the real frames are added once per side).
Real reference: the recorded panoramas of clip frames 3..34 from the frozen eval dataset (val pkl, fps_stride 10,5),
stitched with stitch_pano_halves and PANDWM_PANO_BLEND_W=128.  FID-torchmetrics-stitched-pano over the 32 frames,
FVD-I3D-sector-6-16 over frames 0-15 (window 1).  Depth: the expected depth is read at every pack LiDAR point inside
the band (recon/band.py at (H, W)) and not NaN, per image all / static points, as eval_gs_shift.depth_frame.

    python -m dwm_pano.recon.score_shift_panos -c <gs config> --data-pkl <val pkl> --pano-dir <dir>
        --depth-pack <dir> --i3d <i3d> --clips 0:213 --state <state.pt> [--merge s1.pt s2.pt --output report.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import einops
import numpy as np
import torch

from dwm_pano.eval_gs_shift import CONTEXT, build_metrics, load_into, save_state

SHIFTS = (0, 1, 2, 4)


def parse_clips(text):
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


def depth_rows_for(pack, depth, info_cams, test_index):
    from dwm_pano.recon.band import band_geometry
    from dwm_pano.recon.score import depth_metrics
    from dwm_pano.recon.targets import CAM_ORDER, pose
    f_count, _, h, w = depth.shape
    sx, sy = w / 1600.0, h / 900.0
    offsets = np.concatenate([[0], np.cumsum(pack["counts"].reshape(-1))])
    rows = []
    for fi, t in enumerate(pack["frames"].tolist()):
        cams = info_cams[t]
        s2e = np.stack([pose(cams[n]["sensor2ego_translation"], cams[n]["sensor2ego_rotation"]) for n in CAM_ORDER])
        band = band_geometry(s2e, pack["K"][fi].astype(np.float64), [[1600, 900]] * 6, (h, w))["valid"]
        d = depth[fi].float().numpy()
        for c in range(6):
            q = fi * 6 + c
            sl = slice(int(offsets[q]), int(offsets[q + 1]))
            uv = pack["uv16"][sl].astype(np.float64) / 16.0 - 0.5
            z = pack["z"][sl].astype(np.float64)
            static = pack["static"][sl]
            ui = np.round((uv[:, 0] + 0.5) * sx - 0.5).astype(np.int64)
            vi = np.round((uv[:, 1] + 0.5) * sy - 0.5).astype(np.int64)
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            ui, vi, z, static = ui[ok], vi[ok], z[ok], static[ok]
            val = d[c, vi, ui]
            keep = band[c, vi, ui] & np.isfinite(val) & (val > 1e-3)
            d_hat, z, static = val[keep].astype(np.float64), z[keep], static[keep]
            rows.append({"test_index": int(test_index), "frame": int(t), "camera": CAM_ORDER[c],
                         "all": depth_metrics(d_hat, z), "static": depth_metrics(d_hat[static], z[static])})
    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--pano-dir", required=True, type=Path)
    p.add_argument("--depth-pack", type=Path, default=None)
    p.add_argument("--manifest", required=True, type=Path, help="tests213_manifest_20261018.json")
    p.add_argument("--i3d", required=True)
    p.add_argument("--clips", default="0:213", help="test_index list or a:b")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--merge", nargs="*", default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--method", default="external")
    args = p.parse_args()
    os.environ.setdefault("PANDWM_PANO_BLEND_W", "128")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    arms = [f"4d_{d}" for d in SHIFTS]                       # same arm names as our run, so the states line up
    metrics = build_metrics(arms, args.i3d, device)
    done, depth_rows = [], []
    if args.merge is not None:
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            load_into(metrics, part, device)
            done += part["done"]
            depth_rows += part["extra"].get("depth_rows", [])
        for arm in arms:
            for m in metrics[arm].values():
                if not hasattr(m, "orig_dtype"):
                    m.orig_dtype = torch.float32
        out = {"method": args.method, "clips": len(done), "arms": {a: {n: float(metrics[a][n].compute())
                                                                        for n in ("fid", "fvd_w1")} for a in arms},
               "metric_names": {"fid": "FID-torchmetrics-stitched-pano, 32 generated frames",
                                "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 (window 1)"}}
        if depth_rows:
            out["depth"] = {}
            for tag in ("all", "static"):
                vals = [r[tag] for r in depth_rows if r[tag] is not None]
                out["depth"][tag] = {k: float(np.mean([v[k] for v in vals])) for k in
                                     ("abs_rel", "rmse", "delta1", "abs_rel_med")} if vals else None
                if vals:
                    out["depth"][tag].update(images=len(vals), points=int(sum(v["points"] for v in vals)))
        print(json.dumps(out, indent=2), flush=True)
        if args.output is not None:
            args.output.write_text(json.dumps(out, indent=2) + "\n")
        return
    if args.state.is_file():
        st = torch.load(args.state, map_location="cpu", weights_only=False)
        load_into(metrics, st, device)
        done, depth_rows = st["done"], st["extra"].get("depth_rows", [])
    import dwm.common
    from dwm.datasets.common import CollateFnIgnoring
    from dwm_pano.evaluate_gs import build_dataset_config, install_strict_pano_hardseam_stub, load_config
    from dwm_pano.export_gs_batch149 import FRAMES
    from dwm_pano.pano_stitch import stitch_pano_halves
    install_strict_pano_hardseam_stub()
    config = load_config(args.config)
    cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 5]])
    cfg["base_dataset"]["sequence_length"] = FRAMES
    dataset = dwm.common.create_instance_from_config(cfg)
    base = dataset.base_dataset
    manifest = {int(r["test_index"]): r for r in json.loads(args.manifest.read_text())["clips"]}
    by_key = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in enumerate(zip(base.clips, base.clip_audit))}
    for test_index in parse_clips(args.clips):
        if test_index in done:
            continue
        src = args.pano_dir / f"clip{test_index:03d}"
        row = manifest[test_index]
        ds_index = by_key[(row["scene_token"], int(row["anchor_timestamp_us"]))]
        batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [ds_index]), batch_size=1,
                                                      shuffle=False, num_workers=0,
                                                      collate_fn=CollateFnIgnoring(keys=[]))))
        real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
        for d in SHIFTS:
            arm = f"4d_{d}"
            for sign in ((1, -1) if d > 0 else (1,)):
                pano = torch.load(src / f"dy{sign * d}.pt", map_location="cpu", weights_only=True)
                if tuple(pano.shape) != (FRAMES - CONTEXT, 3, real.shape[-2], real.shape[-1]) or pano.dtype != torch.uint8:
                    raise ValueError(f"{src}/dy{sign * d}.pt: {tuple(pano.shape)} {pano.dtype}, expected "
                                     f"({FRAMES - CONTEXT}, 3, {real.shape[-2]}, {real.shape[-1]}) uint8")
                fake = (pano.to(device).float() / 255.0)[None, :, None]               # (1, 32, 1, 3, H, W)
                metrics[arm]["fid"].update(real.flatten(0, 2), real=True)
                metrics[arm]["fid"].update(fake.flatten(0, 2), real=False)
                metrics[arm]["fvd_w1"].update(einops.rearrange(real[:, :16], "b t v c h w -> (b v) t c h w"), real=True)
                metrics[arm]["fvd_w1"].update(einops.rearrange(fake[:, :16], "b t v c h w -> (b v) t c h w"), real=False)
        if args.depth_pack is not None and (src / "depth.pt").is_file():
            pack = np.load(args.depth_pack / f"clip{test_index:03d}.npz")
            depth = torch.load(src / "depth.pt", map_location="cpu", weights_only=True)
            infos = base.clips[ds_index]
            depth_rows += depth_rows_for(pack, depth, {t: infos[t]["cams"] for t in pack["frames"].tolist()},
                                         test_index)
        done.append(test_index)
        save_state(args.state, arms, done, metrics, {}, {"depth_rows": depth_rows, "method": args.method})
        print(f"[score-shift-panos] clip {test_index} done", flush=True)
    print(f"SHARD_DONE {len(done)} clips", flush=True)


if __name__ == "__main__":
    main()
