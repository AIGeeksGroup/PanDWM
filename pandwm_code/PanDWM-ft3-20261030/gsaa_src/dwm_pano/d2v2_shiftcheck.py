"""Does the refiner keep the lateral shift? (2026-10-31, D2V2_PLAN_20261031.md section 6.4)

The shift protocol scores refined views unpaired against the recorded (0 m) frames, so a refiner that drifted back
to the original viewpoint would score well without doing the task.  For each clip / frame / shift d this compares
the decoded refined view at +-d with the export's raw render at the same +-d and with the raw render at 0 m, after a
16 x 16 average pool (layout / geometry, blind to the detail the refiner adds), on the stitch-valid pixels:

    psnr_same = PSNR(pool(refined_d), pool(raw_d))      should stay high
    psnr_zero = PSNR(pool(refined_d), pool(raw_0))      should stay below psnr_same, as for the raw render itself
    raw_gap   = PSNR(pool(raw_d), pool(raw_0))          reference: how different the two viewpoints are

and writes panels (raw 0 m | raw d | refined d) of one frame per clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import dwm.common
from dwm_pano.d2v2_eval import decode_latents
from dwm_pano.evaluate_gs import ensure_single_rank_process_group, load_config


def pool_psnr(a, b, mask):
    pa, pb = F.avg_pool2d(a[None], 16)[0], F.avg_pool2d(b[None], 16)[0]
    pm = F.avg_pool2d(mask[None, None].float(), 16)[0, 0] > 0.99
    mse = ((pa - pb) ** 2).mean(0)[pm].mean()
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--refined", required=True, type=Path)
    p.add_argument("--clips", default="0,25,64,122,5,47,100,140")
    p.add_argument("--frames", default="10,20,30")
    p.add_argument("--shifts", default="1,-1,2,-2,4,-4")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        config = load_config(args.config)
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        frames = [int(x) for x in args.frames.split(",")]
        shifts = [float(x) for x in args.shifts.split(",")]
        rows, panels = [], []
        args.out.mkdir(parents=True, exist_ok=True)
        for clip in [int(x) for x in args.clips.split(",")]:
            d0 = torch.load(args.inputs / f"clip{clip:03d}" / "render_dy0.pt", map_location="cpu", weights_only=True)
            raw0 = d0["pixels"][frames].to(device).float() / 255
            valid = torch.ones(raw0.shape[-2:], dtype=torch.bool, device=device)
            for d in shifts:
                pk = torch.load(args.inputs / f"clip{clip:03d}" / f"render_dy{d:g}.pt", map_location="cpu",
                                weights_only=True)
                rawd = pk["pixels"][frames].to(device).float() / 255
                ref = torch.load(args.refined / f"clip{clip:03d}" / f"d2_dy{d:g}.pt", map_location="cpu",
                                 weights_only=True)["latents"][frames].float()
                fix = decode_latents(pipe, ref)
                if fix.shape[-2:] != rawd.shape[-2:]:
                    fix = F.interpolate(fix, size=rawd.shape[-2:], mode="bilinear", align_corners=False)
                for i, t in enumerate(frames):
                    rows.append({"clip": clip, "frame": t, "dy": d,
                                 "psnr_same": pool_psnr(fix[i], rawd[i], valid),
                                 "psnr_zero": pool_psnr(fix[i], raw0[i], valid),
                                 "raw_gap": pool_psnr(rawd[i], raw0[i], valid)})
                if d == 4.0:
                    i = frames.index(20) if 20 in frames else 0
                    panels.append(torch.cat([raw0[i], rawd[i], fix[i]], dim=-2).cpu())
        summary = {}
        for d in shifts:
            sel = [r for r in rows if r["dy"] == d]
            summary[f"{d:g}"] = {k: round(sum(r[k] for r in sel) / len(sel), 2)
                                 for k in ("psnr_same", "psnr_zero", "raw_gap")}
        summary["frac_closer_to_zero"] = round(sum(r["psnr_zero"] > r["psnr_same"] for r in rows) / len(rows), 3)
        (args.out / "shiftcheck.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
        print(json.dumps(summary, indent=1), flush=True)
        from PIL import Image
        img = torch.cat(panels, dim=-2)
        img = F.interpolate(img[None], scale_factor=0.5, mode="area")[0]
        Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")).save(
            args.out / "panels_4m.png")
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
