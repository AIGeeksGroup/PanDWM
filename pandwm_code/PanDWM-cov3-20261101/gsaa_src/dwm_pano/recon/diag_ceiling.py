"""Where does the 3D Gaussian quality go?  Ceiling decomposition on Test-S (2026-10-24, diagnostic, not a table).

For a subset of tab:gs_recon samples, the 3d scene of render_ours.py (v4 settings, frame 20 of the recorded clip) is
rendered at the input frame's own six cameras ("self") and at the next frame's ("next") with three colourings of
the SAME Gaussians (same geometry, same opacities):
  model   the decoder's colours (what tab:gs_recon scores);
  vae     colours read from the VAE reconstruction of the input panorama at each Gaussian's own pixel;
  oracle  colours read from the recorded input panorama at each Gaussian's own pixel.
The Gaussian grid is the halves' pixel grid (pipe.pixel_directions), so pixel i of the halves owns Gaussian i before
the keep mask.  Also measured in the panorama itself: the VAE round trip (recorded halves -> posterior mean -> decode)
and the 0 m self render of the band canvas.  Scores: band mask only (no sky mask), alpha >= 0.5 covered pixels,
PSNR covered / full (over 0.5 gray) and SSIM, at 224x400.  Output: one json of per-sample rows + means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dwm.common
from dwm_pano.d2_render import encode_halves_mean
from dwm_pano.evaluate_gs import build_dataset_config, ensure_single_rank_process_group, \
    install_strict_pano_hardseam_stub, load_config
from dwm_pano.gs_scene_v3 import SceneBuilder, SceneOptions
from dwm_pano.recon.render_ours import INPUT_FRAME
from dwm_pano.recon.score import gaussian_window, psnr, ssim_map

HW = (224, 400)


def masked_mean(x, m):
    n = m.sum()
    return None if n == 0 else float((x * m).sum() / n)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--masks", required=True, type=Path)
    p.add_argument("--samples", default="0:213:5")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--da3-src", required=True)
    p.add_argument("--da3-anyview", required=True)
    p.add_argument("--da3-metric", required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    data = json.loads(args.targets.read_text())
    a, b, s = (int(x) for x in args.samples.split(":"))
    positions = list(range(a, b, s))
    rigs = json.loads((args.masks / "rigs.json").read_text())["target_rig"]
    from dwm_pano_4d.da3_depth import Da3Ranger
    from dwm_pano.calibrated_stitch import prepare_gsplat_intrinsics
    da3 = Da3Ranger(args.da3_src, args.da3_anyview, args.da3_metric, device)
    started = ensure_single_rank_process_group(device)
    window = gaussian_window(device=device)
    rows = []
    try:
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        pipe = dwm.common.create_instance_from_config(config["pipeline"], output_path=str(args.out.parent),
                                                      config=config, device=device, resume_from=None)
        pipe.vae.eval()
        pipe.decoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipe.decoder.eval()
        pipe.decoder_wrapper.eval()
        pipe.rasterize_mode = "antialiased"
        pipe.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        cfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=[[10, 5.0]])
        cfg["base_dataset"]["sequence_length"] = 35
        dataset = dwm.common.create_instance_from_config(cfg)
        base = dataset.base_dataset
        by_key = {(a_["scene_token"], int(c[0]["timestamp"])): i for i, (c, a_) in
                  enumerate(zip(base.clips, base.clip_audit))}
        from dwm.datasets.common import CollateFnIgnoring
        original_wrapper = pipe.decoder_wrapper
        colour_override = {}

        class Wrapper(torch.nn.Module):
            def forward(self, *x, **kw):
                out = original_wrapper(*x, **kw)
                if "rgb" in colour_override:
                    if out[0].shape[1] != colour_override["rgb"].shape[0]:
                        raise RuntimeError(f"{out[0].shape} Gaussians vs {colour_override['rgb'].shape} pixels")
                    g = out[0].clone()
                    g[0, :, 3:6] = colour_override["rgb"].to(g.dtype)
                    out = (g,) + tuple(out[1:])
                return out
        for pos in positions:
            smp = data["samples"][pos]
            index = by_key[(smp["scene_token"], int(smp["anchor_timestamp_us"]))]
            batch = next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, [index]), batch_size=1,
                                                          shuffle=False, num_workers=0,
                                                          collate_fn=CollateFnIgnoring(keys=[]))))
            clip = base.clips[index]
            halves = batch["vae_images"][0, :INPUT_FRAME + 1].to(device).float()
            latents = encode_halves_mean(pipe, halves).cpu()
            rec = halves[INPUT_FRAME]                                                    # (2, 3, H, W)
            vae = pipe.decode_half_latents(latents[INPUT_FRAME:INPUT_FRAME + 1].to(device, pipe.latent_directions.dtype)
                                           ).float()[0].clamp(0, 1)
            row = {"test_index": smp["test_index"],
                   "pano_vae_psnr": float(-10 * torch.log10(((vae - rec) ** 2).mean()))}
            for colour in ("model", "vae", "oracle"):
                colour_override.clear()
                if colour != "model":
                    src = rec if colour == "oracle" else vae
                    colour_override["rgb"] = src.permute(0, 2, 3, 1).reshape(-1, 3)
                pipe.decoder_wrapper = Wrapper() if colour != "model" else original_wrapper
                opts = SceneOptions(da3_relabel=True, da3_mode="scale", drop_bottom_rows=2, da3_gap=0,
                                    window=(-20, -15, -10, -5))
                scene = SceneBuilder(pipe, batch, clip[:INPUT_FRAME + 1], latents, opts, device, da3=da3)
                for t in range(INPUT_FRAME + 1):
                    scene._canvas[t] = pipe._unpadded_stitch_canvas(halves[t:t + 1])[0].clamp(0, 1)
                g = scene.decode(INPUT_FRAME)
                pipe.decoder_wrapper = original_wrapper
                for tname, entry in (("self", smp["input"]), ("next", smp["targets"]["next"])):
                    cams = entry["cams"]
                    c2w = torch.as_tensor(np.asarray([c["c2w"] for c in cams]), device=device, dtype=torch.float32)
                    K = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(c["K"], dtype=torch.float64),
                                                               (900, 1600), HW) for c in cams]).float().to(device)
                    rgb, depth, alpha = pipe.render_targets(g, target_c2w=c2w, target_intrinsics=K, width=HW[1],
                                                            height=HW[0])
                    rid = rigs[f"{smp['test_index']}/next"]
                    band = torch.from_numpy(np.load(args.masks / f"band_{rid}_{HW[0]}x{HW[1]}.npz")["valid"]).to(device)
                    gt = torch.stack([torch.from_numpy(np.asarray(Image.open(c["image"]).convert("RGB").resize(
                        (HW[1], HW[0]), Image.BICUBIC)).copy()).permute(2, 0, 1) for c in cams]).to(device).float() / 255
                    comp = rgb.clamp(0, 1) + (1 - alpha[:, None]) * 0.5
                    cov = alpha >= 0.5
                    err = (comp - gt) ** 2
                    ss = ssim_map(comp, gt, window)
                    vals = {"psnr_cov": [], "psnr_full": [], "ssim_cov": [], "coverage": []}
                    for c in range(6):
                        m, mc = band[c].float(), (band[c] & cov[c]).float()
                        vals["psnr_cov"].append(psnr(err[c], mc))
                        vals["psnr_full"].append(psnr(err[c], m))
                        vals["ssim_cov"].append(masked_mean(ss[c], mc))
                        vals["coverage"].append(masked_mean(cov[c].float(), m))
                    for k, v in vals.items():
                        v = [x for x in v if x is not None]
                        row[f"{tname}_{colour}_{k}"] = float(np.mean(v)) if v else None
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if started:
            torch.distributed.destroy_process_group()
    keys = [k for k in rows[0] if k != "test_index"]
    means = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}
    args.out.write_text(json.dumps({"samples": len(rows), "means": means, "rows": rows}, indent=1) + "\n")
    print("DIAG_MEANS " + json.dumps(means, indent=1), flush=True)


if __name__ == "__main__":
    main()
