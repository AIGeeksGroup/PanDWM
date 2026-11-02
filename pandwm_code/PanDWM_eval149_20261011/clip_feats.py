"""Per-frame Inception and per-window pano-patch I3D features of 35-frame clip runs (2026-10-18, Test-S / Dev sets).

The frozen-protocol instrument of compute_eval149_metrics.py, but saving the features of every clip instead of
accumulating metric states, so FID / FVD can be computed per seed, bootstrapped over scenes and compared at equal n:
  * generated frames: the clip's latents.pt (fp16) decoded with the configured SD3.5 VAE, stitched (blend from
    PANDWM_PANO_BLEND_W), frames 3..34 (32 = 2 windows); real frames: the dataset's vae_images of the same clip;
  * inc: torchmetrics FrechetInceptionDistance(normalize=True) features (2048) of every frame, i.e. the metric's own
    (x * 255).byte() -> Inception;
  * i3d: PanoPatchFrechetVideoDistance features (6 Voronoi camera patches x 400 I3D logits) per 16-frame window.
A self-check on the first clip compares the feature sums with the metric objects' own states.
  python3 clip_feats.py --base-config cd_eval_frozen149_hpc3.json --manifest M.json --dataset-stride 5 \
      --out OUT.npz (--arm-root <dir with clipNNN/latents.pt> | --real)
--real writes the real features only (the same for every arm of a manifest).
"""
import argparse
import json
import os
import time
from pathlib import Path

import einops
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import dwm.common
import dwm.functional
from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
from dwm_pano.pano_stitch import stitch_pano_halves
from long_feats import I3D, i3d_features

FRAMES, CONTEXT = 35, 3


@torch.no_grad()
def inc_float(fid, frames, batch=32):
    """(N, 3, H, W) in [0, 1] -> (N, 2048), exactly FrechetInceptionDistance(normalize=True).update's path."""
    out = []
    for s in range(0, frames.shape[0], batch):
        out.append(fid.inception((frames[s:s + batch] * 255).byte()).float())
    return torch.cat(out)


@torch.no_grad()
def features(fid, fvd, pano):
    """(1, 32, 1, 3, H, W) stitched clip -> inc (32, 2048), i3d (2, 6, 400)."""
    frames = pano[0, :, 0]
    inc = inc_float(fid, frames)
    i3d = torch.stack([i3d_features(fvd, frames[16 * w:16 * w + 16][None]) for w in range(2)])
    return inc, i3d


def self_check(fid, fvd, pano, inc, i3d, device):
    clip = pano[:, :16]
    m = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(device)
    m.update(einops.rearrange(clip, "b t v c h w -> (b v) t c h w"), real=True)
    ok_v = torch.allclose(m.real_features_sum, i3d[0].double().sum(0), rtol=1e-4, atol=1e-3)
    f = FrechetInceptionDistance(normalize=True).to(device)
    f.update(pano[:, :16].flatten(0, 2), real=True)
    # like for like: the metric saw a batch of 16 frames, so compare with this code on the same batch of 16; the saved
    # features come from batches of 32, whose cuDNN kernels differ in the last float bits (reported, not gated)
    same16 = inc_float(fid, pano[0, :16, 0]).double()
    ok_i = torch.allclose(f.real_features_sum, same16.sum(0), rtol=1e-4, atol=1e-3)
    batch_diff = (same16 - inc[:16].double()).abs().max().item()
    print(f"SELF_CHECK i3d {'ok' if ok_v else 'FAIL'} inception {'ok' if ok_i else 'FAIL'} "
          f"(batch-32 vs batch-16 max abs feature diff {batch_diff:.2e})", flush=True)
    if batch_diff > 1e-2:
        raise RuntimeError(f"batch-size dependence of the Inception features too large: {batch_diff}")
    if not (ok_v and ok_i):
        raise RuntimeError("feature extraction differs from the metric objects")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset-stride", type=float, required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm-root")
    g.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f"SKIP {a.out} (exists)", flush=True)
        return
    manifest = json.load(open(a.manifest))
    clips = manifest["clips"]
    config = json.loads(Path(a.base_config).read_text())
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != FRAMES or ds["fps_stride_tuples"] != [[10, 20]]:
        raise RuntimeError("expected the frozen 35-frame protocol config")
    ds["fps_stride_tuples"] = [[10, a.dataset_stride]]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if len(dataset) != manifest["dataset_len"]:
        raise RuntimeError(f"protocol drift: {len(dataset)} clips, manifest says {manifest['dataset_len']}")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fvd = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(device)
    if not a.real:
        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor
        pc = config["pipeline"]
        vae = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
        processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
        shift = vae.config.shift_factor or 0
        chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

        def decode(latents):  # (T, V, C, h, w) -> (T, V, 3, H, W) in [0, 1], as compute_eval149_metrics.py
            t, v = latents.shape[:2]
            image = dwm.functional.memory_efficient_split_call(
                vae, latents.flatten(0, 1).to(device, dtype=vae.dtype),
                lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
            return processor.postprocess(image, output_type="pt").float().unflatten(0, (t, v))
    n = len(clips)
    inc = np.zeros((n, 32, 2048), np.float32)
    i3d = np.zeros((n, 2, 6, 400), np.float32)
    meta = {"test_index": [], "scene_token": [], "seed": []}
    checked = False
    t0 = time.time()
    for k, row in enumerate(clips):
        pos = int(row["test_index"])
        if a.real:
            batch = collate([dataset[int(row["dataset_index"])]])
            if int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
                raise RuntimeError(f"clip {pos}: dataset clip is not the manifest clip")
            pano = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
            seed = -1
        else:
            d = Path(a.arm_root) / f"clip{pos:03d}"
            summary = json.load(open(d / "summary.json"))
            if summary["scene_token"] != row["scene_token"] or \
                    int(summary["anchor_timestamp_us"]) != int(row["anchor_timestamp_us"]):
                raise RuntimeError(f"{d}: not the manifest clip")
            payload = torch.load(d / "latents.pt", map_location="cpu", weights_only=True)
            latents = payload["latents"]
            if latents.shape[0] != FRAMES or payload.get("context_frames") != CONTEXT:
                raise ValueError(f"{d}: latents {tuple(latents.shape)}")
            pano = stitch_pano_halves(decode(latents[CONTEXT:])[None]).clamp(0, 1)
            seed = int(summary["seed"])
        f_inc, f_i3d = features(fid, fvd, pano)
        if not checked:
            self_check(fid, fvd, pano, f_inc, f_i3d, device)
            checked = True
        inc[k], i3d[k] = f_inc.cpu().numpy(), f_i3d.cpu().numpy()
        meta["test_index"].append(pos)
        meta["scene_token"].append(row["scene_token"])
        meta["seed"].append(seed)
        if (k + 1) % 20 == 0 or k + 1 == n:
            print(f"[feats] {k + 1}/{n} {time.time() - t0:.0f}s", flush=True)
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, inc=inc, i3d=i3d, test_index=np.array(meta["test_index"]),
             scene_token=np.array(meta["scene_token"]), seed=np.array(meta["seed"]),
             manifest=os.path.basename(a.manifest), arm_root=a.arm_root or "real")
    os.replace(tmp, a.out)
    print(f"SAVED {a.out}", flush=True)


if __name__ == "__main__":
    main()
