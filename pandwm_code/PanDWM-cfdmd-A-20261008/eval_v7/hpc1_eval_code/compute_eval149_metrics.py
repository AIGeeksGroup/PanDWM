"""FID / FVD for the saved 149-clip runs, with the frozen protocol's metric code.

Same instruments as ``cd_eval_hpc3.json`` / ``evaluate_pipeline``:
  * FID  = torchmetrics FrechetInceptionDistance(normalize=True) on stitched
    V=1 panoramas (PANDWM_PANO_BLEND_W as in generation), every generated
    frame (32 per clip, the 3 context frames excluded, as with
    generate_frames_for_reference=false);
  * FVD  = PanoPatchFrechetVideoDistance(sequence_count=16): six Voronoi
    camera patches, I3D 400-logit features, first 16 generated frames.
The frozen FVD therefore only sees window 1.  Two extra numbers are reported
under their own names: FVD on window 2 (generated frames 16..31) and FID per
window.

Generated frames are decoded from each clip's saved ``latents.pt`` (fp16) with
the configured SD3.5 VAE, exactly as the 4-step sampler decodes; real frames
are the dataset's ``vae_images``.  Metric states are checkpointed after every
clip so the run can be split over 30-minute jobs (``--state``).

  python compute_eval149_metrics.py --base-config cd_eval_hpc3.json \
      --arm b0=<root> --arm b0_anchor=<root> --state S.pt --output R.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import einops
import torch

import dwm.common
import dwm.functional
from dwm_pano.pano_stitch import stitch_pano_halves

FRAMES, CONTEXT = 35, 3
METRICS = ("fid_all", "fid_w1", "fid_w2", "fvd_w1", "fvd_w2")
REPORT_NAMES = {
    "fid_all": "FID-torchmetrics-stitched-pano, 32 generated frames x 149 (frozen protocol)",
    "fvd_w1": "FVD-I3D-sector-6-16, generated frames 0-15 = window 1 (frozen protocol)",
    "fvd_w2": "FVD-I3D-sector-6-16, generated frames 16-31 = window 2 (extra)",
    "fid_w1": "FID, window 1 frames only (extra)",
    "fid_w2": "FID, window 2 frames only (extra)",
}


def build_metrics(config, device):
    spec = config["pipeline"]["metrics"]
    out = {}
    for name in METRICS:
        m = dwm.common.create_instance_from_config(spec["fid" if name.startswith("fid") else "fvd"])
        out[name] = m.to(device)
    return out


def write_report(args, arms, metrics):
    report = {"protocol": "frozen 149-clip val, 35 frames = 3 GT + 2 x 16 generated",
              "pano_blend_w": os.environ.get("PANDWM_PANO_BLEND_W"),
              "latents": "saved fp16 latents.pt, decoded with the configured VAE",
              "metric_names": REPORT_NAMES, "arms": {}}
    for arm, ms in metrics.items():
        report["arms"][arm] = {"clips_root": arms[arm],
                               **{n: float(ms[n].compute()) for n in METRICS}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("METRICS_DONE", flush=True)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", required=True)
    p.add_argument("--arm", action="append", required=True, help="name=clips_root")
    p.add_argument("--state", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--clip-start", type=int, default=0)
    p.add_argument("--clip-end", type=int, default=149)
    p.add_argument("--merge", nargs="*", default=None,
                   help="shard state files to sum (metric states are additive) instead of computing")
    args = p.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm)
    t_start = time.monotonic()
    config = json.loads(Path(args.base_config).read_text())
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)

    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    pc = config["pipeline"]
    vae = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    shift = vae.config.shift_factor or 0
    chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

    def decode(latents):  # (T, V, C, h, w) -> (T, V, 3, H, W) in [0, 1]
        t, v = latents.shape[:2]
        image = dwm.functional.memory_efficient_split_call(
            vae, latents.flatten(0, 1).to(device, dtype=vae.dtype),
            lambda block, x: block.decode(x / block.config.scaling_factor + shift,
                                          return_dict=False)[0], chunk)
        return processor.postprocess(image, output_type="pt").float().unflatten(0, (t, v))

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if len(dataset) != 149 or \
            config["validation_dataset"]["base_dataset"]["sequence_length"] != FRAMES:
        raise RuntimeError("protocol drift: expected 149 clips of 35 frames")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]

    metrics = {arm: build_metrics(config, device) for arm in arms}
    if args.merge is not None:
        covered = []
        for path in args.merge:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if part["arms"] != arms:
                raise ValueError(f"{path}: different arm set")
            covered += part["done"]
            for arm in arms:
                for name in METRICS:
                    for key, value in part["metrics"][arm][name].items():
                        current = getattr(metrics[arm][name], key)
                        setattr(metrics[arm][name], key, current + value.to(device))
        for arm in arms:   # FID records its input dtype in update(); merged states never ran it
            for name in METRICS:
                if not hasattr(metrics[arm][name], "orig_dtype"):
                    metrics[arm][name].orig_dtype = torch.float32
        if sorted(covered) != list(range(149)):
            raise RuntimeError(f"shards cover {len(set(covered))} clips, {len(covered)} entries")
        write_report(args, arms, metrics)
        return
    done = []
    if args.state.is_file():
        state = torch.load(args.state, map_location="cpu", weights_only=False)
        if state["arms"] != arms:
            raise ValueError("state belongs to a different arm set")
        for arm in arms:
            for name in METRICS:
                for key, value in state["metrics"][arm][name].items():
                    setattr(metrics[arm][name], key, value.to(device))
        done = state["done"]
        print(f"[metrics] resumed with {len(done)} clips", flush=True)

    def save_state():
        tmp = args.state.with_suffix(".tmp")
        torch.save({"arms": arms, "done": done,
                    "metrics": {a: {n: {k: v.detach().cpu() for k, v in m.metric_state.items()}
                                    for n, m in ms.items()}
                                for a, ms in metrics.items()}}, tmp)
        tmp.replace(args.state)

    last = 0.0
    for index in range(args.clip_start, args.clip_end):
        if index in done:
            continue
        if args.time_budget_s and last and \
                time.monotonic() - t_start + 1.5 * last > args.time_budget_s:
            print(f"[metrics] budget reached before clip {index}", flush=True)
            break
        t0 = time.monotonic()
        for arm, root in arms.items():
            if not (Path(root) / f"clip{index:03d}" / "latents.pt").is_file():
                raise FileNotFoundError(f"{arm}: clip {index} has no latents.pt")
        batch = collate([dataset[index]])
        real = stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device)).clamp(0, 1)
        for arm, root in arms.items():
            payload = torch.load(Path(root) / f"clip{index:03d}" / "latents.pt",
                                 map_location="cpu", weights_only=True)
            latents = payload["latents"]
            if latents.shape[0] != FRAMES or payload.get("context_frames") != CONTEXT:
                raise ValueError(f"{arm}: clip {index} latents {tuple(latents.shape)}")
            fake = stitch_pano_halves(decode(latents[CONTEXT:])[None]).clamp(0, 1)
            m = metrics[arm]
            for name, sl in (("fid_all", slice(0, 32)), ("fid_w1", slice(0, 16)),
                             ("fid_w2", slice(16, 32))):
                m[name].update(real[:, sl].flatten(0, 2), real=True)
                m[name].update(fake[:, sl].flatten(0, 2), real=False)
            for name, sl in (("fvd_w1", slice(0, 16)), ("fvd_w2", slice(16, 32))):
                m[name].update(einops.rearrange(real[:, sl], "b t v c h w -> (b v) t c h w"), real=True)
                m[name].update(einops.rearrange(fake[:, sl], "b t v c h w -> (b v) t c h w"), real=False)
            del fake
        done.append(index)
        if len(done) % 10 == 0:
            save_state()
        last = time.monotonic() - t0
        print(f"[metrics] clip {index} ({len(done)}/149) {last:.1f}s", flush=True)

    save_state()
    if len(done) == args.clip_end - args.clip_start:
        print("SHARD_DONE", flush=True)
        if args.clip_start == 0 and args.clip_end == 149:
            write_report(args, arms, metrics)


if __name__ == "__main__":
    main()
