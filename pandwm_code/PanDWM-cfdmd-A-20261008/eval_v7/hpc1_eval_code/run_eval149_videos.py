"""Render the frozen 149-clip validation protocol (35 frames) with one model load.

Same clips as the FID/FVD protocol (``cd_eval_hpc3.json`` validation dataset:
35 frames, fps_stride [[10, 20]], one window per val scene).  Each clip uses
the three real context frames, then two closed-loop 16-frame windows.  The
sampler generator is reseeded per clip from (seed, dataset index), so a clip's
output does not depend on how the 149 clips are split into jobs.

Per clip: gt_vs_generated.mp4 (stitched pano, GT top / generated bottom),
latents.pt (35 latent frames, fp16, for the Gaussian decoder) and summary.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
import dwm.utils.preview
from dwm_pano import evaluate_4step
from dwm_pano.pano_stitch import stitch_pano_halves
from run_history_probe import _encode_rgb, _psnr_by_frame, _window_batch

WINDOWS = 2
CLIP = 3 + 16 * WINDOWS


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-mode", choices=("clean", "anchor_sink"),
                        default="clean",
                        help="anchor_sink: from window 2, the oldest history slot "
                             "is the last real context frame (as in run_long_rollout)")
    parser.add_argument("--time-budget-s", type=float, default=0.0,
                        help="stop between clips before this many seconds")
    args = parser.parse_args()
    job_start = time.monotonic()
    last_clip_s = 0.0

    config = json.loads(Path(args.base_config).read_text())
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = config["pipeline"]
    pc["model_checkpoint_path"] = args.checkpoint
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != CLIP or ds["fps_stride_tuples"] != [[10, 20]]:
        raise ValueError("expected the frozen 35-frame, stride-20 protocol")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline = dwm.common.create_instance_from_config(
        pc, output_path=str(args.out_root), config=config, device=device)
    # 4-step sampling uses pipeline.model only; drop the frozen teacher and
    # EMA copies so the student fits a 48 GB GPU (no RNG or output change).
    for name in ("teacher_model", "teacher_model_wrapper",
                 "ema_model", "ema_model_wrapper"):
        if getattr(pipeline, name, None) is not None:
            setattr(pipeline, name, None)
    torch.cuda.empty_cache()
    sampler = evaluate_4step.make_4step_inference(
        pipeline, evaluate_4step.compute_denoising_sigmas(pipeline))

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    total = len(dataset)
    if total != 149:
        raise RuntimeError(f"protocol drift: {total} clips, expected 149")
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    collate = loader_args.get("collate_fn") or torch.utils.data.default_collate
    exceptions = pc["inference_config"].get(
        "autoregression_data_exception_for_take_sequence", [])
    args.out_root.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, min(args.end, total)):
        audit = dataset.base_dataset.clip_audit[index]
        out = args.out_root / f"clip{index:03d}"
        if (out / "summary.json").is_file():
            print(f"[eval149] skip {index}", flush=True)
            continue
        if (args.time_budget_s and last_clip_s and
                time.monotonic() - job_start + 1.2 * last_clip_s > args.time_budget_s):
            print(f"[eval149] budget reached before clip {index}; rerun to resume",
                  flush=True)
            break
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        pipeline.generator.manual_seed(args.seed * 1000 + index)
        # Single-process load: re-creating a multi-worker DataLoader per clip
        # hung after the first clip of every job on HPC2.
        batch = collate([dataset[index]])
        reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
        latent_shape = (1, 19, *reference.shape[2:])
        latents = [reference[0]]
        anchor = reference[:, 2:3].clone()
        frames, psnr_windows = [], []
        for window in range(WINDOWS):
            start = window * 16
            cond = _window_batch(batch, start, 19, exceptions)
            history = reference
            if args.history_mode == "anchor_sink" and window > 0:
                history = torch.cat([anchor, reference[:, 1:]], dim=1)
            result = sampler(latent_shape, cond, "pt", history,
                             reference_frame_count=3, reference_timestep=0)
            reference = result["latents"][:, -3:].detach()
            latents.append(result["latents"][0, 3:].detach())
            fake = stitch_pano_halves(
                result["images"].unflatten(0, (1, 19, 2))[:, 3:].to(device)).clamp(0, 1)
            real = stitch_pano_halves(
                batch["vae_images"][:, start + 3:start + 19].to(device)).clamp(0, 1)
            psnr = _psnr_by_frame(fake, real)
            psnr_windows.append(round(sum(psnr) / len(psnr), 4))
            pair = torch.cat([real[0, :, 0], fake[0, :, 0]], dim=-2)
            frames.extend(pair.mul(255).round().byte().cpu().unbind(0))
            del result, fake, real, pair
        torch.save({"latents": torch.cat(latents).half().cpu(),
                    "context_frames": 3}, out / "latents.pt")
        dwm.utils.preview.save_tensor_to_video(
            str(out / "gt_vs_generated.mp4"), "libx264", 10, frames)
        summary = {
            "protocol": "frozen 149-clip val, 35 frames (3 GT + 2x16 generated)",
            "clip_index": index, "dataset_index": index,
            "scene_token": audit["scene_token"],
            "anchor_timestamp_us": int(batch["anchor_timestamp_us"][0]),
            "checkpoint": args.checkpoint, "seed": args.seed,
            "history_mode": args.history_mode,
            "generator_seed": args.seed * 1000 + index,
            "psnr_mean_db_per_window": psnr_windows,
            "runtime_seconds": round(time.monotonic() - t0, 1),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        last_clip_s = summary["runtime_seconds"]
        print(f"[eval149] clip {index} psnr={psnr_windows} "
              f"t={summary['runtime_seconds']}s", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
