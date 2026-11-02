"""Render a manifest of 35-frame clips (3 frames of history + 2 closed-loop 16-frame windows), one model load.
2026-10-18, for Test-S and the setting-G check (PAPER_COMPARISON_PLAN_20261012.md sections 2.2, 4.2, 5).

run_eval149_videos.py generalised in four ways, everything else is the same code path:
  * clips come from a manifest (tests213_manifest_20261018.json): the frozen eval dataset with fps_stride_tuples
    [[10, --dataset-stride]], clip = manifest["clips"][i]["dataset_index"], checked against its scene token and
    anchor timestamp; the sampler generator is reseeded per clip from seed * 1000 + test_index;
  * --engine teacher: the 50-step guided causal teacher (or --teacher-steps 4), as run_long_rollout_multi.py builds it;
    --bidirectional drops the block-causal mask (Stage 0);
  * --history none: setting G, window 1 generates all 19 frames from noise (no recorded frame enters), window 2
    continues from the last 3 generated frames; the scored frames are 3..34 in both settings;
  * --student-steps: the student step list (default = the training list [1000, 750, 500, 250], the paper setting).
With a manifest whose test_index equals the Dev-149 clip index, --dataset-stride 20 and the defaults, the output equals
run_eval149_videos.py's bit for bit (checked by scripts/verify_clips_runner.sbatch).

Per clip: <out>/clip<test_index:03d>/latents.pt (35 latent frames, fp16, frames 0-2 = encoded recorded history or,
for --history none, generated), gt_vs_generated.mp4 and summary.json.
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
TEACHER_KEYS = ("common_config", "training_config", "inference_config", "pretrained_model_name_or_path", "model",
                "model_dtype", "model_load_state_args", "metrics")


def build_pipeline(args, config, out_root, device):
    pc = config["pipeline"]
    if args.engine == "teacher":
        pc = {key: pc[key] for key in TEACHER_KEYS if key in pc}
        pc["_class_name"] = "dwm_pano.pipelines.ctsd_pano.PanoCrossviewTemporalSD"
        pc["model_checkpoint_path"] = args.checkpoint
        pc["inference_config"]["inference_steps"] = args.teacher_steps
        if args.teacher_guidance_scale <= 0:
            pc["inference_config"].pop("guidance_scale", None)
        else:
            pc["inference_config"]["guidance_scale"] = args.teacher_guidance_scale
        if args.bidirectional:
            pc["inference_config"].pop("temporal_attention_mask", None)
        elif pc["inference_config"].get("temporal_attention_mask") != "causal":
            raise ValueError("the causal teacher needs inference_config temporal_attention_mask = causal")
        config["pipeline"] = pc
    else:
        if args.bidirectional:
            raise ValueError("--bidirectional is for the teacher engine")
        pc["model_checkpoint_path"] = args.checkpoint
        pc["denoising_step_list"] = list(args.student_steps)
    pipeline = dwm.common.create_instance_from_config(pc, output_path=str(out_root), config=config, device=device)
    if args.engine == "teacher":
        return pipeline, pipeline.inference_pipeline
    # 4-step sampling uses pipeline.model only; drop the frozen teacher and EMA copies (no RNG or output change).
    for name in ("teacher_model", "teacher_model_wrapper", "ema_model", "ema_model_wrapper"):
        if getattr(pipeline, name, None) is not None:
            setattr(pipeline, name, None)
    torch.cuda.empty_cache()
    return pipeline, evaluate_4step.make_4step_inference(pipeline, evaluate_4step.compute_denoising_sigmas(pipeline))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-stride", type=float, required=True)
    parser.add_argument("--positions", type=int, nargs=2, required=True, help="manifest positions [start, end)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engine", choices=("student", "teacher"), default="student")
    parser.add_argument("--student-steps", type=int, nargs=4, default=[1000, 750, 500, 250])
    parser.add_argument("--teacher-steps", type=int, default=50)
    parser.add_argument("--teacher-guidance-scale", type=float, default=3.0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--history", choices=("recorded", "none"), default="recorded")
    parser.add_argument("--time-budget-s", type=float, default=0.0, help="stop between clips before this many seconds")
    args = parser.parse_args()
    job_start = time.monotonic()
    last_clip_s = 0.0

    manifest = json.load(open(args.manifest))
    clips = manifest["clips"][args.positions[0]:args.positions[1]]
    config = json.loads(Path(args.base_config).read_text())
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != CLIP or ds["fps_stride_tuples"] != [[10, 20]]:
        raise ValueError("expected the frozen 35-frame protocol config")
    ds["fps_stride_tuples"] = [[10, args.dataset_stride]]
    if manifest.get("dataset_fps_stride_tuples", [[10, args.dataset_stride]]) != [[10, args.dataset_stride]]:
        raise ValueError("manifest was built with another dataset stride")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline, sampler = build_pipeline(args, config, args.out_root, device)
    pc = config["pipeline"]

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    if "dataset_len" in manifest and len(dataset) != manifest["dataset_len"]:
        raise RuntimeError(f"protocol drift: {len(dataset)} clips, manifest says {manifest['dataset_len']}")
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    collate = loader_args.get("collate_fn") or torch.utils.data.default_collate
    exceptions = pc["inference_config"].get("autoregression_data_exception_for_take_sequence", [])
    args.out_root.mkdir(parents=True, exist_ok=True)
    settings = {
        "engine": args.engine, "history": args.history, "bidirectional": args.bidirectional,
        "student_steps": list(args.student_steps) if args.engine == "student" else None,
        "teacher_steps": args.teacher_steps if args.engine == "teacher" else None,
        "teacher_guidance_scale": args.teacher_guidance_scale if args.engine == "teacher" else None,
    }
    for row in clips:
        position, index = int(row["test_index"]), int(row["dataset_index"])
        out = args.out_root / f"clip{position:03d}"
        if (out / "summary.json").is_file():
            print(f"[clips] skip {position}", flush=True)
            continue
        if (args.time_budget_s and last_clip_s and
                time.monotonic() - job_start + 1.2 * last_clip_s > args.time_budget_s):
            print(f"[clips] budget reached before clip {position}; rerun to resume", flush=True)
            break
        t0 = time.monotonic()
        pipeline.generator.manual_seed(args.seed * 1000 + position)
        # Single-process load: re-creating a multi-worker DataLoader per clip hung after the first clip on HPC2.
        batch = collate([dataset[index]])
        audit = dataset.base_dataset.clip_audit[index]
        if audit["scene_token"] != row["scene_token"] or \
                int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"clip {position}: dataset index {index} is not the manifest clip")
        out.mkdir(parents=True, exist_ok=True)
        if args.history == "recorded":
            reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
            latent_shape = (1, 19, *reference.shape[2:])
            latents = [reference[0]]
        else:
            h, w = batch["vae_images"].shape[-2:]
            latent_shape = (1, 19, batch["vae_images"].shape[2], pipeline.vae.config.latent_channels, h // 8, w // 8)
            reference = None
            latents = []
        frames, psnr_windows = [], []
        for window in range(WINDOWS):
            start = window * 16
            cond = _window_batch(batch, start, 19, exceptions)
            kwargs = {"reference_frame_count": 3 if reference is not None else 0}
            if args.engine == "student":
                kwargs["reference_timestep"] = 0
            result = sampler(latent_shape, cond, "pt", reference, **kwargs)
            if reference is None:
                latents.append(result["latents"][0, :3].detach())
            reference = result["latents"][:, -3:].detach()
            latents.append(result["latents"][0, 3:].detach())
            fake = stitch_pano_halves(result["images"].unflatten(0, (1, 19, 2))[:, 3:].to(device)).clamp(0, 1)
            real = stitch_pano_halves(batch["vae_images"][:, start + 3:start + 19].to(device)).clamp(0, 1)
            psnr = _psnr_by_frame(fake, real)
            psnr_windows.append(round(sum(psnr) / len(psnr), 4))
            pair = torch.cat([real[0, :, 0], fake[0, :, 0]], dim=-2)
            frames.extend(pair.mul(255).round().byte().cpu().unbind(0))
            del result, fake, real, pair
        latents = torch.cat(latents)
        if latents.shape[0] != CLIP:
            raise RuntimeError(f"clip {position}: {latents.shape[0]} latent frames")
        torch.save({"latents": latents.half().cpu(), "context_frames": 3, "history": args.history},
                   out / "latents.pt")
        dwm.utils.preview.save_tensor_to_video(str(out / "gt_vs_generated.mp4"), "libx264", 10, frames)
        summary = {
            "protocol": "35 frames (3 history + 2x16 generated), manifest " + Path(args.manifest).name,
            "test_index": position, "dataset_index": index, "scene_token": audit["scene_token"],
            "offset_s": row.get("offset_s"), "anchor_timestamp_us": int(batch["anchor_timestamp_us"][0]),
            "checkpoint": args.checkpoint, "seed": args.seed, "generator_seed": args.seed * 1000 + position,
            **settings, "psnr_mean_db_per_window": psnr_windows,
            "runtime_seconds": round(time.monotonic() - t0, 1),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        last_clip_s = summary["runtime_seconds"]
        print(f"[clips] clip {position} psnr={psnr_windows} t={summary['runtime_seconds']}s", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
