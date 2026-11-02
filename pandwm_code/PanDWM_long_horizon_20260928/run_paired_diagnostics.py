"""Matched 195-frame history ladder or inference-time condition ablation.

All target RGB stays outside the sampler batch. Each ladder branch reuses the
closed-loop window's exact torch.Generator state and future conditions.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import dwm.common
from dwm_pano import evaluate_4step
from dwm_pano.history_noise import make_history_for_window
from dwm_pano.pano_stitch import stitch_pano_halves
from run_history_probe import _encode_rgb, _window_batch
from uncompressed_metrics import (
    boundary_metrics, summarize_window, window_metrics)


def fingerprint(state):
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()[:16]


def save_review_crop(pano, path):
    """Selected lossless RGB crop for manual QA; never used for metrics."""
    from PIL import Image
    frame = pano[0, 8, 0, :, :, 686:2057]
    array = (frame.float().clamp(0, 1).mul(255).round().byte()
             .permute(1, 2, 0).cpu().numpy())
    Image.fromarray(array).save(path)


def masked_conditions(cond, mask, color):
    if mask == "full":
        return cond
    result = dict(cond)
    keys = {"no_map": ("hdmap_images",),
            "no_box": ("3dbox_images",),
            "no_map_box": ("hdmap_images", "3dbox_images")}[mask]
    for key in keys:
        if key not in result:
            raise KeyError(f"condition ablation requested missing {key}")
        result[key] = torch.full_like(result[key], color)
    return result


@torch.no_grad()
def run(args):
    if not 1 <= args.windows <= 12:
        raise ValueError("--windows must be 1..12")
    if args.mode == "ladder" and (args.history_mode != "clean" or
                                  args.condition_mask != "full"):
        raise ValueError("history ladder uses clean, fully conditioned B0")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.base_config).read_text())
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = config["pipeline"]
    if args.engine == "teacher":
        base_keys = ("common_config", "training_config", "inference_config",
                     "pretrained_model_name_or_path", "model", "model_dtype",
                     "model_load_state_args", "metrics")
        pc = {key: pc[key] for key in base_keys if key in pc}
        pc["_class_name"] = "dwm_pano.pipelines.ctsd_pano.PanoCrossviewTemporalSD"
        pc["model_checkpoint_path"] = args.checkpoint
        pc["inference_config"]["inference_steps"] = 50
        pc["inference_config"].pop("guidance_scale", None)
        config["pipeline"] = pc
    else:
        pc["_class_name"] = (
            "dwm_pano.pipelines.ctsd_pano_consistency_distillation."
            "PanoCrossviewTemporalConsistencyDistillation")
        pc["model_checkpoint_path"] = args.checkpoint
        pc["consistency_ema_checkpoint_path"] = args.checkpoint
        pc["denoising_step_list"] = [1000, 625, 250, 42]
    pc["inference_config"].update({
        "sequence_length_per_iteration": 19,
        "reference_frame_count": 3,
        "generate_frames_for_reference": False,
    })
    ds = config["validation_dataset"]["base_dataset"]
    ds["sequence_length"] = 195
    ds["fps_stride_tuples"] = [[10, 20]]
    config.setdefault("informations", {}).update({
        "experiment": "matched history ladder or condition ablation",
        "mode": args.mode,
        "scene_token": args.scene_token,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "engine": args.engine,
        "future_rgb_in_sampler": False,
        "metric_source": "VAE-decoded float RGB before video encoding",
        "lpips": "AlexNet LPIPS v0.1, six 192x512 panorama sectors",
        "history_mode": args.history_mode,
        "noise_sigma": args.noise_sigma,
        "condition_mask": args.condition_mask,
        "windows": args.windows,
    })
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline = dwm.common.create_instance_from_config(
        pc, output_path=str(output), config=config, device=device)
    if args.engine == "teacher":
        sampler = pipeline.inference_pipeline
    else:
        sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
        sampler = evaluate_4step.make_4step_inference(pipeline, sigmas)

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    matches = [i for i, item in enumerate(dataset.base_dataset.clip_audit)
               if item["scene_token"] == args.scene_token]
    if len(matches) != 1:
        raise ValueError(f"scene token has {len(matches)} 195-frame matches")
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    loader_args.pop("batch_size", None)
    batch = next(iter(torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, matches),
        batch_size=1, **loader_args)))
    if tuple(batch["vae_images"].shape[:3]) != (1, 195, 2):
        raise ValueError("expected one 195-frame, two-half panorama")
    reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
    latent_shape = (1, 19, *reference.shape[2:])
    exceptions = pc["inference_config"].get(
        "autoregression_data_exception_for_take_sequence", [])
    initial_stats = (
        reference.float().mean(dim=(1, 2, 4, 5), keepdim=True),
        reference.float().std(dim=(1, 2, 4, 5), keepdim=True))
    lpips_metric = None
    if not args.skip_lpips:
        import lpips
        lpips_metric = lpips.LPIPS(net="alex", version="0.1",
                                  verbose=False).to(device).eval()
        lpips_metric.requires_grad_(False)

    summary = {
        "mode": args.mode, "scene_token": args.scene_token,
        "dataset_index": matches[0], "seed": args.seed,
        "checkpoint": args.checkpoint,
        "engine": args.engine,
        "history_mode": args.history_mode,
        "condition_mask": args.condition_mask,
        "noise_sigma": args.noise_sigma,
        "requested_windows": args.windows,
        "initial_gt_frames": [0, 2],
        "target_gt_usage": "scoring and diagnostic oracle history only",
        "windows": [], "complete": False,
    }

    def sample(history, cond, gt, gt_prev, fake_prev, rng_state,
               window, label, timestep=0):
        pipeline.generator.set_state(rng_state.clone())
        t0 = time.monotonic()
        sampler_kwargs = {"reference_frame_count": 3}
        if args.engine == "student":
            sampler_kwargs["reference_timestep"] = timestep
        elif timestep != 0:
            raise ValueError("Stage-1 teacher has no reference-timestep input")
        result = sampler(latent_shape, cond, "pt", history,
                         **sampler_kwargs)
        next_history = result["latents"][:, -3:].detach()
        future_latents = result["latents"][:, 3:].detach()
        decoded = result["images"].unflatten(0, (1, 19, 2))[:, 3:]
        fake = stitch_pano_halves(decoded.to(device)).clamp(0, 1)
        if fake.shape != gt.shape:
            raise ValueError(f"generated shape mismatch: {fake.shape} != {gt.shape}")
        frame = window_metrics(fake, gt, lpips_metric)
        boundary = boundary_metrics(fake[:, :1], fake_prev, gt[:, :1], gt_prev)
        record = summarize_window(frame, boundary)
        record.update({
            "window": window + 1,
            "future_frames": [16 * window + 3, 16 * window + 18],
            "history_frames": [16 * window, 16 * window + 2],
            "history_timestep": timestep,
            "sampler_rng_sha256_prefix": fingerprint(rng_state),
            "runtime_seconds": round(time.monotonic() - t0, 3),
        })
        if args.save_review and window + 1 in (1, 4, 8, 12):
            save_review_crop(fake, output / f"w{window + 1:02d}_{label}.png")
            if label == "closed_loop" or (args.mode == "condition" and
                                          label == "condition"):
                save_review_crop(gt, output / f"w{window + 1:02d}_gt.png")
        last_fake = fake[:, -1:].detach()
        first_fake = fake[:, :1].detach()
        del result, decoded, fake
        return record, next_history, last_fake, first_fake, future_latents

    closed_history = reference
    prev_closed_fake = stitch_pano_halves(
        batch["vae_images"][:, 2:3].to(device)).clamp(0, 1)
    prev_gt_branch_history = None
    prev_gt_branch_fake = None
    for window in range(args.windows):
        start = window * 16
        base_cond = _window_batch(batch, start, 19, exceptions)
        cond = masked_conditions(
            base_cond, args.condition_mask,
            pc["common_config"].get("uncondition_image_color", 0))
        gt = stitch_pano_halves(
            batch["vae_images"][:, start + 3:start + 19].to(device)
        ).clamp(0, 1)
        gt_prev = stitch_pano_halves(
            batch["vae_images"][:, start + 2:start + 3].to(device)
        ).clamp(0, 1)
        state_before = pipeline.generator.get_state().clone()
        if args.mode == "condition":
            history, timestep = make_history_for_window(
                closed_history, initial_stats, args.history_mode,
                args.noise_sigma, 0.5, args.seed, window)
            record, closed_history, prev_closed_fake, _, _ = sample(
                history, cond, gt, gt_prev, prev_closed_fake,
                state_before, window, "condition", timestep)
            summary["windows"].append(record)
        else:
            closed_record, next_closed, next_closed_fake, _, closed_latents = sample(
                closed_history, cond, gt, gt_prev, prev_closed_fake,
                state_before, window, "closed_loop")
            state_after_closed = pipeline.generator.get_state().clone()
            if window == 0:
                gt_record = copy.deepcopy(closed_record)
                single_record = copy.deepcopy(closed_record)
                next_gt_history = next_closed
                next_gt_fake = next_closed_fake
            else:
                gt_history = _encode_rgb(
                    pipeline, batch["vae_images"][:, start:start + 3]).detach()
                gt_record, next_gt_history, next_gt_fake, _, _ = sample(
                    gt_history, cond, gt, gt_prev, gt_prev,
                    state_before, window, "gt_history")
                single_record, _, _, _, single_latents = sample(
                    prev_gt_branch_history, cond, gt, gt_prev,
                    prev_gt_branch_fake, state_before, window,
                    "one_step_history")
                if window == 1:
                    if not torch.equal(single_latents, closed_latents) or \
                            single_record["psnr_per_frame_db"] != \
                            closed_record["psnr_per_frame_db"]:
                        raise AssertionError(
                            "window 2 one-step and closed loop must be identical")
                del gt_history, single_latents
            if not (gt_record["sampler_rng_sha256_prefix"] ==
                    single_record["sampler_rng_sha256_prefix"] ==
                    closed_record["sampler_rng_sha256_prefix"]):
                raise AssertionError("paired branches used different sampler RNG")
            prev_gt_branch_history = next_gt_history
            prev_gt_branch_fake = next_gt_fake
            closed_history = next_closed
            prev_closed_fake = next_closed_fake
            pipeline.generator.set_state(state_after_closed)
            summary["windows"].append({
                "window": window + 1,
                "gt_history": gt_record,
                "one_step_history": single_record,
                "closed_loop": closed_record,
            })
            del closed_latents
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("[paired-diagnostics]", args.mode, args.scene_token,
              window + 1, flush=True)
    summary["complete"] = True
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--mode", choices=("ladder", "condition"), required=True)
    parser.add_argument("--engine", choices=("student", "teacher"),
                        default="student")
    parser.add_argument("--history-mode", choices=("clean", "flow_noise"),
                        default="clean")
    parser.add_argument("--noise-sigma", type=float, default=0)
    parser.add_argument("--condition-mask", choices=(
        "full", "no_map", "no_box", "no_map_box"), default="full")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--save-review", action="store_true")
    args = parser.parse_args()
    if args.history_mode == "flow_noise" and not 0 < args.noise_sigma <= 1:
        parser.error("flow_noise needs --noise-sigma in (0,1]")
    if args.history_mode == "clean" and args.noise_sigma != 0:
        parser.error("clean mode requires --noise-sigma 0")
    run(args)


if __name__ == "__main__":
    main()
