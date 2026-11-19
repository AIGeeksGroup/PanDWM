"""Evaluate a closed-loop PanDWM rollout with generated history only.

Ground-truth RGB is encoded only for the first three frames. All later history
latents come from the previous sampler window; later RGB is used for scoring.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
import dwm.utils.preview
from dwm_pano import evaluate_4step
from dwm_pano.history_noise import make_history_for_window
from dwm_pano.pano_stitch import stitch_pano_halves

from run_history_probe import _encode_rgb, _psnr_by_frame, _window_batch


def latent_channel_stats(latents):
    values = latents.float()
    return {
        "mean": values.mean(dim=(1, 2, 4, 5))[0].cpu().tolist(),
        "std": values.std(dim=(1, 2, 4, 5))[0].cpu().tolist(),
    }


@torch.no_grad()
def run(args):
    if not 1 <= args.windows <= 12:
        raise ValueError("windows must be 1..12 for the 20-second scenes")
    if args.history_mode == "flow_noise" and not 0 < args.noise_sigma <= 1:
        raise ValueError("flow_noise needs --noise-sigma in (0,1]")
    if args.history_mode == "moment_align" and not 0 < args.align_alpha <= 1:
        raise ValueError("moment_align needs --align-alpha in (0,1]")
    clip_length = 3 + 16 * args.windows
    output = Path(args.output)
    resume_path = output / "resume_state.pt"
    if output.exists() and not (args.resume and resume_path.is_file()):
        raise FileExistsError(f"{output} exists (use --resume to continue it)")
    output.mkdir(parents=True, exist_ok=True)

    with open(args.base_config, encoding="utf-8") as handle:
        config = json.load(handle)
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = config["pipeline"]
    if args.engine == "teacher":
        # The Stage-1 checkpoint belongs to the base flow model. The Stage-2
        # config contains many CD-only keys and would also allocate an unused
        # teacher and EMA, so retain only the base pipeline constructor keys.
        base_keys = ("common_config", "training_config", "inference_config",
                     "pretrained_model_name_or_path", "model", "model_dtype",
                     "model_load_state_args", "metrics")
        pc = {key: pc[key] for key in base_keys if key in pc}
        pc["_class_name"] = "dwm_pano.pipelines.ctsd_pano.PanoCrossviewTemporalSD"
        pc["model_checkpoint_path"] = args.checkpoint
        pc["inference_config"]["inference_steps"] = 50
        if args.teacher_guidance_scale <= 0:
            pc["inference_config"].pop("guidance_scale", None)
        else:
            pc["inference_config"]["guidance_scale"] = args.teacher_guidance_scale
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
    ds["sequence_length"] = clip_length
    ds["fps_stride_tuples"] = [[10, 20]]
    config.setdefault("informations", {}).update({
        "experiment": "closed-loop long rollout; generated history only",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "windows": args.windows,
        "scene_token": args.scene_token,
        "gt_rgb_usage": "only first 3 frames for input; rest for scoring",
        "future_conditions": "dataset camera, ego, HD map, 3D boxes, text",
        "engine": args.engine,
        "sampler": "native_50_step" if args.engine == "teacher" else "stochastic",
        "schedule": 50 if args.engine == "teacher" else [1000, 625, 250, 42],
        "teacher_guidance_scale": (
            args.teacher_guidance_scale if args.engine == "teacher" else None),
        "pano_blend_width": 128,
        "history_mode": args.history_mode,
        "history_noise_sigma": args.noise_sigma,
        "history_align_alpha": args.align_alpha,
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
        # 4-step inference runs pipeline.model only.  The CD pipeline also
        # builds a frozen teacher and an EMA copy (2 x 13.6 GB) that sampling
        # never touches; free them so the student fits a 48 GB GPU.  They are
        # dropped after construction, so no RNG draw or output changes.
        for name in ("teacher_model", "teacher_model_wrapper",
                     "ema_model", "ema_model_wrapper"):
            if getattr(pipeline, name, None) is not None:
                setattr(pipeline, name, None)
        torch.cuda.empty_cache()
        sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
        sampler = evaluate_4step.make_4step_inference(pipeline, sigmas)

    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    matches = [i for i, a in enumerate(dataset.base_dataset.clip_audit)
               if a["scene_token"] == args.scene_token]
    if len(matches) != 1:
        raise ValueError(f"scene token has {len(matches)} matches for {clip_length} frames")
    scene_index = matches[0]
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    loader_args.pop("batch_size", None)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [scene_index]),
        batch_size=1, **loader_args)
    batch = next(iter(loader))
    assert batch["vae_images"].shape[:3] == (1, clip_length, 2)
    reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
    # anchor_sink keeps the last real context frame as a persistent anchor
    # (Deep Forcing / FLEX "attention sink" idea, adapted to a fixed 3-frame
    # history): from window 2 on, the oldest history slot is replaced by it.
    anchor = reference[:, 2:3].clone()
    initial_stats = (
        reference.float().mean(dim=(1, 2, 4, 5), keepdim=True),
        reference.float().std(dim=(1, 2, 4, 5), keepdim=True))
    latent_shape = (1, 19, *reference.shape[2:])
    exceptions = pc["inference_config"].get(
        "autoregression_data_exception_for_take_sequence", [])
    summary = {
        "scene_token": args.scene_token,
        "dataset_index": scene_index,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "initial_gt_frames": [0, 2],
        "generated_frames": [3, clip_length - 1],
        "fps": 10,
        "windows": [],
        "history_mode": args.history_mode,
        "engine": args.engine,
        "teacher_guidance_scale": (
            args.teacher_guidance_scale if args.engine == "teacher" else None),
    }
    first_window = 0
    if args.resume and resume_path.is_file():
        # Continue a rollout cut by a job time limit: restore the exact fp32
        # history and sampler RNG so the result matches an uninterrupted run.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["scene_token"] != args.scene_token or state["seed"] != args.seed:
            raise ValueError("resume state belongs to a different rollout")
        first_window = int(state["next_window"])
        reference = state["reference"].to(device)
        pipeline.generator.set_state(state["generator_state"])
        summary["windows"] = state["windows"]
        print(f"[long-rollout] resume at window {first_window + 1}", flush=True)
    budget_start = time.monotonic()
    for window in range(first_window, args.windows):
        start = window * 16
        cond = _window_batch(batch, start, 19, exceptions)
        raw_history = reference
        if args.history_mode == "anchor_sink":
            history = raw_history if window == 0 else torch.cat(
                [anchor, raw_history[:, 1:]], dim=1)
            reference_timestep = 0.0
        else:
            history, reference_timestep = make_history_for_window(
                raw_history, initial_stats, args.history_mode,
                args.noise_sigma, args.align_alpha, args.seed, window)
        t0 = time.monotonic()
        sampler_kwargs = {"reference_frame_count": 3}
        if args.engine == "student":
            sampler_kwargs["reference_timestep"] = reference_timestep
        elif reference_timestep != 0:
            raise ValueError("native teacher sampler requires clean history")
        result = sampler(latent_shape, cond, "pt", history, **sampler_kwargs)
        reference = result["latents"][:, -3:].detach()
        decoded_images = result["images"].unflatten(0, (1, 19, 2))
        if args.history_mode == "vae_roundtrip":
            reference = _encode_rgb(pipeline, decoded_images[:, -3:]).detach()
        images = decoded_images[:, 3:]
        fake = stitch_pano_halves(images.to(device)).clamp(0, 1)
        real = stitch_pano_halves(
            batch["vae_images"][:, start + 3:start + 19].to(device)
        ).clamp(0, 1)
        assert fake.shape == real.shape and fake.shape[1] == 16
        psnr = _psnr_by_frame(fake, real)
        record = {
            "window": window + 1,
            "history_frames": [start, start + 2],
            "future_frames": [start + 3, start + 18],
            "psnr_per_frame_db": [round(float(x), 4) for x in psnr],
            "psnr_mean_db": round(sum(psnr) / len(psnr), 4),
            "history_timestep": reference_timestep,
            "history_latent_raw_stats": latent_channel_stats(raw_history),
            "history_latent_used_stats": latent_channel_stats(history),
            "next_history_latent_stats": latent_channel_stats(reference),
            "runtime_seconds": round(time.monotonic() - t0, 2),
        }
        summary["windows"].append(record)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        torch.save({
            "history_latent": history.cpu().to(torch.float16),
            "raw_history_latent": raw_history.cpu().to(torch.float16),
            "future_latent": result["latents"][:, 3:].cpu().to(torch.float16),
        }, output / f"window{window + 1:02d}.pt")
        pair = torch.cat([real[0, :, 0], fake[0, :, 0]], dim=-2)
        dwm.utils.preview.save_tensor_to_video(
            str(output / f"window{window + 1:02d}.mp4"), "libx264", 10,
            list((pair.mul(255).round().byte().cpu()).unbind(0)))
        torch.save({
            "scene_token": args.scene_token, "seed": args.seed,
            "next_window": window + 1, "reference": reference.float().cpu(),
            "generator_state": pipeline.generator.get_state(),
            "windows": summary["windows"],
        }, resume_path)
        print("[long-rollout]", args.scene_token,
              {k: record[k] for k in ("window", "psnr_mean_db", "runtime_seconds")},
              flush=True)
        del result, decoded_images, images, fake, real, pair
        elapsed = time.monotonic() - budget_start
        if (args.time_budget_s and window + 1 < args.windows
                and elapsed + 1.3 * record["runtime_seconds"] > args.time_budget_s):
            print(f"[long-rollout] INCOMPLETE stop after window {window + 1} "
                  f"({elapsed:.0f}s used); rerun with --resume", flush=True)
            torch.distributed.destroy_process_group()
            return

    summary["overall_psnr_mean_db"] = round(
        sum(w["psnr_mean_db"] for w in summary["windows"]) / args.windows, 4)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    import av
    video_frames = []
    for window in range(args.windows):
        with av.open(str(output / f"window{window + 1:02d}.mp4")) as container:
            for frame in container.decode(video=0):
                video_frames.append(torch.from_numpy(
                    frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    if len(video_frames) != 16 * args.windows:
        raise RuntimeError(f"expected {16 * args.windows} frames, got {len(video_frames)}")
    dwm.utils.preview.save_tensor_to_video(
        str(output / "gt_vs_generated.mp4"), "libx264", 10, video_frames)
    resume_path.unlink(missing_ok=True)
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=11)
    parser.add_argument("--history-mode", choices=(
        "clean", "flow_noise", "vae_roundtrip", "moment_align", "anchor_sink"),
        default="clean")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--align-alpha", type=float, default=0.5)
    parser.add_argument("--engine", choices=("student", "teacher"),
                        default="student")
    parser.add_argument("--teacher-guidance-scale", type=float, default=3.0,
                        help="0 disables CFG in the native 50-step teacher")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted rollout in --output")
    parser.add_argument("--time-budget-s", type=float, default=0.0,
                        help="stop at a window boundary before this many seconds")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
