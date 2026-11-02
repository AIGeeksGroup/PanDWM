"""Several closed-loop long rollouts in ONE process (2026-10-14): run_long_rollout_v2.py with the pipeline, the
sampler and the dataset built once.  Every scene restarts from the exact RNG state a fresh single-scene run has at that
point (pipeline generator, global CPU and CUDA RNG, snapshotted after construction), so each output equals
run_long_rollout_v2.py's for the same scene, seed and arguments; this skips the ~2.7 min of loading per video.
  python lowmem_launch.py run_long_rollout_multi.py --base-config C --checkpoint K --output-root R --tag T \
      --scene-list dev76_long10_20261014.json --indices 0 1 2 --seed 42 --windows 10
Output R/<T>_scene<dev76_index>_seed<seed>, the same files as run_long_rollout_v2.py.  A finished video is skipped;
an interrupted one with resume_state.pt is resumed; a directory without either is moved aside (.incomplete_<time>).
--debug-windows N stops every scene after N windows without writing the final video (for the equality check).
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



def build(args, scene_token_for_config):
    if not 1 <= args.windows <= 12:
        raise ValueError("windows must be 1..12 for the 20-second scenes")
    if not 1 <= args.start_window <= args.windows:
        raise ValueError("start-window must be in 1..windows")
    if args.history_mode == "flow_noise" and not 0 < args.noise_sigma <= 1:
        raise ValueError("flow_noise needs --noise-sigma in (0,1]")
    if args.history_mode == "moment_align" and not 0 < args.align_alpha <= 1:
        raise ValueError("moment_align needs --align-alpha in (0,1]")
    clip_length = 3 + 16 * args.windows
    with open(args.base_config, encoding="utf-8") as handle:
        config = json.load(handle)
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
        pc["denoising_step_list"] = list(args.student_steps)
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
        "scene_token": scene_token_for_config,
        "gt_rgb_usage": "only first 3 frames for input; rest for scoring",
        "future_conditions": "dataset camera, ego, HD map, 3D boxes, text",
        "engine": args.engine,
        "sampler": "native_50_step" if args.engine == "teacher" else "stochastic",
        "schedule": 50 if args.engine == "teacher" else list(args.student_steps),
        "teacher_guidance_scale": (
            args.teacher_guidance_scale if args.engine == "teacher" else None),
        "pano_blend_width": 128,
        "history_mode": args.history_mode,
        "history_noise_sigma": args.noise_sigma,
        "history_align_alpha": args.align_alpha,
        "start_window": args.start_window,
        "runner": "run_long_rollout_multi.py",
    })
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline = dwm.common.create_instance_from_config(
        pc, output_path=str(root / f"_log_{args.tag}"), config=config, device=device)
    if args.engine == "teacher":
        sampler = pipeline.inference_pipeline
    else:
        for name in ("teacher_model", "teacher_model_wrapper",
                     "ema_model", "ema_model_wrapper"):
            if getattr(pipeline, name, None) is not None:
                setattr(pipeline, name, None)
        torch.cuda.empty_cache()
        sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
        sampler = evaluate_4step.make_4step_inference(pipeline, sigmas)
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    rng = {"gen": pipeline.generator.get_state(), "cpu": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state(device)}
    return config, pc, pipeline, sampler, dataset, device, clip_length, rng


@torch.no_grad()
def rollout_one(args, config, pc, pipeline, sampler, dataset, device, clip_length, rng, scene_token, output,
                budget_start):
    """One scene; returns False when the job's time budget stopped it (resume later)."""
    start0 = 16 * (args.start_window - 1)
    resume_path = output / "resume_state.pt"
    if output.exists() and not resume_path.is_file():
        aside = output.with_name(output.name + f".incomplete_{int(time.time())}")
        output.rename(aside)
        print(f"[multi] moved an incomplete run aside: {aside}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    config["informations"]["scene_token"] = scene_token
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    # the RNG state of a fresh single-scene process at this point
    pipeline.generator.set_state(rng["gen"])
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"], device)
    matches = [i for i, a in enumerate(dataset.base_dataset.clip_audit) if a["scene_token"] == scene_token]
    if len(matches) != 1:
        raise ValueError(f"scene token has {len(matches)} matches for {clip_length} frames")
    scene_index = matches[0]
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    loader_args.pop("batch_size", None)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [scene_index]), batch_size=1, **loader_args)
    batch = next(iter(loader))
    assert batch["vae_images"].shape[:3] == (1, clip_length, 2)
    reference = _encode_rgb(pipeline, batch["vae_images"][:, start0:start0 + 3])
    anchor = reference[:, 2:3].clone()
    initial_stats = (
        reference.float().mean(dim=(1, 2, 4, 5), keepdim=True),
        reference.float().std(dim=(1, 2, 4, 5), keepdim=True))
    latent_shape = (1, 19, *reference.shape[2:])
    exceptions = pc["inference_config"].get("autoregression_data_exception_for_take_sequence", [])
    summary = {
        "scene_token": scene_token, "dataset_index": scene_index, "seed": args.seed,
        "checkpoint": args.checkpoint, "initial_gt_frames": [start0, start0 + 2],
        "generated_frames": [start0 + 3, clip_length - 1], "start_window": args.start_window, "fps": 10,
        "windows": [], "history_mode": args.history_mode, "engine": args.engine,
        "teacher_guidance_scale": (args.teacher_guidance_scale if args.engine == "teacher" else None),
    }
    first_window = args.start_window - 1
    if resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["scene_token"] != scene_token or state["seed"] != args.seed:
            raise ValueError("resume state belongs to a different rollout")
        first_window = int(state["next_window"])
        reference = state["reference"].to(device)
        pipeline.generator.set_state(state["generator_state"])
        summary["windows"] = state["windows"]
        print(f"[long-rollout] resume at window {first_window + 1}", flush=True)
    last_window = args.windows if not args.debug_windows else min(args.windows, args.debug_windows)
    for window in range(first_window, last_window):
        start = window * 16
        cond = _window_batch(batch, start, 19, exceptions)
        raw_history = reference
        if args.history_mode == "anchor_sink":
            history = raw_history if window == args.start_window - 1 else torch.cat(
                [anchor, raw_history[:, 1:]], dim=1)
            reference_timestep = 0.0
        else:
            history, reference_timestep = make_history_for_window(
                raw_history, initial_stats, args.history_mode,
                args.noise_sigma, args.align_alpha, args.seed,
                window - (args.start_window - 1))
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
            "scene_token": scene_token, "seed": args.seed,
            "next_window": window + 1, "reference": reference.float().cpu(),
            "generator_state": pipeline.generator.get_state(),
            "windows": summary["windows"],
        }, resume_path)
        print("[long-rollout]", scene_token,
              {k: record[k] for k in ("window", "psnr_mean_db", "runtime_seconds")},
              flush=True)
        del result, decoded_images, images, fake, real, pair
        elapsed = time.monotonic() - budget_start
        if (args.time_budget_s and window + 1 < args.windows
                and elapsed + 1.3 * record["runtime_seconds"] > args.time_budget_s):
            print(f"[long-rollout] INCOMPLETE stop after window {window + 1} "
                  f"({elapsed:.0f}s used); rerun with --resume", flush=True)
            return False

    if args.debug_windows:
        return True
    summary["overall_psnr_mean_db"] = round(
        sum(w["psnr_mean_db"] for w in summary["windows"]) / len(summary["windows"]), 4)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    import av
    video_frames = []
    for window in range(args.start_window - 1, args.windows):
        with av.open(str(output / f"window{window + 1:02d}.mp4")) as container:
            for frame in container.decode(video=0):
                video_frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1))
    expected = 16 * (args.windows - args.start_window + 1)
    if len(video_frames) != expected:
        raise RuntimeError(f"expected {expected} frames, got {len(video_frames)}")
    dwm.utils.preview.save_tensor_to_video(str(output / "gt_vs_generated.mp4"), "libx264", 10, video_frames)
    resume_path.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scene-list", required=True, help="json with scenes[i] = {dev76_index, scene_token}")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--start-window", type=int, default=1)
    parser.add_argument("--history-mode", choices=(
        "clean", "flow_noise", "vae_roundtrip", "moment_align", "anchor_sink"), default="clean")
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--align-alpha", type=float, default=0.5)
    parser.add_argument("--engine", choices=("student", "teacher"), default="student")
    # 2026-10-17: the student step list was hard-coded to [1000, 625, 250, 42]; that stays the default so earlier
    # runs reproduce. The paper uses the training list [1000, 750, 500, 250] (user decision 2026-10-17).
    parser.add_argument("--student-steps", type=int, nargs=4, default=[1000, 625, 250, 42])
    parser.add_argument("--teacher-guidance-scale", type=float, default=3.0)
    parser.add_argument("--time-budget-s", type=float, default=0.0)
    parser.add_argument("--debug-windows", type=int, default=0)
    args = parser.parse_args()
    rows = {int(r["dev76_index"]): r["scene_token"] for r in json.load(open(args.scene_list))["scenes"]}
    todo = []
    for idx in args.indices:
        out = Path(args.output_root) / f"{args.tag}_scene{idx}_seed{args.seed}"
        if (out / "gt_vs_generated.mp4").is_file() and not args.debug_windows:
            print(f"SCENE_SKIP {args.tag} {idx}", flush=True)
            continue
        todo.append((idx, rows[idx], out))
    if not todo:
        print("MULTI_NOTHING_TO_DO", flush=True)
        return
    built = build(args, todo[0][1])
    budget_start = time.monotonic()
    for idx, token, out in todo:
        t0 = time.monotonic()
        done = rollout_one(args, *built, token, out, budget_start)
        print(f"SCENE_DONE {args.tag} {idx} {'ok' if done else 'INCOMPLETE'} {time.monotonic() - t0:.0f}s", flush=True)
        if not done:
            break
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
