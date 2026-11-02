"""Paired fourth-window history probe for the frozen PanDWM 4-step student.

The first three windows are generated once. Each fourth-window arm receives
the same conditions and exactly the same sampler RNG state. Future RGB is
used only for the oracle-history diagnostic and post-generation scoring.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed

import dwm.common
import dwm.functional
import dwm.utils.preview
from dwm_pano import evaluate_4step
from dwm_pano.pano_stitch import stitch_pano_halves
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd


_original_auxiliary_builder = (
    cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model)


def _single_gpu_auxiliary(self, checkpoint_path, model_load_state_args,
                          requires_grad, role):
    # Evaluation never uses the frozen teacher. Avoid loading a second 3B model.
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("[probe] alias unused teacher to main model", flush=True)
        return self.model, self.model_wrapper
    return _original_auxiliary_builder(
        self, checkpoint_path, model_load_state_args, requires_grad, role)


cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = \
    _single_gpu_auxiliary


def _window_batch(batch, start, length, exceptions):
    result = {
        key: (value if key in exceptions else
              dwm.functional.take_sequence_clip(value, start, start + length))
        for key, value in batch.items()
    }
    # Only the initial three real images are encoded. No target RGB may enter
    # the sampler's conditioning path, including the oracle diagnostic arm.
    result.pop("vae_images", None)
    assert result["pts"].shape[1] == length
    return result


@torch.no_grad()
def _encode_rgb(pipeline, rgb):
    batch_size, frame_count, view_count = rgb.shape[:3]
    image = pipeline.image_processor.preprocess(
        rgb.flatten(0, 2).to(pipeline.device))
    shift = pipeline.vae.config.shift_factor or 0
    if pipeline.is_temporal_vae:
        raise RuntimeError("this probe expects the configured 2D SD3.5 VAE")
    latent = dwm.functional.memory_efficient_split_call(
        pipeline.vae, image,
        lambda block, tensor: (
            block.encode(tensor).latent_dist.mode() - shift
        ) * block.config.scaling_factor,
        pipeline.common_config.get("memory_efficient_batch", -1))
    return latent.unflatten(0, (batch_size, frame_count, view_count))


def _psnr_by_frame(fake, real):
    # Both are B,T,V,C,H,W panoramas on the same target timestamps.
    mse = (fake.float() - real.float()).square().mean(dim=(0, 2, 3, 4, 5))
    return (-10 * torch.log10(mse.clamp_min(1e-12))).cpu().tolist()


@torch.no_grad()
def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with open(args.base_config, encoding="utf-8") as handle:
        config = json.load(handle)
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pipeline_config = config["pipeline"]
    pipeline_config["_class_name"] = (
        "dwm_pano.pipelines.ctsd_pano_consistency_distillation."
        "PanoCrossviewTemporalConsistencyDistillation")
    pipeline_config["model_checkpoint_path"] = args.checkpoint
    pipeline_config["consistency_ema_checkpoint_path"] = args.checkpoint
    pipeline_config["denoising_step_list"] = [1000, 625, 250, 42]
    pipeline_config["inference_config"].update({
        "sequence_length_per_iteration": 19,
        "reference_frame_count": 3,
        "generate_frames_for_reference": False,
    })
    dataset_config = config["validation_dataset"]["base_dataset"]
    dataset_config["sequence_length"] = 67
    dataset_config["fps_stride_tuples"] = [[10, 20]]
    config.setdefault("informations", {}).update({
        "experiment": "fourth-window history probe",
        "checkpoint": args.checkpoint,
        "scene_index": args.scene_index,
        "seed": args.seed,
        "sampler": "stochastic",
        "schedule": [1000, 625, 250, 42],
        "pano_blend_width": 128,
        "oracle_history_is_diagnostic_only": True,
        "anchor_is_stale_frame_slot_probe_not_deep_forcing": True,
    })
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(
            value)

    print("[probe] loading checkpoint", args.checkpoint, flush=True)
    pipeline = dwm.common.create_instance_from_config(
        pipeline_config, output_path=str(output), config=config, device=device)
    sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
    sampler = evaluate_4step.make_4step_inference(pipeline, sigmas)
    print("[probe] schedule", [int(x) for x in pipeline.denoising_step_list],
          "sigmas", [float(x) for x in sigmas], flush=True)

    dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if not 0 <= args.scene_index < len(dataset):
        raise IndexError("scene index outside 67-frame validation set")
    loader_args = dwm.common.instantiate_config(
        config["validation_dataloader"])
    loader_args.pop("batch_size", None)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [args.scene_index]),
        batch_size=1, **loader_args)
    batch = next(iter(loader))
    assert batch["vae_images"].shape[:3] == (1, 67, 2)
    initial = _encode_rgb(pipeline, batch["vae_images"][:, :3])
    latent_shape = (1, 19, *initial.shape[2:])
    exceptions = pipeline.inference_config.get(
        "autoregression_data_exception_for_take_sequence", [])
    print("[probe] input", tuple(batch["vae_images"].shape),
          "latent window", latent_shape, flush=True)

    reference = initial
    for window in range(3):
        start = window * 16
        cond = _window_batch(batch, start, 19, exceptions)
        t0 = time.monotonic()
        generated = sampler(
            latent_shape, cond, "latent", reference,
            reference_frame_count=3)["latents"]
        reference = generated[:, -3:].detach()
        print("[probe] shared window", window + 1,
              "frames", start + 3, "to", start + 18,
              "seconds", round(time.monotonic() - t0, 2), flush=True)

    # With a 16-frame stride, the fourth window is f48..66. Its reference
    # slots are f48..50 and its scored future is exactly f51..66.
    oracle = _encode_rgb(pipeline, batch["vae_images"][:, 48:51])
    anchor = torch.cat([initial[:, :1], reference[:, 1:]], dim=1)
    histories = {"generated": reference, "oracle_gt": oracle,
                 "early_anchor": anchor}
    fourth_cond = _window_batch(batch, 48, 19, exceptions)
    fourth_rng_state = pipeline.generator.get_state()
    ground_truth = stitch_pano_halves(
        batch["vae_images"][:, 51:67].to(device)).clamp(0, 1)
    assert ground_truth.shape[1] == 16
    summary = {
        "scene_index": args.scene_index,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "generated_target_frames": [51, 66],
        "history_frames": [48, 50],
        "history_policy": {
            "generated": "generated f48,f49,f50",
            "oracle_gt": "GT f48,f49,f50; diagnostic upper bound",
            "early_anchor": "GT f0 placed in f48 slot; generated f49,f50; "
                            "stale-slot probe only, not Deep Forcing",
        },
        "metrics": {},
    }
    for name, history in histories.items():
        pipeline.generator.set_state(fourth_rng_state.clone())
        t0 = time.monotonic()
        result = sampler(
            latent_shape, fourth_cond, "pt", history,
            reference_frame_count=3)
        images = result["images"].unflatten(0, (1, 19, 2))[:, 3:]
        fake = stitch_pano_halves(images.to(device)).clamp(0, 1)
        assert fake.shape == ground_truth.shape
        psnr = _psnr_by_frame(fake, ground_truth)
        arm_dir = output / name
        arm_dir.mkdir()
        torch.save({
            "history_latent": history.cpu().to(torch.float16),
            "future_latent": result["latents"][:, 3:].cpu().to(torch.float16),
            "future_pano_uint8": (fake[0, :, 0].cpu() * 255).round().byte(),
        }, arm_dir / "output.pt")
        two_row = torch.cat([
            ground_truth[0, :, 0].cpu(), fake[0, :, 0].cpu()
        ], dim=-2)
        dwm.utils.preview.save_tensor_to_video(
            str(arm_dir / "gt_vs_generated.mp4"), "libx264", 10, two_row)
        summary["metrics"][name] = {
            "psnr_per_frame_db": [round(float(x), 4) for x in psnr],
            "psnr_mean_db": round(sum(psnr) / len(psnr), 4),
            "runtime_seconds": round(time.monotonic() - t0, 2),
        }
        print("[probe]", name, summary["metrics"][name], flush=True)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
