"""Cache real Stage-1 teacher closed-loop prefixes from train clips only."""

import argparse
import json
from pathlib import Path

import torch

import dwm.common
from dwm_pano.generated_latent_cache import atomic_torch_save, scalar_int
from dwm_pano.pipelines.ctsd_pano_srr import history_cache_path
from run_history_probe import _encode_rgb, _window_batch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-target-window", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards or \
            args.max_target_window not in (1, 2, 3):
        parser.error("invalid shard or target-window count")
    config = json.loads(Path(args.config).read_text())
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = config["pipeline"]
    pc["_class_name"] = "dwm_pano.pipelines.ctsd_pano.PanoCrossviewTemporalSD"
    pc.pop("srr_config", None)
    pc["model_checkpoint_path"] = args.checkpoint
    pc["inference_config"]["inference_steps"] = 50
    pc["inference_config"].pop("guidance_scale", None)
    pc["inference_config"].update({
        "reference_frame_count": 3,
        "sequence_length_per_iteration": 19,
        "generate_frames_for_reference": False,
    })
    dataset_config = config["training_dataset"]
    ds = dataset_config["base_dataset"]
    if ds["split"] != "train" or ds["sequence_length"] != 67:
        raise ValueError("cache config must be train split with 67 frames")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline = dwm.common.create_instance_from_config(
        pc, output_path=args.cache_dir, config=config, device=device)
    dataset = dwm.common.create_instance_from_config(dataset_config)
    if len(dataset) == 0:
        raise RuntimeError("no eligible train clips")
    scene_manifest = json.loads(Path(args.scene_manifest).read_text())
    forbidden = {item["scene_token"] for key in ("dev", "heldout")
                 for item in scene_manifest[key]}
    overlap = {item["scene_token"] for item in dataset.base_dataset.clip_audit} & forbidden
    if overlap:
        raise RuntimeError(f"train cache overlaps dev/heldout: {sorted(overlap)}")
    indices = [i for i in range(len(dataset)) if i % args.shards == args.shard]
    if args.max_items:
        indices = indices[:args.max_items]
    loader_config = dwm.common.instantiate_config(config["training_dataloader"])
    loader_config.update({"batch_size": 1, "shuffle": False})
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices), **loader_config)
    exceptions = pc["inference_config"].get(
        "autoregression_data_exception_for_take_sequence", [])
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "source": "stage1_teacher_50step_closed_loop",
        "train_pkl": ds["pkl_path"],
        "shard": args.shard, "shards": args.shards,
        "max_target_window": args.max_target_window,
        "seed": args.seed,
        "items": len(indices),
    }
    (cache_dir / f"manifest_shard{args.shard}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    for position, batch in enumerate(loader):
        if batch["split"][0] != "train":
            raise RuntimeError("cache generation encountered non-train clip")
        index = scalar_int(batch["dataset_index"])
        anchor = scalar_int(batch["anchor_timestamp_us"])
        reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
        latent_shape = (1, 19, *reference.shape[2:])
        # The teacher sees only three real frames at the beginning.  Every
        # later reference comes from the immediately previous generated window.
        for target_window in range(1, args.max_target_window + 1):
            source_window = target_window - 1
            cond = _window_batch(batch, source_window * 16, 19, exceptions)
            result = pipeline.inference_pipeline(
                latent_shape, cond, "pt", reference,
                reference_frame_count=3)
            reference = result["latents"][:, -3:].detach()
            start = target_window * 16
            payload = {
                "schema_version": 1,
                "source": "stage1_teacher_50step_closed_loop",
                "checkpoint": args.checkpoint,
                "dataset_index": index,
                "anchor_timestamp_us": anchor,
                "scene_token": batch["scene_token"][0],
                "split": batch["split"][0],
                "target_window": target_window,
                "history_timestamps_us": batch["frame_timestamps_us"][
                    0, start:start + 3].cpu().contiguous(),
                "generated_history": reference[0].half().cpu().contiguous(),
            }
            atomic_torch_save(
                payload,
                history_cache_path(cache_dir, index, anchor, target_window))
            del result
        print(f"[srr-cache] shard={args.shard} {position + 1}/{len(indices)} "
              f"index={index} windows={args.max_target_window}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
