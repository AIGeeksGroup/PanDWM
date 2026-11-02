"""Cache the CF-DMD student's own closed-loop history on TRAIN clips (2026-10-15; teacher fine-tune v3, report 30).

The teacher fine-tune (SRR-style recovery) trains the Stage-1 teacher to predict the REAL target window from the
history the student actually produces.  For every train clip (83 frames = 3 GT context frames + 5 windows), the
student rolls out windows 1..W (W = --max-target-window, default 4) from the 3 GT frames with the deployed sampler
(4-step stochastic, schedule [1000, 625, 250, 42], blend 128, same as the Dev-76-L rollouts) and the dataset's own
future conditions.  After window j the last 3 generated latent frames are the history of target window j (its GT
frames 16j .. 16j+18 stay the training target); they are saved as
  <cache>/<dataset_index:08d>_<anchor_timestamp_us>_w<j:02d>.pt
with the clip identity and frame timestamps, so the trainer can check that a cache entry belongs to the clip it reads.
Each clip uses its own sampler seed (seed + dataset_index), so shards and reruns are reproducible; existing entries
are skipped.  Only the train split is accepted.
  python lowmem_launch.py cache_student_history.py --base-config cd_eval_frozen149_hpc3.json --checkpoint K \
      --train-config <cfdmd train config> --cache-dir D --shard 0 --shards 8
"""
import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
from dwm_pano import evaluate_4step

from run_history_probe import _encode_rgb, _window_batch

SOURCE = "cfdmd_student_4step_closed_loop"


def cache_path(cache_dir, dataset_index, anchor, window):
    return Path(cache_dir) / f"{int(dataset_index):08d}_{int(anchor)}_w{int(window):02d}.pt"


def scalar(x):
    return int(x.reshape(-1)[0].item()) if isinstance(x, torch.Tensor) else int(x)


def atomic_save(payload, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="frozen eval config (student pipeline, val dataset)")
    ap.add_argument("--train-config", required=True, help="CF-DMD train config: the train dataset paths are read")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--max-target-window", type=int, default=4)
    ap.add_argument("--stride-s", type=float, default=20.0, help="clip start stride (fps_stride_tuples)")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()
    if not 0 <= a.shard < a.shards or not 1 <= a.max_target_window <= 6:
        ap.error("invalid shard or window count")
    clip_length = 16 * a.max_target_window + 19
    config = json.load(open(a.base_config, encoding="utf-8"))
    train_ds = json.load(open(a.train_config, encoding="utf-8"))["training_dataset"]["base_dataset"]
    config["generator_seed"] = a.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = config["pipeline"]
    pc["_class_name"] = ("dwm_pano.pipelines.ctsd_pano_consistency_distillation."
                         "PanoCrossviewTemporalConsistencyDistillation")
    pc["model_checkpoint_path"] = a.checkpoint
    pc["consistency_ema_checkpoint_path"] = a.checkpoint
    pc["denoising_step_list"] = [1000, 625, 250, 42]
    pc["inference_config"].update({"sequence_length_per_iteration": 19, "reference_frame_count": 3,
                                   "generate_frames_for_reference": False})
    dsc = json.loads(json.dumps(config["validation_dataset"]))
    ds = dsc["base_dataset"]
    for key in ("pkl_path", "rgb_dir", "bbox_dir", "hdmap_dir", "nuscenes_meta_dir"):
        ds[key] = train_ds[key]
    ds.update({"split": "train", "sequence_length": clip_length, "fps_stride_tuples": [[10, a.stride_s]],
               "bad_timestamps_us": train_ds.get("bad_timestamps_us", [])})
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    out = Path(a.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = dwm.common.create_instance_from_config(pc, output_path=str(out / "_log"), config=config,
                                                      device=device)
    for name in ("teacher_model", "teacher_model_wrapper", "ema_model", "ema_model_wrapper"):
        if getattr(pipeline, name, None) is not None:
            setattr(pipeline, name, None)
    torch.cuda.empty_cache()
    sampler = evaluate_4step.make_4step_inference(pipeline, evaluate_4step.compute_denoising_sigmas(pipeline))
    dataset = dwm.common.create_instance_from_config(dsc)
    n = len(dataset)
    indices = [i for i in range(n) if i % a.shards == a.shard]
    if a.max_items:
        indices = indices[:a.max_items]
    manifest = {"schema_version": 1, "source": SOURCE, "checkpoint": a.checkpoint, "train_pkl": ds["pkl_path"],
                "clip_length": clip_length, "stride_s": a.stride_s, "max_target_window": a.max_target_window,
                "seed": a.seed, "dataset_len": n, "shard": a.shard, "shards": a.shards, "items": len(indices),
                "sampler": "4-step stochastic [1000, 625, 250, 42]"}
    (out / f"manifest_shard{a.shard}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"CACHE_START shard={a.shard}/{a.shards} clips={len(indices)} of {n} length={clip_length}", flush=True)
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    loader_args.update({"batch_size": 1, "shuffle": False})
    loader_args.pop("persistent_workers", None)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices), **loader_args)
    exceptions = pc["inference_config"].get("autoregression_data_exception_for_take_sequence", [])
    done = skipped = 0
    t_start = time.monotonic()
    for pos, batch in enumerate(loader):
        if batch["split"][0] != "train":
            raise RuntimeError("cache generation met a non-train clip")
        index, anchor = scalar(batch["dataset_index"]), scalar(batch["anchor_timestamp_us"])
        paths = [cache_path(out, index, anchor, w) for w in range(1, a.max_target_window + 1)]
        if all(p.is_file() for p in paths):
            skipped += 1
            continue
        assert batch["vae_images"].shape[:3] == (1, clip_length, 2), batch["vae_images"].shape
        pipeline.generator.manual_seed(a.seed + index)
        reference = _encode_rgb(pipeline, batch["vae_images"][:, :3])
        latent_shape = (1, 19, *reference.shape[2:])
        for window in range(1, a.max_target_window + 1):
            cond = _window_batch(batch, 16 * (window - 1), 19, exceptions)
            result = sampler(latent_shape, cond, "latent", reference, reference_frame_count=3)
            reference = result["latents"][:, -3:].detach()
            if not bool(torch.isfinite(reference).all()):
                raise RuntimeError(f"non-finite latent, clip {index} window {window}")
            start = 16 * window
            atomic_save({
                "schema_version": 1, "source": SOURCE, "checkpoint": a.checkpoint,
                "dataset_index": index, "anchor_timestamp_us": anchor, "scene_token": batch["scene_token"][0],
                "split": "train", "target_window": window,
                "history_timestamps_us": batch["frame_timestamps_us"][0, start:start + 3].cpu().contiguous(),
                "generated_history": reference[0].half().cpu().contiguous(),
            }, paths[window - 1])
            del result
        done += 1
        if done % 10 == 1 or pos + 1 == len(indices):
            rate = (time.monotonic() - t_start) / max(done, 1)
            print(f"[student-cache] shard={a.shard} {pos + 1}/{len(indices)} done={done} skipped={skipped} "
                  f"{rate:.1f}s/clip", flush=True)
    print(f"CACHE_DONE shard={a.shard} done={done} skipped={skipped} clips={len(indices)}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
