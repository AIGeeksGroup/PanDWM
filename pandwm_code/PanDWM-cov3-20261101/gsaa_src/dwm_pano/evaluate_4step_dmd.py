"""4-step distilled FID/FVD evaluation for a Stage 3 (Asymmetric DMD) ckpt.

Unlike `evaluate_4step.py` for Stage 2 CD, no `inference_pipeline` monkey-patch
is needed — `PanoCrossviewTemporalDMD.inference_pipeline` IS the 4-step impl
(see `ctsd_pano_dmd.py:inference_pipeline`). The base
`autoregressive_inference_pipeline` polymorphically routes through it.

DMD-specific quirks handled here:
  - --ema: override `model_checkpoint_path` to `<step>_ema.pth` and skip
    `resume_from` so we don't need critic / optimizer artifacts.
  - non-ema (default): override to `<step>.pth` likewise.
Either way the critic stays loaded from the config's
`fake_score_checkpoint_path` (only used for training KL grad — irrelevant
for inference). The bidirectional `real_score` teacher is also unused for
4-step generation but the DMD pipeline ctor still builds it; it's loaded
from the config and held frozen.

Launched via:
    scripts/eval_ctsd_unimlvg_stage3_tirda_causal_nwa_dmd_4step.sh
"""
import argparse
import json
import os

import torch
import torch.distributed

import dwm.common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--run", type=str, required=True,
                        help="Stage 3 DMD run dir containing checkpoints/<step>.pth")
    parser.add_argument("-s", "--step", type=int, required=True)
    parser.add_argument("--ema", action="store_true",
                        help="Use <step>_ema.pth as generator weights")
    args = parser.parse_args()

    config = json.loads(open(args.config).read())
    # Mesh override: training config targets 2 nodes; this script defaults to
    # the 1-node auxiliary alloc to avoid stealing GPUs from the live train
    # job. Bump to [2, 4] in the config if you launch on the main 2-node alloc.
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 4]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    for k, v in config["global_state"].items():
        dwm.common.global_state[k] = dwm.common.create_instance_from_config(v)

    # Override model_checkpoint_path to the requested step's weights and skip
    # resume_from. Skipping resume_from avoids:
    #   - loading optimizer state for gen + critic (irrelevant for eval)
    #   - DMD ctor's critic ckpt redirect to `<step>_critic.pth` (we leave
    #     the config's `fake_score_checkpoint_path` instead — critic is
    #     unused in inference anyway).
    suffix = "_ema" if args.ema else ""
    ckpt_path = os.path.join(
        args.run, "checkpoints", f"{args.step}{suffix}.pth")
    config["pipeline"]["model_checkpoint_path"] = ckpt_path

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=args.run, config=config, device=device)

    # Subset the val set to a length divisible by total batch_size so every
    # rank gets the same count (clean FID/FVD gather), then DistributedSampler.
    val_ds = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    total_batch_size = (
        int(os.environ["WORLD_SIZE"])
        * config["validation_dataloader"]["batch_size"])
    dataset_length = len(val_ds) // total_batch_size * total_batch_size
    val_ds = torch.utils.data.Subset(val_ds, range(0, dataset_length))
    sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
    loader = torch.utils.data.DataLoader(
        val_ds,
        **dwm.common.instantiate_config(config["validation_dataloader"]),
        sampler=sampler)

    pipeline.evaluate_pipeline(args.step, len(val_ds), loader, sampler)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
