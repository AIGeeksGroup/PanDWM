"""4-step distilled FID/FVD evaluation for a DMD-distilled checkpoint.

Unlike evaluate_4step.py (consistency distillation), no inference_pipeline patch is needed: the DMD pipeline's own
inference_pipeline is the 4-step sampler, and the base autoregressive_inference_pipeline routes through it.

The checkpoint is selected by overriding model_checkpoint_path with <step>.pth (or <step>_ema.pth with --ema) and
skipping resume_from, so no critic or optimizer state is loaded. The critic and the real-score teacher that the DMD
constructor builds are not used for generation.
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
                        help="DMD run dir containing checkpoints/<step>.pth")
    parser.add_argument("-s", "--step", type=int, required=True)
    parser.add_argument("--ema", action="store_true",
                        help="Use <step>_ema.pth as generator weights")
    args = parser.parse_args()

    config = json.loads(open(args.config).read())
    # the training config targets 2 nodes; this script uses a single-node mesh
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 4]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    for k, v in config["global_state"].items():
        dwm.common.global_state[k] = dwm.common.create_instance_from_config(v)

    # load the requested step's weights directly; skipping resume_from avoids loading optimizer / critic state
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
