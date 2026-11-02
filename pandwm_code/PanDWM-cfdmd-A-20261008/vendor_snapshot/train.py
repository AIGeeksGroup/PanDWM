import warnings
warnings.filterwarnings("ignore")
# Import torch BEFORE transformers / diffusers / huggingface_hub: their import
# chains transitively call into torch.cuda's lazy init from a context where
# `_check_capability` sees `num_gpus=` empty on this cluster, triggering
# `device=N, num_gpus= INTERNAL ASSERT FAILED` at the later `set_device`.
# Plain `import torch` first sidesteps the bad init ordering.
import torch
import transformers.utils.logging as _tfm_log
_tfm_log.set_verbosity_error()
_tfm_log.disable_progress_bar()
import diffusers.utils.logging as _diff_log
_diff_log.set_verbosity_error()
_diff_log.disable_progress_bar()
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

import argparse
import dwm.common
import itertools
import json
import os
from dwm.utils.sampler import VariableVideoBatchSampler


class _InitialSamplerOffset(torch.utils.data.Sampler):
    """Skip sampler items once, without loading the skipped dataset items.

    This is used for an exact mid-epoch resume.  Subsequent epochs delegate to
    the complete wrapped sampler again, so a fixed-order sampler resumes the
    same sequence an uninterrupted run would have consumed.
    """

    def __init__(self, sampler, offset: int):
        self.sampler = sampler
        self.initial_offset = offset

    def __iter__(self):
        offset = self.initial_offset
        self.initial_offset = 0
        return itertools.islice(iter(self.sampler), offset, None)

    def __len__(self):
        return max(0, len(self.sampler) - self.initial_offset)

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "--log-steps", default=100, type=int,
        help="The step count to print log and update the tensorboard.")
    parser.add_argument(
        "--preview-steps", default=400, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--checkpointing-steps", default=10000, type=int,
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--evaluation-steps", default=10000, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--resume-from", default=None, type=int,
        help="The step to resume from")
    parser.add_argument(
        "--resume-data-offset", default=0, type=int,
        help="Skip this many sampler items once after restoring a checkpoint. "
             "Requires DDP without mix_config and training batch_size=1.")
    parser.add_argument(
        "--initial-data-offset", default=0, type=int,
        help="Skip this many sampler items once for a fresh/warm-start run. "
             "This does not change global_step and cannot be combined with "
             "--resume-from. Requires DDP without mix_config and batch_size=1.")
    parser.add_argument(
        "--max-steps", default=None, type=int,
        help="Stop cleanly after completing this global step (inclusive of "
             "logging, preview, checkpointing, and evaluation at that step).")
    parser.add_argument(
        "--wandb", action="store_true",
        help="Use wandb to log the training process.")
    parser.add_argument(
        "--wandb-project", type=str, default="dwm",
        help="The wandb project name.")
    parser.add_argument(
        "--wandb-run-name", type=str, default="train",
        help="The wandb run name.")
    parser.add_argument(
        "--eval-on-save", action="store_true",
        help="At each ckpt step: save (no upload) → eval(EMA, 4-step) → "
             "upload+prune. Pipeline must implement evaluate_4step_ema_pipeline.")
    parser.add_argument(
        "--keep-last-ckpts", type=int, default=0,
        help="Local ckpt retention after each upload: 0 = keep all (default); "
             "N>0 = keep only the most recent N step numbers locally.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.resume_data_offset < 0:
        parser.error("--resume-data-offset must be non-negative")
    if args.resume_data_offset and args.resume_from is None:
        parser.error("--resume-data-offset requires --resume-from")
    if args.initial_data_offset < 0:
        parser.error("--initial-data-offset must be non-negative")
    if args.initial_data_offset and args.resume_from is not None:
        parser.error("--initial-data-offset cannot be combined with --resume-from")
    if args.initial_data_offset and args.resume_data_offset:
        parser.error(
            "--initial-data-offset and --resume-data-offset are mutually exclusive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if (
        args.max_steps is not None and args.resume_from is not None and
        args.max_steps <= args.resume_from
    ):
        parser.error("--max-steps must be greater than --resume-from")

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    torch.manual_seed(config["generator_seed"])

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        # 60 min NCCL timeout (default 10 min). `--eval-on-save` runs a full
        # val-set eval inside the cycle (~27 min for 60 batches × 27 s) which
        # would otherwise trip NCCL watchdog on collectives that interleave
        # with rank-0-only work (eval restore, prune).
        import datetime
        torch.distributed.init_process_group(
            backend=config["ddp_backend"],
            timeout=datetime.timedelta(minutes=60))
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    output_path = config["output_path"] if args.output_path is None else args.output_path

    # Direct wandb.log() pattern from official quickstart
    # (https://docs.wandb.ai/quickstart/). NO sync_tensorboard because mixing
    # TB-sync with explicit run.log(step=N) causes wandb to silently drop the
    # step= parameter (warning: "Step cannot be set when using tensorboard
    # syncing"). pipeline.log() in ctsd_pano.py calls run.log() directly with
    # explicit step=global_step. Order doesn't matter without sync_tensorboard.
    if args.wandb and should_save:
        import wandb
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=config)

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=output_path, config=config,
        device=device, resume_from=args.resume_from)

    if should_log:
        print("The pipeline is loaded.")

    # load the dataset
    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:

        if "mix_config" in config.keys():
            if args.resume_data_offset or args.initial_data_offset:
                raise ValueError(
                    "data offsets do not support mix_config")
            process_group = torch.distributed.group.WORLD

            training_datasampler = VariableVideoBatchSampler(
                training_dataset,
                config["mix_config"],
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=config["data_shuffle"],
                seed=config["generator_seed"]
            )

            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                batch_sampler=training_datasampler)

        else:
            training_datasampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset, shuffle=config["data_shuffle"],
                seed=config["generator_seed"])
            data_offset = args.resume_data_offset or args.initial_data_offset
            if data_offset:
                batch_size = config["training_dataloader"].get(
                    "batch_size", 1)
                if batch_size != 1:
                    raise ValueError(
                        "data offsets currently require training "
                        "batch_size=1, got {}".format(batch_size))
                sampler_length = len(training_datasampler)
                if data_offset >= sampler_length:
                    raise ValueError(
                        "data offset {} must be smaller than sampler length {}"
                        .format(data_offset, sampler_length))
                if args.resume_data_offset:
                    expected_offset = args.resume_from % sampler_length
                    if args.resume_data_offset != expected_offset:
                        raise ValueError(
                            "resume data offset {} does not match resume step {} "
                            "modulo sampler length {} (expected {})".format(
                                args.resume_data_offset, args.resume_from,
                                sampler_length, expected_offset))
                training_datasampler = _InitialSamplerOffset(
                    training_datasampler, data_offset)
            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                sampler=training_datasampler)

        # make equal sample count for each process to simplify the result
        # gathering
        total_batch_size = int(os.environ["WORLD_SIZE"]) * \
            config["validation_dataloader"]["batch_size"]
        dataset_length = len(validation_dataset) // \
            total_batch_size * total_batch_size
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(0, dataset_length))
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        if args.resume_data_offset or args.initial_data_offset:
            raise ValueError(
                "data offsets currently require the DDP sampler path")
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            **dwm.common.instantiate_config(config["training_dataloader"]),
            shuffle=config["data_shuffle"])
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["preview_dataloader"])) if \
        "preview_dataloader" in config else None
    if preview_dataloader is not None:
        preview_data_iterator = iter(preview_dataloader)

    if should_log:
        print("The training dataset is loaded with {} items.".format(
            len(training_dataset)))
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))
        if args.resume_data_offset:
            print("The first resumed epoch will skip {} sampler items.".format(
                args.resume_data_offset))
        if args.initial_data_offset:
            print("The warm-start epoch will skip {} sampler items.".format(
                args.initial_data_offset))

    # train loop
    global_step = 0 if args.resume_from is None else args.resume_from
    stop_training = False
    for epoch in range(config["train_epochs"]):

        if ddp:
            # Fixing training data order reduces the accessed objects per rank,
            # therefore reduces the upper-bound of memory usage comsumed by the
            # Python reference counting of objects.
            sampler_epoch = 0 if config.get("fix_training_data_order", False) \
                else epoch
            training_datasampler.set_epoch(sampler_epoch)

        # Restore RNG after DataLoader iterator creation but before loading the
        # first resumed batch; otherwise its base-seed draw perturbs training.
        training_iterator = iter(training_dataloader)
        if hasattr(pipeline, "restore_pending_rng"):
            pipeline.restore_pending_rng()
        for batch in training_iterator:
            if args.initial_data_offset and global_step == 0 and should_log:
                dataset_index = batch.get("dataset_index")
                anchor_timestamp = batch.get("anchor_timestamp_us")
                if isinstance(dataset_index, torch.Tensor) \
                        and isinstance(anchor_timestamp, torch.Tensor):
                    print(
                        "First warm-start batch after offset: "
                        "dataset_index={} anchor_timestamp_us={}.".format(
                            int(dataset_index.reshape(-1)[0]),
                            int(anchor_timestamp.reshape(-1)[0])),
                        flush=True)
            pipeline.train_step(batch, global_step)
            global_step += 1

            # log
            if global_step % args.log_steps == 0:
                pipeline.log(global_step, args.log_steps)

            # preview
            if global_step % args.preview_steps == 0:
                if preview_dataloader is None:
                    pipeline.preview_pipeline(batch, output_path, global_step)
                else:
                    try:
                        preview_batch = next(preview_data_iterator)
                    except StopIteration:
                        preview_data_iterator = iter(preview_dataloader)
                        preview_batch = next(preview_data_iterator)

                    pipeline.preview_pipeline(
                        preview_batch, output_path, global_step)

            # save (no upload) → eval(EMA, 4-step) → upload+prune cycle, opt-in
            # via `--eval-on-save`. The pipeline must implement
            # `evaluate_4step_ema_pipeline` (CD / DMD); eval acts as the gate
            # before HF upload of the just-saved ckpt.
            if global_step % args.checkpointing_steps == 0:
                if args.eval_on_save:
                    pipeline.save_checkpoint(
                        output_path, global_step, upload=False)
                    pipeline.evaluate_4step_ema_pipeline(
                        global_step, len(validation_dataset),
                        validation_dataloader, validation_datasampler)
                    if should_save:
                        pipeline._upload_and_prune(
                            output_path, global_step,
                            keep_last_n=args.keep_last_ckpts)
                else:
                    pipeline.save_checkpoint(
                        output_path, global_step,
                        keep_last_n=args.keep_last_ckpts)

            # standalone evaluation at its own cadence
            if (
                args.evaluation_steps > 0 and
                global_step % args.evaluation_steps == 0 and
                not (args.eval_on_save and
                     global_step % args.checkpointing_steps == 0)
            ):
                pipeline.evaluate_pipeline(
                    global_step, len(validation_dataset),
                    validation_dataloader, validation_datasampler)

            if args.max_steps is not None and global_step >= args.max_steps:
                if should_log:
                    print("Reached max step {}; stopping cleanly.".format(
                        args.max_steps), flush=True)
                stop_training = True
                break

        if should_log:
            print("Epoch {} done.".format(epoch))
        if stop_training:
            break

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
