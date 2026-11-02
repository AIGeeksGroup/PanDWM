"""4-step distilled FID/FVD evaluation for a Stage 2 (consistency distillation) ckpt.

Hot-swaps the pipeline's `inference_pipeline` with a 4-step
(`denoising_step_list = [1000, 750, 500, 250]`) implementation, then runs the
standard `PanoCrossviewTemporalSD.evaluate_pipeline` over the full validation
set so FID/FVD reflect the actual few-step regime (the training-loop
`evaluate_pipeline` would inherit the parent's 50-step path, which is
mathematically wrong for a distilled generator).

Launched via:
    scripts/eval_ctsd_unimlvg_stage3_tirda_causal_nwa_consistency_distillation_4step.sh
"""
import argparse
import copy
import json
import math
import os

import torch
import torch.distributed

import dwm.common
import dwm.functional
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD


def compute_denoising_sigmas(pipeline):
    """FlowMatch sigma at each `pipeline.denoising_step_list` timestep.

    Pipeline only stores the 4 timestep VALUES, not the sigmas. Look them up
    once from a 1000-step scheduler so the inference loop can convert model
    velocity → predicted x0 via `x0 = noisy + (-sigma) * velocity`.
    """
    lookup = copy.deepcopy(pipeline.test_scheduler)
    lookup.set_timesteps(pipeline.num_train_timestep, pipeline.device)
    lookup_t = lookup.timesteps.to(pipeline.device)
    lookup_s = lookup.sigmas.to(pipeline.device)
    return torch.stack([
        lookup_s[(lookup_t - t).abs().argmin()]
        for t in pipeline.denoising_step_list
    ]).to(pipeline.device)


def _sampler_mode():
    """Which multistep transition the 4-step sampler uses.

    "stochastic" (default) re-noises the x0 prediction with a fresh independent
    draw -- the CM-style sampler this file has always used. Align Your Flow
    Thm 3.1 shows that family degrades past some NFE, and the teacher is scored
    through a 50-step DDPM path instead, so a teacher/student comparison run
    this way confounds model quality with sampler choice.

    "deterministic" takes the flow-matching ODE Euler step, which is also the
    transition the CD teacher supervises against during training.

    Default stays "stochastic" so historical numbers remain reproducible; set
    PANDWM_4STEP_SAMPLER=deterministic to switch.
    """
    mode = os.environ.get("PANDWM_4STEP_SAMPLER", "stochastic").strip().lower()
    if mode not in ("stochastic", "deterministic"):
        raise ValueError(
            "PANDWM_4STEP_SAMPLER must be 'stochastic' or 'deterministic', "
            "got: {}".format(mode))
    return mode


def make_4step_inference(
    pipeline, sigmas, sampler_mode=None, random_generator=None
):
    """Build an `inference_pipeline` callable that does 4-step denoising.

    Signature matches `PanoCrossviewTemporalSD.inference_pipeline` so it can
    be hot-swapped onto the pipeline instance; `autoregressive_inference_pipeline`
    (and hence `preview_pipeline`) will route through it transparently. CFG is
    OFF because the Stage 2 generator is trained on conditional inputs only.
    """
    def inference_pipeline(
        latent_shape, batch, output_type, image_latents=None,
        reference_frame_count=0, start_timestep=0, stop_timestep=None,
        take_time=0, reference_timestep=0,
    ):
        pipeline.model_wrapper.eval()
        mode = _sampler_mode() if sampler_mode is None else sampler_mode
        if mode not in ("stochastic", "deterministic"):
            raise ValueError("unknown 4-step sampler mode: {}".format(mode))
        rng = pipeline.generator if random_generator is None else random_generator
        # Vanilla history guidance (DFoT, Song et al. 2025), off by default.
        # The student saw reference-dropout during training (refs noised and
        # denoised with the targets), so the no-history branch is in
        # distribution: v = v_nohist + w * (v_hist - v_nohist) on targets.
        hg_weight = float(os.environ.get("PANDWM_HISTORY_GUIDANCE", "0") or 0)
        use_hg = (hg_weight > 0 and hg_weight != 1.0
                  and image_latents is not None and reference_frame_count > 0)
        latents = torch.randn(
            latent_shape, generator=rng
        ).to(pipeline.device)

        cond = CrossviewTemporalSD.get_conditions(
            pipeline.model, pipeline.text_encoders, pipeline.tokenizers,
            pipeline.common_config, latent_shape, batch, pipeline.device,
            pipeline.model_dtype,
            do_classifier_free_guidance=False,
            latents_shape=latents.shape)
        pipeline._add_temporal_attention_mask(cond, latent_shape)

        for i in range(len(pipeline.denoising_step_list)):
            t_val = pipeline.denoising_step_list[i]
            sigma_t = sigmas[i].to(dtype=latents.dtype)

            timesteps = torch.full(
                latent_shape[:3], float(t_val.item()),
                dtype=t_val.dtype, device=pipeline.device)
            inp = latents
            if image_latents is not None and reference_frame_count > 0:
                inp = torch.cat([
                    image_latents[:, :reference_frame_count],
                    latents[:, reference_frame_count:]
                ], dim=1)
                timesteps[:, :reference_frame_count] = reference_timestep
            inp = inp.to(dtype=pipeline.model_dtype)

            with pipeline.get_autocast_context():
                model_output, _, _ = pipeline.model_wrapper(
                    inp, timesteps, **cond)
            velocity = model_output[0]
            if use_hg:
                weak_timesteps = torch.full(
                    latent_shape[:3], float(t_val.item()),
                    dtype=t_val.dtype, device=pipeline.device)
                with pipeline.get_autocast_context():
                    weak_output, _, _ = pipeline.model_wrapper(
                        latents.to(dtype=pipeline.model_dtype),
                        weak_timesteps, **cond)
                weak = weak_output[0]
                guided = weak + hg_weight * (velocity - weak)
                velocity = torch.cat([
                    velocity[:, :reference_frame_count],
                    guided[:, reference_frame_count:]], dim=1)
            x0_pred = inp + (-sigma_t) * velocity

            if i < len(pipeline.denoising_step_list) - 1:
                sigma_next = sigmas[i + 1].to(dtype=latents.dtype)
                if mode == "deterministic":
                    # Euler step along the flow-matching ODE. Since
                    # x_s = (1 - s) * x0 + s * eps and x0 = inp - sigma_t * v,
                    # this equals re-noising x0_pred with the noise implied by
                    # the current state instead of a fresh draw -- and it is the
                    # same transition the CD teacher uses during training
                    # (next = noisy + (next_sigma - sigma) * velocity).
                    latents = inp + (sigma_next - sigma_t) * velocity
                else:
                    # CM-style stochastic multistep: re-noise with an
                    # independent draw. This is the historical default; every
                    # number reported before 2026-09-01 used it.
                    new_noise = torch.randn(
                        latent_shape, generator=rng
                    ).to(pipeline.device)
                    latents = \
                        (1.0 - sigma_next) * x0_pred + sigma_next * new_noise
            else:
                latents = x0_pred

        if output_type == "latent":
            return {"latents": latents}

        shift_factor = (
            pipeline.vae.config.shift_factor
            if pipeline.vae.config.shift_factor is not None else 0)
        image_tensor = dwm.functional.memory_efficient_split_call(
            pipeline.vae,
            latents.flatten(0, 2).to(dtype=pipeline.vae.dtype),
            lambda block, tensor: block.decode(
                tensor / block.config.scaling_factor + shift_factor,
                return_dict=False)[0],
            pipeline.common_config.get("memory_efficient_batch", -1))
        images = pipeline.image_processor.postprocess(
            image_tensor, output_type=output_type)
        return {"images": images, "latents": latents}

    return inference_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--run", type=str, required=True,
                        help="Stage 2 run dir containing checkpoints/<step>.pth")
    parser.add_argument("-s", "--step", type=int, required=True)
    parser.add_argument("--ema", action="store_true",
                        help="Use ema_model_wrapper instead of model_wrapper")
    args = parser.parse_args()

    config = json.loads(open(args.config).read())
    # Mesh: the config it shares with training targets 2 nodes, so its
    # `mesh_shape` describes the *training* allocation, not whatever this eval
    # was launched on. `init_device_mesh` refuses a mesh larger than the world
    # size, so an unconditional value crashes every allocation but the one it
    # was written for -- which is why this used to be monkeypatched from the
    # outside on single-GPU runs.
    #
    # Honour the config only when it matches the actual allocation, otherwise
    # derive a flat mesh from WORLD_SIZE and say so. This reproduces both
    # historical behaviours exactly: a 4-GPU run keeps [1, 4], and a 1-GPU run
    # gets [1, 1], which is what the external patch forced.
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _mesh_shape = config["global_state"]["device_mesh"].get("mesh_shape")
    _mesh_numel = math.prod(_mesh_shape) if _mesh_shape else 0
    if _mesh_numel != _world_size:
        config["global_state"]["device_mesh"]["mesh_shape"] = [1, _world_size]
        print(
            "[4step] mesh_shape {} does not match WORLD_SIZE {}; using {}"
            .format(_mesh_shape, _world_size, [1, _world_size]), flush=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    for k, v in config["global_state"].items():
        dwm.common.global_state[k] = dwm.common.create_instance_from_config(v)

    # When --ema: skip resume_from and point both model_checkpoint_path and
    # consistency_ema_checkpoint_path at `<step>_ema.pth` so the main model
    # ends up with EMA weights at init time. Avoids needing `<step>.pth` and
    # the optimizer/<step>/ distcp dir on disk — only `<step>_ema.pth` matters.
    # When --ema absent: resume_from triggers normal load of main + EMA + optim.
    if args.ema:
        ema_path = os.path.join(args.run, "checkpoints", f"{args.step}_ema.pth")
        config["pipeline"]["model_checkpoint_path"] = ema_path
        config["pipeline"]["consistency_ema_checkpoint_path"] = ema_path
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"],
            output_path=args.run, config=config, device=device)
    else:
        pipeline = dwm.common.create_instance_from_config(
            config["pipeline"],
            output_path=args.run, config=config, device=device,
            resume_from=args.step)

    sigmas = compute_denoising_sigmas(pipeline)
    pipeline.inference_pipeline = make_4step_inference(pipeline, sigmas)
    # Provenance: the multistep transition is a real degree of freedom that
    # changes the reported numbers, so every run states which one it used.
    print("[4step] sampler={}  denoising_step_list={}  sigmas={}".format(
        _sampler_mode(),
        [int(t) for t in pipeline.denoising_step_list],
        ["{:.4f}".format(float(s)) for s in sigmas]), flush=True)

    # Match `src/dwm_pano/evaluate.py`: Subset the val set to a length divisible
    # by total batch_size so every rank gets the same count (clean FID/FVD
    # gather), then a DistributedSampler over that subset.
    val_ds = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    total_batch_size = (
        int(os.environ["WORLD_SIZE"])
        * config["validation_dataloader"]["batch_size"])
    selected_indices = os.environ.get("PANDWM_EVAL_INDICES")
    if selected_indices:
        indices = [
            int(value) for value in selected_indices.replace(":", ",").split(",")]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(
                "PANDWM_EVAL_INDICES must contain unique integer indices")
        if min(indices) < 0 or max(indices) >= len(val_ds):
            raise IndexError(
                "PANDWM_EVAL_INDICES escapes dataset length {}: {}".format(
                    len(val_ds), indices))
        if len(indices) % total_batch_size:
            raise ValueError(
                "selected evaluation count {} must be divisible by global "
                "batch size {}".format(len(indices), total_batch_size))
        print("[4step] selected validation indices={}".format(indices),
              flush=True)
        dataset_length = len(indices)
        val_ds = torch.utils.data.Subset(val_ds, indices)
    else:
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
