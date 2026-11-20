"""4-step distilled inference for a consistency-distillation checkpoint.

Generates one panoramic clip with denoising_step_list = [1000, 750, 500, 250] and saves it via
PanoCrossviewTemporalSD.preview_pipeline. The training-loop preview still samples with 50 steps, so this script
checks the distilled model in the few-step regime.
"""
import argparse
import copy
import json
import os

import torch
import torch.distributed

import dwm.common
import dwm.functional
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD


def compute_denoising_sigmas(pipeline):
    """FlowMatch sigma at each pipeline.denoising_step_list timestep.

    The pipeline only stores the timestep values, so the sigmas are looked up once from a 1000-step scheduler; the sampler
    converts model velocity to predicted x0 via x0 = noisy + (-sigma) * velocity.
    """
    lookup = copy.deepcopy(pipeline.test_scheduler)
    lookup.set_timesteps(pipeline.num_train_timestep, pipeline.device)
    lookup_t = lookup.timesteps.to(pipeline.device)
    lookup_s = lookup.sigmas.to(pipeline.device)
    return torch.stack([
        lookup_s[(lookup_t - t).abs().argmin()]
        for t in pipeline.denoising_step_list
    ]).to(pipeline.device)


def make_4step_inference(pipeline, sigmas):
    """Build an inference_pipeline callable that does 4-step denoising.

    The signature matches PanoCrossviewTemporalSD.inference_pipeline, so it can be swapped onto the pipeline instance and
    autoregressive_inference_pipeline (hence preview_pipeline) routes through it. CFG is off because the generator is
    trained on conditional inputs only.
    """
    def inference_pipeline(
        latent_shape, batch, output_type, image_latents=None,
        reference_frame_count=0, start_timestep=0, stop_timestep=None,
        take_time=0,
    ):
        pipeline.model_wrapper.eval()
        latents = torch.randn(
            latent_shape, generator=pipeline.generator
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
                timesteps[:, :reference_frame_count] = 0
            inp = inp.to(dtype=pipeline.model_dtype)

            with pipeline.get_autocast_context():
                model_output, _, _ = pipeline.model_wrapper(
                    inp, timesteps, **cond)
            velocity = model_output[0]
            x0_pred = inp + (-sigma_t) * velocity

            if i < len(pipeline.denoising_step_list) - 1:
                sigma_next = sigmas[i + 1].to(dtype=latents.dtype)
                new_noise = torch.randn(
                    latent_shape, generator=pipeline.generator
                ).to(pipeline.device)
                latents = (1.0 - sigma_next) * x0_pred + sigma_next * new_noise
            else:
                latents = x0_pred

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
                        help="run dir containing checkpoints/<step>.pth")
    parser.add_argument("-s", "--step", type=int, required=True)
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Where to save preview/<step>.mp4")
    parser.add_argument("--ema", action="store_true",
                        help="Use ema_model_wrapper instead of model_wrapper")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Skip the first N samples of val_ds, then preview the next one")
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

    # resume_from loads <step>.pth into the main model and <step>_ema.pth into the EMA model
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=args.run, config=config, device=device,
        resume_from=args.step)

    if args.ema:
        pipeline.model_wrapper = pipeline.ema_model_wrapper

    sigmas = compute_denoising_sigmas(pipeline)
    pipeline.inference_pipeline = make_4step_inference(pipeline, sigmas)

    val_ds = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if args.scene_index > 0:
        # start the subset at scene_index so rank 0's first batch is the requested sample
        val_ds = torch.utils.data.Subset(
            val_ds, range(args.scene_index, len(val_ds)))
    sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, shuffle=False)
    loader = torch.utils.data.DataLoader(
        val_ds,
        **dwm.common.instantiate_config(config["validation_dataloader"]),
        sampler=sampler)
    batch = next(iter(loader))

    pipeline.preview_pipeline(batch, args.output, args.step)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
