"""Dump the DMD generator/critic/teacher x0 predictions as decoded images.

Loads a Stage-3 DMD checkpoint (generator + trained critic via `resume_from`),
runs one validation batch through the generator to get `x0_pred`, then at a few
noise levels computes the critic's `fake_x0` and the teacher's (CFG-guided)
`real_x0` — exactly the quantities whose difference IS the DMD gradient
(`grad = fake_x0 - real_x0`). Decodes all three through the VAE and saves PNGs
so we can SEE what the DMD loss is pulling the generator toward.

No FID, no fallback paths — straight-line so any mismatch crashes loudly.

Launched the same way as evaluate_4step_dmd.py (1 node x 4 GPU).
"""
import argparse
import json
import os

import torch
import torch.distributed

import dwm.common
import dwm.functional
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD


def decode_to_images(pipeline, latents):
    # latents: (B, F, V, C, H, W) -> list of PIL images (B*F*V).
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
    return pipeline.image_processor.postprocess(image_tensor, output_type="pil")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--run", type=str, required=True)
    parser.add_argument("-s", "--step", type=int, required=True)
    parser.add_argument("-o", "--out", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(open(args.config).read())
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 4]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    rank = torch.distributed.get_rank()

    for k, v in config["global_state"].items():
        dwm.common.global_state[k] = dwm.common.create_instance_from_config(v)

    # resume_from loads the trained generator (<step>.pth), the trained critic
    # (<step>_critic.pth) and EMA (<step>_ema.pth) — we want the LIVE critic.
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"],
        output_path=args.run, config=config, device=device,
        resume_from=args.step)

    pipeline.model_wrapper.eval()
    pipeline.fake_score_wrapper.eval()
    pipeline.real_score_wrapper.eval()

    # Training loader, copied verbatim from train.py (non-mix DDP branch):
    # produces exactly `sequence_length_per_iteration` frames per batch, so the
    # batch matches what train_step / `_run_generator` / `get_conditions`
    # consume (the val loader yields full clips → frame-count mismatch).
    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    training_datasampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset, shuffle=config["data_shuffle"],
        seed=config["generator_seed"])
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        **dwm.common.instantiate_config(config["training_dataloader"]),
        sampler=training_datasampler)
    batch = next(iter(training_dataloader))

    seq_len = pipeline.inference_config["sequence_length_per_iteration"]
    view_count = batch["vae_images"].shape[2]
    latent_shape = pipeline._get_latent_shape(batch, seq_len, view_count)

    image_latents = None
    ref_count = pipeline.inference_config.get("reference_frame_count", 0)
    if (not pipeline.inference_config.get("generate_frames_for_reference", True)
            and ref_count > 0):
        image_latents = pipeline._vae_encode(
            batch["vae_images"][:, :ref_count])

    with torch.no_grad(), pipeline.get_autocast_context():
        cond = CrossviewTemporalSD.get_conditions(
            pipeline.model, pipeline.text_encoders, pipeline.tokenizers,
            pipeline.common_config, latent_shape, batch, pipeline.device,
            pipeline.model_dtype, do_classifier_free_guidance=False,
            latents_shape=latent_shape)
        cond_cfg = CrossviewTemporalSD.get_conditions(
            pipeline.model, pipeline.text_encoders, pipeline.tokenizers,
            pipeline.common_config, latent_shape, batch, pipeline.device,
            pipeline.model_dtype, do_classifier_free_guidance=True,
            latents_shape=latent_shape)

    with torch.no_grad():
        x0_pred = pipeline._run_generator(
            latent_shape, cond, image_latents=image_latents,
            reference_frame_count=ref_count, grad_on_last=False)

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        for i, img in enumerate(decode_to_images(pipeline, x0_pred)):
            img.save(os.path.join(args.out, f"x0_pred_{i:02d}.png"))

    # At each denoising timestep: noise x0_pred, then read the critic's fake_x0
    # and the teacher's CFG real_x0 (same math as `_compute_kl_grad`).
    cond_real = {
        k: v for k, v in cond_cfg.items() if k != "temporal_attention_mask"}
    cond_fake = dict(cond)
    if pipeline.fake_score_causal:
        pipeline._add_temporal_attention_mask(cond_fake, x0_pred.shape)

    for idx in range(len(pipeline.denoising_step_list)):
        t_val = pipeline.denoising_step_list[idx]
        sigma = pipeline.denoising_sigmas[idx].to(dtype=x0_pred.dtype)
        ts = torch.full(
            latent_shape[:3], float(t_val.item()),
            dtype=t_val.dtype, device=device)

        noise = torch.randn_like(x0_pred)
        noisy = sigma * noise + (1.0 - sigma) * x0_pred

        with torch.no_grad(), pipeline.get_autocast_context():
            fake_out, _, _ = pipeline.fake_score_wrapper(
                noisy.to(pipeline.model_dtype), ts, **cond_fake)
        fake_x0 = noisy.float() + (-sigma).float() * fake_out[0].float()

        noisy_cfg = torch.cat([noisy, noisy], dim=0)
        ts_cfg = torch.cat([ts, ts], dim=0)
        with torch.no_grad(), pipeline.get_autocast_context():
            real_out, _, _ = pipeline.real_score_wrapper(
                noisy_cfg.to(pipeline.model_dtype), ts_cfg, **cond_real)
        real_x0_cfg = noisy_cfg.float() + (-sigma).float() * real_out[0].float()
        real_x0_uncond, real_x0_cond = real_x0_cfg.chunk(2, dim=0)
        real_x0 = real_x0_cond + (
            real_x0_cond - real_x0_uncond) * pipeline.real_guidance_scale

        d_real = (x0_pred.float() - real_x0).abs().mean().item()
        d_fake = (x0_pred.float() - fake_x0).abs().mean().item()
        d_grad = (fake_x0 - real_x0).abs().mean().item()
        if rank == 0:
            print(f"[t={int(t_val.item())}] |x0_pred-real_x0|={d_real:.4f} "
                  f"|x0_pred-fake_x0|={d_fake:.4f} |grad(fake-real)|={d_grad:.4f}",
                  flush=True)
            t = int(t_val.item())
            for i, img in enumerate(decode_to_images(pipeline, real_x0)):
                img.save(os.path.join(args.out, f"real_x0_t{t}_{i:02d}.png"))
            for i, img in enumerate(decode_to_images(pipeline, fake_x0.to(x0_pred.dtype))):
                img.save(os.path.join(args.out, f"fake_x0_t{t}_{i:02d}.png"))

    if rank == 0:
        for i, img in enumerate(
                pipeline.image_processor.postprocess(
                    batch["vae_images"].flatten(0, 2).to(device),
                    output_type="pil")):
            img.save(os.path.join(args.out, f"gt_{i:02d}.png"))
        print(f"saved to {args.out}", flush=True)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
