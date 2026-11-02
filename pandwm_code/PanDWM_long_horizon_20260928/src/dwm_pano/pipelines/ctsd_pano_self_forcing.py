"""On-policy history training for the PanDWM four-step CD student.

This is a Self-Forcing-inspired *context-matched CD* experiment. It does not
implement the original paper's holistic DMD/SiD/GAN objective or rolling KV
cache, and must be described with that qualification in reported results.
"""

import random
import time

import dwm.functional
import einops
import torch

from dwm_pano.evaluate_4step import (
    compute_denoising_sigmas, make_4step_inference)
from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import (
    slice_sequence_batch, validate_rollout_config)
from dwm_pano.pipelines.ctsd_pano_consistency_distillation import \
    PanoCrossviewTemporalConsistencyDistillation


class PanoSelfForcingConsistencyDistillation(
    PanoCrossviewTemporalConsistencyDistillation
):
    """Use detached four-step student rollouts as later-window history."""

    def __init__(self, *args, self_forcing=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_forcing = validate_rollout_config(self_forcing or {})
        self._generated_reference_context = None

    def _training_reference_context(self, latents, reference_mask):
        prefix = self._generated_reference_context
        if prefix is None:
            return None
        if prefix.shape != latents[:, :self.self_forcing["reference_frame_count"]].shape:
            raise ValueError("generated prefix shape does not match target window")
        context = torch.zeros_like(latents)
        context[:, :prefix.shape[1]] = prefix.to(
            device=latents.device, dtype=latents.dtype)
        if not torch.all(reference_mask[:, :prefix.shape[1]]):
            raise ValueError("training mask does not expose all history frames")
        if torch.any(reference_mask[:, prefix.shape[1]:]):
            raise ValueError("training mask exposes future frames")
        return context

    @torch.no_grad()
    def _encode_initial_history(self, batch):
        count = self.self_forcing["reference_frame_count"]
        rgb = batch["vae_images"][:, :count]
        batch_size, _, view_count = rgb.shape[:3]
        image = self.image_processor.preprocess(
            rgb.flatten(0, 2).to(self.device))
        shift = self.vae.config.shift_factor or 0
        if self.is_temporal_vae:
            image = einops.rearrange(
                image, "(b t v) c h w -> (b v) c t h w",
                t=count, v=view_count)
        latent = dwm.functional.memory_efficient_split_call(
            self.vae, image,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.mode() - shift
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            return einops.rearrange(
                latent, "(b v) c t h w -> b t v c h w", v=view_count)
        return latent.unflatten(0, (batch_size, count, view_count))

    @torch.no_grad()
    def _rollout_prefix(self, batch, target_window, global_step):
        config = self.self_forcing
        reference = self._encode_initial_history(batch)
        shape = (
            reference.shape[0], config["window_length"],
            *reference.shape[2:])
        # Separate RNG keeps the paired control and on-policy arms' CD noise
        # and timestep draws identical at the same global step.
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(self.config["generator_seed"]) + 1000003 + global_step)
        sampler = make_4step_inference(
            self, compute_denoising_sigmas(self),
            sampler_mode=config["sampler_mode"], random_generator=rng)
        exceptions = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [])
        for window in range(target_window):
            start = window * config["window_stride"]
            window_batch = slice_sequence_batch(
                batch, start, config["window_length"], exceptions)
            # Prefix generation requires layout/time/text conditions, never
            # target RGB. Excluding the key makes accidental visual GT leakage
            # fail loudly if a future condition path tries to read it.
            window_batch.pop("vae_images")
            result = sampler(
                shape, window_batch, "latent", reference,
                reference_frame_count=config["reference_frame_count"])
            reference = result["latents"][:, -config["reference_frame_count"]:].detach()
        return reference

    def train_step(self, batch: dict, global_step: int):
        config = self.self_forcing
        expected = config["window_length"] + (
            config["rollout_windows"] - 1) * config["window_stride"]
        if batch["vae_images"].shape[1] != expected:
            raise ValueError(
                "self-forcing dataset must emit {} frames, got {}".format(
                    expected, batch["vae_images"].shape[1]))
        if self.training_config.get("reference_frame_count") != \
                config["reference_frame_count"]:
            raise ValueError("training and rollout history counts differ")

        # Both paired arms draw the same target window without consuming the
        # training generator's noise/timestep stream.
        curriculum = config["curriculum_steps"]
        upper = 2 if global_step < curriculum else config["rollout_windows"]
        target_window = random.Random(
            int(self.config["generator_seed"]) + 7919 * global_step
        ).randrange(1, upper)
        exceptions = self.inference_config.get(
            "autoregression_data_exception_for_take_sequence", [])
        target_batch = slice_sequence_batch(
            batch, target_window * config["window_stride"],
            config["window_length"], exceptions)

        try:
            prefix_t0 = time.monotonic()
            if config["enabled"]:
                self._generated_reference_context = self._rollout_prefix(
                    batch, target_window, global_step)
            prefix_seconds = time.monotonic() - prefix_t0
            super().train_step(target_batch, global_step)
            self.step_duration += prefix_seconds
            self.loss_report_list[-1].update({
                "self_forcing_enabled": float(config["enabled"]),
                "self_forcing_target_window": float(target_window),
                "self_forcing_prefix_windows": float(
                    target_window if config["enabled"] else 0),
                "self_forcing_prefix_seconds": prefix_seconds,
            })
            if global_step < 3 and self.should_save:
                print(
                    "[self-forcing] step={} target_window={} history={} "
                    "prefix_detached={}".format(
                        global_step, target_window,
                        "student" if config["enabled"] else "GT",
                        bool(self._generated_reference_context is None or
                             not self._generated_reference_context.requires_grad)),
                    flush=True)
        finally:
            self._generated_reference_context = None
