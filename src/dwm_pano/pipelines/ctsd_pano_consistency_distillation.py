"""
OpenDWM pano adaptation of Causal Forcing++ consistency distillation.

The algorithm follows the official Causal-Forcing implementation copied under
``dwm_pano/pipelines/causal_forcing_official_reference``.  This module keeps the
OpenDWM CTSD data path, VAE, SD3 text/layout conditions, pano view dimension,
and temporal causal mask behavior from ``ctsd_pano.PanoCrossviewTemporalSD``.
"""

import copy
import os
import re
import time

import dwm.common
import dwm.functional
import einops
import torch
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD, PanoCrossviewTemporalSD


class PanoCrossviewTemporalConsistencyDistillation(PanoCrossviewTemporalSD):
    """Causal consistency distillation for the OpenDWM pano CTSD model.

    This is the OpenDWM equivalent of Causal-Forcing++ Stage 2.  The trainable
    generator is the normal ``PanoCrossviewTemporalSD`` model loaded from a
    causal 50-step checkpoint.  A frozen causal teacher advances one adjacent
    FlowMatch step, and a frozen EMA copy provides the consistency target.
    """

    def __init__(
        self,
        output_path,
        config: dict,
        device,
        common_config: dict,
        training_config: dict,
        inference_config: dict,
        pretrained_model_name_or_path: str,
        model,
        model_dtype=None,
        model_checkpoint_path=None,
        model_load_state_args: dict = {},
        metrics: dict = {},
        resume_from=None,
        consistency_teacher_checkpoint_path=None,
        consistency_ema_checkpoint_path=None,
        consistency_discrete_steps: int = None,
        consistency_guidance_scale: float = None,
        consistency_ema_decay: float = None,
        consistency_ema_start_step: int = None,
        generator_ckpt=None,
        generator_fsdp_wrap_strategy=None,
        real_score_fsdp_wrap_strategy=None,
        fake_score_fsdp_wrap_strategy=None,
        real_name=None,
        text_encoder_fsdp_wrap_strategy=None,
        denoising_step_list=None,
        warp_denoising_step=True,
        ts_schedule=False,
        num_train_timestep=1000,
        timestep_shift=5.0,
        guidance_scale=None,
        denoising_loss_type="flow",
        mixed_precision=None,
        seed=None,
        wandb_host=None,
        wandb_key=None,
        wandb_entity=None,
        wandb_project=None,
        sharding_strategy=None,
        lr=None,
        lr_critic=None,
        beta1=None,
        beta2=None,
        beta1_critic=None,
        beta2_critic=None,
        batch_size=None,
        ema_weight=None,
        ema_start_step=None,
        total_batch_size=None,
        log_iters=None,
        negative_prompt=None,
        dfake_gen_update_ratio=None,
        image_or_video_shape=None,
        distribution_loss=None,
        trainer=None,
        gradient_checkpointing=None,
        num_frame_per_block=None,
        load_raw_video=None,
        model_kwargs=None,
        data_path=None,
        grad_accum=None,
        spatial_sf=None,
        teacher_forcing=None,
        causal=None,
        is_causal=None,
        discrete_cd_N=None,
        **official_unused_kwargs,
    ):
        if model_checkpoint_path is None and generator_ckpt is not None:
            model_checkpoint_path = generator_ckpt
        if consistency_teacher_checkpoint_path is None and generator_ckpt is not None:
            consistency_teacher_checkpoint_path = generator_ckpt
        if consistency_ema_checkpoint_path is None and generator_ckpt is not None:
            consistency_ema_checkpoint_path = generator_ckpt

        super().__init__(
            output_path=output_path,
            config=config,
            device=device,
            common_config=common_config,
            training_config=training_config,
            inference_config=inference_config,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model=model,
            model_dtype=model_dtype,
            model_checkpoint_path=model_checkpoint_path,
            model_load_state_args=model_load_state_args,
            metrics=metrics,
            resume_from=resume_from)

        self.official_stage2_parameters = {
            "generator_ckpt": generator_ckpt,
            "generator_fsdp_wrap_strategy": generator_fsdp_wrap_strategy,
            "real_score_fsdp_wrap_strategy": real_score_fsdp_wrap_strategy,
            "fake_score_fsdp_wrap_strategy": fake_score_fsdp_wrap_strategy,
            "real_name": real_name,
            "text_encoder_fsdp_wrap_strategy": text_encoder_fsdp_wrap_strategy,
            "denoising_step_list": denoising_step_list,
            "warp_denoising_step": warp_denoising_step,
            "ts_schedule": ts_schedule,
            "num_train_timestep": num_train_timestep,
            "timestep_shift": timestep_shift,
            "guidance_scale": guidance_scale,
            "denoising_loss_type": denoising_loss_type,
            "mixed_precision": mixed_precision,
            "seed": seed,
            "sharding_strategy": sharding_strategy,
            "lr": lr,
            "lr_critic": lr_critic,
            "beta1": beta1,
            "beta2": beta2,
            "beta1_critic": beta1_critic,
            "beta2_critic": beta2_critic,
            "batch_size": batch_size,
            "ema_weight": ema_weight,
            "ema_start_step": ema_start_step,
            "total_batch_size": total_batch_size,
            "log_iters": log_iters,
            "negative_prompt": negative_prompt,
            "dfake_gen_update_ratio": dfake_gen_update_ratio,
            "image_or_video_shape": image_or_video_shape,
            "distribution_loss": distribution_loss,
            "trainer": trainer,
            "gradient_checkpointing": gradient_checkpointing,
            "num_frame_per_block": num_frame_per_block,
            "load_raw_video": load_raw_video,
            "model_kwargs": model_kwargs,
            "data_path": data_path,
            "grad_accum": grad_accum,
            "spatial_sf": spatial_sf,
            "teacher_forcing": teacher_forcing,
            "causal": causal,
            "is_causal": is_causal,
            "discrete_cd_N": discrete_cd_N,
        }
        self.raw_denoising_step_list = denoising_step_list or [1000, 750, 500, 250]
        self.warp_denoising_step = warp_denoising_step
        self.num_train_timestep = num_train_timestep
        self.timestep_shift = timestep_shift
        self.denoising_loss_type = denoising_loss_type
        self.denoising_step_list = self._make_official_denoising_step_list(
            self.raw_denoising_step_list,
            warp_denoising_step=self.warp_denoising_step,
            num_train_timestep=self.num_train_timestep)
        self.consistency_discrete_steps = (
            discrete_cd_N if discrete_cd_N is not None else
            (consistency_discrete_steps if consistency_discrete_steps is not None else 48))
        self.consistency_guidance_scale = (
            guidance_scale if guidance_scale is not None else
            (consistency_guidance_scale if consistency_guidance_scale is not None else
             inference_config.get("guidance_scale", 1.0)))
        self.consistency_ema_decay = (
            ema_weight if ema_weight is not None else
            (consistency_ema_decay if consistency_ema_decay is not None else 0.99))
        self.consistency_ema_start_step = (
            ema_start_step if ema_start_step is not None else
            (consistency_ema_start_step if consistency_ema_start_step is not None else 0))

        teacher_checkpoint = consistency_teacher_checkpoint_path or model_checkpoint_path
        ema_checkpoint = consistency_ema_checkpoint_path or teacher_checkpoint

        self.consistency_scheduler = copy.deepcopy(self.test_scheduler)
        self._set_consistency_timesteps(self.consistency_discrete_steps)

        self.teacher_model, self.teacher_model_wrapper = self._build_auxiliary_model(
            checkpoint_path=teacher_checkpoint,
            model_load_state_args=model_load_state_args,
            requires_grad=False,
            role="teacher")

        # On resume, load EMA from the matching `<resume_from>_ema.pth`. The
        # path must be passed in BEFORE FSDP wrap (inside `_build_auxiliary_model`)
        # because post-wrap the module's parameters become `_flat_param` and
        # raw-key state dicts no longer load with strict=True.
        ema_load_path = (
            os.path.join(output_path, "checkpoints", f"{resume_from}_ema.pth")
            if resume_from is not None else ema_checkpoint
        )
        self.ema_model, self.ema_model_wrapper = self._build_auxiliary_model(
            checkpoint_path=ema_load_path,
            model_load_state_args=model_load_state_args,
            requires_grad=False,
            role="ema")

    def _set_consistency_timesteps(self, num_steps: int):
        self.consistency_scheduler.set_timesteps(num_steps)
        self.consistency_scheduler.timesteps = \
            self.consistency_scheduler.timesteps.to(self.device)
        self.consistency_scheduler.sigmas = \
            self.consistency_scheduler.sigmas.to(self.device)

    def _make_official_denoising_step_list(
        self,
        denoising_step_list,
        warp_denoising_step: bool,
        num_train_timestep: int,
    ):
        raw_steps = torch.tensor(
            denoising_step_list, dtype=torch.long, device=self.device)
        if not warp_denoising_step:
            return raw_steps

        scheduler = copy.deepcopy(self.test_scheduler)
        try:
            scheduler.set_timesteps(num_train_timestep, self.device)
        except TypeError:
            scheduler.set_timesteps(num_train_timestep)
        timesteps = torch.cat((
            scheduler.timesteps.detach().cpu(),
            torch.tensor([0], dtype=torch.float32),
        )).to(self.device)
        indices = num_train_timestep - raw_steps
        if torch.any(indices < 0) or torch.any(indices >= timesteps.numel()):
            raise ValueError(
                f"denoising_step_list={denoising_step_list} is incompatible "
                f"with num_train_timestep={num_train_timestep}")
        return timesteps[indices]

    def _build_auxiliary_model(
        self,
        checkpoint_path: str,
        model_load_state_args: dict,
        requires_grad: bool,
        role: str,
    ):
        model = dwm.common.create_instance_from_config(
            self.config["pipeline"]["model"])
        model = model.to(dtype=self.model_dtype)
        if requires_grad:
            model.enable_gradient_checkpointing()
        if checkpoint_path is not None:
            state_dict = CrossviewTemporalSD.load_state(checkpoint_path)
            model.load_state_dict(state_dict, **model_load_state_args)
        model.requires_grad_(requires_grad)
        wrapper = self._wrap_auxiliary_model(model)
        wrapper.eval()
        return model, wrapper

    def _wrap_auxiliary_model(self, model):
        distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if (
            not torch.distributed.is_initialized()
            or distribution_framework == "ddp"
        ):
            model.to(self.device)
            return model

        ignored_modules = None
        if "fsdp_ignored_module_pattern" in self.common_config:
            pattern = re.compile(self.common_config["fsdp_ignored_module_pattern"])
            ignored_modules = [
                module for name, module in model.named_modules()
                if pattern.match(name) is not None
            ]

        return FSDP(
            model,
            device_id=torch.cuda.current_device(),
            ignored_modules=ignored_modules,
            **self.common_config["ddp_wrapper_settings"])

    @staticmethod
    def _duplicate_batch_conditions(conditions: dict, batch_size: int):
        duplicated = {}
        for key, value in conditions.items():
            if torch.is_tensor(value) and value.shape[0] == batch_size:
                duplicated[key] = torch.cat([value, value], dim=0)
            else:
                duplicated[key] = value
        return duplicated

    def _add_temporal_attention_mask(self, model_conditions: dict, latent_shape):
        tam_setting = self.training_config.get("temporal_attention_mask")
        if tam_setting is None:
            return

        sequence_length = latent_shape[1]
        attn_type = self.model.temporal_attention_type
        if attn_type == "pointwise":
            causal = torch.zeros(
                sequence_length, sequence_length,
                device=self.device, dtype=self.model_dtype)
            causal.masked_fill_(
                torch.triu(
                    torch.ones(sequence_length, sequence_length,
                               dtype=torch.bool, device=self.device),
                    diagonal=1),
                float("-inf"))
            model_conditions["temporal_attention_mask"] = causal
        elif attn_type in ("rowwise", "full"):
            from dwm_pano.models.flex_block_causal import \
                make_blockwise_causal_block_mask
            hp = latent_shape[4] // self.model.config.patch_size
            wp = latent_shape[5] // self.model.config.patch_size
            frame_seqlen = wp if attn_type == "rowwise" else hp * wp
            model_conditions["temporal_attention_mask"] = \
                make_blockwise_causal_block_mask(
                    self.device,
                    num_frames=sequence_length,
                    frame_seqlen=frame_seqlen)

    def _model_velocity(self, model_wrapper, noisy_latents, timesteps, conditions):
        model_output, _, _ = model_wrapper(
            noisy_latents.to(dtype=self.model_dtype), timesteps, **conditions)
        return model_output[0]

    @staticmethod
    def _predicted_x0(noisy_latents, velocity, sigma):
        return velocity * (-sigma) + noisy_latents

    @torch.no_grad()
    def _update_ema_model(self):
        decay = self.consistency_ema_decay
        for ema_param, model_param in zip(
            self.ema_model_wrapper.parameters(),
            self.model_wrapper.parameters()
        ):
            ema_param.data.mul_(decay).add_(
                model_param.detach().data, alpha=1.0 - decay)

    def save_checkpoint(self, output_path: str, steps: int, upload: bool = True,
                        keep_last_n: int = 0):
        # Save model + optimizer (no upload yet), then EMA, then upload+prune
        # in one cycle so the EMA file is included alongside the main ckpt.
        # When `upload=False`, skip the HF upload/prune so the caller (e.g.,
        # train.py with save→eval→upload sequencing) can run eval first.
        super().save_checkpoint(output_path, steps, upload=False)

        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            ema_state_dict = torch.distributed.checkpoint.state_dict\
                .get_model_state_dict(self.ema_model_wrapper, options=options)
        elif self.should_save:
            ema_state_dict = self.ema_model.state_dict()

        if self.should_save:
            ema_path = os.path.join(
                output_path, "checkpoints", f"{steps}_ema.pth")
            torch.save(ema_state_dict, ema_path)
            print(f"EMA consistency checkpoint saved to {ema_path}")
            if upload:
                self._upload_and_prune(
                    output_path, steps, keep_last_n=keep_last_n)

    def evaluate_4step_ema_pipeline(
        self, global_step, dataset_length, dataloader, datasampler
    ):
        # Eval with EMA weights + 4-step inference (matches distilled regime).
        # Snapshot pipeline state, swap model_wrapper → ema_model_wrapper +
        # monkey-patch inference_pipeline → 4-step, run base evaluate_pipeline,
        # restore. Reuses helpers from `dwm_pano.evaluate_4step`.
        from dwm_pano.evaluate_4step import (
            compute_denoising_sigmas, make_4step_inference)

        _orig_model_wrapper = self.model_wrapper
        _orig_inference_pipeline = self.inference_pipeline

        sigmas = compute_denoising_sigmas(self)
        self.model_wrapper = self.ema_model_wrapper
        self.inference_pipeline = make_4step_inference(self, sigmas)
        try:
            self.evaluate_pipeline(
                global_step, dataset_length, dataloader, datasampler)
        finally:
            self.model_wrapper = _orig_model_wrapper
            self.inference_pipeline = _orig_inference_pipeline

    def train_step(self, batch: dict, global_step: int):
        self.model_wrapper.train()
        self.teacher_model_wrapper.eval()
        self.ema_model_wrapper.eval()

        t0 = time.time()

        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(self.device))

        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor, "(b t v) c h w -> (b v) c t h w",
                t=sequence_length, v=view_count)

        shift_factor = self.vae.config.shift_factor \
            if self.vae.config.shift_factor is not None else 0
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))

        if self.is_temporal_vae:
            latents = einops.rearrange(
                latents, "(b v) c t h w -> b t v c h w", v=view_count)
        else:
            latents = einops.rearrange(
                latents, "(b t v) c h w -> b t v c h w",
                t=sequence_length, v=view_count)

        noise = torch.randn(
            latents.shape, generator=self.generator).to(self.device)
        step_count = len(self.consistency_scheduler.timesteps)
        timestep_indices = torch.randint(
            0, step_count - 1, latents.shape[:1],
            generator=self.generator).to(self.device)
        next_timestep_indices = timestep_indices + 1

        timesteps = self.consistency_scheduler.timesteps[timestep_indices]
        next_timesteps = self.consistency_scheduler.timesteps[
            next_timestep_indices]
        sigma = CrossviewTemporalSD.sd3_get_sigmas(
            self.consistency_scheduler, timestep_indices,
            n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
        next_sigma = CrossviewTemporalSD.sd3_get_sigmas(
            self.consistency_scheduler, next_timestep_indices,
            n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)

        noisy_latents = sigma * noise + (1.0 - sigma) * latents
        model_timesteps = timesteps.view(batch_size, 1, 1).repeat(
            1, sequence_length, view_count)
        next_model_timesteps = next_timesteps.view(batch_size, 1, 1).repeat(
            1, sequence_length, view_count)

        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("text_prompt_condition_ratio", 1.0))\
            .tolist()
        box_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("3dbox_condition_ratio", 1.0))\
            .to(self.device)
        hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("hdmap_condition_ratio", 1.0))\
            .to(self.device)
        action_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("action_condition_ratio", 1.0))
        if self.common_config.get("explicit_view_modeling", False):
            explicit_view_modeling_mask = (
                torch.rand((batch_size,), generator=self.generator) <
                self.training_config.get("explicit_view_modeling_ratio", 1.0))\
                .to(self.device)
        else:
            explicit_view_modeling_mask = None

        loss_dict = {}
        with self.get_autocast_context():
            conditional_conditions = CrossviewTemporalSD.get_conditions(
                self.model,
                self.text_encoders,
                self.tokenizers,
                self.common_config, batch["vae_images"].shape, batch,
                self.device, self.model_dtype, text_condition_mask,
                box_condition_mask, hdmap_condition_mask,
                action_condition_mask, explicit_view_modeling_mask,
                latents_shape=latents.shape)
            cfg_conditions = CrossviewTemporalSD.get_conditions(
                self.model,
                self.text_encoders,
                self.tokenizers,
                self.common_config, batch["vae_images"].shape, batch,
                self.device, self.model_dtype, text_condition_mask,
                box_condition_mask, hdmap_condition_mask,
                action_condition_mask, explicit_view_modeling_mask,
                do_classifier_free_guidance=True,
                latents_shape=latents.shape)

            reference_latent_count = self.get_reference_latent_count()
            noisy_input, model_timesteps, additional_conditions, \
                reference_frame_indicator = \
                CrossviewTemporalSD.try_make_input_for_prediction(
                    noisy_latents, latents, model_timesteps,
                    self.training_config, self.common_config,
                    generator=self.generator,
                    reference_latent_count=reference_latent_count)

            if additional_conditions is not None:
                conditional_conditions.update(additional_conditions)
                cfg_conditions.update(
                    self._duplicate_batch_conditions(
                        additional_conditions, batch_size))

            if getattr(self.model_wrapper, "mask_module", None) is not None:
                conditional_conditions["noise"] = noise
                cfg_conditions["noise"] = torch.cat([noise, noise], dim=0)

            self._add_temporal_attention_mask(
                conditional_conditions, latents.shape)
            self._add_temporal_attention_mask(
                cfg_conditions, latents.shape)

            with torch.no_grad():
                cfg_noisy_input = torch.cat([noisy_input, noisy_input], dim=0)
                cfg_timesteps = torch.cat([model_timesteps, model_timesteps], dim=0)
                teacher_velocity = self._model_velocity(
                    self.teacher_model_wrapper,
                    cfg_noisy_input,
                    cfg_timesteps,
                    cfg_conditions)
                teacher_uncond, teacher_cond = teacher_velocity.chunk(2)
                guided_teacher_velocity = teacher_uncond + \
                    self.consistency_guidance_scale * (
                        teacher_cond - teacher_uncond)
                next_noisy_latents = noisy_input + \
                    (next_sigma - sigma) * guided_teacher_velocity

                reference_mask = reference_frame_indicator.view(
                    *latents.shape[:3], 1, 1, 1).to(latents.device)
                next_noisy_input = torch.where(
                    reference_mask, latents, next_noisy_latents)
                next_model_timesteps = torch.where(
                    reference_frame_indicator.to(next_model_timesteps.device),
                    torch.zeros_like(next_model_timesteps),
                    next_model_timesteps)

            generator_velocity = self._model_velocity(
                self.model_wrapper,
                noisy_input,
                model_timesteps,
                conditional_conditions)

            with torch.no_grad():
                ema_velocity = self._model_velocity(
                    self.ema_model_wrapper,
                    next_noisy_input,
                    next_model_timesteps,
                    conditional_conditions)

            sigma_for_model = torch.where(
                reference_mask, torch.zeros_like(sigma), sigma)
            next_sigma_for_model = torch.where(
                reference_mask, torch.zeros_like(next_sigma), next_sigma)
            generator_x0 = self._predicted_x0(
                noisy_input, generator_velocity, sigma_for_model)
            ema_x0 = self._predicted_x0(
                next_noisy_input, ema_velocity, next_sigma_for_model)

            diff = (generator_x0.float() - ema_x0.float().detach()).square()
            if self.training_config.get("disable_reference_frame_loss", False):
                loss_mask = torch.logical_not(reference_mask).float()
                loss = (diff * loss_mask).sum() / \
                    loss_mask.expand_as(diff).sum().clamp_min(1.0)
            else:
                loss = diff.mean()
            loss_dict["consistency_distillation_loss"] = loss

        loss = torch.stack([i[1] for i in loss_dict.items()]).sum()
        loss_report = {
            "loss": loss.item(),
            "consistency_distillation_loss":
                loss_dict["consistency_distillation_loss"].item(),
            "consistency_timestep": timesteps.float().mean().item(),
            "consistency_next_timestep": next_timesteps.float().mean().item(),
            "denoising_step_count": float(self.denoising_step_list.numel()),
        }
        self.loss_report_list.append(loss_report)

        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        should_optimize = \
            "gradient_accumulation_steps" not in self.training_config or \
            (global_step + 1) % \
            self.training_config["gradient_accumulation_steps"] == 0
        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.unscale_(self.optimizer)

                if (
                    torch.distributed.is_initialized() and
                    self.distribution_framework == "fsdp"
                ):
                    self.model_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config["max_norm_for_grad_clip"])

            if self.training_config.get("enable_grad_scaler", False):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            if global_step + 1 >= self.consistency_ema_start_step:
                self._update_ema_model()
            self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.step_duration += time.time() - t0
