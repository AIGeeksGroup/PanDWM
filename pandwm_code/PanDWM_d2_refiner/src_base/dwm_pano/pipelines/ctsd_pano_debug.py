"""Numeric-debug fork of `PanoCrossviewTemporalSD`.

At global_step == 0 on rank 0, runs a one-shot audit through every node of the
forward chain with stats (shape/dtype/finite/mean/std/min/max) printed for each
intermediate tensor. Specifically validates the causal AR retrofit:

  1) Loads + VAE-encodes the batch, prints latent stats
  2) Builds SD3 conditioning via `CrossviewTemporalSD.get_conditions`, prints
     encoder_hidden_states / pooled_projections / condition_image_tensor /
     camera_intrinsics_norm / camera2referego stats
  3) Runs reference-frame injection (`try_make_input_for_prediction`), prints
     noisy_latents + timesteps + reference_frame_indicator stats
  4) Constructs the causal `BlockMask` via `make_blockwise_causal_block_mask`,
     asserts it's a `BlockMask`, prints a small (T_vis, F_vis) dense reference
     mask + verifies no upper-triangular block-attention violations
  5) Runs TWO forwards — with mask vs without — and prints `max_abs_diff` +
     `mean_rel_diff`; nonzero diff proves the mask is wired through

Then defers to parent `train_step` so step 1+ runs normally.

Enable by setting `pipeline._class_name` to
`dwm_pano.pipelines.ctsd_pano_debug.PanoCrossviewTemporalSDDebug` in a config.
"""
from __future__ import annotations

import einops
import torch

import dwm.functional
from dwm_pano.pipelines.ctsd_pano import (
    CrossviewTemporalSD, PanoCrossviewTemporalSD,
)


def _is_rank0():
    return (not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0)


def _stats(name: str, t):
    """Print shape/dtype/finite/mean/std/min/max for tensor t (rank 0 only)."""
    if not _is_rank0():
        return
    if t is None:
        _print0(f"[dbg] {name:34s} = None", flush=True)
        return
    if not torch.is_tensor(t):
        _print0(f"[dbg] {name:34s} = (not tensor: {type(t).__name__})", flush=True)
        return
    finite = bool(torch.isfinite(t).all().item())
    f = t.float()
    _print0(
        f"[dbg] {name:34s} shape={tuple(t.shape)} dtype={str(t.dtype):>14s} "
        f"finite={str(finite):>5s} "
        f"mean={f.mean().item():+.3e} "
        f"std={f.std().item():+.3e} "
        f"min={f.min().item():+.3e} "
        f"max={f.max().item():+.3e}",
        flush=True)


def _print0(*args, **kwargs):
    """print() that fires only on rank 0."""
    if _is_rank0():
        print(*args, **kwargs)


class PanoCrossviewTemporalSDDebug(PanoCrossviewTemporalSD):
    """One-shot numeric audit of the conditioning + causal-mask forward chain.

    Audit runs at `global_step == 0` on rank 0. Then defers to parent
    `train_step` so the rest of training is identical to the production
    `PanoCrossviewTemporalSD`.
    """

    def train_step(self, batch: dict, global_step: int):
        # All ranks must participate in the audit: FSDP-wrapped modules
        # (T5 text_encoder_3, DiT model_wrapper) require synchronized
        # collectives. Print is gated to rank 0 via _stats / _print0.
        if global_step == 0:
            try:
                self._numeric_audit(batch)
            except Exception as e:
                if _is_rank0():
                    import traceback
                    _print0(f"[dbg] AUDIT EXCEPTION: {e!r}", flush=True)
                    traceback.print_exc()

        super().train_step(batch, global_step)

    @torch.no_grad()
    def _numeric_audit(self, batch: dict):
        bar = "=" * 78
        _print0(f"\n[dbg] {bar}", flush=True)
        _print0(f"[dbg] NUMERIC AUDIT — global_step=0", flush=True)
        _print0(f"[dbg] {bar}", flush=True)

        device = self.device
        was_training = self.model_wrapper.training
        self.model_wrapper.eval()

        try:
            self._audit_chain(batch, device)
        finally:
            if was_training:
                self.model_wrapper.train()

        _print0(f"[dbg] {bar}", flush=True)
        _print0(f"[dbg] AUDIT DONE — proceeding to normal train_step", flush=True)
        _print0(f"[dbg] {bar}\n", flush=True)

    def _audit_chain(self, batch: dict, device):
        B, T, V = batch["vae_images"].shape[:3]
        _print0(f"[dbg] batch dims: B={B}  T={T}  V={V}", flush=True)
        _stats("batch.vae_images",        batch["vae_images"])
        _stats("batch.3dbox_images",      batch.get("3dbox_images"))
        _stats("batch.hdmap_images",      batch.get("hdmap_images"))
        _stats("batch.camera_intrinsics", batch.get("camera_intrinsics"))
        _stats("batch.camera_transforms", batch.get("camera_transforms"))
        _stats("batch.ego_transforms",    batch.get("ego_transforms"))

        # ----- 1) VAE encode -----
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(device))
        if self.is_temporal_vae:
            image_tensor = einops.rearrange(
                image_tensor, "(b t v) c h w -> (b v) c t h w", t=T, v=V)
        shift_factor = (self.vae.config.shift_factor
                        if self.vae.config.shift_factor is not None else 0)
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        if self.is_temporal_vae:
            latents = einops.rearrange(
                latents, "(b v) c t h w -> b t v c h w", v=V)
        else:
            latents = einops.rearrange(
                latents, "(b t v) c h w -> b t v c h w", t=T, v=V)
        _stats("vae_latents (B,T,V,C,H,W)", latents)
        H_lat, W_lat = latents.shape[-2], latents.shape[-1]
        _print0(f"[dbg] latent grid: H={H_lat} W={W_lat}", flush=True)

        # ----- 2) noise + flow-match sigma + timestep -----
        noise = torch.randn(latents.shape, generator=self.generator).to(device)
        u = CrossviewTemporalSD.sd3_compute_density_for_timestep_sampling(
            weighting_scheme=self.training_config.get(
                "weighting_scheme", "logit_normal"),
            size=latents.shape[:1],
            logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
        ts_idx = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[ts_idx].to(device)
        sigmas = CrossviewTemporalSD.sd3_get_sigmas(
            self.train_scheduler, ts_idx, n_dim=latents.ndim,
            dtype=latents.dtype, device=device)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        _stats("noise",         noise)
        _stats("sigmas",        sigmas)
        _stats("noisy_latents", noisy_latents)
        while len(timesteps.shape) < 3:
            timesteps = timesteps.unsqueeze(-1).repeat_interleave(
                latents.shape[len(timesteps.shape)], -1)
        _stats("timesteps (B,T,V)", timesteps)

        # ----- 3) conditioning -----
        cond = CrossviewTemporalSD.get_conditions(
            self.model, self.text_encoders, self.tokenizers,
            self.common_config, batch["vae_images"].shape, batch,
            device, self.model_dtype, latents_shape=latents.shape)
        _print0(f"[dbg] cond keys: {sorted(cond.keys())}", flush=True)
        _stats("cond.encoder_hidden_states",  cond.get("encoder_hidden_states"))
        _stats("cond.pooled_projections",     cond.get("pooled_projections"))
        _stats("cond.condition_image_tensor", cond.get("condition_image_tensor"))
        _stats("cond.camera_intrinsics_norm", cond.get("camera_intrinsics_norm"))
        _stats("cond.camera2referego",        cond.get("camera2referego"))

        # ----- 4) reference-frame injection -----
        reference_latent_count = self.get_reference_latent_count()
        noisy_in, ts_in, additional, ref_ind = \
            CrossviewTemporalSD.try_make_input_for_prediction(
                noisy_latents, latents, timesteps,
                self.training_config, self.common_config,
                generator=self.generator,
                reference_latent_count=reference_latent_count)
        if additional is not None:
            cond.update(additional)
        _stats("post_ref.noisy_in",                  noisy_in)
        _stats("post_ref.timesteps",                 ts_in)
        ref_fraction = ref_ind.float().mean().item()
        _print0(f"[dbg] reference_frame_indicator: shape={tuple(ref_ind.shape)} "
              f"True_fraction={ref_fraction:+.3f}", flush=True)

        # ----- 5) causal mask construction -----
        from dwm_pano.models.flex_block_causal import \
            make_blockwise_causal_block_mask
        from torch.nn.attention.flex_attention import BlockMask

        attn_type = self.model.temporal_attention_type
        patch = self.model.config.patch_size
        Hp, Wp = H_lat // patch, W_lat // patch
        frame_seqlen = Wp if attn_type == "rowwise" else Hp * Wp
        _print0(f"[dbg] temporal_attention_type={attn_type} "
              f"frame_seqlen={frame_seqlen} (Hp={Hp} Wp={Wp})", flush=True)

        mask = make_blockwise_causal_block_mask(
            device, num_frames=T, frame_seqlen=frame_seqlen)
        _print0(f"[dbg] mask type = {type(mask).__name__}", flush=True)
        assert isinstance(mask, BlockMask), \
            f"expected BlockMask, got {type(mask).__name__}"

        # Build reference dense block-causal mask for visualization +
        # causality check on a (T_vis, F_vis) toy grid.
        T_vis = min(T, 5)
        F_vis = min(frame_seqlen, 4)
        total = T_vis * F_vis
        ends = torch.zeros(total, dtype=torch.long)
        for tmp in range(0, total, F_vis):
            ends[tmp:tmp + F_vis] = tmp + F_vis
        dense = torch.zeros(total, total, dtype=torch.bool)
        for q in range(total):
            for k in range(total):
                dense[q, k] = (k < ends[q].item()) or (q == k)
        _print0(
            f"[dbg] reference block-causal mask "
            f"(T_vis={T_vis}, F_vis={F_vis}, total={total}):",
            flush=True)
        for r in range(total):
            row = "".join("1" if dense[r, c].item() else "."
                          for c in range(total))
            tb = r // F_vis
            marker = "←diag" if r % F_vis == 0 else ""
            _print0(f"[dbg]   t={tb} q={r:2d} | {row} {marker}", flush=True)
        violations = sum(
            1 for q in range(total) for k in range(total)
            if dense[q, k].item() and (k // F_vis) > (q // F_vis) and q != k)
        _print0(f"[dbg] block-causal violations on toy grid: {violations} "
              f"(must be 0)", flush=True)

        # ----- 6) forward with vs without mask -----
        cond_no_mask = dict(cond)
        cond_with_mask = dict(cond)
        cond_with_mask["temporal_attention_mask"] = mask

        with self.get_autocast_context():
            out_no_mask, _, _ = self.model_wrapper(
                noisy_in.to(dtype=self.model_dtype), ts_in, **cond_no_mask)
            out_with_mask, _, _ = self.model_wrapper(
                noisy_in.to(dtype=self.model_dtype), ts_in, **cond_with_mask)
        _stats("forward_no_mask",  out_no_mask[0])
        _stats("forward_with_mask", out_with_mask[0])
        diff = (out_with_mask[0].float() - out_no_mask[0].float())
        max_abs_diff = diff.abs().max().item()
        mean_rel_diff = (
            diff.abs() / (out_no_mask[0].float().abs() + 1e-9)
        ).mean().item()
        _print0(
            f"[dbg] |causal − bidirectional|  max_abs={max_abs_diff:+.4e}  "
            f"mean_rel={mean_rel_diff:+.4e}",
            flush=True)
        if max_abs_diff < 1e-6:
            _print0("[dbg] !!! WARNING: causal mask did NOT change forward output. "
                  "Mask injection is likely broken.", flush=True)
        else:
            _print0("[dbg] PASS: causal mask changes forward output → "
                  "mask is in effect.", flush=True)

        # ----- 7) loss (mock — verify gradient signal would be finite) -----
        sd_pred = out_with_mask[0]
        sd_pred_latent = sd_pred * (-sigmas) + noisy_in
        target = latents
        loss = torch.nn.functional.mse_loss(
            sd_pred_latent.float(), target.float(), reduction="mean")
        _print0(
            f"[dbg] mock_sd_loss = {loss.item():+.4e}  "
            f"finite={bool(torch.isfinite(loss).item())}",
            flush=True)
