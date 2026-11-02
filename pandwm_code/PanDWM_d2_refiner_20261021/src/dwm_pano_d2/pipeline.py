"""D2 refiner pipeline (2026-10-21, PanDWM-d2-refiner-20261021/D2_PLAN_20261021.md Task 2).

Training: every step takes ONE 19-frame window [s, s + 19) of a 35-frame train clip (s = d2_window_start(step,
seed), 0..16, the same on every rank) together with the matching frames of the clip's degraded GS render condition
(batch["d2_render"], (1, 35, 2, 17, h, w): 16 render-latent channels + coverage, from D2RenderDataset), and runs
the native Stage-1 flow-matching train_step of PanoCrossviewTemporalSD on it.  The render condition reaches the
model through a stand-in for model_wrapper that adds render_condition= to the single forward call of that
train_step.  With probability render_dropout (hash of seed, step, rank) the render condition is all zeros, which
is also the "no render" input of render guidance at inference.

Render-init samples (report section 43.3, from the continuation after the pilot): with probability
render_init_prob (hash of seed, step, rank; never on a render-dropped step) the noisy input of every non-reference
frame whose sigma >= render_init_min_sigma is built from the RENDER instead of the ground truth,
x = sigma * eps + (1 - sigma) * render, while the target stays the ground-truth latent.  This is the input the
sampler sees when it starts from the noised render (refine_window init_sigma), so the model learns to correct the
render there instead of copying its blur.

Inference: refine_window() is inference_pipeline's full-sequence denoising path (same scheduler, causal mask,
reference-frame injection at timestep 0) plus the render condition; the 3D-box and HD-map conditions are dropped
by default because they are drawn for the original trajectory, not the shifted one.
"""

import hashlib
import json
from pathlib import Path

import torch

from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD, PanoCrossviewTemporalSD
from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import slice_sequence_batch

WINDOW = 19
CLIP = 35
LATENT_CHANNELS = 16
RENDER_CHANNELS = LATENT_CHANNELS + 1
GUIDANCE_MODES = ("none", "all", "text")


def _unit_hash(*parts):
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2 ** 64


def d2_window_start(step, seed, max_start=CLIP - WINDOW):
    """Window start for a train step, arm- and rank-independent; consumes no model RNG."""
    digest = hashlib.sha256(f"{seed}:{step}:d2_window".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (int(max_start) + 1)


def d2_render_dropped(step, seed, rank, p):
    """Bernoulli(p) per (step, rank) for the render-condition dropout; consumes no model RNG."""
    if p <= 0:
        return False
    return _unit_hash(seed, step, rank, "d2_render_drop") < p


def take_window(batch, start, exceptions):
    """Pop batch["d2_render"] and cut the batch and the render condition to frames [start, start + WINDOW)."""
    render = batch.pop("d2_render")
    window = slice_sequence_batch(batch, start, WINDOW, exceptions)
    return window, render[:, start:start + WINDOW]


def d2_render_init(step, seed, rank, p):
    """Bernoulli(p) per (step, rank) for the render-init samples; consumes no model RNG."""
    if p <= 0:
        return False
    return _unit_hash(seed, step, rank, "d2_render_init") < p


def render_init_input(noisy_input, latents, timesteps, render_latents, min_sigma, num_train_timesteps=1000):
    """sigma * eps + (1 - sigma) * x0  ->  sigma * eps + (1 - sigma) * render on the (b, t, v) entries whose
    sigma = timestep / num_train_timesteps >= min_sigma.  Returns (input, used (b, t, v, 1, 1, 1) bool)."""
    sigma = (timesteps.float() / float(num_train_timesteps)).view(*timesteps.shape[:3], 1, 1, 1)
    sigma = sigma.to(device=noisy_input.device, dtype=noisy_input.dtype)
    used = sigma >= float(min_sigma)
    swapped = noisy_input + (1.0 - sigma) * (render_latents.to(noisy_input) - latents.to(noisy_input))
    return torch.where(used, swapped, noisy_input), used


class swap_input_builder:
    """Within the block, CrossviewTemporalSD.try_make_input_for_prediction gets its noisy input passed through
    transform(noisy_input, latents, timesteps) first; the original static method is restored on exit."""

    def __init__(self, transform):
        self.transform = transform
        self.calls = 0

    def __enter__(self):
        self.original = CrossviewTemporalSD.__dict__["try_make_input_for_prediction"]
        base = self.original.__func__ if isinstance(self.original, staticmethod) else self.original
        outer = self

        def patched(noisy_input, latents, timesteps, *args, **kwargs):
            outer.calls += 1
            return base(outer.transform(noisy_input, latents, timesteps), latents, timesteps, *args, **kwargs)

        setattr(CrossviewTemporalSD, "try_make_input_for_prediction", staticmethod(patched))
        return self

    def __exit__(self, *exc):
        setattr(CrossviewTemporalSD, "try_make_input_for_prediction", self.original)
        return False


class RenderInjector:
    """Stands in for model_wrapper during one train_step: every call gets render_condition=render."""

    def __init__(self, wrapper, render):
        object.__setattr__(self, "_wrapper", wrapper)
        object.__setattr__(self, "_render", render)
        object.__setattr__(self, "calls", 0)

    def __call__(self, *args, **kwargs):
        if "render_condition" in kwargs:
            raise RuntimeError("render_condition is already set by the caller")
        object.__setattr__(self, "calls", self.calls + 1)
        return self._wrapper(*args, render_condition=self._render, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapper"), name)


def causal_temporal_mask(model, device, dtype, frames, latent_height, latent_width):
    """The temporal attention mask of inference_pipeline / train_step for `frames` frames."""
    attn_type = getattr(model, "temporal_attention_type", "pointwise")
    if attn_type == "pointwise":
        mask = torch.zeros(frames, frames, device=device, dtype=dtype)
        mask.masked_fill_(torch.triu(torch.ones(frames, frames, dtype=torch.bool, device=device), diagonal=1),
                          float("-inf"))
        return mask
    if attn_type in ("rowwise", "full"):
        from dwm_pano.models.flex_block_causal import make_blockwise_causal_block_mask
        hp = latent_height // model.config.patch_size
        wp = latent_width // model.config.patch_size
        return make_blockwise_causal_block_mask(device, num_frames=frames,
                                                frame_seqlen=wp if attn_type == "rowwise" else hp * wp)
    raise ValueError(f"unsupported temporal_attention_type={attn_type!r}")


class PanoD2RefinerSD(PanoCrossviewTemporalSD):
    def __init__(self, *args, d2_config=None, **kwargs):
        dc = dict(d2_config or {})
        self.d2_render_dropout = float(dc.get("render_dropout", 0.1))
        if not 0 <= self.d2_render_dropout < 1:
            raise ValueError("render_dropout must be in [0, 1)")
        self.d2_max_start = int(dc.get("max_start", CLIP - WINDOW))
        if not 0 <= self.d2_max_start <= CLIP - WINDOW:
            raise ValueError(f"max_start must be 0..{CLIP - WINDOW}")
        manifest_path = dc.get("forbidden_scene_manifest")
        if not manifest_path:
            raise ValueError("D2 training needs the frozen dev/heldout scene manifest")
        manifest = json.loads(Path(manifest_path).read_text())
        self.d2_forbidden_scenes = {item["scene_token"] for key in ("dev", "heldout") for item in manifest[key]}
        self.d2_render_init_prob = float(dc.get("render_init_prob", 0.0))
        self.d2_render_init_min_sigma = float(dc.get("render_init_min_sigma", 0.5))
        if not 0 <= self.d2_render_init_prob <= 1:
            raise ValueError("render_init_prob must be in [0, 1]")
        self.d2_counts = {"steps": 0, "render_dropped": 0, "render_init": 0}
        super().__init__(*args, **kwargs)
        if not hasattr(self.model, "render_embed"):
            raise TypeError("PanoD2RefinerSD needs dwm_pano_d2.model.D2RenderDiT")

    # ------------------------------------------------------------------ training
    def train_step(self, batch, global_step):
        if tuple(batch["vae_images"].shape[1:3]) != (CLIP, 2):
            raise ValueError(f"expected {CLIP}-frame two-view clips, got {tuple(batch['vae_images'].shape)}")
        if batch["split"][0] != "train":
            raise ValueError("D2 trains on the train split only")
        if batch["scene_token"][0] in self.d2_forbidden_scenes:
            raise ValueError("training clip overlaps dev/heldout")
        render = batch["d2_render"]
        if tuple(render.shape[:3]) != tuple(batch["vae_images"].shape[:3]) or render.shape[3] != RENDER_CHANNELS:
            raise ValueError(f"render condition {tuple(render.shape)} does not fit the clip")
        seed = self.config["generator_seed"]
        start = d2_window_start(global_step, seed, self.d2_max_start)
        exceptions = self.inference_config.get("autoregression_data_exception_for_take_sequence", [])
        window, render = take_window(batch, start, exceptions)
        render = render.to(self.device)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dropped = d2_render_dropped(global_step, seed, rank, self.d2_render_dropout)
        if dropped:
            render = torch.zeros_like(render)
        use_init = (not dropped) and d2_render_init(global_step, seed, rank, self.d2_render_init_prob)
        self.d2_counts["steps"] += 1
        self.d2_counts["render_dropped"] += int(dropped)
        self.d2_counts["render_init"] += int(use_init)
        init_used = {"fraction": 0.0}
        render_latents = render[:, :, :, :LATENT_CHANNELS]
        num_train_timesteps = int(self.train_scheduler.config.num_train_timesteps)

        def to_render_init(noisy_input, latents, timesteps):
            out, used = render_init_input(noisy_input, latents, timesteps, render_latents,
                                          self.d2_render_init_min_sigma, num_train_timesteps)
            init_used["fraction"] = float(used.float().mean())
            return out

        wrapper = self.model_wrapper
        injector = RenderInjector(wrapper, render)
        builder = swap_input_builder(to_render_init) if use_init else None
        self.model_wrapper = injector
        try:
            if builder is not None:
                with builder:
                    result = super().train_step(window, global_step)
            else:
                result = super().train_step(window, global_step)
        finally:
            self.model_wrapper = wrapper
        if injector.calls != 1:
            raise RuntimeError(f"train_step called the model {injector.calls} times, expected 1")
        if builder is not None and builder.calls != 1:
            raise RuntimeError(f"render-init input builder ran {builder.calls} times, expected 1")
        if global_step % 10 == 0 and getattr(self, "should_save", True):
            coverage = float(render[:, :, :, -1].float().mean())
            print(f"D2 step={global_step} start={start} render_dropped={int(dropped)} render_init={int(use_init)} "
                  f"init_fraction={init_used['fraction']:.2f} coverage={coverage:.3f} counts={self.d2_counts}",
                  flush=True)
        return result

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def refine_window(self, batch, render, reference=None, reference_count=0, steps=20, guidance_scale=1.0,
                      guidance_mode="none", drop_layout=True, init_sigma=0.0):
        """One window of the refiner.

        batch: collated window conditions (slice_sequence_batch of the clip, WINDOW frames);
        render: (1, WINDOW, 2, 17, h, w) render condition; reference: (1, >= reference_count, 2, 16, h, w) clean
        latents of the window's first reference_count frames (or None).  Returns (1, WINDOW, 2, 16, h, w) float
        latents with the reference frames copied in.  init_sigma > 0 (section 43.2): start from the render latents
        noised to the first scheduler sigma <= init_sigma, x = (1 - sigma) * render + sigma * noise, and run only
        the remaining steps (SDEdit start on the refiner); 0 = pure noise, all steps.
        """
        if guidance_mode not in GUIDANCE_MODES:
            raise ValueError(f"guidance_mode must be one of {GUIDANCE_MODES}")
        if render.ndim != 6 or render.shape[3] != RENDER_CHANNELS:
            raise ValueError(f"render condition must be (b, t, v, {RENDER_CHANNELS}, h, w)")
        if reference_count and (reference is None or reference.shape[1] < reference_count):
            raise ValueError("reference_count needs that many reference frames")
        self.model_wrapper.eval()
        b, t, v, _, h, w = render.shape
        latent_shape = (b, t, v, LATENT_CHANNELS, h, w)
        do_cfg = guidance_mode != "none" and float(guidance_scale) != 1.0
        layout_mask = torch.zeros((b,), dtype=torch.bool, device=self.device) if drop_layout else None
        conditions = CrossviewTemporalSD.get_conditions(
            self.model, self.text_encoders, self.tokenizers, self.common_config, latent_shape, batch,
            self.device, self.model_dtype, _3dbox_condition_mask=layout_mask, hdmap_condition_mask=layout_mask,
            do_classifier_free_guidance=do_cfg, latents_shape=latent_shape)
        if self.inference_config.get("temporal_attention_mask") == "causal":
            conditions["temporal_attention_mask"] = causal_temporal_mask(self.model, self.device, self.model_dtype,
                                                                         t, h, w)
        render = render.to(self.device)
        if do_cfg:
            uncond_render = torch.zeros_like(render) if guidance_mode == "all" else render
            render_input = torch.cat([uncond_render, render])
        else:
            render_input = render
        if reference_count:
            reference = reference[:, :reference_count].to(self.device, dtype=torch.float32)
        self.test_scheduler.set_timesteps(int(steps), self.device)
        latents = torch.randn(latent_shape, generator=self.generator).to(self.device) * \
            getattr(self.test_scheduler, "init_noise_sigma", 1)
        timesteps_all = self.test_scheduler.timesteps
        if init_sigma > 0:
            sigmas = self.test_scheduler.sigmas[:len(timesteps_all)].to(self.device)
            start = int((sigmas <= float(init_sigma)).nonzero()[0, 0])
            sigma0 = sigmas[start].to(latents.dtype)
            latents = (1.0 - sigma0) * render[:, :, :, :LATENT_CHANNELS].float() + sigma0 * latents
            timesteps_all = timesteps_all[start:]
            if hasattr(self.test_scheduler, "set_begin_index"):
                self.test_scheduler.set_begin_index(start)
        for step_t in timesteps_all:
            timesteps = step_t.reshape(1, 1, 1).repeat(b, t, v)
            model_input = latents
            if reference_count:
                model_input = torch.cat([reference, latents[:, reference_count:]], 1)
                timesteps = torch.cat([torch.zeros_like(timesteps[:, :reference_count]),
                                       timesteps[:, reference_count:]], 1)
            model_input = model_input.to(dtype=self.model_dtype)
            if hasattr(self.test_scheduler, "scale_model_input"):
                model_input = self.test_scheduler.scale_model_input(model_input, step_t).to(dtype=self.model_dtype)
            if do_cfg:
                model_input = torch.cat([model_input, model_input])
                timesteps = torch.cat([timesteps, timesteps])
            with self.get_autocast_context():
                output, _, _ = self.model_wrapper(model_input, timesteps, render_condition=render_input,
                                                  **conditions)
            prediction = output[0]
            if do_cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + float(guidance_scale) * (cond - uncond)
            latents = self.test_scheduler.step(prediction, step_t, latents).prev_sample
        if reference_count:
            latents = torch.cat([reference, latents[:, reference_count:]], 1)
        return latents.float()
