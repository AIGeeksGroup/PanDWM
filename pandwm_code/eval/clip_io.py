"""Clip helpers shared by the closed-loop clip renderers (run_clips_videos.py, run_long_rollout_multi.py).

_window_batch cuts one 19-frame window out of a batch, _encode_rgb encodes recorded frames to latents and
_psnr_by_frame scores generated against recorded panoramas.  Importing this module also replaces the
consistency-distillation pipeline's auxiliary-model builder (_single_gpu_auxiliary), so that evaluation does not
load a second copy of the teacher.
"""

import torch

import dwm.functional
from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd


_original_auxiliary_builder = (
    cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model)


def _single_gpu_auxiliary(self, checkpoint_path, model_load_state_args,
                          requires_grad, role):
    # Evaluation never uses the frozen teacher. Avoid loading a second 3B model.
    if role == "teacher" and hasattr(self, "model_wrapper"):
        print("[probe] alias unused teacher to main model", flush=True)
        return self.model, self.model_wrapper
    return _original_auxiliary_builder(
        self, checkpoint_path, model_load_state_args, requires_grad, role)


cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = \
    _single_gpu_auxiliary


def _window_batch(batch, start, length, exceptions):
    result = {
        key: (value if key in exceptions else
              dwm.functional.take_sequence_clip(value, start, start + length))
        for key, value in batch.items()
    }
    # Only the initial three real images are encoded. No target RGB may enter
    # the sampler's conditioning path, including the oracle diagnostic arm.
    result.pop("vae_images", None)
    assert result["pts"].shape[1] == length
    return result


@torch.no_grad()
def _encode_rgb(pipeline, rgb):
    batch_size, frame_count, view_count = rgb.shape[:3]
    image = pipeline.image_processor.preprocess(
        rgb.flatten(0, 2).to(pipeline.device))
    shift = pipeline.vae.config.shift_factor or 0
    if pipeline.is_temporal_vae:
        raise RuntimeError("this probe expects the configured 2D SD3.5 VAE")
    latent = dwm.functional.memory_efficient_split_call(
        pipeline.vae, image,
        lambda block, tensor: (
            block.encode(tensor).latent_dist.mode() - shift
        ) * block.config.scaling_factor,
        pipeline.common_config.get("memory_efficient_batch", -1))
    return latent.unflatten(0, (batch_size, frame_count, view_count))


def _psnr_by_frame(fake, real):
    # Both are B,T,V,C,H,W panoramas on the same target timestamps.
    mse = (fake.float() - real.float()).square().mean(dim=(0, 2, 3, 4, 5))
    return (-10 * torch.log10(mse.clamp_min(1e-12))).cpu().tolist()
