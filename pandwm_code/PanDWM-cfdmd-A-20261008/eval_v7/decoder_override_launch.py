"""Run a PanDWM evaluation script with a fine-tuned SD3.5 VAE decoder, without editing the script (2026-10-12):
  PANDWM_DECODER_CKPT=<decoder_6000.pt> python decoder_override_launch.py compute_eval149_metrics.py <its args...>
AutoencoderKL.from_pretrained is wrapped: after the normal load, the decoder weights are replaced by the checkpoint
(strict).  The encoder, and so the latent space, stay the original ones.
"""
import os
import runpy
import sys

import torch
from diffusers import AutoencoderKL

CKPT = os.environ["PANDWM_DECODER_CKPT"]
_original = AutoencoderKL.from_pretrained


def _from_pretrained(*args, **kwargs):
    vae = _original(*args, **kwargs)
    vae.decoder.load_state_dict(torch.load(CKPT, map_location="cpu"))
    print(f"DECODER_OVERRIDE {CKPT}", flush=True)
    return vae


AutoencoderKL.from_pretrained = _from_pretrained

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
