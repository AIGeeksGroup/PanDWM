"""Temporally correlated sampler noise for the 4-step evaluation, without editing evaluate_4step.py (2026-10-12).

Inference-only test for "static-scene boiling": the stochastic 4-step sampler re-noises x0_pred with an independent
draw for every frame (at the last step the input is half fresh noise), so static content can be re-synthesised
differently in every frame.  Here every noise draw of the sampler is
    n = sqrt(rho) * shared + sqrt(1 - rho) * independent,     shared: one draw per sample/view, same for all frames,
which keeps each element N(0, 1) and only adds correlation along time.
  PANDWM_RENOISE_CORR=rho   applies to the three re-noise draws (default 0 = the historical sampler);
  PANDWM_INIT_CORR=rho      applies to the initial noise draw (default 0).
The patch replaces the module-global `torch` seen by dwm_pano.evaluate_4step with a shim whose randn is the
correlated draw (everything else is forwarded to torch), and wraps make_4step_inference to count draws per call.
Then it runs lowmem_launch.py exactly as before:
  python corr_noise_launch.py lowmem_launch.py run_eval149_videos.py <its arguments...>
"""
import atexit
import math
import os
import runpy
import sys
import types

import torch

import dwm_pano.evaluate_4step as e4

RHO_RENOISE = float(os.environ.get("PANDWM_RENOISE_CORR", "0") or 0)
RHO_INIT = float(os.environ.get("PANDWM_INIT_CORR", "0") or 0)
assert 0.0 <= RHO_RENOISE < 1.0 and 0.0 <= RHO_INIT < 1.0
_state = {"call": 0, "draws": 0, "correlated": 0}


def _corr_randn(*size, generator=None, **kwargs):
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
    _state["call"] += 1
    _state["draws"] += 1
    rho = RHO_INIT if _state["call"] == 1 else RHO_RENOISE
    independent = torch.randn(shape, generator=generator, **kwargs)
    if rho <= 0 or len(shape) != 6:                       # (B, T, V, C, H, W) latents only
        return independent
    shared = torch.randn((shape[0], 1) + shape[2:], generator=generator, **kwargs)
    _state["correlated"] += 1
    return math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * independent


class _TorchShim(types.ModuleType):
    def __getattr__(self, name):
        return getattr(torch, name)


_shim = _TorchShim("torch_corr_shim")
_shim.randn = _corr_randn
e4.torch = _shim
_make = e4.make_4step_inference


def _make_corr(*args, **kwargs):
    inner = _make(*args, **kwargs)

    def inference_pipeline(*a, **k):
        _state["call"] = 0
        return inner(*a, **k)

    return inference_pipeline


e4.make_4step_inference = _make_corr
print(f"CORR_NOISE active: renoise rho={RHO_RENOISE} init rho={RHO_INIT}", flush=True)
atexit.register(lambda: print(f"CORR_NOISE draws={_state['draws']} correlated={_state['correlated']}", flush=True))

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
