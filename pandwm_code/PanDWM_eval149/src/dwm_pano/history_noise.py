"""Flow-consistent corruption of visible history, independent of model imports."""

import torch


def flow_corrupt_reference_context(latents, reference_mask, noise, sigma):
    """Corrupt visible history using x_sigma=(1-sigma)*x0+sigma*epsilon."""
    if noise.shape != latents.shape:
        raise ValueError("history noise must match latent shape")
    if sigma.shape != (latents.shape[0], 1, 1, 1, 1, 1):
        raise ValueError("history sigma must have shape (B,1,1,1,1,1)")
    if bool(torch.any((sigma < 0) | (sigma > 1))):
        raise ValueError("history sigma must be in [0,1]")
    mixed = (1 - sigma) * latents + sigma * noise.to(
        device=latents.device, dtype=latents.dtype)
    return torch.where(reference_mask, mixed, latents)


def make_history_for_window(reference, initial_stats, mode, sigma, alpha,
                            seed, window):
    """Transform generated history while leaving sampler RNG untouched."""
    if window == 0 or mode in ("clean", "vae_roundtrip"):
        return reference, 0.0
    if mode == "flow_noise":
        if not 0 < sigma <= 1:
            raise ValueError("flow history sigma must be in (0,1]")
        rng = torch.Generator(device="cpu").manual_seed(
            int(seed) * 1000003 + window)
        noise = torch.randn(reference.shape, generator=rng).to(
            device=reference.device, dtype=reference.dtype)
        return (1 - sigma) * reference + sigma * noise, 1000.0 * sigma
    if mode == "moment_align":
        if not 0 < alpha <= 1:
            raise ValueError("moment alignment alpha must be in (0,1]")
        values = reference.float()
        src_mean = values.mean(dim=(1, 2, 4, 5), keepdim=True)
        src_std = values.std(dim=(1, 2, 4, 5), keepdim=True).clamp_min(1e-6)
        dst_mean, dst_std = initial_stats
        ratio = (dst_std / src_std).clamp(0.5, 2.0)
        aligned = (values - src_mean) * ratio + dst_mean
        return (values * (1 - alpha) + aligned * alpha).to(
            reference.dtype), 0.0
    raise ValueError(f"unknown history mode {mode}")
