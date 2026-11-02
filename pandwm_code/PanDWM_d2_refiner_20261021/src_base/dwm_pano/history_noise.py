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


def history_noise_draws(shape, settings, step, seed, rank):
    """Per-sample history noise levels and noise for one training step.

    2026-10-15 teacher v3.  settings = {"prob": p, "sigma_max": s1, "sigma_min": s0}:
    a sample is noised with probability p, with s ~ U(s0, s1).  The generator is
    seeded by (seed, step, rank) only, so the draws never touch the model RNG.
    """
    prob = float(settings.get("prob", 0.5))
    smin = float(settings.get("sigma_min", 0.0))
    smax = float(settings.get("sigma_max", 0.2))
    if not 0 <= prob <= 1 or not 0 <= smin <= smax <= 1:
        raise ValueError(f"bad history_noise settings {settings}")
    g = torch.Generator(device="cpu").manual_seed(
        (int(seed) * 1000003 + int(step) * 7919 + int(rank) * 104729 + 17)
        % (2 ** 62))
    b = int(shape[0])
    on = torch.rand(b, generator=g) < prob
    s = smin + (smax - smin) * torch.rand(b, generator=g)
    sigma = torch.where(on, s, torch.zeros_like(s))
    eps = torch.randn(tuple(shape), generator=g)
    return sigma, eps


def apply_reference_noise(noisy_latents, timesteps, reference_mask, sigma, eps):
    """x_s = (1 - s) x + s eps on the visible reference frames; their timesteps become 1000 s.

    noisy_latents (b, t, v, c, h, w); timesteps and reference_mask (b, t, v);
    sigma (b,); frames that are not visible references are returned unchanged.
    """
    b, t, v = reference_mask.shape
    s = sigma.to(device=noisy_latents.device, dtype=torch.float32)
    sel = reference_mask.to(device=noisy_latents.device).bool() & (s.view(b, 1, 1) > 0)
    if not bool(sel.any()):
        return noisy_latents, timesteps
    s6 = s.view(b, 1, 1, 1, 1, 1)
    mixed = ((1 - s6) * noisy_latents.float()
             + s6 * eps.to(device=noisy_latents.device, dtype=torch.float32))
    out = torch.where(sel.view(b, t, v, 1, 1, 1), mixed.to(noisy_latents.dtype),
                      noisy_latents)
    level = (1000.0 * s).view(b, 1, 1).to(device=timesteps.device, dtype=timesteps.dtype).expand_as(timesteps)
    return out, torch.where(sel.to(timesteps.device), level, timesteps)
