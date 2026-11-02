"""Window metrics on VAE-decoded float RGB, before video encoding."""

import torch
import torch.nn.functional as F


ANALYSIS_SIZE = (192, 512)  # Each of six azimuth sectors keeps its aspect ratio.


def _sectors(pano):
    """B,T,1,3,H,W -> (B*T*6,3,192,512), without MP4 quantization."""
    if pano.ndim != 6 or pano.shape[2] != 1 or pano.shape[3] != 3:
        raise ValueError(f"expected B,T,1,3,H,W panorama, got {pano.shape}")
    b, t = pano.shape[:2]
    pieces = []
    for part in torch.tensor_split(pano[:, :, 0], 6, dim=-1):
        if part.shape[-1] < 8:
            raise ValueError("panorama too narrow for six sectors")
        flat = part.flatten(0, 1).float()
        pieces.append(F.interpolate(
            flat, size=ANALYSIS_SIZE, mode="bilinear", align_corners=False))
    return torch.stack(pieces, dim=1).reshape(b * t * 6, 3, *ANALYSIS_SIZE)


@torch.no_grad()
def window_metrics(fake, real, lpips_metric=None, lpips_chunk=8):
    """Return per-frame PSNR, LPIPS, high-frequency energy and correlation."""
    if fake.shape != real.shape or fake.shape[0] != 1:
        raise ValueError("paired panorama tensors must have same shape and B=1")
    fake = fake.float().clamp(0, 1)
    real = real.float().clamp(0, 1)
    t = fake.shape[1]
    mse = (fake - real).square().mean(dim=(0, 2, 3, 4, 5))
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    f = _sectors(fake)
    r = _sectors(real)
    luminance = torch.tensor([0.2126, 0.7152, 0.0722],
                             device=f.device).view(1, 3, 1, 1)
    fy = (f * luminance).sum(dim=1, keepdim=True)
    ry = (r * luminance).sum(dim=1, keepdim=True)
    # The same local high-pass filter is applied to GT and generated RGB.
    fh = fy - F.avg_pool2d(fy, 5, stride=1, padding=2)
    rh = ry - F.avg_pool2d(ry, 5, stride=1, padding=2)
    # Discard filter border pixels; sector edges and panorama seams are not
    # treated as real image detail.
    fh, rh = fh[..., 2:-2, 2:-2], rh[..., 2:-2, 2:-2]
    fe = fh.square().mean(dim=(1, 2, 3))
    re = rh.square().mean(dim=(1, 2, 3))
    ratio = (fe / re.clamp_min(1e-12)).reshape(t, 6).mean(dim=1)
    corr = ((fh * rh).mean(dim=(1, 2, 3)) /
            (fe * re).sqrt().clamp_min(1e-12)).reshape(t, 6).mean(dim=1)
    out = {
        "psnr_per_frame_db": psnr.cpu().tolist(),
        "high_frequency_energy_ratio_per_frame": ratio.cpu().tolist(),
        "high_frequency_correlation_per_frame": corr.cpu().tolist(),
        "analysis_sector_size": list(ANALYSIS_SIZE),
    }
    if lpips_metric is not None:
        scores = []
        for start in range(0, len(f), lpips_chunk):
            end = start + lpips_chunk
            scores.append(lpips_metric(
                2 * f[start:end] - 1, 2 * r[start:end] - 1,
                normalize=False).reshape(-1).float())
        out["lpips_per_frame"] = torch.cat(scores).reshape(t, 6).mean(1).cpu().tolist()
    return out


@torch.no_grad()
def boundary_metrics(first_fake, prev_fake, first_real, prev_real):
    """Compare the generated join to the motion at the same GT timestamps."""
    if any(x.shape != first_fake.shape for x in
           (prev_fake, first_real, prev_real)):
        raise ValueError("boundary frames must have the same shape")
    fake_delta = first_fake.float() - prev_fake.float()
    real_delta = first_real.float() - prev_real.float()
    return {
        "generated_boundary_l1": float(fake_delta.abs().mean()),
        "gt_boundary_l1": float(real_delta.abs().mean()),
        "boundary_motion_residual_l1": float(
            (fake_delta - real_delta).abs().mean()),
    }


def summarize_window(frame_metrics, boundary=None):
    result = {}
    for key, values in frame_metrics.items():
        if key.endswith("_per_frame") or key == "psnr_per_frame_db":
            result[key] = [round(float(x), 6) for x in values]
            base = key.replace("_per_frame", "")
            result[f"{base}_mean"] = round(sum(values) / len(values), 6)
        else:
            result[key] = values
    if boundary is not None:
        result.update({key: round(value, 6) for key, value in boundary.items()})
    return result
