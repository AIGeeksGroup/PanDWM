"""Does the error accumulate across the generated window?

The evaluation is *paired*: every generated frame has the ground-truth frame
it was conditioned to produce (same clip, same bbox/HD-map layout, same three
context frames). So the fake-vs-real distance can be read per frame index,
which no distribution metric can do.

The window is denoised jointly under a block-causal temporal mask, so frame t
attends to frames <= t of the *current iterate*, not to ground truth. Error
accumulation inside the window is therefore possible even though the protocol
never rolls out across chunks. `rCM` names exactly this -- error accumulation
under a consistency objective -- as the reason few-step consistency models lose
fine detail, so it is worth measuring before paying for any retraining.

Reads out three things:

    pl1_*    paired L1 between fake and real, by frame index
    sharp_*  spatial gradient magnitude, by frame index and arm
    *_slope  ordinary least squares slope per frame index

A flat `pl1` profile means the deficit is uniform over the window and drift is
not the mechanism. A rising one means it is.
"""
import torch
import torchmetrics


class PanoFrameProfile(torchmetrics.Metric):

    # Frames scored per clip by the frozen protocol.
    MAX_FRAMES = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ("pl1", "sharp_real", "sharp_fake", "count"):
            self.add_state(
                name, default=torch.zeros(self.MAX_FRAMES),
                dist_reduce_fx="sum")

    def update(self, fake: torch.Tensor, real: torch.Tensor):
        """Args:
            fake, real: ``(B, T, C, H, W)`` stitched panoramas, paired.
        """
        if fake.shape != real.shape:
            raise ValueError(
                "PanoFrameProfile needs paired shapes, got {} vs {}"
                .format(tuple(fake.shape), tuple(real.shape)))
        if fake.dim() != 5:
            raise ValueError(
                "PanoFrameProfile expects (B, T, C, H, W), got {}"
                .format(tuple(fake.shape)))

        f = fake.detach().to(dtype=torch.float32)
        r = real.detach().to(dtype=torch.float32)
        frames = f.shape[1]
        if frames > self.MAX_FRAMES:
            raise ValueError(
                "sequence of {} frames exceeds MAX_FRAMES={}"
                .format(frames, self.MAX_FRAMES))

        # Mean over everything but the frame index.
        pl1 = (f - r).abs().mean(dim=(0, 2, 3, 4))

        def sharpness(x):
            gx = (x - x.roll(1, dims=-1)).abs().mean(dim=(0, 2, 3, 4))
            gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(0, 2, 3, 4))
            return 0.5 * (gx + gy)

        weight = float(f.shape[0])
        self.pl1 = self.pl1.index_add(
            0, torch.arange(frames, device=self.pl1.device),
            (pl1 * weight).to(self.pl1))
        self.sharp_fake = self.sharp_fake.index_add(
            0, torch.arange(frames, device=self.sharp_fake.device),
            (sharpness(f) * weight).to(self.sharp_fake))
        self.sharp_real = self.sharp_real.index_add(
            0, torch.arange(frames, device=self.sharp_real.device),
            (sharpness(r) * weight).to(self.sharp_real))
        self.count = self.count.index_add(
            0, torch.arange(frames, device=self.count.device),
            torch.full((frames,), weight, device=self.count.device,
                       dtype=self.count.dtype))

    @staticmethod
    def _slope(values: torch.Tensor):
        """OLS slope per frame index."""
        n = values.numel()
        if n < 2:
            return 0.0
        idx = torch.arange(n, dtype=torch.float64)
        y = values.to(dtype=torch.float64)
        idx = idx - idx.mean()
        return float((idx * (y - y.mean())).sum() / (idx * idx).sum())

    def compute(self):
        used = self.count > 0
        if int(used.sum()) == 0:
            raise RuntimeError("PanoFrameProfile got no samples")

        counts = self.count[used]
        pl1 = (self.pl1[used] / counts).cpu()
        sf = (self.sharp_fake[used] / counts).cpu()
        sr = (self.sharp_real[used] / counts).cpu()
        n = pl1.numel()
        head = max(1, n // 4)

        out = {
            "pl1_first4": float(pl1[:head].mean()),
            "pl1_last4": float(pl1[-head:].mean()),
            "pl1_growth": float(
                pl1[-head:].mean() / pl1[:head].mean().clamp_min(1e-12)),
            "pl1_slope": self._slope(pl1),
            "sharp_fake_first4": float(sf[:head].mean()),
            "sharp_fake_last4": float(sf[-head:].mean()),
            "sharp_real_first4": float(sr[:head].mean()),
            "sharp_real_last4": float(sr[-head:].mean()),
            "frames": float(n),
        }
        # Does the fake arm lose sharpness *faster* than the real arm does?
        # Real video also changes over a window, so the real decay is the
        # baseline the fake decay has to be read against.
        fake_decay = out["sharp_fake_last4"] \
            / max(out["sharp_fake_first4"], 1e-12)
        real_decay = out["sharp_real_last4"] \
            / max(out["sharp_real_first4"], 1e-12)
        out["sharp_decay_ratio"] = fake_decay / max(real_decay, 1e-12)
        return out
