"""Direct test of the over-smoothing hypothesis for the few-step student.

`InstanceAppearanceConsistency` measured `iac_ratio = 0.928`, i.e. the same
object's appearance drifts *less* in generated video than in real video. That
is compatible with two opposite readings and IAC alone cannot separate them:

    over-smoothing  -> generated frames change too little over time and carry
                       less spatial detail
    identity gain   -> the model genuinely preserves objects better

This metric measures the change itself rather than an object's appearance, on
three axes that respond differently to the two readings:

    tdelta   mean |x[t] - x[t-1]|           first-order temporal energy
    tjitter  mean |x[t+1] - 2x[t] + x[t-1]| second-order (flicker) energy
    sgrad    mean spatial gradient magnitude (both axes)

Over-smoothing predicts `tdelta_ratio < 1` *and* `sgrad_ratio < 1`. Temporal
flicker -- the opposite defect -- predicts `tjitter_ratio > 1` while
`tdelta_ratio` may be near 1. A model that is genuinely fine on this axis
gives all three near 1.0.
"""
import torch
import torchmetrics


class PanoTemporalSmoothness(torchmetrics.Metric):

    _AXES = ("tdelta", "tjitter", "sgrad")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for axis in self._AXES:
            for arm in ("real", "fake"):
                self.add_state(
                    "{}_{}".format(axis, arm),
                    default=torch.tensor(0.0), dist_reduce_fx="sum")
        for arm in ("real", "fake"):
            self.add_state(
                "n_{}".format(arm),
                default=torch.tensor(0.0), dist_reduce_fx="sum")
        # Sum of squares of the per-clip motion energy, so the *dispersion*
        # across clips can be reported alongside its mean. This is the direct
        # test of the divergence diagnosis: a forward-KL (mode-covering)
        # objective predicts an over-dispersed motion distribution
        # (ratio > 1), a reverse-KL (mode-seeking) one predicts under-
        # dispersion (< 1). The means alone cannot tell those apart.
        for arm in ("real", "fake"):
            self.add_state(
                "tdelta_sq_{}".format(arm),
                default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, frames: torch.Tensor, real: bool):
        """Args:
            frames: ``(B, T, C, H, W)`` stitched panoramas, V already dropped.
            real: which arm the batch belongs to.
        """
        x = frames.detach().to(dtype=torch.float32)
        if x.dim() != 5:
            raise ValueError(
                "PanoTemporalSmoothness expects (B, T, C, H, W), got {}"
                .format(tuple(x.shape)))
        if x.shape[1] < 3:
            raise ValueError(
                "the jitter term needs at least 3 frames, got {}"
                .format(x.shape[1]))

        batch_size = float(x.shape[0])
        values = {
            "tdelta": (x[:, 1:] - x[:, :-1]).abs().mean(),
            "tjitter":
                (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]).abs().mean(),
            # Horizontal difference is circular: the canvas is a closed 360
            # panorama, so the wrap column is a real neighbour pair.
            "sgrad": 0.5 * (
                (x - x.roll(1, dims=-1)).abs().mean()
                + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()),
        }
        arm = "real" if real else "fake"
        for axis, value in values.items():
            name = "{}_{}".format(axis, arm)
            setattr(self, name, getattr(self, name) + value * batch_size)
        # Per-clip, so the batch dimension must be kept out of the mean.
        per_clip = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(1, 2, 3, 4))
        sq_name = "tdelta_sq_{}".format(arm)
        setattr(
            self, sq_name,
            getattr(self, sq_name) + per_clip.square().sum())
        setattr(
            self, "n_{}".format(arm),
            getattr(self, "n_{}".format(arm)) + batch_size)

    def compute(self):
        if float(self.n_real) <= 0.0 or float(self.n_fake) <= 0.0:
            raise RuntimeError(
                "PanoTemporalSmoothness got no samples on one arm")

        out = {}
        for axis in self._AXES:
            real = float(getattr(self, "{}_real".format(axis))) \
                / float(self.n_real)
            fake = float(getattr(self, "{}_fake".format(axis))) \
                / float(self.n_fake)
            out["{}_real".format(axis)] = real
            out["{}_fake".format(axis)] = fake
            out["{}_ratio".format(axis)] = fake / max(real, 1e-12)

        stds = {}
        for arm in ("real", "fake"):
            n = float(getattr(self, "n_{}".format(arm)))
            mean = float(getattr(self, "tdelta_{}".format(arm))) / n
            second = float(getattr(self, "tdelta_sq_{}".format(arm))) / n
            stds[arm] = max(second - mean * mean, 0.0) ** 0.5
            out["tdelta_std_{}".format(arm)] = stds[arm]
        out["tdelta_std_ratio"] = stds["fake"] / max(stds["real"], 1e-12)
        return out
