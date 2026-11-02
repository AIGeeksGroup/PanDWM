"""Per-column fake-vs-real discontinuity profile on the V=1 stitched panorama.

Answers the one question the per-camera FVD decomposition structurally
cannot. The panorama is sliced into two 201.6-degree halves whose joints land
at *exactly* the front and back azimuths, so "worse at the seams" and "worse
straight ahead / straight behind" are the same statement in a six-region
readout. The discriminator is the *shape* of the discrepancy along the
azimuth axis:

    narrow peak at the two joint columns -> seam hypothesis
    broad hump over the front/back sectors -> scene-content hypothesis

Protocol matches the offline seam attribution (`jump` at the joint column
divided by `base`, the same statistic over the surrounding +-20 columns), so
the numbers here are directly comparable to the 12-24 sample offline study --
only measured over the full 149-clip protocol instead.

The canvas is a closed 360-degree panorama: column 0 is adjacent to column
W-1, so the wrap joint is measured with a circular difference rather than
being dropped.
"""
import math
import os

import torch
import torchmetrics


class PanoSeamProfile(torchmetrics.Metric):

    # V=1 stitched pano width (= 2 * 2304 - 494 overlap).
    PANO_W_V1 = 4114
    # Left half's right edge: out[2303] comes from the left half, out[2304]
    # from the right half, so the joint is the difference *at* column 2304.
    OVERLAP_W = 494
    SEAM_LR = 2304
    # The canvas wraps, so the joint is the difference at column 0 against
    # column W-1.
    SEAM_WRAP = 0
    # Same neighbourhood the offline study used for its `base` denominator.
    BASE_HALFWIDTH = 20
    # Window inside which the half-maximum width of the excess is measured.
    PEAK_HALFWIDTH = 64
    # 30-degree azimuth sectors; sector k spans [-180 + 30k, -180 + 30(k+1)).
    SECTOR_COUNT = 12

    def __init__(self, save_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        w = self.PANO_W_V1
        self.add_state(
            "sum_real", default=torch.zeros(w), dist_reduce_fx="sum")
        self.add_state(
            "sum_fake", default=torch.zeros(w), dist_reduce_fx="sum")
        self.add_state(
            "n_real", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_fake", default=torch.tensor(0.0), dist_reduce_fx="sum")
        for name in ("dis_real", "dis_fake", "dis_n_real", "dis_n_fake"):
            self.add_state(
                name, default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, frames: torch.Tensor, real: bool):
        """Accumulate the per-column horizontal gradient.

        Args:
            frames: ``(B, T, C, H, W)`` stitched panoramas, V already dropped.
            real: which arm the batch belongs to.
        """
        if frames.shape[-1] != self.PANO_W_V1:
            raise ValueError(
                "PanoSeamProfile expects the V=1 stitched width {}, got {}"
                .format(self.PANO_W_V1, frames.shape[-1]))

        x = frames.detach().to(dtype=torch.float32)
        x = x.flatten(0, 1) if x.dim() == 5 else x
        # d[..., c] = |x[..., c] - x[..., c - 1]|, circular so d[..., 0] is
        # the wrap joint rather than a boundary artefact.
        d = (x - x.roll(1, dims=-1)).abs()
        # Sum over samples, channels and rows -> one number per column.
        column_sum = d.sum(dim=(0, 1, 2))
        weight = float(x.shape[0] * x.shape[1] * x.shape[2])

        if real:
            self.sum_real = self.sum_real + column_sum.to(self.sum_real)
            self.n_real = self.n_real + weight
        else:
            self.sum_fake = self.sum_fake + column_sum.to(self.sum_fake)
            self.n_fake = self.n_fake + weight

    def update_halves(self, halves: torch.Tensor, real: bool):
        """Measure how much the two independently generated halves disagree.

        The halves overlap by `PANO_OVERLAP_W` columns imaging the same 43.2
        degrees, so this is the one place in the panorama where two model
        outputs have a ground-truth requirement to be identical. Real frames
        give the floor (two independent LANCZOS resizes of one source image);
        anything above it on the fake arm is the DiT failing to keep its two
        views consistent -- the same mechanism that would produce a wrap-joint
        artefact, measured where it is directly observable.

        Args:
            halves: ``(B, T, V=2, C, H, W)`` before stitching.
            real: which arm the batch belongs to.
        """
        x = halves.detach().to(dtype=torch.float32)
        if x.dim() != 6 or x.shape[2] != 2:
            raise ValueError(
                "update_halves expects (B, T, 2, C, H, W), got {}"
                .format(tuple(x.shape)))

        width = x.shape[-1]
        left_overlap = x[:, :, 0, ..., width - self.OVERLAP_W:]
        right_overlap = x[:, :, 1, ..., :self.OVERLAP_W]
        disagree = (left_overlap - right_overlap).abs().mean()
        weight = float(x.shape[0])

        if real:
            self.dis_real = self.dis_real + disagree.to(self.dis_real) * weight
            self.dis_n_real = self.dis_n_real + weight
        else:
            self.dis_fake = self.dis_fake + disagree.to(self.dis_fake) * weight
            self.dis_n_fake = self.dis_n_fake + weight

    def _ring_indices(self, center: int, halfwidth: int):
        w = self.PANO_W_V1
        return [(center + o) % w for o in range(-halfwidth, halfwidth + 1)]

    def _seam_ratio(self, profile: torch.Tensor, center: int):
        idx = self._ring_indices(center, self.BASE_HALFWIDTH)
        base = torch.stack([profile[i] for i in idx if i != center]).mean()
        jump = profile[center]
        return float(jump), float(base), float(jump / base.clamp_min(1e-12))

    def _half_max_width(self, ratio: torch.Tensor, center: int):
        """Columns around `center` whose excess exceeds half the peak excess.

        A one-column joint artefact gives ~1; a broad sector-wide elevation
        fills the whole window and gives ~2 * PEAK_HALFWIDTH + 1.
        """
        peak_excess = float(ratio[center]) - 1.0
        if peak_excess <= 0.0:
            return 0.0
        threshold = 1.0 + 0.5 * peak_excess
        idx = self._ring_indices(center, self.PEAK_HALFWIDTH)
        return float(sum(1 for i in idx if float(ratio[i]) > threshold))

    def compute(self):
        if float(self.n_real) <= 0.0 or float(self.n_fake) <= 0.0:
            raise RuntimeError("PanoSeamProfile got no samples on one arm")

        pr = self.sum_real / self.n_real
        pf = self.sum_fake / self.n_fake
        ratio = pf / pr.clamp_min(1e-12)

        out = {}
        for name, center in (
                ("lr", self.SEAM_LR), ("wrap", self.SEAM_WRAP)):
            _, _, r_real = self._seam_ratio(pr, center)
            _, _, r_fake = self._seam_ratio(pf, center)
            out["seam_{}_real".format(name)] = r_real
            out["seam_{}_fake".format(name)] = r_fake
            # How much of the joint artefact is *not* common-mode. This is the
            # only part a teacher-vs-student comparison can possibly see.
            out["seam_{}_excess".format(name)] = \
                r_fake / max(r_real, 1e-12)
            out["seam_{}_width".format(name)] = \
                self._half_max_width(ratio, center)

        # Broad-hump readout. Sector 0 starts at azimuth -180 (straight back)
        # and sectors advance towards the front, which sits at sector 6.
        step = self.PANO_W_V1 / self.SECTOR_COUNT
        for k in range(self.SECTOR_COUNT):
            lo, hi = int(round(k * step)), int(round((k + 1) * step))
            out["sect{:02d}".format(k)] = float(
                pf[lo:hi].mean() / pr[lo:hi].mean().clamp_min(1e-12))

        if float(self.dis_n_real) > 0.0 and float(self.dis_n_fake) > 0.0:
            dis_real = float(self.dis_real) / float(self.dis_n_real)
            dis_fake = float(self.dis_fake) / float(self.dis_n_fake)
            out["overlap_real"] = dis_real
            out["overlap_fake"] = dis_fake
            out["overlap_ratio"] = dis_fake / max(dis_real, 1e-12)

        if self.save_path:
            try:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(
                    {
                        "profile_real": pr.detach().cpu(),
                        "profile_fake": pf.detach().cpu(),
                        "ratio": ratio.detach().cpu(),
                    }, self.save_path)
            except OSError as e:
                print("[seam] profile not saved: {}".format(e), flush=True)

        return out
