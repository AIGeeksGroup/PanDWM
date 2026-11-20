"""Pano-patch FVD for a single-view stitched panorama (4114 px wide).

Cuts the stitched pano into six camera-aligned Voronoi patches and runs standard TATS FVD on the pooled
patch features. Each patch contributes one sample to the same Frechet distance pool (views are flattened
into the batch, as in the multi-view FVD of the base pipeline).

Patch boundaries are the Voronoi cuts of the 9600-wide source pano, mapped to the 4114-wide stitched
canvas (two resized halves with a 494-px overlap dropped, see `ctsd_pano.py:_stitch_pano`).

CAM_BACK is wrap-merged (right tail + left head) so all six cameras stay whole. CAM_FRONT contains the
left/right half seam (column 2304); it affects real and generated frames alike.
"""
import torch
import torch.distributed
import torchmetrics

# PYTHONPATH includes ${workspaceFolder}/externals/TATS/tats/fvd
import pytorch_i3d


def _compute_fid(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor,
    sigma2: torch.Tensor
):
    # The same implementation as torchmetrics.image.fid._compute_fid()

    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class PanoPatchFrechetVideoDistance(torchmetrics.Metric):

    target_resolution = (224, 224)
    i3d_min = 10

    # Stitched pano width after `ctsd_pano.py:_stitch_pano`
    # (= 2 * 2304 - _PANO_OVERLAP_W (494) = 4114).
    PANO_W_V1 = 4114

    # Inner Voronoi cuts on the 4114 axis, mapped from the boundaries
    # [1229, 2625, 4069, 5557, 7038, 8421] (on the 9600 axis) through the
    # two-segment left/right half resize geometry.
    # Cam ranges (each `[lo, hi)` on the 4114 axis):
    #   CAM_BACK         = [3609, 4114) U [0, 527)   wrap-concat, W=1032
    #                      (505 + 527; the six widths sum to 4114)
    #   CAM_BACK_LEFT    = [527,  1125)              W= 598
    #   CAM_FRONT_LEFT   = [1125, 1744)              W= 619
    #   CAM_FRONT        = [1744, 2381)              W= 637  (crosses the left/right
    #                                                          seam at column 2304)
    #   CAM_FRONT_RIGHT  = [2381, 3016)              W= 635
    #   CAM_BACK_RIGHT   = [3016, 3609)              W= 593
    PANO_PATCH_COLS = [527, 1125, 1744, 2381, 3016, 3609]

    # Must stay in the same order as the `patches` list built in `update`.
    CAM_NAMES = (
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT")

    def __init__(
        self, inception_3d_checkpoint_path: str, sequence_count: int = -1,
        num_classes: int = 400, equal_patch_width: int = 0, **kwargs
    ):
        super().__init__(**kwargs)

        # Optional (0 = off): the six patches have different widths, so they
        # reach 224 px through different horizontal squeezes and per-camera
        # values are not comparable across cameras. If set, every patch is
        # center-cropped to this common width.
        self.equal_patch_width = int(equal_patch_width)

        self.inception = pytorch_i3d.InceptionI3d(num_classes)
        self.inception.eval()
        state_dict = torch.load(
            inception_3d_checkpoint_path, map_location="cpu",
            weights_only=True)
        self.inception.load_state_dict(state_dict)
        self.sequence_count = sequence_count

        mx_num_feats = (num_classes, num_classes)
        self.add_state(
            "real_features_sum", torch.zeros(num_classes).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "real_features_cov_sum", torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_sum", torch.zeros(num_classes).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_cov_sum", torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum")
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(),
            dist_reduce_fx="sum")

        # Per-camera accumulators, kept alongside the pooled ones: a pooled
        # Gaussian over all six patches cannot be derived from six per-patch
        # Gaussians.
        cam_count = len(self.CAM_NAMES)
        for side in ("real", "fake"):
            self.add_state(
                f"{side}_cam_sum",
                torch.zeros(cam_count, num_classes).double(),
                dist_reduce_fx="sum")
            self.add_state(
                f"{side}_cam_cov_sum",
                torch.zeros(cam_count, *mx_num_feats).double(),
                dist_reduce_fx="sum")
            self.add_state(
                f"{side}_cam_num_samples",
                torch.zeros(cam_count).long(), dist_reduce_fx="sum")

    def update(self, frames, real=True):
        """Update the state with extracted features.

        Args:
            frames: Video tensor of V=1 stitched panorama in the shape of
                `(batch_size, sequence_length, channels, height, W=4114)`.
                Pixel values in [0, 1].
            real: Whether given frames are real or fake.
        """

        if self.sequence_count >= 0:
            frames = frames[:, :self.sequence_count]

        assert frames.shape[1] >= self.i3d_min

        # Cut six camera-aligned patches by the Voronoi boundaries.
        # CAM_BACK wraps across the panorama seam (columns 0 and PANO_W_V1),
        # so its head and tail are concatenated into one patch.
        b0, b1, b2, b3, b4, b5 = self.PANO_PATCH_COLS
        W = self.PANO_W_V1
        patches = [
            torch.cat([frames[..., b5:W], frames[..., 0:b0]], dim=-1),  # CAM_BACK
            frames[..., b0:b1],   # CAM_BACK_LEFT
            frames[..., b1:b2],   # CAM_FRONT_LEFT
            frames[..., b2:b3],   # CAM_FRONT
            frames[..., b3:b4],   # CAM_FRONT_RIGHT
            frames[..., b4:b5],   # CAM_BACK_RIGHT
        ]

        if self.equal_patch_width > 0:
            w = self.equal_patch_width
            cropped = []
            for p in patches:
                assert p.shape[-1] >= w, (
                    "equal_patch_width={} exceeds patch width {}".format(
                        w, p.shape[-1]))
                start = (p.shape[-1] - w) // 2
                cropped.append(p[..., start:start + w])
            patches = cropped

        self._accumulate_patches(patches, real)

    def _accumulate_patches(self, patches, real):
        """Feed an already-cut list of six views into the shared I3D path.

        Split out of :meth:`update` so that a sibling metric that obtains its
        six views differently (e.g. by resampling them out of the pano instead
        of slicing columns) reuses the same normalization, resize, I3D weights
        and Frechet computation.
        """
        # Per-patch normalize + bilinear resize to (224, 224), then stack
        # into one (6*B, T, C, 224, 224) tensor so I3D forward runs once.
        resized = []
        for p in patches:
            # normalize from [0, 1] to [-1, 1]
            p = p * 2 - 1
            p = torch.nn.functional.interpolate(
                p.flatten(0, 1), size=self.target_resolution,
                mode="bilinear"
            ).unflatten(0, p.shape[:2])
            resized.append(p)
        stacked = torch.cat(resized, dim=0)

        features = self.inception(stacked.transpose(1, 2))
        self.orig_dtype = features.dtype
        features = features.double()

        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]

        # `stacked` was cat'd patch-major, so rows [i*B:(i+1)*B] belong to
        # patch i -- that is what makes this unflatten the right split.
        per_cam = features.unflatten(0, (len(patches), -1))
        side = "real" if real else "fake"
        getattr(self, f"{side}_cam_sum").add_(per_cam.sum(dim=1))
        getattr(self, f"{side}_cam_cov_sum").add_(
            per_cam.transpose(1, 2) @ per_cam)
        getattr(self, f"{side}_cam_num_samples").add_(per_cam.shape[1])

    def compute(self):
        if (
            self.real_features_num_samples < 2 or
            self.fake_features_num_samples < 2
        ):
            raise RuntimeError(
                "More than one sample is required for both the real and fake "
                "distributed to compute FVD")

        mean_real = (
            self.real_features_sum / self.real_features_num_samples
        ).unsqueeze(0)
        mean_fake = (
            self.fake_features_sum / self.fake_features_num_samples
        ).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - \
            self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - \
            self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def compute_per_camera(self):
        """FVD computed separately inside each of the six Voronoi cam patches.

        Same I3D features and Frechet formula as `compute`, but each camera
        region gets its own Gaussian instead of one pooled over all six. Use it
        to see where a pooled FVD change comes from; the per-camera values do
        not average to the pooled value.

        Returns:
            dict mapping camera name -> float FVD, in `CAM_NAMES` order.
        """
        if int(self.real_cam_num_samples.min()) < 2 or \
                int(self.fake_cam_num_samples.min()) < 2:
            raise RuntimeError(
                "per-camera FVD needs at least two samples per camera; got "
                f"real={self.real_cam_num_samples.tolist()} "
                f"fake={self.fake_cam_num_samples.tolist()}")

        result = {}
        for i, name in enumerate(self.CAM_NAMES):
            values = []
            for side in ("real", "fake"):
                count = getattr(self, f"{side}_cam_num_samples")[i]
                mean = (getattr(self, f"{side}_cam_sum")[i]
                        / count).unsqueeze(0)
                cov_num = getattr(self, f"{side}_cam_cov_sum")[i] \
                    - count * mean.t().mm(mean)
                values.append((mean.squeeze(0), cov_num / (count - 1)))
            (mean_real, cov_real), (mean_fake, cov_fake) = values
            result[name] = float(
                _compute_fid(mean_real, cov_real, mean_fake, cov_fake))
        return result
