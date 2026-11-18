"""Pano-patch FVD for V=1 stitched panorama (W=4114).

Cuts the stitched pano into 6 cam-aligned Voronoi patches and runs
standard TATS FVD on the pooled patch features. Each cam patch
contributes one sample to the same Frechet distance pool — mirrors
the official 6-view pattern in `dwm.pipelines.ctsd:1962-1980`
(`einops.rearrange("b t v c h w -> (b v) t c h w")` flattens views
into batch).

Patch boundaries derived from PanDWM
`omninwm/utils/pano_hardseam.py:22` Voronoi cuts on the 9600-wide
source pano, mapped to V=1 4114-wide after the LEFT-half LANCZOS
resize + RIGHT-half LANCZOS resize + 494-px dedup overlap drop in
`ctsd_pano.py:_stitch_pano`.

CAM_BACK is wrap-merged (RIGHT tail + LEFT head concat'd) so all 6
cams remain whole; alternative split-wrap would over-weight CAM_BACK.
CAM_FRONT inherits the LEFT/RIGHT half seam internally (at V=1 col
2304) — this is a real model-output property; FVD will reflect any
seam artifact symmetrically for real and fake.
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

    # V=1 stitched pano width after `ctsd_pano.py:_stitch_pano`
    # (= 2 * 2304 - _PANO_OVERLAP_W (494) = 4114).
    PANO_W_V1 = 4114

    # 6 inner Voronoi cuts on V=1 4114, mapping
    # `VORONOI_BOUNDARIES = [1229, 2625, 4069, 5557, 7038, 8421]` (on 9600)
    # via the two-segment LEFT/RIGHT half resize geometry.
    # Cam ranges (each `[lo, hi)` on the 4114 axis):
    #   CAM_BACK         = [3609, 4114) U [0, 527)   wrap-concat, W=1032
    #                      (505 + 527; the six widths sum to 4114)
    #   CAM_BACK_LEFT    = [527,  1125)              W= 598
    #   CAM_FRONT_LEFT   = [1125, 1744)              W= 619
    #   CAM_FRONT        = [1744, 2381)              W= 637  (crosses LEFT/RIGHT
    #                                                          seam at V=1 col 2304)
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

        # Instrument control, default off. Every nuScenes camera owns a
        # different azimuth wedge, so the six Voronoi patches have different
        # widths (CAM_BACK 1032 vs CAM_BACK_RIGHT 593) and therefore reach
        # 224 through a different horizontal squeeze (4.61x vs 2.65x). That
        # makes the per-camera values incomparable ACROSS cameras. Setting
        # this center-crops every patch to one common width so the six become
        # geometrically identical. 0 keeps the historical behaviour bit-exact.
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

        # Per-camera accumulators, kept ALONGSIDE the pooled ones above so the
        # pooled number stays bit-identical to every previously reported run.
        # A pooled Gaussian over all six patches is not the same object as six
        # per-patch Gaussians, so these cannot be derived from each other.
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

        # Cut 6 cam-aligned patches by Voronoi boundaries on the V=1 axis.
        # CAM_BACK wraps across the panorama seam (cols 0 and PANO_W_V1),
        # so its head and tail are concat'd into one whole-cam patch.
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

        Split out of :meth:`update` so a sibling metric that obtains its six
        views some other way -- e.g. the real-camera protocol bridge, which
        resamples them out of the pano instead of slicing columns -- reuses
        byte-identical normalization, resize, I3D weights and Frechet math.
        Anything that bypasses this method is a different instrument and its
        numbers are not comparable (see the change log's metric-naming rule).
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

        Same I3D features and same Frechet formula as `compute`, but each
        camera region gets its own Gaussian instead of one pooled over all
        six. Use this to ask *where* a pooled FVD change came from; the
        per-camera values are not expected to average to the pooled one
        (a pooled Gaussian is a different object).

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
