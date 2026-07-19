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
    #   CAM_BACK         = [3609, 4114) U [0, 527)   wrap-concat, W=1033
    #   CAM_BACK_LEFT    = [527,  1125)              W= 598
    #   CAM_FRONT_LEFT   = [1125, 1744)              W= 619
    #   CAM_FRONT        = [1744, 2381)              W= 637  (crosses LEFT/RIGHT
    #                                                          seam at V=1 col 2304)
    #   CAM_FRONT_RIGHT  = [2381, 3016)              W= 635
    #   CAM_BACK_RIGHT   = [3016, 3609)              W= 593
    PANO_PATCH_COLS = [527, 1125, 1744, 2381, 3016, 3609]

    def __init__(
        self, inception_3d_checkpoint_path: str, sequence_count: int = -1,
        num_classes: int = 400, **kwargs
    ):
        super().__init__(**kwargs)

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
