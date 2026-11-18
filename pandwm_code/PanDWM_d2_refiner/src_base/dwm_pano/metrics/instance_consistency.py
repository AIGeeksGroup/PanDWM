"""Instance appearance consistency: does one object keep its look across frames?

FID/FVD are distribution metrics over whole frames; neither is designed to
notice that a particular car changed colour between frame 3 and frame 7.  The
identity-preserving driving world models (ConsisDrive, InstaDrive) both
validate instance consistency with *tracking* metrics rather than FID/FVD for
exactly this reason.  This metric is the cheapest instrument that measures the
property directly, without needing a detector or a tracker: the GT boxes and
their persistent `gt_box_ids` already tell us where each object is in every
frame, so we can simply compare an object's own crops over time.

Score definition
----------------
For every instance visible in at least two frames, crop its region in each
frame, resize to a fixed patch, and L2-normalise the flattened RGB into a
descriptor.  ``drift`` is the mean distance between descriptors of the *same*
instance in consecutive frames.

Real video sets the achievable floor -- a real car legitimately changes
appearance as it turns, becomes occluded, or moves through shadow -- so the
reported number is the ratio

    IAC = drift(generated) / drift(real)

A value of 1.0 means the generated video drifts exactly as much as reality;
above 1.0 means excess identity drift.  ``separation`` is a sanity companion:
the mean distance between *different* instances in the same frame.  If
same-instance and different-instance distances are comparable, the descriptor
is not discriminative and the ratio should not be trusted.

Descriptor choice is the load-bearing assumption
------------------------------------------------
The original descriptor is a 32x32 bilinear downsample of the raw RGB crop.
That is a *low-order appearance* statistic: it moves a lot when lighting or
pose changes and barely at all when the object keeps its colour but swaps
identity (a sedan becoming a hatchback of the same tone).  A near-1.0 reading
from that descriptor therefore cannot by itself close the "does this model
preserve instance identity" question -- the instrument may simply be blind to
the failure mode.  ``descriptor`` selects what the crops are embedded with:

``"patch"``      32x32 RGB flatten (the original; kept as the control arm).
``"inception"``  2048-d pool features of the FID Inception-V3.  Semantic, and
                 the exact space FID itself lives in.
``"vgg"``        ImageNet VGG16 through ``relu3_3`` -- the LPIPS perceptual
                 backbone, sensitive to texture and shape rather than tone.

Both CNN weights already sit in the torch hub cache used by the FID metric, so
no download is needed.  Run several instances side by side on one evaluation:
the frames are deterministic, so the ``"patch"`` arm must reproduce its earlier
number exactly, which doubles as an instrument check on the others.
"""
import numpy as np
import torch
import torchmetrics

from dwm_pano.metrics.instance_boxes import PANO_W_V1, frame_instance_boxes


# Input side each backbone expects.  None = use the caller's patch_size.
_DESCRIPTOR_INPUT = {"patch": None, "inception": 299, "vgg": 224}

# One backbone per (descriptor, device); several metric instances in the same
# evaluation would otherwise each hold their own copy of VGG16.
_BACKBONE_CACHE = {}


def _build_backbone(descriptor: str, device: str):
    """Load a frozen embedding network from the local torch hub cache.

    Both checkpoints are already present wherever the FID metric has run
    (``weights-inception-2015-12-05-6726825d.pth`` and ``vgg16-*.pth``), so
    this must not require network access.  Failing loudly beats silently
    falling back to the raw-pixel descriptor, which is the very thing these
    arms exist to check.
    """
    key = (descriptor, device)
    if key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[key]

    if descriptor == "inception":
        from torchmetrics.image.fid import NoTrainInceptionV3
        net = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=["2048"])
    elif descriptor == "vgg":
        import torchvision
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        # features[:16] ends at relu3_3, the LPIPS layer.
        net = torch.nn.Sequential(
            *list(torchvision.models.vgg16(weights=weights).features[:16]))
    else:
        raise ValueError(f"no backbone for descriptor {descriptor!r}")

    net.eval().to(device)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    _BACKBONE_CACHE[key] = net
    return net


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InstanceAppearanceConsistency(torchmetrics.Metric):

    def __init__(self, pkl_path: str, patch_size: int = 32,
                 min_pixels: int = 8, descriptor: str = "patch",
                 backbone_device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        if descriptor not in _DESCRIPTOR_INPUT:
            raise ValueError(
                "descriptor must be one of {}, got {!r}".format(
                    sorted(_DESCRIPTOR_INPUT), descriptor))
        self.pkl_path = pkl_path
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.descriptor = descriptor
        self.backbone_device = backbone_device
        self._backbone = None
        self._index = None
        for side in ("real", "fake"):
            self.add_state(f"{side}_drift_sum", torch.tensor(0.0).double(),
                           dist_reduce_fx="sum")
            self.add_state(f"{side}_drift_num", torch.tensor(0).long(),
                           dist_reduce_fx="sum")
            self.add_state(f"{side}_sep_sum", torch.tensor(0.0).double(),
                           dist_reduce_fx="sum")
            self.add_state(f"{side}_sep_num", torch.tensor(0).long(),
                           dist_reduce_fx="sum")
        self.add_state("wrap_skipped", torch.tensor(0).long(),
                       dist_reduce_fx="sum")

    def _timestamp_index(self):
        if self._index is None:
            import pickle
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
            infos = data["infos"] if isinstance(data, dict) else data
            self._index = {int(i["timestamp"]): i for i in infos}
        return self._index

    def _crop_side(self):
        side = _DESCRIPTOR_INPUT[self.descriptor]
        return self.patch_size if side is None else side

    def _embed(self, crops):
        """crops: (N, 3, S, S) float in [0, 1] -> (N, D) raw embeddings."""
        if self.descriptor == "patch":
            return crops.flatten(1).double()

        if self._backbone is None:
            self._backbone = _build_backbone(
                self.descriptor, self.backbone_device)
        batch = crops.to(self.backbone_device)
        with torch.no_grad():
            if self.descriptor == "inception":
                # torch-fidelity's extractor asserts a uint8 input and does its
                # own [-1, 1] rescale; feeding it floats silently returns
                # garbage on some versions, so convert explicitly.
                batch = (batch.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                out = self._backbone(batch)
            else:
                mean = torch.tensor(
                    _IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
                std = torch.tensor(
                    _IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
                out = self._backbone((batch - mean) / std).flatten(1)
        return out.float().cpu().double()

    def _descriptors(self, frames, boxes):
        """frames: (T, C, H, W); boxes: (T, 4) -> (T, D) L2-normalised.

        Crop geometry and the L2 normalisation are identical across
        descriptors; only the embedding of each crop changes.  That is what
        makes the ``"patch"`` arm reproduce the historical number bit for bit
        while the CNN arms answer the same question in a different space.
        """
        side = self._crop_side()
        crops = []
        _, _, height, width = frames.shape
        for t in range(frames.shape[0]):
            x0, y0, x1, y1 = boxes[t]
            x0 = max(0, min(width - 1, int(np.floor(x0))))
            x1 = max(x0 + 1, min(width, int(np.ceil(x1))))
            y0 = max(0, min(height - 1, int(np.floor(y0))))
            y1 = max(y0 + 1, min(height, int(np.ceil(y1))))
            crop = frames[t:t + 1, :, y0:y1, x0:x1]
            crops.append(torch.nn.functional.interpolate(
                crop, size=(side, side), mode="bilinear", align_corners=False))
        vectors = self._embed(torch.cat(crops, dim=0))
        return vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-8)

    def update(self, frames, frame_timestamps_us, real=True):
        """frames: (B, T, C, H, W=4114) in [0, 1]; timestamps: (B, T) int64."""
        if frames.shape[-1] != PANO_W_V1:
            raise ValueError(
                f"expected the V=1 stitched canvas (W={PANO_W_V1}), "
                f"got {frames.shape[-1]}")
        index = self._timestamp_index()
        side = "real" if real else "fake"
        frames = frames.detach().float().cpu()
        timestamps = frame_timestamps_us.detach().cpu().tolist()

        for b in range(frames.shape[0]):
            # instance token -> {frame: box}
            tracks, per_frame = {}, []
            for t, ts in enumerate(timestamps[b]):
                info = index.get(int(ts))
                if info is None:
                    per_frame.append(([], np.zeros((0, 4))))
                    continue
                ids, boxes, skipped = frame_instance_boxes(
                    info, min_pixels=self.min_pixels)
                self.wrap_skipped += skipped
                per_frame.append((ids, boxes))
                for token, box in zip(ids, boxes):
                    tracks.setdefault(token, {})[t] = box

            for token, by_frame in tracks.items():
                times = sorted(by_frame)
                if len(times) < 2:
                    continue
                boxes = np.stack([by_frame[t] for t in times])
                descriptors = self._descriptors(frames[b, times], boxes)
                # consecutive-frame drift of this one instance
                distance = (descriptors[1:] - descriptors[:-1]).norm(dim=-1)
                getattr(self, f"{side}_drift_sum").add_(distance.sum())
                getattr(self, f"{side}_drift_num").add_(distance.numel())

            # separation companion: different instances, same frame
            for t, (ids, boxes) in enumerate(per_frame):
                if len(ids) < 2:
                    continue
                descriptors = self._descriptors(
                    frames[b, t].unsqueeze(0).expand(len(ids), -1, -1, -1),
                    boxes)
                pair = torch.cdist(descriptors, descriptors)
                upper = pair[torch.triu(torch.ones_like(pair), 1) > 0]
                getattr(self, f"{side}_sep_sum").add_(upper.sum())
                getattr(self, f"{side}_sep_num").add_(upper.numel())

    def compute(self):
        def mean(side, kind):
            total = getattr(self, f"{side}_{kind}_sum")
            count = getattr(self, f"{side}_{kind}_num")
            return float(total / count) if int(count) > 0 else float("nan")

        real_drift, fake_drift = mean("real", "drift"), mean("fake", "drift")
        # Several arms run side by side in one evaluation, and the printer
        # flattens every dict into one line, so the labels have to carry the
        # arm identity themselves -- an unprefixed `iac_ratio` from four
        # metrics is four indistinguishable numbers. The prefix is built from
        # the two settings that define an arm rather than from the config key,
        # which the metric never sees. Values stay numeric: the printer
        # formats with `{:.3f}`.
        tag = "{}{}".format(self.descriptor, self.min_pixels)
        return {
            "{}_iac_ratio".format(tag):
                fake_drift / real_drift if real_drift > 0 else float("nan"),
            "{}_drift_real".format(tag): real_drift,
            "{}_drift_fake".format(tag): fake_drift,
            "{}_separation_real".format(tag): mean("real", "sep"),
            "{}_separation_fake".format(tag): mean("fake", "sep"),
            "{}_pairs_real".format(tag): int(self.real_drift_num),
            "{}_pairs_fake".format(tag): int(self.fake_drift_num),
            "{}_wrap_skipped".format(tag): int(self.wrap_skipped),
        }
