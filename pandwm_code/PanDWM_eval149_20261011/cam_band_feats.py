"""Six-camera band features at 256x448 (2026-10-20): the resolution-matched comparison of the paper's evaluation space.

The paper scores every method in its panoramic band; baselines enter it by stitching their six views (OneBEV map). The
second comparison promised in the paper goes the other way, at UniMLVG's native 256x448 per camera, restricted to the
camera pixels the band covers: our canvases are resampled into each real camera with the clip's calibration, and the
native views of the six-camera methods receive the same mask.

Per Val-150 clip (150 scene-start clips, test_index = Val-150 position, scene-0018 = 149), clip frames 3..34:
  * output grid: the FULL native 1600x900 frame at pixel centres u = (j + 0.5) * 1600 / 448, v = (i + 0.5) * 900 / 256,
    i.e. exactly the pixel grid of a 1600x900 -> 448x256 resize (= intrinsics scaled with the half-pixel convention);
    no render at another size, no resize. Calibration: the clip's REAL six-camera rig from the dataset batch
    (pano_camera_transforms / pano_camera_intrinsics / pano_camera_image_size, frame 0; asserted constant over the clip).
  * geometry (--geometry, comma list; several share one decode, --out-dir then needs {geometry}):
      bridge              dwm_pano.nuscenes6_bridge.build_rig_sampling_grids(output_size=(256, 448), band_limit=False,
                          source_axes=canvas_source_axes): the ideal calibrated sphere. Validation (cam_band_check.py,
                          2026-10-20) shows it is misregistered against the recorded views (own-column MAE 13-22/255,
                          best shift >= 8 px, CAM_BACK ~10% zoom), because the recorded panoramas were made with the
                          OneBEV legacy warp, not the ideal sphere;
      onebev_inverse      exact inverse of that OneBEV legacy warp (LegacyInverseGeometry; own-column MAE ~1/255,
                          best shift 0, zoom 1.00), mask = in-band set;
      onebev_inverse_own  the same with the mask restricted to pixels whose canvas column the camera owns (drops the
                          neighbour-owned edge strips, where the canvas holds the neighbour's warp; MAE ~25-38/255).
  * mask[c] = the geometry's in-band set of camera c (where the resample lands inside the canvas band). One mask per
    clip and camera, applied to EVERY arm, the recorded reference included: pixels outside it are set to 0.5 gray
    before Inception / I3D. --mask-top-native-rows 52 also masks MagicDrive-V2's 26 padded rows (all arms).
  * arm kinds (--arm NAME=KIND[:ARG]):
      latents:ROOT[+ROOT2...]  our clip runs (clipNNN/latents.pt + summary.json, NNN = test_index): latents decoded
                               with the configured SD3.5 VAE and stitched exactly as clip_feats.py (blend --blend, the
                               paper's 128), then resampled into the six cameras;
      recband                  the recorded canvas of the same clip, exactly as clip_feats.py --real (the paper's
                               real.npz), resampled the same way -- the resample's cost on our side;
      views:RUNDIR             a six-camera run (clipNNN/{fake.pt uint8 [T, 6, 3, h, w], calib.pt, meta.json}, NNN =
                               OpenDWM/MagicDrive val index, mapped by --map; camera order OpenDWM FL, F, FR, BR, B, BL
                               unless meta.json has "camera_order"); frames picked by meta["generated_frames"]; views
                               that are not 256x448 are resized with bilinear antialias (MagicDrive-V2 450x800 is
                               0.5x of 1600x900 including its 26 edge-replicated top rows, so the resize lands on the
                               same 448x256 pixel grid; 0-2.7% of the masked pixels fall in those rows, see above).
    calib.pt of a views run must match the batch calibration (camera order check), otherwise the clip fails.
    --source-ref NAME=npz checks on every clip that the decoded / recorded canvas gives the paper's pano features.
  * interchange: every arm becomes uint8 [32, 6, 3, 256, 448] (resample output rounded like the saved baseline views),
    then x = u8 / 255 with x[~mask] = 0.5.
  * inc [n, 32, 6, 2048]: torchmetrics FrechetInceptionDistance(normalize=True) features of every frame and camera
    (the metric's own (x * 255).byte() -> Inception), as clip_feats.inc_float;
  * i3d [n, 2, 6, 400]: TATS I3D-400 logits per camera on windows frames 3-18 / 19-34, through the path of
    PanoPatchFrechetVideoDistance._accumulate_patches = OpenDWM FrechetVideoDistance (x * 2 - 1, bilinear 224x224
    squash, I3D); NuScenes6BandFrechetVideoDistance.update gives the same features (self-check).
  * dataset: frozen eval config + relaxed_sync (150 clips), with the 3D-box / HD-map renders switched off (light).
Camera axis of inc / i3d: bridge CAM_ORDER (CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
CAM_BACK_RIGHT). Analyse with cam_band_analyze.py (tests_analyze.py with FID pooled over 6 cameras x frames).
  python3 cam_band_feats.py --base-config cd_eval_frozen149_hpc3.json --out-dir 'D_{geometry}' \
      --geometry bridge,onebev_inverse,onebev_inverse_own --arm rec448=views:RUN --arm recband=recband ...
Writes D_<geometry>/feats_<NAME>.npz; existing files are kept (arm skipped).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

import relaxed_sync
from clip_feats import features as pano_features
from clip_feats import inc_float
from long_feats import I3D
from dwm_pano.datasets.pano_projection import pano_column_camera_indices
from dwm_pano.metrics.fvd_pano import PanoPatchFrechetVideoDistance
from dwm_pano.metrics.nuscenes6_fvd import NuScenes6BandFrechetVideoDistance
from dwm_pano.nuscenes6_bridge import (
    CAM_ORDER, PANO_CANVAS_OVERLAP_COLUMNS, apply_camera_sampling_grid, build_rig_sampling_grids, canvas_source_axes,
    ego_directions_to_pano_pixels, real_camera_pixel_directions_ego)
from dwm_pano.pano_stitch import stitch_pano_halves

FRAMES, CONTEXT = 35, 3
FIRST, LAST = 3, 34                     # scored clip frames, windows 3-18 and 19-34
OUT_HW = (256, 448)                     # UniMLVG native per-camera size
CANVAS_HW = (256, 4114)
GRAY = 0.5
ODWM_ORDER = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT")
MANIFESTS = ("dev149_manifest_20261018.json", "scene0018_manifest_20261019.json")
PROTOCOL = {"name": "nusc6band-256x448-masked", "resolution_hw": list(OUT_HW), "cams": list(CAM_ORDER),
            "frames": [FIRST, LAST], "windows": [[3, 18], [19, 34]], "mask": "in-band set of the geometry",
            "outside_mask": GRAY, "interchange": "uint8", "inception": "torchmetrics FID normalize=True",
            "i3d": "TATS i3d_pretrained_400, x*2-1, bilinear 224x224"}


# ----------------------------------------------------------------------------------------------- data and geometry
def load_val150(base_config, stride=20.0, light=True):
    """The 150-clip val dataset (frozen eval config + relaxed_sync) and the Val-150 rows with their dataset index.

    light: skip the 3D-box and HD-map renders (not used here; they cost ~8 s of the ~11 s per clip); the RGB panorama,
    its resize to the 256x2304 halves and the camera calibration are untouched (recband is checked against real.npz).
    """
    relaxed_sync.install()
    import dwm.common
    config = json.loads(Path(base_config).read_text())
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != FRAMES or ds["fps_stride_tuples"] != [[10, 20]]:
        raise RuntimeError("expected the frozen 35-frame protocol config")
    ds["fps_stride_tuples"] = [[10, stride]]
    if light:
        ds["enable_3dbox"] = ds["enable_hdmap"] = False
        tl = config["validation_dataset"]["transform_list"]
        config["validation_dataset"]["transform_list"] = [
            t for t in tl if t.get("old_key") not in ("3dbox_images", "hdmap_images")]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    base = dataset.base_dataset
    where = {(a["scene_token"], int(c[0]["timestamp"])): i for i, (c, a) in enumerate(zip(base.clips, base.clip_audit))}
    rows = []
    for m in MANIFESTS:
        for r in json.load(open(m))["clips"]:
            key = (r["scene_token"], int(r["anchor_timestamp_us"]))
            if key not in where:
                raise RuntimeError(f"{m} clip {r['test_index']} not in the relaxed dataset")
            rows.append(dict(r, relaxed_index=where[key]))
    rows.sort(key=lambda r: int(r["test_index"]))
    if [int(r["test_index"]) for r in rows] != list(range(150)) or len(dataset) != 150:
        raise RuntimeError(f"expected Val-150 = test_index 0..149 over a 150-clip dataset, got {len(dataset)}")
    collate = dwm.common.instantiate_config(config["validation_dataloader"])["collate_fn"]
    return config, dataset, collate, rows


def batch_rig(batch):
    """Real six-camera rig of a clip (CAM_ORDER), from frame 0; asserted constant over the clip."""
    rig = [batch[k][0].double().cpu().numpy() for k in
           ("pano_camera_transforms", "pano_camera_intrinsics", "pano_camera_image_size")]
    for x in rig:
        if not np.allclose(x, x[:1], atol=1e-6):
            raise RuntimeError("camera calibration changes within the clip")
    return [x[0] for x in rig]


class CamGeometry:
    """Bridge grids, band mask and canvas-column owner of every 256x448 camera pixel, cached per rig."""

    def __init__(self):
        rows, cols = CANVAS_HW
        self.axes = canvas_source_axes(half_size=(rows, (cols + PANO_CANVAS_OVERLAP_COLUMNS) // 2))
        self.cache = {}

    def __call__(self, c2e, K, size):
        key = (tuple(np.round(c2e, 4).ravel()), tuple(np.round(K, 3).ravel()), tuple(np.asarray(size).ravel()))
        if key not in self.cache:
            if len(self.cache) >= 16:
                raise RuntimeError("more than 16 distinct rigs; check the calibration")
            grids, valid, reports = build_rig_sampling_grids(
                c2e, K, size, CANVAS_HW, OUT_HW, band_limit=False, source_axes=self.axes)
            owner = np.zeros(valid.shape, np.int64)
            for c in range(6):  # owner of the canvas column a pixel samples (fixed Voronoi cuts of the hard seam)
                w, h = int(size[c][0]), int(size[c][1])
                d = real_camera_pixel_directions_ego(c2e[c], K[c], OUT_HW[0], OUT_HW[1], row_stride=h / OUT_HW[0],
                                                     column_stride=w / OUT_HW[1])
                tx, _, in_band = ego_directions_to_pano_pixels(d)
                if not np.array_equal(in_band, valid[c]):
                    raise RuntimeError("bridge mask and direction recomputation disagree")
                owner[c] = pano_column_camera_indices(tx - 0.5)  # continuous column, bucketized as the stitch does
            self.cache[key] = {"grids": grids, "valid": valid, "owner": owner, "reports": reports}
        return self.cache[key]


def _legacy_pixels(T, Km, c, rows, cols, crop=None):
    """OneBEV legacy camera pixel (OpenCV index coords at 900x1600) of pano points (pre-crop row, 9600 column).

    The arithmetic of dwm_pano.calibrated_stitch._onebev_legacy_grid_for_camera (the map that made the recorded
    panoramas and that stitch_views.py uses for the baselines), on 1-D point arrays, returning the raw pixel
    coordinates (that function replaces invalid points by 2.0, which a Newton solver cannot use) and the warp's crop
    constants. crop=None computes them from the given points, which must then be the full canvas mesh, as the stitch
    does. LegacyInverseGeometry checks this copy against the original on the canvas mesh.
    """
    import math
    from dwm_pano.calibrated_stitch import (
        ONEBEV_BACK_FOV_DEG, ONEBEV_OTHER_FOV_DEG, PANO_H_PRE_CROP, PANO_W, VERT_FOV_HALF_RAD, _onebev_angles)
    source_height, source_width = 900, 1600
    angles = _onebev_angles(T)
    angle_x, angle_y, roll_deg = angles[0, c, 0], angles[0, c, 1], angles[0, c, 2]
    tr = T[..., :3, 3]
    off = tr - tr.mean(dim=1, keepdim=True)
    du = Km[..., 0, 0] * (off[..., 0] / 5.5)
    dv = Km[..., 1, 1] * (off[..., 1] / 5.5)
    r_off = torch.sqrt(du.square() + dv.square())[0, c]
    fov = ONEBEV_BACK_FOV_DEG if c == 0 else ONEBEV_OTHER_FOV_DEG
    radius = source_width / 2.0 / math.tan(math.radians(fov) / 2.0) + r_off
    az = ((cols - PANO_W / 2.0 - 0.5) / PANO_W) * (2.0 * math.pi)
    el = -((rows - PANO_H_PRE_CROP / 2.0 - 0.5) / PANO_H_PRE_CROP) * (2.0 * VERT_FOV_HALF_RAD)
    x0 = radius * torch.cos(angle_y) * torch.sin(angle_x)
    y0 = radius * torch.cos(angle_y) * torch.cos(angle_x)
    z0 = radius * torch.sin(angle_y)
    alpha, beta, gamma = torch.cos(el) * torch.sin(az), torch.cos(el) * torch.cos(az), torch.sin(el)
    division = x0 * alpha + y0 * beta + z0 * gamma
    safe = torch.where(division.abs() > 1e-8, division, torch.ones_like(division))
    scale = radius.square() / safe
    vx, vy, vz = scale * alpha - x0, scale * beta - y0, scale * gamma - z0
    pxx, pxy = torch.cos(angle_x), -torch.sin(angle_x)
    delta_x = vx * pxx + vy * pxy
    pyx, pyy, pyz = -z0 * pxy, z0 * pxx, x0 * pxy - y0 * pxx
    norm = torch.sqrt(pyx.square() + pyy.square() + pyz.square()).clamp_min(1e-8)
    delta_y = (vx * pyx + vy * pyy + vz * pyz) / norm
    pixel_x = delta_x + (source_width + 1.0) / 2.0
    pixel_y = delta_y + (source_height + 1.0) / 2.0
    if crop is None:
        min_x = torch.maximum(torch.ones_like(pixel_x.amin()), torch.floor(pixel_x.amin()) - 1.0)
        min_y = torch.maximum(torch.ones_like(pixel_y.amin()), torch.floor(pixel_y.amin()) - 1.0)
        max_x = torch.minimum(torch.full_like(min_x, float(source_width)), torch.ceil(pixel_x.amax()) + 1.0)
        max_y = torch.minimum(torch.full_like(min_y, float(source_height)), torch.ceil(pixel_y.amax()) + 1.0)
        crop = (min_x, min_y, max_x, max_y)
    min_x, min_y, max_x, max_y = crop
    ccx, ccy = (max_x - min_x) / 2.0, (max_y - min_y) / 2.0
    dx, dy = pixel_x - min_x + 1.0 - ccx, pixel_y - min_y + 1.0 - ccy
    rr = roll_deg * (math.pi / 180.0)
    cos, sin = torch.cos(rr), torch.sin(rr)
    out_x = cos * dx - sin * dy + ccx + min_x
    out_y = sin * dx + cos * dy + ccy + min_y
    return out_x, out_y, division, crop


class LegacyInverseGeometry(CamGeometry):
    """Exact inverse of the OneBEV legacy stitch (2026-10-20, alternative to the nuscenes6_bridge geometry).

    The recorded panoramas were made with the OneBEV legacy warp (fixed 110 / 70 degree FOV sphere radius plus a
    translation heuristic, principal point at the image centre), not with the ideal calibrated sphere the
    nuscenes6_bridge inverts; on a nuScenes rig the two put the same pano point 20-65 native pixels apart. This finds,
    for every 256x448 camera pixel q (native OpenCV index coords u = (j + 0.5) * 1600 / 448 - 0.5, v likewise), the
    pano point that camera c's legacy warp maps to q (Newton on the analytic warp, started at the bridge's ideal
    solution), and samples the canvas there with the bridge's own canvas-axis inversion and sampler. Mask = converged,
    in front of the camera and inside the band (pre-crop pano row centre in [149.5, 749.5), the bridge's band rule).
    Where the solution's canvas column is owned by a neighbour (fixed Voronoi cuts) the canvas holds the neighbour's
    warp of its own image, so a residue remains there, as with the bridge.
    """

    def __call__(self, c2e, K, size):
        key = (tuple(np.round(c2e, 4).ravel()), tuple(np.round(K, 3).ravel()), tuple(np.asarray(size).ravel()))
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= 16:
            raise RuntimeError("more than 16 distinct rigs; check the calibration")
        from dwm_pano.calibrated_stitch import _onebev_legacy_grid_for_camera, prepare_gsplat_intrinsics
        from dwm_pano.datasets.pano_projection import PANO_CROP_TOP
        if any(tuple(int(v) for v in s) != (1600, 900) for s in size):
            raise RuntimeError("the legacy warp is defined at native 1600x900")
        ideal = super().__call__(c2e, K, size)
        T = torch.as_tensor(c2e, dtype=torch.float64)[None]
        Km = torch.stack([prepare_gsplat_intrinsics(torch.as_tensor(K[i], dtype=torch.float64), (900, 1600),
                                                    (900, 1600)) for i in range(6)])[None]
        rows_pre = torch.as_tensor(self.axes[0] + PANO_CROP_TOP, dtype=torch.float64)
        cols = torch.as_tensor(self.axes[1], dtype=torch.float64)
        R, C = torch.meshgrid(rows_pre, cols, indexing="ij")
        h, w = OUT_HW
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        qx = torch.as_tensor((jj + 0.5) * 1600.0 / w - 0.5)
        qy = torch.as_tensor((ii + 0.5) * 900.0 / h - 0.5)
        grids, valids, owners, reports = [], [], [], []
        for c in range(6):
            ref, ref_valid = _onebev_legacy_grid_for_camera(T, Km, c, R, C, (900, 1600))
            px, py, div, crop = _legacy_pixels(T, Km, c, R.flatten(), C.flatten())
            mine = torch.stack([(px + 0.5) * (2.0 / 1600) - 1.0, (py + 0.5) * (2.0 / 900) - 1.0], -1)
            v = ref_valid.flatten()
            if not torch.allclose(mine[v], ref.reshape(-1, 2)[v], atol=1e-9, rtol=0):
                raise RuntimeError(f"legacy warp copy differs from calibrated_stitch for camera {c}")
            # start: the ideal (calibrated-sphere) pano point of every pixel, the bridge's solution without band limit
            d = real_camera_pixel_directions_ego(c2e[c], K[c], h, w, row_stride=900.0 / h, column_stride=1600.0 / w)
            tx, ty, _ = ego_directions_to_pano_pixels(d)
            k = torch.as_tensor(tx - 0.5)
            r = torch.as_tensor(ty - 0.5 + PANO_CROP_TOP)
            g = torch.as_tensor(ideal["grids"][c], dtype=torch.float64)
            col = ((g[..., 0] + 1.0) / 2.0 * (CANVAS_HW[1] + 2.0) - 1.5).numpy()
            row = ((g[..., 1] + 1.0) / 2.0 * CANVAS_HW[0] - 0.5).numpy()
            eps = 0.25
            for _ in range(40):
                fx, fy, _, _ = _legacy_pixels(T, Km, c, r, k, crop)
                ex, ey = fx - qx, fy - qy
                fxr, fyr, _, _ = _legacy_pixels(T, Km, c, r + eps, k, crop)
                fxc, fyc, _, _ = _legacy_pixels(T, Km, c, r, k + eps, crop)
                a, b = (fxr - fx) / eps, (fxc - fx) / eps
                cc, d = (fyr - fy) / eps, (fyc - fy) / eps
                det = a * d - b * cc
                det = torch.where(det.abs() > 1e-12, det, torch.full_like(det, 1e-12))
                r = r - (d * ex - b * ey) / det
                k = k - (-cc * ex + a * ey) / det
                r = r.clamp(-2000.0, 2900.0)
            fx, fy, div, _ = _legacy_pixels(T, Km, c, r, k, crop)
            resid = torch.sqrt((fx - qx) ** 2 + (fy - qy) ** 2)
            kw = torch.remainder(k, 9600.0)
            valid = (resid < 1e-3) & (div > 1e-8) & (r >= PANO_CROP_TOP - 0.5) & (r < PANO_CROP_TOP + 600 - 0.5)
            row_c = np.interp(r.numpy().ravel() - PANO_CROP_TOP, self.axes[0], np.arange(CANVAS_HW[0]))
            col_c = np.interp(kw.numpy().ravel(), self.axes[1], np.arange(CANVAS_HW[1]))
            gx = np.where(valid.numpy().ravel(), (col_c + 1.5) / (CANVAS_HW[1] + 2.0) * 2.0 - 1.0, 0.0)
            gy = np.where(valid.numpy().ravel(), (row_c + 0.5) / CANVAS_HW[0] * 2.0 - 1.0, 0.0)
            grids.append(np.stack([gx, gy], -1).reshape(h, w, 2))
            valids.append(valid.numpy())
            owners.append(pano_column_camera_indices(kw.numpy()))
            reports.append({"camera": CAM_ORDER[c], "newton_max_resid_px_in_mask": float(resid[valid].max()),
                            "unconverged_fraction": float((resid >= 1e-3).double().mean()),
                            "mean_shift_vs_bridge_canvas_px": float(np.hypot(
                                (col_c.reshape(h, w) - col)[valid.numpy() & ideal["valid"][c]],
                                (row_c.reshape(h, w) - row)[valid.numpy() & ideal["valid"][c]]).mean()),
                            "valid_fraction": float(valid.double().mean()),
                            "bridge_valid_fraction": float(ideal["valid"][c].mean())})
        self.cache[key] = {"grids": np.stack(grids), "valid": np.stack(valids), "owner": np.stack(owners),
                           "reports": reports}
        return self.cache[key]


class LegacyInverseOwnGeometry(LegacyInverseGeometry):
    """LegacyInverseGeometry with the mask restricted to camera pixels whose canvas column the camera owns itself.

    There the resample is exact (validated: MAE ~1/255 against the recorded views); the neighbour-owned strips at the
    camera edges hold the neighbour's warp of its own image (MAE ~25-38/255 on recorded data) and are masked for every
    arm instead.
    """

    def __call__(self, c2e, K, size):
        key = ("own",) + (tuple(np.round(c2e, 4).ravel()), tuple(np.round(K, 3).ravel()), tuple(np.asarray(size).ravel()))
        if key not in self.cache:
            geo = super().__call__(c2e, K, size)
            own = geo["owner"] == np.arange(6)[:, None, None]
            self.cache[key] = dict(geo, valid=geo["valid"] & own)
        return self.cache[key]


GEOMETRIES = {"bridge": CamGeometry, "onebev_inverse": LegacyInverseGeometry,
              "onebev_inverse_own": LegacyInverseOwnGeometry}


def canvas_to_cams(canvas, geo):
    """canvas [T, 3, 256, 4114] in [0, 1] -> uint8 [T, 6, 3, 256, 448] in CAM_ORDER (zero outside the band)."""
    views = [apply_camera_sampling_grid(canvas, geo["grids"][c], geo["valid"][c]) for c in range(6)]
    return (torch.stack(views, 1).clamp(0, 1) * 255).round().to(torch.uint8)


def recorded_canvas(batch, device, blend):
    """The recorded canvas of clip frames 3..34, as clip_feats.py --real (stitch blend from tests_feats.sbatch)."""
    return stitch_pano_halves(batch["vae_images"][:, CONTEXT:FRAMES].to(device), blend_width=blend)[0, :, 0].clamp(0, 1)


def make_decoder(config, device):
    import dwm.functional
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    pc = config["pipeline"]
    vae = AutoencoderKL.from_pretrained(pc["pretrained_model_name_or_path"], subfolder="vae").to(device).eval()
    processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    shift = vae.config.shift_factor or 0
    chunk = pc.get("common_config", {}).get("memory_efficient_batch", -1)

    def decode(latents):  # (T, V, C, h, w) -> (T, V, 3, H, W) in [0, 1], as clip_feats.py
        t, v = latents.shape[:2]
        image = dwm.functional.memory_efficient_split_call(
            vae, latents.flatten(0, 1).to(device, dtype=vae.dtype),
            lambda block, x: block.decode(x / block.config.scaling_factor + shift, return_dict=False)[0], chunk)
        return processor.postprocess(image, output_type="pt").float().unflatten(0, (t, v))
    return decode


def ours_canvas(roots, row, decode, blend):
    pos = int(row["test_index"])
    d = next((Path(r) / f"clip{pos:03d}" for r in roots if (Path(r) / f"clip{pos:03d}" / "latents.pt").exists()), None)
    if d is None:
        raise FileNotFoundError(f"clip{pos:03d} not in {roots}")
    summary = json.load(open(d / "summary.json"))
    if summary["scene_token"] != row["scene_token"] or int(summary["anchor_timestamp_us"]) != int(row["anchor_timestamp_us"]):
        raise RuntimeError(f"{d}: not the manifest clip")
    payload = torch.load(d / "latents.pt", map_location="cpu", weights_only=True)
    latents = payload["latents"]
    if latents.shape[0] != FRAMES or payload.get("context_frames") != CONTEXT:
        raise ValueError(f"{d}: latents {tuple(latents.shape)}")
    canvas = stitch_pano_halves(decode(latents[CONTEXT:])[None], blend_width=blend)[0, :, 0].clamp(0, 1)
    return canvas, int(summary["seed"]), str(d)


def run_views(run, clip_index, row, rig, device):
    """uint8 [32, 6, 3, 256, 448] CAM_ORDER views of a six-camera run for clip frames 3..34, and the run's seed."""
    d = Path(run) / f"clip{clip_index:03d}"
    meta = json.loads((d / "meta.json").read_text())
    order = tuple(meta.get("camera_order", ODWM_ORDER))
    perm = [order.index(c) for c in CAM_ORDER]
    fake = torch.load(d / "fake.pt", map_location="cpu", weights_only=True)
    frames = list(meta["generated_frames"])
    if fake.dtype != torch.uint8 or fake.ndim != 5 or fake.shape[0] != len(frames) or fake.shape[1] != 6:
        raise ValueError(f"{d}: fake {fake.dtype} {tuple(fake.shape)} vs {len(frames)} frames")
    views = fake[[frames.index(f) for f in range(FIRST, LAST + 1)]][:, perm].to(device)
    calib_path = d / "calib.pt"
    if calib_path.exists():  # camera order and rig check against the batch calibration
        calib = torch.load(calib_path, weights_only=True)
        c2e = calib["camera_transforms"].double().numpy()[perm]
        K = calib["camera_intrinsics"].double().numpy()[perm]
        size = calib["image_size"].numpy()[perm]
        if np.abs(c2e[:, :3, :3] - rig[0][:, :3, :3]).max() > 1e-3 or np.abs(c2e[:, :3, 3] - rig[0][:, :3, 3]).max() > 1e-2 \
                or np.abs(K - rig[1]).max() > 0.5 or not np.array_equal(size, rig[2].astype(np.int64)):
            raise RuntimeError(f"{d}: calib.pt does not match the clip's rig (camera order {order})")
    if tuple(views.shape[-2:]) != OUT_HW:
        x = views.flatten(0, 1).float() / 255.0
        x = F.interpolate(x, size=OUT_HW, mode="bilinear", antialias=True, align_corners=False)
        views = (x.clamp(0, 1) * 255).round().to(torch.uint8).unflatten(0, views.shape[:2])
    return views, int(meta.get("seed", -1)), str(d)


# ------------------------------------------------------------------------------------------------------- features
def masked_float(views_u8, valid):
    """uint8 [T, 6, 3, H, W] -> float in [0, 1] with every pixel outside the clip's band mask set to 0.5."""
    x = views_u8.float() / 255.0
    m = torch.as_tensor(valid, device=x.device)[None, :, None]
    return torch.where(m, x, torch.full_like(x, GRAY))


@torch.no_grad()
def cam_i3d(fvd, clip):
    """(16, 6, 3, H, W) in [0, 1] -> (6, 400): PanoPatchFrechetVideoDistance._accumulate_patches up to I3D."""
    resized = []
    for c in range(clip.shape[1]):
        p = clip[None, :, c] * 2 - 1
        p = F.interpolate(p.flatten(0, 1), size=fvd.target_resolution, mode="bilinear").unflatten(0, p.shape[:2])
        resized.append(p)
    return fvd.inception(torch.cat(resized, dim=0).transpose(1, 2)).float()


@torch.no_grad()
def cam_features(fid, fvd, x):
    """x (32, 6, 3, H, W) masked, in [0, 1] -> inc (32, 6, 2048), i3d (2, 6, 400)."""
    t, v = x.shape[:2]
    inc = inc_float(fid, x.flatten(0, 1)).unflatten(0, (t, v))
    i3d = torch.stack([cam_i3d(fvd, x[16 * w:16 * w + 16]) for w in range(2)])
    return inc, i3d


def cam_self_check(fid, fvd, x, inc, i3d, device):
    """The saved features must give NuScenes6BandFrechetVideoDistance's and FrechetInceptionDistance's own states."""
    m = NuScenes6BandFrechetVideoDistance(I3D, sequence_count=16).to(device)
    m.update(x[None, :16], real=True)
    ok_v = torch.allclose(m.real_features_sum, i3d[0].double().sum(0), rtol=1e-4, atol=1e-3) and \
        torch.allclose(m.real_cam_sum, i3d[0].double(), rtol=1e-4, atol=1e-3)
    f = FrechetInceptionDistance(normalize=True).to(device)
    f.update(x[:16].flatten(0, 1), real=True)
    same = inc_float(fid, x[:16].flatten(0, 1), batch=16 * x.shape[1]).double()
    ok_i = torch.allclose(f.real_features_sum, same.sum(0), rtol=1e-4, atol=1e-3)
    diff = (same - inc[:16].flatten(0, 1).double()).abs().max().item()
    print(f"SELF_CHECK cam i3d {'ok' if ok_v else 'FAIL'} inception {'ok' if ok_i else 'FAIL'} "
          f"(batch-32 vs batch-96 max abs feature diff {diff:.2e})", flush=True)
    if diff > 1e-2 or not (ok_v and ok_i):
        raise RuntimeError("camera feature extraction differs from the metric objects")


_REF_CACHE = {}


def pano_source_check(name, canvas, ref_path, pos, fid, fvd_pano, verbose):
    """Our canvas must be the canvas the paper's Val-150 features were computed from (decode + stitch identity)."""
    if ref_path not in _REF_CACHE:
        z = np.load(ref_path)
        _REF_CACHE[ref_path] = {"test_index": list(z["test_index"]), "inc": z["inc"], "i3d": z["i3d"]}
    z = _REF_CACHE[ref_path]
    k = z["test_index"].index(pos)
    inc, i3d = pano_features(fid, fvd_pano, canvas[None, :, None])
    di = np.abs(inc.cpu().numpy() - z["inc"][k]).max()
    dv = np.abs(i3d.cpu().numpy() - z["i3d"][k]).max()
    scale_i, scale_v = np.abs(z["inc"][k]).mean(), np.abs(z["i3d"][k]).mean()
    # same batching as clip_feats.py, so only kernel-level float noise is allowed (batch-size noise there is ~1e-3)
    ok = di < 1e-2 and dv < 1e-2 * max(1.0, scale_v)
    if verbose or not ok:
        print(f"SOURCE_CHECK {name} clip{pos:03d} vs {ref_path}: inc max abs diff {di:.3e} (mean |x| {scale_i:.3f}), "
              f"i3d {dv:.3e} (mean |x| {scale_v:.3f}) {'ok' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise RuntimeError(f"{name}: canvas differs from the one behind {ref_path}")
    return float(di), float(dv)


# ------------------------------------------------------------------------------------------------------------ main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--out-dir", required=True, help="may contain {geometry}")
    ap.add_argument("--arm", action="append", required=True, help="NAME=latents:R1[+R2] | NAME=recband | NAME=views:RUN")
    ap.add_argument("--map", default="val150_scene_map.json", help="six-camera run clip index -> Val-150 test_index")
    ap.add_argument("--blend", type=int, default=128, help="stitch cross-fade of our and the recorded canvas (paper 128)")
    ap.add_argument("--source-ref", action="append", default=[], help="NAME=pano feature npz to check the canvas against")
    ap.add_argument("--positions", default="", help="smoke tests only: a subset of test indices")
    ap.add_argument("--geometry", default="bridge",
                    help="comma list of " + ",".join(sorted(GEOMETRIES)) + ": bridge = nuscenes6_bridge (ideal "
                         "calibrated sphere); onebev_inverse = inverse of the OneBEV legacy warp that made the recorded "
                         "panoramas. Several geometries share one decode; --out-dir must then contain {geometry}")
    ap.add_argument("--mask-top-native-rows", type=int, default=0,
                    help="also mask camera rows whose centre lies in native rows [0, N) for every arm (MagicDrive-V2: "
                         "52 = its 26 edge-replicated rows at 450x800)")
    a = ap.parse_args()
    geoms = a.geometry.split(",")
    top = ((np.arange(OUT_HW[0]) + 0.5) * 900.0 / OUT_HW[0] < a.mask_top_native_rows)[None, :, None]
    if any(g not in GEOMETRIES for g in geoms) or (len(geoms) > 1 and "{geometry}" not in a.out_dir):
        raise ValueError(f"bad --geometry {a.geometry} / --out-dir {a.out_dir}")
    out_dirs = {g: Path(a.out_dir.format(geometry=g)) for g in geoms}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    arms = {}
    for spec in a.arm:
        name, kind = spec.split("=", 1)
        kind, _, arg = kind.partition(":")
        if kind not in ("latents", "recband", "views"):
            raise ValueError(spec)
        todo = [g for g in geoms if not (out_dirs[g] / f"feats_{name}.npz").exists()]
        for g in set(geoms) - set(todo):
            print(f"SKIP {name} {g} (exists)", flush=True)
        if todo:
            arms[name] = {"kind": kind, "arg": arg, "geoms": todo, "seed": [], "src": [],
                          "inc": {g: [] for g in todo}, "i3d": {g: [] for g in todo}}
    if not arms:
        return
    refs = dict(s.split("=", 1) for s in a.source_ref)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    config, dataset, collate, rows = load_val150(a.base_config)
    if a.positions:
        keep = {int(p) for p in a.positions.split(",")}
        rows = [r for r in rows if int(r["test_index"]) in keep]
    by_test = {}
    for k, v in json.load(open(a.map)).items():
        by_test[int(v["test_index"])] = (int(k), v["scene_token"])
    decode = make_decoder(config, device) if any(v["kind"] == "latents" for v in arms.values()) else None
    fid = FrechetInceptionDistance(normalize=True).to(device)
    fvd = PanoPatchFrechetVideoDistance(I3D, sequence_count=16).to(device)
    geometry = {g: GEOMETRIES[g]() for g in geoms}
    checked = set()
    t0 = time.time()
    for n, row in enumerate(rows):
        pos = int(row["test_index"])
        batch = collate([dataset[int(row["relaxed_index"])]])
        if int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"clip {pos}: dataset clip is not the manifest clip")
        rig = batch_rig(batch)
        geo = {g: geometry[g](*rig) for g in geoms}
        if a.mask_top_native_rows:
            geo = {g: dict(v, valid=v["valid"] & ~top) for g, v in geo.items()}
        for name, arm in arms.items():
            canvas = u8 = None
            if arm["kind"] == "views":
                ci, token = by_test[pos]
                if token != row["scene_token"]:
                    raise RuntimeError(f"map: clip {ci} is scene {token}, Val-150 {pos} is {row['scene_token']}")
                u8, seed, src = run_views(arm["arg"], ci, row, rig, device)
            else:
                if arm["kind"] == "recband":
                    canvas, seed, src = recorded_canvas(batch, device, a.blend), -1, "recorded canvas"
                else:
                    canvas, seed, src = ours_canvas(arm["arg"].split("+"), row, decode, a.blend)
                if name in refs:  # every clip: the canvas is the one behind the paper's Val-150 pano features
                    d = pano_source_check(name, canvas, refs[name], pos, fid, fvd, name not in checked)
                    arm["srcdiff"] = [max(x, y) for x, y in zip(arm.get("srcdiff", [0.0, 0.0]), d)]
            for g in arm["geoms"]:
                views = u8 if canvas is None else canvas_to_cams(canvas, geo[g])
                x = masked_float(views, geo[g]["valid"])
                inc, i3d = cam_features(fid, fvd, x)
                if (name, g) not in checked:
                    cam_self_check(fid, fvd, x, inc, i3d, device)
                    checked.add((name, g))
                arm["inc"][g].append(inc.cpu().numpy())
                arm["i3d"][g].append(i3d.cpu().numpy())
            checked.add(name)
            arm["seed"].append(seed)
            arm["src"].append(src)
        if (n + 1) % 10 == 0 or n + 1 == len(rows):
            print(f"[camfeats] {n + 1}/{len(rows)} {time.time() - t0:.0f}s rigs="
                  + ",".join(f"{g}:{len(geometry[g].cache)}" for g in geoms), flush=True)
    for name, arm in arms.items():
        roots = sorted({str(Path(s).parent) for s in arm["src"]}) if arm["kind"] != "recband" else ["recorded canvas"]
        for g in arm["geoms"]:
            out = out_dirs[g] / f"feats_{name}.npz"
            tmp = str(out) + ".tmp.npz"
            np.savez(tmp, inc=np.stack(arm["inc"][g]).astype(np.float32), i3d=np.stack(arm["i3d"][g]).astype(np.float32),
                     test_index=np.array([int(r["test_index"]) for r in rows]),
                     scene_token=np.array([r["scene_token"] for r in rows]), seed=np.array(arm["seed"]),
                     manifest="+".join(MANIFESTS), arm_root=f"{arm['kind']}:" + "+".join(roots),
                     protocol=json.dumps(dict(PROTOCOL, blend=a.blend, geometry=g,
                                              mask_top_native_rows=a.mask_top_native_rows)))
            os.replace(tmp, out)
            print(f"SAVED {out} clips={len(rows)} seeds={sorted(set(arm['seed']))}"
                  + (f" source check max abs diff inc {arm['srcdiff'][0]:.2e} i3d {arm['srcdiff'][1]:.2e} over all "
                     f"clips" if "srcdiff" in arm else ""), flush=True)
    print("CAMFEATS_DONE", flush=True)


if __name__ == "__main__":
    main()
