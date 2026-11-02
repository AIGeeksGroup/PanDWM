# Data preparation

PanDWM is trained on the nuScenes train split (700 scenes) and evaluated on the validation split (150 scenes). The panoramic bands, the condition images and the 12 Hz index files below are derived from nuScenes; use them under the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).

## Expected layout

The configs and job scripts read data from `$ROOT/pandwm_data/` and `$ROOT/pandwm_full_train_700_20260929/` (`ROOT=/path/to/pandwm_root`):

```text
pandwm_data/
  stable-diffusion-3.5-medium/                                # VAE and text encoders
  nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl          # validation index (evaluation)
  rgb/      bbox/      hdmap/                                  # validation panoramas and conditions
pandwm_full_train_700_20260929/
  rgb/      bbox/      hdmap/                                  # train panoramas and conditions
pandwm_recovered_metadata_20260929/
  nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl        # train index
```

Panoramas are named by timestamp; every RGB timestamp has a matching `bbox` and `hdmap` image.

## Sources

| Data | Source | Notes |
|---|---|---|
| Train panoramas | [`zeyuren2002/nuscene-360_train`](https://huggingface.co/datasets/zeyuren2002/nuscene-360_train), revision `72a55450cf3c7040657d45828ff11f5051f83d55`, `train/` | 166 tar shards, 218.6 GB, 165,280 frames |
| Box and map conditions (train and validation) | same repository, `bbox/` and `hdmap/` | 201 shards each, 52.9 GB together, 200,644 frames each |
| Validation panoramas | [`zeyuren2002/nuscene_360_val`](https://huggingface.co/datasets/zeyuren2002/nuscene_360_val), `rgb.tar` | 46.2 GB, 35,364 frames |
| Train index | [`Arlolo0/OmniNWM`](https://huggingface.co/Arlolo0/OmniNWM), revision `223f11f1728b7147ad115ca1aa45d85f3fe2a442`, `pickle/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl` | SHA-256 `cf4da7995c68a356a7604790c92d6c50d6458281c75948a11ff33ec9cfca399e` |
| Validation index | TODO: release location | 341,967,258 bytes, SHA-256 `66aa82b0d766ed4766a22526a372a4a1ee94d021319802adcc65c273f2e41dd4`; 35,364 frames of 150 scenes |

The validation index used for every reported number differs from the public file of the same name in `Arlolo0/OmniNWM` (347,419,822 bytes). Use the file with the checksum above.

Each index entry holds the six camera calibrations, the ego pose, the 12 Hz interpolated 3D boxes with instance ids, the location and a text caption.

## Evaluation clips

The clip lists are in `pandwm_code/PanDWM_eval149_20261011/`. Clips are 35 frames at 10 Hz (3 recorded history frames and two 16-frame windows), except Test-L.

| Set | Use | File |
|---|---|---|
| Val-150 | Comparison with other video models | `dev149_manifest_20261018.json` (149 scenes) + `scene0018_manifest_20261019.json` |
| Dev-78 | Ablations and checkpoint selection | `dev78_manifest_20261018.json` |
| Test-S | Test-set check and both Gaussian tables | `tests249_manifest_20261024.json` (= `tests213_manifest_20261018.json` + `tests_ext36_manifest_20261024.json`) |
| Test-L | Long videos, 3 recorded + 12 × 16 generated frames | `review149_test71_long12_20261016.json` + `testl_ext12_long12_20261024.json` (`testl83_long12_20261024.json` lists all 83) |

The generated-scene Gaussian table uses the 83 Test-S clips that start 10 s into each scene.

## Other inputs

- **Gaussian decoder training** uses the train index without the 5 held-out labelled scenes (`nuscenes_interp_12Hz_infos_train_minus_holdout5.pkl`, 695 scenes).
- **LiDAR depth** for the depth metrics comes from the official nuScenes LiDAR data (`dwm_pano/recon/` in `PanDWM-cov3-20261101`; `gen_depth_pack.py` packs it for the generated scenes).
