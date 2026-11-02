# PanDWM 4D Gaussian decoder training (2026-10-02)

WorldSplat-style 4D training for the existing PanDWM Gaussian decoder. **New files only**: the
package `dwm_pano_4d` subclasses / imports `dwm_pano` from
[`../PanDWM-gs-aa-ft`](../PanDWM-gs-aa-ft) and changes nothing there.

## Why

The current decoder (G2-AA) is trained one frame at a time at its own cameras. Its round-trip
PSNR is high, but its geometry is barely constrained (decoder depth error is about 5 m), so
pooling frames into a 4D scene ghosts. WorldSplat (`references/worldsplat`,
`gs_decoder/model.py::render_with_static_dynamic`) instead trains the decoder so that **static
Gaussians from several frames, moved into a common frame, must explain each frame together**.
Ego motion between those frames provides the parallax the single-frame loss lacks.

## What a training step does (`src/dwm_pano_4d/pipeline.py`)

F = 3 frames of a 19-frame clip, stride 2–4 frames (0.2–0.4 s apart at 10 Hz, so the full span
is 0.4–0.8 s), are decoded from GT latents, each in its own ego frame, with gradients kept for
all of them. A Gaussian is **dynamic** if its centre lies in a 3D box of its frame (the classes
in `dataset.py`, plus a 0.3 m margin); otherwise it is **static**.

| Loss | Origin | Definition |
|---|---|---|
| `own` | G2 | per-frame RGB L1 + LPIPS (+ depth/range L1 when a depth mirror exists) |
| `composite` | WorldSplat | static(all F frames) moved into frame t, plus dynamic(t), rendered at t's cameras, compared with frame t |
| `loo` | new | static of the **other** frames only, rendered at t, on static, covered pixels: a held-out-view loss |
| `motion` | new, off by default | dynamic Gaussians of frame s carried to t by the rigid motion of the same tracked box, on t's dynamic pixels |

Differences from WorldSplat:
- The dynamic split comes from 3D boxes, not a SegFormer-supervised segmentation head. This
  dataset has no semantic masks, and boxes are already model conditions.
- Panorama rays and the nuScenes six-camera stitched supervision are kept from G2.
- The leave-one-out and box-motion terms are additions.

## Files

| Path | Role |
|---|---|
| `src/dwm_pano_4d/geometry.py` | quaternion / rigid transforms of (N,14) Gaussians, points-in-boxes, box poses |
| `src/dwm_pano_4d/dataset.py` | `PanoNuScenesBoxDataset`: adds per-frame ego-frame boxes and clip-local track ids |
| `src/dwm_pano_4d/pipeline.py` | `PanoGaussianDecoder4DPipeline` (only `train_step` overridden) |
| `src/dwm_pano_4d/evaluate_4d.py` | held-out-frame novel-view PSNR / coverage, for G2 vs 4D checkpoints |
| `make_config.py` | builds configs from the G2-AA config (warm start from its `250.pth`) |
| `configs/gs4d_hpc2*.json`, `configs/gs4d_hpc3.json` | HPC2 (no depth mirror, depth losses off), HPC3 (`--depth`) |
| `scripts/*.sbatch` | smoke (debug, 20 steps + 5-clip eval), train (A800), eval |
| `tests/` | CPU tests: geometry, and box extraction vs the repo's corner convention |

## Run

```bash
python3 tests/test_geometry.py && python3 tests/test_dataset_boxes.py      # local, CPU
sbatch scripts/smoke_hpc2.sbatch                                            # HPC2 debug
RUN_TAG=train_v1 STEPS=15000 sbatch scripts/train_hpc2.sbatch               # HPC2 A800
CHECKPOINT=$HOME/pandwm_runs/gs4d_20261002/train_v1/checkpoints/15000.pth NAME=v1_15000 \
  sbatch scripts/eval_hpc2.sbatch
```

The acceptance metric is `psnr_pooled` / `coverage_pooled` from `evaluate_4d.py` on the frozen
val clips, compared with the G2-AA baseline under the same command. `psnr_own` (round trip)
must not drop materially. Phase 2 sets `--motion 0.5` in `make_config.py` once phase 1 helps.
