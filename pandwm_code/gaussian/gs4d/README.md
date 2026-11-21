# 4D Gaussian decoder training

The 4D training decodes several frames of a clip together. Each step decodes F = 3 frames (stride 2-4 frames) of a 19-frame clip from their latents, each in its own ego frame, with gradients kept for all of them. A Gaussian is dynamic if its centre lies in a 3D box of its frame, otherwise it is static. The static Gaussians of the frames are moved into a common frame, and three losses are applied:

- `own`: per-frame RGB L1 + LPIPS (plus depth / range L1 where depth labels exist);
- `composite`: the static Gaussians of all frames moved into frame t plus the dynamic ones of t, rendered at t's cameras and compared with frame t;
- `loo` (leave-one-out): the static Gaussians of the other frames only, rendered at t on static, covered pixels;
- `motion` (optional, off by default): the dynamic Gaussians of frame s carried to t by the rigid motion of the same tracked box, compared on t's dynamic pixels.

Code: `pandwm_code/src/dwm_pano/gs4d/` (`pipeline.py` is the training step; `pipeline_depth.py`, `pipeline_da3.py` and `pipeline_relabel.py` add log-depth terms, Depth Anything 3 pseudo-depth and the test-time relabel; `dataset.py`, `geometry.py` and `evaluate_4d.py` hold the box dataset, the rigid transforms and the held-out-frame evaluation). Configs are in `configs/` (written by `make_config.py`, `make_formal_config.py` and `make_c2_config.py`; `make_config.py --motion` switches on the box motion term); the job scripts are in `scripts/` and `scripts_hpc3/`.

Run (after adjusting the paths in the job scripts and their `common_env*.sh`):

```bash
RUN_TAG=train_v1 STEPS=15000 sbatch scripts/train_hpc2.sbatch                  # CONFIG=<file in configs/> selects another config
CHECKPOINT=<run dir>/checkpoints/15000.pth NAME=v1_15000 sbatch scripts/eval_hpc2.sbatch
STEPS=5000 sbatch scripts_hpc3/train_c2.sbatch                               # Depth Anything 3 variant (also scripts_hpc3/train_formal_F1.sbatch)
```
