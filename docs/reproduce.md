# Reproducing the paper results

All commands are SLURM job scripts as we ran them. Every script starts with a header comment that documents its variables; the commands below give the settings used for the paper.

Conventions:

- `ROOT=/path/to/pandwm_root` (see [Paths](../README.md#paths)); code is read from `$ROOT/pandwm_code/<tree>`, outputs go to `$ROOT/pandwm_runs/...`.
- `CF=$ROOT/pandwm_code/PanDWM-cfdmd-A-20261008`, `EV=$ROOT/pandwm_code/PanDWM_eval149_20261011`.
- Video checkpoint: `pandwm_video_v84i4250_ema.pth`; Gaussian decoder: `pandwm_gs_decoder_G3b_2000.pth`; refiner: `pandwm_render_refiner_v1_4000.pth`.

## 1. Video generation

Fixed settings of the paper model: EMA weights at iteration 4250, stochastic four-step sampler with steps `[1000, 750, 500, 250]`, no classifier-free guidance, panorama blend width 128, clean history.

| Set | Command (run from `$CF`) |
|---|---|
| Test-S, first 213 clips | `TAG=v84i4250 CKPT=<video ckpt> SEED=42 sbatch --array=0-7%4 scripts/tests_gen.sbatch` |
| Test-S, 36 extension clips | same with `MANIFEST=$EV/tests_ext36_manifest_20261024.json` |
| Val-150 (Dev-149 part) | same with `MANIFEST=$EV/dev149_manifest_20261018.json STRIDE=20`, seeds 42, 1000 and 2026 |
| Val-150 (scene-0018) | same with `MANIFEST=$EV/scene0018_manifest_20261019.json`, run through `relaxed_sync.py` (see its docstring) |
| Dev-78 | `MANIFEST=$EV/dev78_manifest_20261018.json STRIDE=20` |
| Test-L, 71 scenes | `TAG=v84i4250 CKPT=<video ckpt> SEED=42 sbatch --array=0-7%4 scripts/long_video_testl.sbatch` |
| Test-L, 12 extension scenes | `LIST=$EV/testl_ext12_long12_20261024.json ... sbatch scripts/long_video_testl_ext.sbatch` |
| 50-step causal teacher (reference rows) | `ENGINE=teacher TAG=teacher50 CKPT=<teacher ckpt> CHUNK=4` with the same scripts |
| Setting G (no recorded history) | `TAG=v84i4250G HISTORY=none` |

Each clip directory holds `latents.pt`, `gt_vs_generated.mp4` and `summary.json`.

## 2. Video metrics

| Result | Scripts |
|---|---|
| FID, FVD-1, FVD-2 (panoramic band) | Features: `scripts/tests_feats.sbatch` (`clip_feats.py`); metrics: `$EV/tests_analyze.py`. Val-150 merge and analysis: `scripts/val150_analyze.sbatch`. The frozen 149-clip protocol is `$EV/compute_eval149_metrics.py` with `$EV/cd_eval_frozen149_hpc3.json`. |
| Six-camera metrics (supplementary) | `scripts/camspace_feats.sbatch`, `scripts/camspace_analyze.sbatch` |
| Flicker and sharpness ratios | `$EV/flicker_diag.py` (`flicker_diag_subset.py` for subsets) |
| Controllability | `scripts/ctrl_pred.sbatch` (`$EV/ctrl_predict.py`, `ctrl_gt.py`), then `$EV/ctrl_analyze.py` |
| Long-video per-window metrics (Test-L) | `scripts/testl_analyze.sbatch`, `scripts/testl_analyze_ext.sbatch` (`long_feats.py`, `long_analyze2.py`) |
| Ablations (Dev-78, 500 iterations each) | `pandwm_code/PanDWM-ablation-20261021`: configs from `scripts/make_ablation_configs.py`, training `scripts/train_abl.sbatch`, evaluation `scripts/abl_analyze.sbatch`, table `scripts/abl_table.py` |
| Speed | `pandwm_code/PanDWM_long_horizon_20260928/src/dwm_pano/benchmark_speed.py` |

Baseline outputs (UniMLVG, OpenDWM (SD3.5), DreamForge, DiST-4D, OmniNWM, and the front-view models of the qualitative comparison) were generated with their official code and scored with the same feature, flicker and controllability scripts; the baseline runners are not part of this release.

## 3. Gaussian reconstruction (249 Test-S clips)

Code: `pandwm_code/PanDWM-cov3-20261101`. Scene construction is `gsaa_src/dwm_pano/gs_scene_v3.py` (`SceneBuilder`); rendering and scoring are in `gsaa_src/dwm_pano/recon/` (`targets.py`, `masks.py`, `render_ours.py`, `score.py`).

1. Render (`scripts/gaussian/tab3_reconstruction/g3b_render_v2.sbatch`, three array tasks) with `TAG=g3bf CODE=PanDWM-cov3-20261101` and the decoder checkpoint, using these scene overrides:
   - 3D: `{"plane_offset_px": 1.0, "seam_bridge": 16, "seam_bridge_search": 6, "seam_bridge_layer": "under", "edge_extend": 8, "band_extend": 8}` (Depth Anything 3 per frame);
   - 4D: the same plus `{"da3_window": "clip", "da3_fuse": true}` (frames 0 to 20 of each clip).
2. Score: `JOBFILE=score_jobs_g3bf.txt sbatch --array=0-11 g3b_score.sbatch`.
3. Merge the 213-clip and 36-clip parts: `python3 merge_g3b.py g3bf`. It reuses the summary functions of `merge_249.py`, which is not included; the aggregation is the `summarize` code of `recon/score.py`.

Only the recorded frames up to the input frame are loaded. Baselines (Omni-Scene, ReconDrive, DrivingForward, MCSplat) were rendered with their official code and scored with the same `score.py`, targets and masks.

## 4. Gaussians on generated scenes (83 Test-S clips)

Code on `PYTHONPATH`: `PanDWM-ft3-20261030` first, then `PanDWM-gs-aa-ft` and `PanDWM-gs4d-20261002`; decoder config `PanDWM-gs4d-20261002/configs/gs_review_hpc3.json`. Input: the Test-S generations of section 1 (`tests_20261018/v84i4250_s42`).

- **Gaussians only:** `gsaa_src/dwm_pano/eval_gs_shift.py` with `--modes 4d --da3-relabel --da3-mode scale --da3-window clip --da3-fuse --plane-offset-px 1 --drop-bottom-rows 2 --fps-stride 10,5 --box-reading mmdet3d_v0x --camera-time-poses --shifts 0,1,2,4`. Run `scripts/gaussian/tab4_generated_scenes/tab5_ft3_hpc3.sbatch` for the 71 clips and `gsgen83_hpc3.sbatch` with `MODE=g3b` for the 12 extension clips and the 83-clip merge.
- **With the render refiner:**
  1. export the renders: `MODE=export SET=test` (and `SET=testext`) `sbatch --array=0-5 d2v2_gs_hpc3.sbatch` (in `PanDWM-ft3-20261030/scripts/` and `scripts/gaussian/tab4_generated_scenes/`);
  2. refine: `scripts/gaussian/render_refiner/d2v2_refine.sbatch` with `CK=<refiner ckpt> STEPS=20 GMODE=all GSCALE=2 INIT_SIGMA=0.8`, seed 42;
  3. score: `MODE=eval TAG=<tag>` with `d2v2_gs_hpc3.sbatch`, then `MODE=merge TAG=<tag> sbatch gsgen83_hpc3.sbatch`.
- **Shift preservation and multi-view consistency (supplementary):** `MODE=shiftcheck` of `d2v2_gs_hpc3.sbatch`; `pandwm_code/PanDWM-fuse-20261031/gsaa_src/dwm_pano/consistency_eval.py` with `consist_hpc3.sbatch`. Only `consistency_eval.py` of that tree is included; it runs on top of the `PanDWM-ft3-20261030` code.

The DiST-4D and DiST-4D + ReconDrive rows were computed with their official code and the same `eval_gs_shift` metrics.

## 5. Training

| Model | Entry |
|---|---|
| Video student, 4 stages, 4250 iterations | `$CF/scripts/train_v7.sbatch` → `train_v8_1.sbatch` → `train_v8_3.sbatch` / `train_ext.sbatch` / `train_ext2.sbatch` → `train_v8_4.sbatch` (see the README) |
| Gaussian decoder, final stage G3b | `PanDWM-ft3-20261030/scripts/train_ft3_hpc2.sbatch` with `ARM=G3b`, config `configs/gs4dd_hpc2_ft3_G3b.json`, 2000 steps from the previous stage |
| Earlier decoder stages | 3D: `PanDWM-gs-aa-ft/configs/ctsd/pandwm/h20/` and `scripts/h20/`; 4D: `PanDWM-gs4d-20261002/configs/` (`make_c2_config.py` and the other `make_*_config.py` scripts) |
| Render refiner | `scripts/gaussian/render_refiner/train_d2.sbatch`, config `PanDWM_d2_refiner_20261021/configs/d2_pilot_hpc3_8gpu.json`, 4000 steps |

The decoder and the refiner are trained on the nuScenes train split only; `PanDWM_d2_refiner_20261021/src/dwm_pano_d2/pipeline.py` rejects any training clip from the 83 evaluation scenes.
