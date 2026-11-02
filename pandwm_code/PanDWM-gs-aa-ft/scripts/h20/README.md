# PanDWM P0 Bootstrap on h20

All commands below run on h20 from:

```bash
cd /path/to/pandwm_root/code/PanDWM
```

## Order of operations

1. Create or refresh the evaluation environment:

   ```bash
   bash scripts/h20/bootstrap_env.sh
   ```

2. Wait for every download to finish. At minimum, verify the SD3.5 directory,
   Stage-2 checkpoint, val tars, 201 bbox tars, 201 hdmap tars, both source
   pkls, nuScenes metadata, caption zip, and TATS files. Partial tar files must
   not be extracted.

3. Inspect and extract the data. Tars are retained by default:

   ```bash
   KEEP_TARS=1 bash scripts/h20/extract_data.sh
   ```

   After validating file counts and several samples, use `KEEP_TARS=0` on a
   rerun to remove archives that have successful extraction markers:

   ```bash
   KEEP_TARS=0 bash scripts/h20/extract_data.sh
   ```

4. Inspect the downloaded caption schema and the val pkl:

   ```bash
   /path/to/pandwm_root/envs/pandwm/bin/python3 \
     scripts/h20/make_pkl_with_caption.py \
     --pkl-in /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid.pkl \
     --caption-zip /path/to/pandwm_root/data/captions/nuscenes_v1.0-trainval_caption_v2.zip \
     --nuscenes-meta-dir /path/to/pandwm_root/data/nuscenes_meta/v1.0-trainval \
     --inspect
   ```

   Then build the val pkl used by the h20 config:

   ```bash
   /path/to/pandwm_root/envs/pandwm/bin/python3 \
     scripts/h20/make_pkl_with_caption.py \
     --pkl-in /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid.pkl \
     --caption-zip /path/to/pandwm_root/data/captions/nuscenes_v1.0-trainval_caption_v2.zip \
     --nuscenes-meta-dir /path/to/pandwm_root/data/nuscenes_meta/v1.0-trainval \
     --pkl-out /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
   ```

5. Run the single-GPU, single-process 4-step Stage-2 evaluation:

   ```bash
   bash scripts/h20/eval_4step_h20.sh
   ```

   `STEP=10000`, `EMA=1`, and
   `RUN=/path/to/pandwm_root/runs/cd_baseline` are the defaults. Set
   `CUDA_VISIBLE_DEVICES` before launch to reserve a known GPU; otherwise the
   launcher selects the GPU with the most free memory at that instant. Use a
   different `MASTER_PORT` if port 29504 is busy. Logs are written under
   `/path/to/pandwm_root/logs`.

## Expected disk use

Approximate values depend on tar compression and the SD3.5 snapshot:

- Downloaded val, bbox, and hdmap tars: about 105 GB.
- Extracted RGB, segmentation, bbox, and hdmap files: about 105 GB.
- Stage-2 checkpoint: 13.6 GB.
- SD3.5-medium, conda environment, metadata, pkls, and captions: about
  35-60 GB.
- Total with tars retained: roughly 260-285 GB before evaluation outputs.
- Total after `KEEP_TARS=0`: roughly 155-180 GB before evaluation outputs.

Run `du -sh /path/to/pandwm_root/* /path/to/pandwm_root/data/*` before extraction and
evaluation. Keep additional headroom for logs, TorchInductor caches, and metric
artifacts.

## G1 GS domain-gap evaluation

G1 keeps the P2 GS decoder and target/metric protocol fixed while changing only
the latent source. The four arms are:

- `gt`: GT VAE latents (the regression arm).
- `stage2`: four-step Stage-2 latents with three clean reference frames.
- `stage2_reencode`: Stage-2 latents followed by a VAE decode/GS-VAE encode round trip.
- `gt_reencode`: GT latents followed by the same round trip, giving the VAE noise floor.

For the Stage-2 arms, `evaluate_gs.py` derives the holdout/validation dataset
config in memory from the GS config, adding captions, pre-rendered bbox/HD-map
images, and the all-ones cross-view mask; this keeps clip enumeration in one
place.

Run a one-clip-per-split co-residency smoke before a full evaluation:

```bash
ARM=stage2 SMOKE=1 STEP=15000 bash scripts/h20/eval_gs_g1_h20.sh
```

The smoke uses `--smoke` to evaluate one holdout and one validation clip and
writes a two-pair grid. If it is healthy, run each arm (the Stage-2 arms use
`EMA=1` and `STAGE2_STEP=10000` by default):

```bash
ARM=gt STEP=15000 bash scripts/h20/eval_gs_g1_h20.sh
ARM=stage2 STEP=15000 bash scripts/h20/eval_gs_g1_h20.sh
ARM=stage2_reencode STEP=15000 bash scripts/h20/eval_gs_g1_h20.sh
ARM=gt_reencode STEP=15000 bash scripts/h20/eval_gs_g1_h20.sh
```

Use `EMA=0` only when a raw Stage-2 checkpoint and its resume state are
available. Set `PANDWM_MIN_FREE_MB` no lower than 70000 on a shared H20; the
launcher refuses the best reported GPU below that floor. `metrics.json` stores
headline metrics and a `per_frame` breakdown keyed by frame index. Frame 2 is
the clean context frame for Stage-2 arms; interpret the headline domain gap
from frames 7, 11, and 16. The `gt_reencode` minus `gt` delta is the VAE floor,
so compare `stage2_reencode` against `gt_reencode` when separating generation
from encode/decode loss.

Use a different `MASTER_PORT` if the default 29506 is already occupied.

Metric key suffixes are `-gt`, `-stage2ref3`, `-stage2ref3reencode`, and
`-gtreencode`, respectively; the split name (for example `holdout5`) remains
in the key before the arm suffix (for example
`PSNR-6view-800x300-holdout5-stage2ref3`).

Stage-2 context frames are encoded with the CD VAE posterior `mode()` call,
matching the autoregressive inference path; the GT arm uses the GS VAE
posterior `sample()` call.  The selected context frame (index 2) is therefore
kept in the per-frame report when interpreting the small mode/sample offset.

## WARNING: assumptions and first-run checks

- The h20 tar member layouts were not available locally. The extractor probes
  every archive, selects the first numeric timestamp filename, and strips its
  leading components. It assumes all data files in that archive use the same
  depth. Verify the printed count and three sample names for every target.
- The extractor assumes RGB uses numeric `.jpg` names and segmentation,
  bbox, and hdmap use numeric `.png` names. This matches
  `PanoNuScenesDataset` for RGB/bbox/hdmap, but the actual archives must still
  be checked. It also assumes timestamp basenames do not conflict across
  shards.
- The public caption zip was inspected from hf-mirror.com on 2026-08-05. It
  contained caption keys of the form
  `scene_token|camera_name|camera_timestamp`; each value had `time`,
  `weather`, `environment`, `objects`, and `image_description`. Confirm that
  the h20 zip produces the same `--inspect` output.
- A pano info has one description, while OpenDWM captions are per camera. The
  injection script uses the nearest `CAM_FRONT` environment, then falls back
  through the other cameras. This representative-camera rule could not be
  compared with the unavailable cluster-generated caption pkl.
- The script assumes `info["token"][:32]` is the parent keyframe sample token
  for 33-character interpolation tokens. This is documented by
  `_group_infos_by_scene`, but the downloaded public pkl was not available for
  a local end-to-end check.
- Bare-list pkl input is normalized to the `{"infos": [...]}` top-level form
  required by `PanoNuScenesDataset`. Dict input preserves all other top-level
  fields, including `metadata`.
- `evaluate_4step.py` always enters DDP and hardcodes a `[1, 4]` FSDP mesh.
  The launcher creates a one-rank NCCL group and narrows only that runtime mesh
  to `[1, 1]`. This path was code-reviewed but could not be executed without
  an H20 and the checkpoints.
- `PanoNuScenesDataset` imports an unavailable online stitcher even when
  pre-rendered files are configured. The launcher supplies a runtime stub that
  raises if any requested bbox or hdmap PNG is absent. A complete pre-rendered
  layout set is therefore mandatory.
- The CD constructor always builds main, teacher, and EMA models. All four
  checkpoint path settings point to `cd_10000_ema.pth`; teacher weights are
  unused during evaluation. Confirm one H20 has enough memory for all three
  model copies and the SD3.5 text encoders.
- `evaluate_4step.py` constructs only `validation_dataset`. The training
  dataset paths in the derived config are val-data placeholders and are never
  opened by this entry point.
- With the default run directory, a link inside `runs/cd_baseline/checkpoints`
  needs three parent traversals to reach the root `ckpts` directory. The
  launcher computes the relative link target with `realpath` so a `RUN`
  override also resolves correctly. The requested two-parent spelling would
  resolve under `runs/ckpts` and would be broken for the stated layout.
- The EMA loading behavior was verified from source: `--ema` replaces the main
  and consistency-EMA paths with `RUN/checkpoints/10000_ema.pth` and skips
  optimizer resume. Actual key compatibility and historical metric parity
  remain first-run checks.
- The Tuna PyPI torch 2.5.1 wheel is expected to be a CUDA 12.4 build. The
  bootstrap script fails if `torch.version.cuda` is not exactly `12.4` or if
  `torchmetrics` is not exactly `1.6.0`.
- Single-process sample selection can differ from the historical four-rank
  evaluation when dataset length is not divisible by four. P0 manifest
  freezing is a separate follow-up item in the approved design.

## P1 Gaussian decoder G0 smoke training

Run all commands from `/path/to/pandwm_root/code/PanDWM`.

### CRITICAL: depth-free launcher shim

`PanoGaussianDecoderPipeline.train_step()` currently calls
`_load_frame_ranges()` unconditionally before it checks `use_range_prior` or
any loss weight. It also computes render-depth and range diagnostics
unconditionally. Therefore the JSON config alone cannot run without depth h5
files: direct use of `src/dwm_pano/train.py` with the smoke config will fail.

The h20 launcher installs a narrow runtime shim that returns a zero range
tensor without opening a file. It activates only when all of these guards
hold: `range_dir=null`, `weight_render_depth=0`, `weight_range_l1=0`,
`use_range_prior=false`, and latent refinement is disabled. The zero target is
used only by zero-weight depth/range diagnostics and never enters the decoder.
Use `scripts/h20/train_gs_smoke_h20.sh`; do not invoke `train.py` directly for
this depth-free run. A future source change should make the loader and the two
diagnostics conditional, after which this shim can be removed.

### CRITICAL: splat scale clamp (OOM root cause, found 2026-08-05)

`PanoGaussianDecoder` maps its raw scale head through an unbounded
`exp(raw) * scale_base`. Before the head has trained, some splats come out
screen-covering, and the `gsplat.rasterization` tile-intersection buffer
(`isect_tiles`) grows with covered tiles: at step ~190 a single 7.92 GiB
allocation OOMed a 95 GiB H20. The fix is the optional `scale_max` kwarg on
`PanoGaussianDecoder` (default `None`, which preserves cluster behavior
bit-for-bit); the smoke config sets `"scale_max": 5.0` meters. Because
`fix_training_data_order` plus a fixed `generator_seed` replays the same batch
order, the offending batch recurs at the same step on a rerun -- the clean
rerun past step 190 is the regression proof. Any new GS config for shared
machines should set `scale_max`.

### Order of operations

1. Wait for all selected RGB and segmentation shards to finish downloading.
   Do not extract a growing `.tar` file. The intended inputs are the 20 files
   under each of `data/train_train_tars` and `data/train_seg_tars`.

2. Extract the subset. Tars are retained by default, and extraction markers
   make reruns safe:

   ```bash
   KEEP_TARS=1 bash scripts/h20/extract_train_subset.sh
   ```

3. Build a pkl containing only complete scenes. This exact interpreter is
   mandatory because it has NumPy 1.26; the script refuses other NumPy minor
   versions:

   ```bash
   /path/to/pandwm_root/envs/pandwm/bin/python3 \
     scripts/h20/filter_pkl_to_available.py \
     --pkl-in /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl \
     --rgb-dir /path/to/pandwm_root/data/train/rgb \
     --seg-dir /path/to/pandwm_root/data/train/seg \
     --pkl-out /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_train_smoke.pkl \
     --min-scene-frames 24
   ```

   The public train pkl was inspected on h20: it is a dict with `infos`,
   `metadata`, and `scene_tokens`; `scene_tokens` is a list of 700 per-scene
   token lists. The filter uses that structure, recognizes interpolation
   tokens through the same `token[:32]` parent semantics as the dataset,
   preserves all top-level fields, and keeps `scene_tokens` aligned with the
   surviving `infos`. It refuses to write when no complete scene survives.

4. Start single-GPU, single-rank DDP training:

   ```bash
   bash scripts/h20/train_gs_smoke_h20.sh
   ```

   The launcher defaults to a 30000 MiB free-memory floor. Override it with
   `PANDWM_MIN_FREE_MB`, or reserve a known card with `CUDA_VISIBLE_DEVICES`.
   It logs every 20 steps, writes previews every 200 steps, checkpoints every
   1000 steps, disables standalone evaluation with `--evaluation-steps 0`,
   and passes neither `--wandb` nor `--eval-on-save`. Resume with, for example:

   ```bash
   RESUME_FROM=1000 bash scripts/h20/train_gs_smoke_h20.sh
   ```

### P1 disk estimate

- Twenty of 166 RGB shards are about 26 GB by proportional estimate; the
  matching segmentation subset is about 3.5 GB.
- Allow about 30 GB for extracted RGB and segmentation. With
  `KEEP_TARS=1`, the P1 data increment is roughly 60 GB; after a validated
  `KEEP_TARS=0` rerun it is roughly 30 GB.
- One fp32 decoder checkpoint is about 0.21 GiB. Its AdamW optimizer state is
  about 0.41 GiB after initialization, so each saved checkpoint pair is about
  0.62 GiB. The launcher keeps all checkpoints.
- Leave additional space for mp4 previews, logs, gsplat/TorchInductor JIT
  caches, and temporary files. Check `du` and `df` before extraction.

### P1 assumptions and first-run checks

- The source GS config and pipeline declare no `global_state` and no
  `device_mesh`. No mesh rewrite is needed for `WORLD_SIZE=1`; the launcher
  creates only a one-rank NCCL process group and wraps the decoder in DDP.
- With `latent_refinement.enabled=false`, the pipeline does not construct the
  Stage-2 DiT and does not load its checkpoint or uncondition pack. It loads
  only the SD3.5 VAE from the model directory for latent encoding. The
  remapped Stage-2 paths are inactive metadata for this run.
- The retained `weight_perceptual=0.05` also constructs a frozen LPIPS VGG16.
  On 2026-08-05, h20 had the small torchmetrics LPIPS linear weights but no
  torchvision VGG16 ImageNet checkpoint in the torch hub cache. Torchvision
  will try to fetch that checkpoint at pipeline startup. Pre-stage it if
  `download.pytorch.org` remains unreachable; do not mistake that download
  failure for a GS or SD3.5 failure.
- Training clips have `sequence_length=19` and
  `fps_stride_tuples=[[10,0.1]]`: enumeration needs a 1.9-second scene span
  and samples 19 frames at 10 fps. A 12 Hz span of that length is about 24
  infos, which is why the filtering command uses `--min-scene-frames 24`.
  The inspected complete train scenes contain 187 to 241 infos. The dataset
  loads all 19 RGB frames in a clip; each optimizer step then chooses four
  distinct random frames (`frames_per_step=4`) and loads their segmentation.
- Validation uses the val pkl/RGB/seg paths with
  `sequence_length=19` and `fps_stride_tuples=[[10,20]]`. Preview uses at most
  eight frames. The dataset stores its `seg_dir`, but GS preview reads only
  RGB because range prior, 3D boxes, and HD map are disabled.
- The 20-shard timestamp coverage is arbitrary. Since the filter requires
  every RGB and segmentation file in a scene, it may retain few scenes or
  refuse with zero scenes. That is a data-coverage result, not a reason to
  weaken the complete-scene rule: partial scenes can enumerate clips whose
  files are missing.
- Both training and validation have `enable_3dbox=false` and
  `enable_hdmap=false`, so the GS data path cannot call the online omninwm
  stitcher and the launcher deliberately has no omninwm stub. Ensure the P0
  lazy-import fallback in `pano_via_stitch.py` is present in the synced h20
  tree so importing the dataset itself remains safe.
- gsplat 1.5.3 compiles CUDA kernels on first use. The installed nvcc is 12.8
  while PyTorch is `2.5.1+cu124`; compilation may take several minutes or fail
  on toolkit/ABI compatibility. Preserve the full first-run log and verify a
  tiny render reaches forward and backward before committing to 1000 steps.
- The tar layout is still not known in advance. The extractor probes each tar
  and assumes all timestamp files in that archive have the same component
  depth. Verify counts and the three printed sample names for both targets,
  then confirm the filter reports nonzero scenes and infos.
- At step 200, inspect the mp4 for structured target/render rows and check all
  reported weighted losses for finite values. At step 1000, verify both
  decoder and optimizer files exist, then test `RESUME_FROM=1000` before
  considering P1 complete.

## P2 GS decoder depth-supervised training and evaluation

Follow the P2 campaign plan in
`docs/plans/2026-08-06-h20-p2-gs-training-plan.md`. Run all commands below
from `/path/to/pandwm_root/code/PanDWM`.

### 1. Release the extracted P1 tar copies

Delete the RGB and segmentation tar directories only after every one of their
20 archives has a valid extraction marker whose first-line probe still exists.
The gate below checks the marker convention used by
`extract_train_subset.sh`; the two exact `rm` commands run only if all checks
pass.

```bash
check_extract_markers() {
  local tar_dir="$1"
  local extracted_dir="$2"
  local archives=()
  local archive marker probe
  shopt -s nullglob
  archives=("$tar_dir"/*.tar)
  shopt -u nullglob
  if (( ${#archives[@]} != 20 )); then
    printf 'Expected 20 tars in %s, found %s\n' \
      "$tar_dir" "${#archives[@]}" >&2
    return 1
  fi
  for archive in "${archives[@]}"; do
    marker="$extracted_dir/.extract-state/$(basename -- "$archive").done"
    [[ -f "$marker" ]] || {
      printf 'Missing extraction marker: %s\n' "$marker" >&2
      return 1
    }
    probe="$(sed -n '1p' "$marker")"
    [[ -n "$probe" && -f "$extracted_dir/$probe" ]] || {
      printf 'Invalid extraction marker probe: %s\n' "$marker" >&2
      return 1
    }
  done
}

check_extract_markers \
  /path/to/pandwm_root/data/train_train_tars \
  /path/to/pandwm_root/data/train/rgb && \
check_extract_markers \
  /path/to/pandwm_root/data/train_seg_tars \
  /path/to/pandwm_root/data/train/seg && {
  rm -rf -- /path/to/pandwm_root/data/train_train_tars
  rm -rf -- /path/to/pandwm_root/data/train_seg_tars
}
```

Do not remove `aslam_backup_20260805`, bbox, or hdmap data.

### 2. Stream the depth subset

Download, inspect, align-check, extract, mark, and then delete each depth tar
before advancing to the next shard:

```bash
bash scripts/h20/fetch_extract_depth_subset.sh
```

The defaults are `SHARD_START=0`, `SHARD_COUNT=20`, and `KEEP_TARS=0`.
Each download uses one connection because the signed Xet URLs reject
multi-connection Range requests. Set `KEEP_TARS=1` only when enough extra disk
space is available. The script aborts below 15 GiB free space, on an unsafe or
misaligned archive, or when the first extracted/tar size ratio exceeds 1.3.

### 3. Refilter the pkl and make the five-scene holdout

Use the NumPy 1.26 environment to require RGB, segmentation, and depth for
every info in a surviving scene. The lexicographically last five stable scene
keys go only to the holdout pkl:

```bash
/path/to/pandwm_root/envs/pandwm/bin/python3 \
  scripts/h20/filter_pkl_to_available.py \
  --pkl-in /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl \
  --rgb-dir /path/to/pandwm_root/data/train/rgb \
  --seg-dir /path/to/pandwm_root/data/train/seg \
  --require-depth /path/to/pandwm_root/data/train/depth \
  --pkl-out /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_train_p2.pkl \
  --holdout-scenes 5 \
  --holdout-out /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl \
  --min-scene-frames 24
```

Record the printed holdout keys and train/holdout scene and info counts before
training.

### 4. Run the 100-step dataloader speed test

Start the real P2 launcher and read the `s/step` value printed at step 100:

```bash
bash scripts/h20/train_gs_p2_h20.sh
```

Interrupt after the step-100 log for this measurement. If it is still above
10 s/step, stop and decide whether the plan's level-2 pre-resize cache work is
needed. Level 2 is intentionally not implemented in P2 yet. If the speed is
acceptable, restart the same launcher for the full deterministic run.

### 5. Launch the full P2 run

The launcher logs every 20 steps, previews every 500, checkpoints every 2500,
disables in-training evaluation, keeps every checkpoint, and requires 45000
MiB free GPU memory by default:

```bash
bash scripts/h20/train_gs_p2_h20.sh
```

The JSON retains six epochs so the run has enough headroom; stop after the
15000-step checkpoint as specified by the P2 plan. Resume only from an
existing decoder and optimizer checkpoint pair, for example:

```bash
RESUME_FROM=2500 bash scripts/h20/train_gs_p2_h20.sh
```

### 6. Evaluate every P2 checkpoint

Run the standalone evaluator for each planned checkpoint. It evaluates every
disjoint 10 fps holdout clip, the first 32 validation clips, writes
`metrics.json`, and writes a two-row-per-clip `grid.png` under each step
directory.

```bash
for STEP in 2500 5000 7500 10000 12500 15000; do
  /path/to/pandwm_root/envs/pandwm/bin/python3 \
    src/dwm_pano/evaluate_gs.py \
    -c configs/ctsd/pandwm/h20/gs_decoder_p2_h20.json \
    -r /path/to/pandwm_root/runs/gs_p2 \
    -s "$STEP" \
    --holdout-pkl /path/to/pandwm_root/data/pickle/nuscenes_interp_12Hz_infos_holdout_p2.pkl \
    --val-limit 32
done
```

Compare the `-gt`-suffixed `DepthL1-pano-holdout5` and
`AbsRel-pano-holdout5` keys against the same keys in `trivial_baseline`, and
track the suffixed holdout5 and val32 RGB metrics across checkpoints.
