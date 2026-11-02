"""D2 refiner on the review clips (2026-10-21, D2_PLAN_20261021.md Tasks 8 / 10 / 11).

For every review clip and signed lateral offset dy, the render condition of PanDWM-gs-aa-ft
dwm_pano/d2_export_shift.py (<inputs>/clipNNN/render_dy<d>.pt) and the clip's own conditions from the frozen
149-clip protocol (validation_dataset of cd_eval_frozen149_hpc3.json, rows of dev149_manifest_20261018.json: text
and cameras of the ORIGINAL trajectory; 3D boxes and HD maps dropped because they are drawn for it) go through
PanoD2RefinerSD.refine_window twice, as the recorded rollout runs:
  window 1 = frames 0..18 without history, window 2 = frames 16..34 with window 1's refined frames 16..18 as the
  clean 3-frame history.  The sampler generator is reseeded per (clip, dy) from seed * 1000 + clip.
No ground-truth frame enters the conditioning path (vae_images is removed before slicing).

    <out-root>/clipNNN/d2_dy<d>.pt   {"latents": (35, 2, 16, 32, 288) fp16, "meta": {...}}
    <out-root>/clipNNN/d2_dy<d>.mp4  rows: GS render (VAE decode) | refined  (only for --video-clips)
"""

import argparse
import json
import time
from pathlib import Path

import torch

import dwm.common
import dwm.functional
import dwm.utils.preview
from dwm_pano.pano_stitch import stitch_pano_halves
from dwm_pano.pipelines.ctsd_pano_self_forcing_utils import slice_sequence_batch

CLIP = 35
WINDOW = 19


def parse_clips(text):
    if ":" in text:
        a, b = (int(x) for x in text.split(":"))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


@torch.no_grad()
def decode(pipe, latents):
    """(T, V, C, h, w) latents -> (T, 3, H, 4114) stitched pano in [0, 1]."""
    shift = pipe.vae.config.shift_factor or 0
    images = dwm.functional.memory_efficient_split_call(
        pipe.vae, latents.flatten(0, 1).to(pipe.device, dtype=pipe.vae.dtype),
        lambda block, tensor: block.decode(tensor / block.config.scaling_factor + shift, return_dict=False)[0],
        pipe.common_config.get("memory_efficient_batch", -1))
    images = ((images.float() + 1.0) / 2.0).clamp(0, 1).unflatten(0, (latents.shape[0], latents.shape[1]))
    return stitch_pano_halves(images[None])[0, :, 0].clamp(0, 1)


def render_condition(pack):
    render, mask = pack["render"].float(), pack["mask"].float() / 255.0
    return torch.cat([render, mask.unsqueeze(2)], dim=2)[None]                    # (1, 35, 2, 17, h, w)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="the D2 training config")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--no-strict", action="store_true", help="smoke only: load a base (teacher) checkpoint")
    p.add_argument("--eval-config", required=True, help="cd_eval_frozen149_hpc3.json")
    p.add_argument("--manifest", required=True, help="dev149_manifest_20261018.json")
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--clips", required=True)
    p.add_argument("--shifts", default="2,-2")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-mode", default="none", choices=["none", "all", "text"])
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--keep-layout", action="store_true", help="ablation: keep the original-trajectory boxes/maps")
    p.add_argument("--init-sigma", type=float, default=0.0, help="section 43.2: start from the noised render")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-clips", default="")
    p.add_argument("--fps-stride", default="10,20",
                   help="2026-10-31 (D2V2_PLAN): the manifest's dataset fps,stride (Test-S tests213: 10,5)")
    args = p.parse_args()

    d2cfg = json.loads(Path(args.config).read_text())
    evalcfg = json.loads(Path(args.eval_config).read_text())
    d2cfg.pop("optimizer", None)
    d2cfg.pop("lr_scheduler", None)
    d2cfg["generator_seed"] = args.seed
    d2cfg["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    pc = d2cfg["pipeline"]
    pc["model_checkpoint_path"] = args.checkpoint
    pc["model_load_state_args"] = {"strict": not args.no_strict}
    ds_cfg = evalcfg["validation_dataset"]
    fps, stride = (float(x) for x in args.fps_stride.split(","))
    if (fps, stride) != (10.0, 20.0):                    # 2026-10-31: Test-S clips come from the stride-5 enumeration
        ds_cfg["base_dataset"]["fps_stride_tuples"] = [[int(fps), stride]]
    if ds_cfg["base_dataset"]["sequence_length"] != CLIP or \
            ds_cfg["base_dataset"]["fps_stride_tuples"] != [[int(fps), stride if stride != 20.0 else 20]]:
        raise ValueError("expected the frozen 35-frame protocol dataset")
    manifest_fs = json.loads(Path(args.manifest).read_text()).get("dataset_fps_stride_tuples")
    if manifest_fs is not None and [[float(a), float(b)] for a, b in manifest_fs] != [[fps, stride]]:
        raise ValueError(f"--fps-stride {args.fps_stride} does not match the manifest's {manifest_fs}")
    rows = {int(r["test_index"]): r for r in json.loads(Path(args.manifest).read_text())["clips"]}
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    if any(float(f"{d:g}") != d for d in shifts) or len({f"{d:g}" for d in shifts}) != len(shifts):
        raise ValueError("shifts must have distinct, exactly representable %g names (e.g. 0.5, 1, -2)")
    video_clips = set(parse_clips(args.video_clips)) if args.video_clips else set()

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in d2cfg["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipe = dwm.common.create_instance_from_config(pc, output_path=str(args.out_root), config=d2cfg, device=device,
                                                  resume_from=None)
    dataset = dwm.common.create_instance_from_config(ds_cfg)
    loader_args = dwm.common.instantiate_config(evalcfg["validation_dataloader"])
    collate = loader_args.get("collate_fn") or torch.utils.data.default_collate
    exceptions = pc["inference_config"].get("autoregression_data_exception_for_take_sequence", [])
    sampler = {"steps": args.steps, "guidance_mode": args.guidance_mode, "guidance_scale": args.guidance_scale,
               "drop_layout": not args.keep_layout, "seed": args.seed, "init_sigma": args.init_sigma}
    print(f"[d2-refine] checkpoint {args.checkpoint} sampler {json.dumps(sampler)}", flush=True)
    for clip in parse_clips(args.clips):
        out = args.out_root / f"clip{clip:03d}"
        todo = []
        for d in shifts:
            path = out / f"d2_dy{d:g}.pt"
            if not path.is_file():
                todo.append(d)
                continue
            old = torch.load(path, map_location="cpu", weights_only=True)["meta"]
            old_sampler = dict(old.get("sampler") or {})
            old_sampler.setdefault("init_sigma", 0.0)             # outputs made before section 43.2 had no key
            if old.get("checkpoint") != args.checkpoint or old_sampler != sampler or \
                    old.get("inputs") != str(args.inputs):
                raise RuntimeError(f"{path} exists from another checkpoint / sampler / input set; use a new --out-root")
        if not todo:
            continue
        row = rows[clip]
        index = int(row["dataset_index"])
        batch = collate([dataset[index]])
        audit = dataset.base_dataset.clip_audit[index]
        if audit["scene_token"] != row["scene_token"] or \
                int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"clip {clip}: dataset index {index} is not the manifest clip")
        frame_ts = [int(x) for x in batch["frame_timestamps_us"][0].tolist()]
        batch.pop("vae_images", None)
        w1_batch = slice_sequence_batch(dict(batch, vae_images=torch.zeros(1, CLIP, 2)), 0, WINDOW, exceptions)
        w2_batch = slice_sequence_batch(dict(batch, vae_images=torch.zeros(1, CLIP, 2)), CLIP - WINDOW, WINDOW,
                                        exceptions)
        for b in (w1_batch, w2_batch):
            b.pop("vae_images")
        out.mkdir(parents=True, exist_ok=True)
        for d in todo:
            t0 = time.monotonic()
            pack = torch.load(args.inputs / f"clip{clip:03d}" / f"render_dy{d:g}.pt", map_location="cpu",
                              weights_only=True)
            if pack["meta"]["frame_timestamps_us"] != frame_ts or int(pack["meta"]["clip_index"]) != clip:
                raise RuntimeError(f"clip {clip} dy {d:g}: render input belongs to another clip")
            render = render_condition(pack)
            pipe.generator.manual_seed(args.seed * 1000 + clip)
            w1 = pipe.refine_window(w1_batch, render[:, :WINDOW], steps=args.steps,
                                    guidance_scale=args.guidance_scale, guidance_mode=args.guidance_mode,
                                    drop_layout=not args.keep_layout, init_sigma=args.init_sigma)
            w2 = pipe.refine_window(w2_batch, render[:, CLIP - WINDOW:], reference=w1[:, CLIP - WINDOW:WINDOW],
                                    reference_count=2 * WINDOW - CLIP, steps=args.steps,
                                    guidance_scale=args.guidance_scale, guidance_mode=args.guidance_mode,
                                    drop_layout=not args.keep_layout, init_sigma=args.init_sigma)
            refined = torch.cat([w1[0], w2[0, 2 * WINDOW - CLIP:]])                 # (35, 2, 16, h, w)
            meta = {"clip_index": clip, "dy": d, "checkpoint": args.checkpoint, "sampler": sampler,
                    "frame_timestamps_us": frame_ts, "inputs": str(args.inputs),
                    "render_coverage_mean": pack["meta"].get("coverage_mean")}
            path = out / f"d2_dy{d:g}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save({"latents": refined.half().cpu(), "meta": meta}, tmp)
            tmp.replace(path)
            if clip in video_clips:
                rows_ = [decode(pipe, pack["render"].float()), decode(pipe, refined)]
                frames = []
                for t in range(CLIP):
                    stacked = torch.cat([r[t] for r in rows_], dim=-2)[None]
                    size = (stacked.shape[-2] // 4 * 2, stacked.shape[-1] // 4 * 2)
                    frames.append(torch.nn.functional.interpolate(stacked, size=size, mode="area")[0].cpu())
                dwm.utils.preview.save_tensor_to_video(str(out / f"d2_dy{d:g}.mp4"), "libx264", 10, frames)
            print(f"[d2-refine] clip {clip} dy {d:g} done ({time.monotonic() - t0:.0f}s)", flush=True)
    torch.distributed.destroy_process_group()
    print("D2_REFINE_DONE", flush=True)


if __name__ == "__main__":
    main()
