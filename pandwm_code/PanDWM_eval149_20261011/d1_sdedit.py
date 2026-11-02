"""D1 of PanDWM-gs4d-20261002/GS_GAP_ANALYSIS_20261020.md: SDEdit of shifted-rig GS pano renders with the locked
video model (v84i4250 EMA student), step 2 of 2 (step 1 = PanDWM-gs-aa-ft export_shift_pano_latents.py on HPC2).

Per clip and lateral offset dy, the GS render latents (35 frames) replace the sampler's starting noise: for start index
k of the student step list [1000, 750, 500, 250] (sigmas 1.0 / 0.9 / 0.75 / 0.5 after the shift-3 schedule) the
window is x_k = (1 - sigma_k) * render + sigma_k * noise and the student runs its remaining steps k..3 with the
stochastic CM transition of evaluate_4step.make_4step_inference, conditioned on the clip's own conditions (text,
boxes, map, cameras of the ORIGINAL trajectory; the shifted ones are not available).  Window 1 = frames 0..18 with no
history; window 2 = frames 16..34 with window 1's refined frames 16..18 as the clean 3-frame history, as the
recorded rollout continues.  The sampler generator is reseeded per clip from seed * 1000 + clip index.

  <out-root>/clipNNN/d1_dy<d>_k<k>.pt   {"latents": (35, 2, 16, 32, 288) fp16, ...}
  <out-root>/clipNNN/d1_dy<d>.mp4       rows: recorded | GS render (VAE round trip) | refined k=... (one row per k)

  python lowmem_launch.py d1_sdedit.py --base-config cd_eval_frozen149_hpc3.json --checkpoint <4250_ema.pth>
      --manifest dev149_manifest_20261018.json --renders <step-1 root> --out-root <dir> --clips 2,60 --shifts 0,2
      --starts 2,3
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
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD
from run_clips_videos import CLIP, build_pipeline
from run_history_probe import _window_batch
from dwm_pano import evaluate_4step


def make_sdedit(pipeline, sigmas):
    def run(init, batch, start_index, reference=None, reference_frame_count=0):
        """init (1, 19, V, C, h, w) render latents; returns the refined (1, 19, V, C, h, w) latents."""
        pipeline.model_wrapper.eval()
        rng = pipeline.generator
        shape = tuple(init.shape)
        cond = CrossviewTemporalSD.get_conditions(
            pipeline.model, pipeline.text_encoders, pipeline.tokenizers, pipeline.common_config, shape, batch,
            pipeline.device, pipeline.model_dtype, do_classifier_free_guidance=False, latents_shape=shape)
        pipeline._add_temporal_attention_mask(cond, shape)
        sigma0 = sigmas[start_index].to(dtype=init.dtype)
        latents = (1.0 - sigma0) * init + sigma0 * torch.randn(shape, generator=rng).to(pipeline.device)
        for i in range(start_index, len(pipeline.denoising_step_list)):
            t_val = pipeline.denoising_step_list[i]
            sigma_t = sigmas[i].to(dtype=latents.dtype)
            timesteps = torch.full(shape[:3], float(t_val.item()), dtype=t_val.dtype, device=pipeline.device)
            inp = latents
            if reference is not None and reference_frame_count > 0:
                inp = torch.cat([reference[:, :reference_frame_count], latents[:, reference_frame_count:]], dim=1)
                timesteps[:, :reference_frame_count] = 0
            inp = inp.to(dtype=pipeline.model_dtype)
            with pipeline.get_autocast_context():
                model_output, _, _ = pipeline.model_wrapper(inp, timesteps, **cond)
            velocity = model_output[0]
            x0_pred = inp + (-sigma_t) * velocity
            if i < len(pipeline.denoising_step_list) - 1:
                sigma_next = sigmas[i + 1].to(dtype=latents.dtype)
                latents = (1.0 - sigma_next) * x0_pred + sigma_next * torch.randn(shape, generator=rng).to(
                    pipeline.device)
            else:
                latents = x0_pred
        if reference is not None and reference_frame_count > 0:
            latents = torch.cat([reference[:, :reference_frame_count], latents[:, reference_frame_count:]], dim=1)
        return latents.float()
    return run


@torch.no_grad()
def decode(pipeline, latents):
    """(T, V, C, h, w) latents -> (T, 3, H, 4114) stitched pano in [0, 1]."""
    shift = pipeline.vae.config.shift_factor or 0
    images = dwm.functional.memory_efficient_split_call(
        pipeline.vae, latents.flatten(0, 1).to(pipeline.device, dtype=pipeline.vae.dtype),
        lambda block, tensor: block.decode(tensor / block.config.scaling_factor + shift, return_dict=False)[0],
        pipeline.common_config.get("memory_efficient_batch", -1))
    images = ((images.float() + 1.0) / 2.0).clamp(0, 1).unflatten(0, (latents.shape[0], latents.shape[1]))
    return stitch_pano_halves(images[None])[0, :, 0].clamp(0, 1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--renders", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--shifts", default="0,2")
    parser.add_argument("--starts", default="2,3", help="student step indices to start from (3 = t250 only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.engine, args.bidirectional, args.student_steps = "student", False, [1000, 750, 500, 250]

    manifest = json.load(open(args.manifest))
    rows = {int(r["test_index"]): r for r in manifest["clips"]}
    config = json.loads(Path(args.base_config).read_text())
    config["generator_seed"] = args.seed
    config["global_state"]["device_mesh"]["mesh_shape"] = [1, 1]
    ds = config["validation_dataset"]["base_dataset"]
    if ds["sequence_length"] != CLIP or ds["fps_stride_tuples"] != [[10, 20]]:
        raise ValueError("expected the frozen 35-frame protocol config")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", 0)
    for key, value in config["global_state"].items():
        dwm.common.global_state[key] = dwm.common.create_instance_from_config(value)
    pipeline, _ = build_pipeline(args, config, args.out_root, device)
    sigmas = evaluate_4step.compute_denoising_sigmas(pipeline)
    sdedit = make_sdedit(pipeline, sigmas)
    pc = config["pipeline"]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    loader_args = dwm.common.instantiate_config(config["validation_dataloader"])
    collate = loader_args.get("collate_fn") or torch.utils.data.default_collate
    exceptions = pc["inference_config"].get("autoregression_data_exception_for_take_sequence", [])
    shifts = [float(s) for s in args.shifts.split(",") if s.strip()]
    starts = [int(s) for s in args.starts.split(",") if s.strip()]
    print(f"[d1] sigmas {[round(float(s), 4) for s in sigmas]} steps {[int(t) for t in pipeline.denoising_step_list]}",
          flush=True)
    for position in [int(x) for x in args.clips.split(",") if x.strip()]:
        row = rows[position]
        index = int(row["dataset_index"])
        batch = collate([dataset[index]])
        audit = dataset.base_dataset.clip_audit[index]
        if audit["scene_token"] != row["scene_token"] or \
                int(batch["anchor_timestamp_us"][0]) != int(row["anchor_timestamp_us"]):
            raise RuntimeError(f"clip {position}: dataset index {index} is not the manifest clip")
        out = args.out_root / f"clip{position:03d}"
        out.mkdir(parents=True, exist_ok=True)
        real = stitch_pano_halves(batch["vae_images"].to(device))[0, :, 0].clamp(0, 1)          # (35, 3, H, W)
        for dy in shifts:
            t0 = time.monotonic()
            pack = torch.load(args.renders / f"clip{position:03d}" / f"render_dy{dy:g}.pt", map_location="cpu")
            render = pack["latents"].to(device).float()                                           # (35, V, C, h, w)
            if render.shape[0] != CLIP:
                raise RuntimeError(f"clip {position} dy {dy}: {render.shape[0]} render frames")
            panels = [real, decode(pipeline, render)]
            for k in starts:
                pipeline.generator.manual_seed(args.seed * 1000 + position)
                w1 = sdedit(render[None, 0:19], _window_batch(batch, 0, 19, exceptions), k)
                w2 = sdedit(render[None, 16:35], _window_batch(batch, 16, 19, exceptions), k,
                            reference=w1[:, 16:19], reference_frame_count=3)
                refined = torch.cat([w1[0], w2[0, 3:]])
                torch.save({"latents": refined.half().cpu(), "dy": dy, "start_index": k, "seed": args.seed,
                            "checkpoint": args.checkpoint, "render": str(pack.get("checkpoint"))},
                           out / f"d1_dy{dy:g}_k{k}.pt")
                panels.append(decode(pipeline, refined))
            frames = []
            for t in range(CLIP):
                stacked = torch.cat([p[t] for p in panels], dim=-2)[None]
                size = (stacked.shape[-2] // 4 * 2, stacked.shape[-1] // 4 * 2)        # even sizes for libx264
                frames.append(torch.nn.functional.interpolate(stacked, size=size, mode="area")[0].cpu())
            dwm.utils.preview.save_tensor_to_video(str(out / f"d1_dy{dy:g}.mp4"), "libx264", 10, frames)
            print(f"[d1] clip {position} dy {dy:g} starts {starts} done ({time.monotonic() - t0:.0f}s)", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
