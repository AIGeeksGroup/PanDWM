"""Diagnostic for report section 36 (2026-10-17): why does eval_label_depth.py measure a decoder-vs-label log-range
error of ~0.48 for the cov 3000 decoder when the depth pilot's training log showed ~0.19 for the same decoder?

For each decoder checkpoint and two clip samplings of the p2 training scenes it decodes one frame per clip and reports
mean / median |log r_decoder - log r_label| with the pipeline's own _load_frame_ranges (bilinear, as in training) and
the training mask (label valid & content ray & 0 < label < far_cut):
  start  = fps_stride [[10, 20]], frame 4: clips begin at each scene's start (what eval_label_depth.py used)
  spread = fps_stride [[10, 0.1]] (the training sampling), sequence 19, 25 evenly spaced clips, frame 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import dwm.common
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--checkpoints", required=True, nargs="+")
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--clips", type=int, default=25)
    p.add_argument("--far-cut", type=float, default=99.0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config["generator_seed"]))
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        pipe = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(args.output.parent), config=config, device=device,
            resume_from=None)
        pipe.vae.eval()
        install_strict_pano_hardseam_stub()
        from dwm.datasets.common import CollateFnIgnoring
        collate = CollateFnIgnoring(keys=[])
        samplings = {}
        for name, stride, length, frame in (("start", [[10, 20]], 20, 4), ("spread", [[10, 0.1]], 19, 9)):
            dcfg = build_dataset_config(config["training_dataset"], pkl_path=args.data_pkl, fps_stride_tuples=stride)
            dcfg["base_dataset"]["sequence_length"] = length
            ds = dwm.common.create_instance_from_config(dcfg)
            step = max(1, len(ds) // args.clips)
            samplings[name] = (ds, list(range(0, len(ds), step))[:args.clips], frame)
            print(f"[diag] sampling {name}: {len(ds)} clips", flush=True)
        report = {}
        for ckpt in args.checkpoints:
            pipe.decoder.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            pipe.decoder.eval()
            pipe.decoder_wrapper.eval()
            for name, (ds, indices, frame) in samplings.items():
                errors, medians, rough_v, rough_h = [], [], [], []
                for index in indices:
                    batch = collate([ds[index]])
                    pipe._prepare_nuscenes6_supervision(batch)
                    sel = torch.tensor([frame])
                    lo, po = pipe._frame_ray_origins(batch, sel)
                    half = batch["vae_images"][:, sel].flatten(0, 1).to(device)
                    out = pipe.decoder_wrapper(
                        pipe.encode_half_latents(half), pipe.latent_directions.unsqueeze(0),
                        pipe.pixel_directions.unsqueeze(0), None, previous_latents=None,
                        temporal_context_mask=0.0, latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)
                    rng = out[1].float()                                       # (1, 2, H, W)
                    gt, valid = pipe._load_frame_ranges(batch, sel, return_valid=True)
                    mask = valid & pipe.pixel_content_ray_valid.unsqueeze(0) & (gt < args.far_cut) & (gt > 0.0)
                    err = (rng.clamp_min(1e-3).log() - gt.float().clamp_min(1e-3).log()).abs()[mask]
                    errors.append(float(err.mean()))
                    lr = rng[0].clamp_min(1e-3).log()                         # roughness, as export_gs_v2_clips v3
                    rough_v.append(float((lr[:, 1:] - lr[:, :-1]).abs().median()))
                    rough_h.append(float((lr[:, :, 1:] - lr[:, :, :-1]).abs().median()))
                    medians.append(float(err.median()))
                key = f"{Path(ckpt).parent.parent.name}/{Path(ckpt).name}:{name}"
                report[key] = {"mean_log_l1": sum(errors) / len(errors), "median_log_l1": sum(medians) / len(medians),
                               "roughness_v": sum(rough_v) / len(rough_v), "roughness_h": sum(rough_h) / len(rough_h),
                               "per_clip_mean": errors}
                print(f"[diag] {key}: mean {report[key]['mean_log_l1']:.4f} median {report[key]['median_log_l1']:.4f} "
                      f"roughness v/h {report[key]['roughness_v']:.4f}/{report[key]['roughness_h']:.4f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
