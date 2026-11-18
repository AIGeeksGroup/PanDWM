"""Re-render Gaussian demos from viewpoints close to the capture rig.

The original demos (``export_gs_long_demo.py`` / ``export_gs_batch149.py``)
put the camera 4 m behind and 2.5 m above the ego vehicle and orbit it at 8 m.
The decoder emits one Gaussian per panorama pixel (a single depth shell seen
through a +/-16.7 deg vertical band), so such viewpoints expose the unobserved
ground under the car (black), disoccluded backs of objects (stretched shells)
and depth seams between cameras (cracks).  This script renders only views the
representation can support: small translations around the rig at rig height,
a pure-rotation pan, and a vertical field of view inside the observed band.

It writes ``gs_clean_views.mp4`` to a new directory and never touches the
earlier outputs.

  long demo : --scene-token T (--rollout-dir D | --latent-source gt) --out DIR
  149 clips : --clips-root R --start a --end b --out-root DIR
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import dwm.common
from dwm.utils.preview import save_tensor_to_video
from dwm_pano.evaluate_gs import (
    build_dataset_config,
    ensure_single_rank_process_group,
    install_strict_pano_hardseam_stub,
    load_config,
)
from dwm_pano.export_gs_long_demo import CONTEXT, load_rollout_latents, look_at_c2w

RIG_HEIGHT = 1.6  # nuScenes cameras sit about 1.5-1.7 m above the ground


def yaw_c2w(position, yaw_rad, pitch_rad=0.0):
    direction = [math.cos(yaw_rad) * math.cos(pitch_rad),
                 math.sin(yaw_rad) * math.cos(pitch_rad),
                 math.sin(pitch_rad)]
    return look_at_c2w(position, np.asarray(position) + 10.0 * np.asarray(direction))


VIEWS = {
    # Pure rotation about the rig: no parallax, so no holes or stretching.
    "pan": ("look around: camera stays at the rig, turns +/-40 deg",
            lambda t: yaw_c2w([0.0, 0.0, RIG_HEIGHT],
                              math.radians(40.0) * math.sin(2 * math.pi * t / 60.0))),
    "chase": ("follow: 1.5 m behind, 0.3 m above the rig",
              lambda t: look_at_c2w([-1.5, 0.0, RIG_HEIGHT + 0.3],
                                    [15.0, 0.0, RIG_HEIGHT - 0.4])),
    "sweep": ("sideways: +/-1 m left-right at rig height",
              lambda t: (lambda y: look_at_c2w([0.0, y, RIG_HEIGHT],
                                               [15.0, y, RIG_HEIGHT - 0.2]))(
                  1.0 * math.sin(2 * math.pi * t / 40.0))),
    # Kept only for the side-by-side test: the old, far viewpoint.
    "old_chase": ("old: 4 m behind, 2.5 m up",
                  lambda t: look_at_c2w([-4.0, 0.0, 2.5], [10.0, 0.0, 1.0])),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, type=Path)
    p.add_argument("--data-pkl", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--views", default="chase,sweep,pan")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--hfov-deg", type=float, default=70.0)
    # Vertical extent above / below the horizon.  The panorama band is
    # +/-16.7 deg, and the ground near the rig is never observed, so the
    # frame stops a few degrees below the horizon instead of showing black.
    p.add_argument("--vfov-up-deg", type=float, default=14.0)
    p.add_argument("--vfov-down-deg", type=float, default=7.0)
    p.add_argument("--fill-width", type=int, default=61,
                   help="inpaint uncovered gaps narrower than this many pixels "
                        "(camera-seam cracks); 0 disables")
    p.add_argument("--max-scale", type=float, default=1.0,
                   help="drop Gaussians whose largest axis exceeds this (m)")
    p.add_argument("--max-frames", type=int, default=0, help="test: render only N frames")
    # long demo
    p.add_argument("--scene-token")
    p.add_argument("--latent-source", choices=("rollout", "gt"), default="rollout")
    p.add_argument("--rollout-dir", type=Path)
    p.add_argument("--out", type=Path)
    # 149 clips
    p.add_argument("--clips-root", type=Path)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--out-root", type=Path)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    args = p.parse_args()
    if (args.scene_token is None) == (args.clips_root is None):
        p.error("give exactly one of --scene-token / --clips-root")
    return args


def fill_cracks(rgb, alpha, width):
    """Inpaint thin uncovered gaps; leave large uncovered regions untouched.

    rgb (3, H, W) in [0, 1], alpha (H, W).  A pixel is filled only if it is
    uncovered and its uncovered region does not survive a morphological
    opening of ``width`` pixels, i.e. it is a crack, not a missing area.
    """
    import cv2
    hole = (alpha.float().numpy() < 0.5).astype(np.uint8)
    if not hole.any():
        return rgb
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    thin = hole & (1 - cv2.morphologyEx(hole, cv2.MORPH_OPEN, kernel))
    if not thin.any():
        return rgb
    image = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    filled = cv2.inpaint(np.ascontiguousarray(image), thin, 5, cv2.INPAINT_TELEA)
    return torch.from_numpy(filled).permute(2, 0, 1).float() / 255.0


class Renderer:
    def __init__(self, args, device):
        self.args = args
        config = load_config(args.config)
        torch.manual_seed(int(config["generator_seed"]))
        self.config = config
        out = args.out or args.out_root
        self.pipeline = pipeline = dwm.common.create_instance_from_config(
            config["pipeline"], output_path=str(out), config=config,
            device=device, resume_from=None)
        pipeline.vae.eval()
        pipeline.decoder.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu"), strict=True)
        pipeline.decoder.eval()
        pipeline.decoder_wrapper.eval()
        pipeline.rasterize_mode = "antialiased"
        pipeline.active_rasterize_eps2d = 0.1
        install_strict_pano_hardseam_stub()
        self.device = device
        self.views = [v for v in args.views.split(",") if v]
        focal = (args.width / 2.0) / math.tan(math.radians(args.hfov_deg) / 2.0)
        up = focal * math.tan(math.radians(args.vfov_up_deg))
        args.height = int(round(up + focal * math.tan(math.radians(args.vfov_down_deg))))
        args.height += args.height % 2
        K = np.array([[focal, 0.0, args.width / 2.0], [0.0, focal, up], [0.0, 0.0, 1.0]])
        self.K = torch.tensor(np.stack([K] * len(self.views)),
                              device=device, dtype=pipeline.pixel_directions.dtype)

    def dataset(self, length):
        cfg = build_dataset_config(self.config["training_dataset"], pkl_path=self.args.data_pkl,
                                   fps_stride_tuples=[[10, 20]])
        cfg["base_dataset"]["sequence_length"] = length
        return dwm.common.create_instance_from_config(cfg)

    @staticmethod
    def load(dataset, index):
        from dwm.datasets.common import CollateFnIgnoring
        return next(iter(torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, [index]), batch_size=1, shuffle=False,
            num_workers=0, collate_fn=CollateFnIgnoring(keys=[]))))

    @torch.no_grad()
    def render(self, batch, latents, path):
        pipe, frames = self.pipeline, latents.shape[0]
        if self.args.max_frames:
            frames = min(frames, self.args.max_frames)
        pipe._prepare_nuscenes6_supervision(batch)
        rows = []
        for t in range(frames):
            selector = torch.tensor([t], dtype=torch.long)
            lo, po = pipe._frame_ray_origins(batch, selector)
            g = pipe.decoder_wrapper(
                latents[t:t + 1].to(self.device, dtype=pipe.latent_directions.dtype),
                pipe.latent_directions.unsqueeze(0), pipe.pixel_directions.unsqueeze(0),
                None, previous_latents=None, temporal_context_mask=0.0,
                latent_origins=lo, pixel_origins=po, camera_origin_scale=1.0)[0][0]
            g = g[g[:, 11:14].amax(dim=-1) <= self.args.max_scale]
            c2w = torch.tensor(np.stack([VIEWS[v][1](t) for v in self.views]),
                               device=self.device, dtype=self.K.dtype)
            views, _, alpha = pipe.render_targets(g, target_c2w=c2w, target_intrinsics=self.K,
                                                  width=self.args.width, height=self.args.height)
            views = views.clamp(0, 1).cpu()
            if self.args.fill_width:
                views = torch.stack([fill_cracks(v, a, self.args.fill_width)
                                     for v, a in zip(views, alpha.cpu())])
            rows.append(torch.cat(list(views), dim=-1))
        save_tensor_to_video(str(path), "libx264", float(batch["fps"][0]), rows)
        return frames

    def metadata(self, **extra):
        a = self.args
        return dict(extra, views=[VIEWS[v][0] for v in self.views],
                    camera={"width": a.width, "height": a.height, "hfov_deg": a.hfov_deg,
                            "vfov_up_deg": a.vfov_up_deg, "vfov_down_deg": a.vfov_down_deg},
                    crack_fill_px=a.fill_width,
                    max_scale_m=a.max_scale, checkpoint=str(a.checkpoint),
                    note="qualitative demo; viewpoints kept near the capture rig")


def run_long(r, args):
    from dwm_pano.export_gs_long_demo import FRAMES
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = r.dataset(FRAMES)
    matches = [i for i, a in enumerate(dataset.base_dataset.clip_audit)
               if a["scene_token"] == args.scene_token]
    if len(matches) != 1:
        raise ValueError(f"scene token has {len(matches)} {FRAMES}-frame clips")
    batch = r.load(dataset, matches[0])
    images = batch["vae_images"]
    encode = lambda frames: torch.cat([r.pipeline.encode_half_latents(
        images[0, s:s + 2].to(r.device)) for s in frames], dim=0)
    with torch.no_grad():
        if args.latent_source == "gt":
            latents = encode(range(0, FRAMES, 2))[:FRAMES].cpu()
        else:
            latents = load_rollout_latents(args.rollout_dir, encode(range(0, CONTEXT, 2))[:CONTEXT])
    n = r.render(batch, latents, args.out / "gs_clean_views.mp4")
    (args.out / "clean_metadata.json").write_text(json.dumps(r.metadata(
        scene_token=args.scene_token, latent_source=args.latent_source,
        rollout_dir=str(args.rollout_dir) if args.rollout_dir else None, frames=n), indent=2) + "\n")


def run_clips(r, args):
    from dwm_pano.export_gs_batch149 import FRAMES
    dataset = r.dataset(FRAMES)
    by_key = {(a["scene_token"], int(c[0]["timestamp"])): i
              for i, (c, a) in enumerate(zip(dataset.base_dataset.clips,
                                             dataset.base_dataset.clip_audit))}
    job_start, last = time.monotonic(), 0.0
    for index in range(args.start, args.end):
        src = args.clips_root / f"clip{index:03d}"
        dst = args.out_root / f"clip{index:03d}"
        if (dst / "gs_clean_views.mp4").is_file():
            continue
        if not (src / "latents.pt").is_file():
            print(f"[gs-clean] missing clip {index}", flush=True)
            continue
        if args.time_budget_s and last and \
                time.monotonic() - job_start + 1.2 * last > args.time_budget_s:
            print(f"[gs-clean] budget reached before clip {index}", flush=True)
            break
        t0 = time.monotonic()
        summary = json.loads((src / "summary.json").read_text())
        batch = r.load(dataset, by_key[(summary["scene_token"],
                                        int(summary["anchor_timestamp_us"]))])
        latents = torch.load(src / "latents.pt", map_location="cpu", weights_only=True)["latents"]
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst / "gs_clean_views.tmp.mp4"
        n = r.render(batch, latents, tmp)
        tmp.replace(dst / "gs_clean_views.mp4")
        (dst / "clean_metadata.json").write_text(json.dumps(r.metadata(
            clip_index=index, scene_token=summary["scene_token"], frames=n), indent=2) + "\n")
        last = time.monotonic() - t0
        print(f"[gs-clean] clip {index} done ({last:.0f}s)", flush=True)


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    started = ensure_single_rank_process_group(device)
    try:
        r = Renderer(args, device)
        (run_long if args.scene_token else run_clips)(r, args)
    finally:
        if started:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
