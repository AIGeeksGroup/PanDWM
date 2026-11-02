"""Zero-shot generative repair of shifted-view renders with NVIDIA difix_ref (2026-10-30, GS_NEARFIX_PLAN_20261030.md
step 2).

Model: huggingface.co/nvidia/difix_ref @ d4830559 (NVIDIA License, research / non-commercial use; SD-Turbo base,
Stability AI Community License), code github.com/nv-tlabs/Difix3D @ c76edc59 (src/pipeline_difix.py), run in a separate
worker process with the versions Difix3D pins (repair_difix_server.py).  One step at timestep 199, prompt "remove
degradation", one reference image; the pipeline stacks input and reference along the batch axis, so each camera is
one call.

Per shifted frame: the six real cameras are rendered (current over 4D fill, as eval_gs_shift does for the stitched
pano), each camera's holes (alpha < 0.5) are push-pull filled, difix repairs it with that camera's 0 m render as the
reference, and two outputs are stitched with the pipeline's own hard seam:
* "fixm" (masked): the repair replaces only the holes, dilated by `dilate` px and feathered over `feather` px;
  covered pixels stay the render (protects colour / fidelity, as ExtraGS / ArtiFixer / DriveX do);
* "fixg" (magnification-aware mask, 10-30 after the --mag-debug diagnostic): holes plus pixels whose content comes
  from Gaussians magnified > mag_thr x against their source view (mag_cull.angular_size, rendered as a colour
  channel), i.e. the low-resolution near content that stretches into smears; same dilation and feather;
* "fixf" (full): the repaired camera image everywhere.
Pixels outside the band (stitch_valid False) are push-pull filled afterwards exactly like the unrepaired pano.
Renders are 640x360 per camera; difix runs at `res` (default 1024x576) and the result is area-resized back.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from dwm_pano.calibrated_stitch import stitch_camera_views_from_owned_grid
from dwm_pano.export_gs_v2_clips import pushpull_fill
from dwm_pano.gs_scene_v3 import composite

PROMPT = "remove degradation"


class DifixRepair:
    """Client of repair_difix_server.py: the pinned Difix3D stack runs in its own interpreter (see that file)."""

    def __init__(self, model_dir, src_dir, device, res=(576, 1024), pkgs_dir=""):
        import os
        import subprocess
        from pathlib import Path
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(x for x in (pkgs_dir, src_dir, env.get("PYTHONPATH", "")) if x)
        server = str(Path(__file__).with_name("repair_difix_server.py"))
        self.proc = subprocess.Popen([sys.executable, server, model_dir, src_dir], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)
        from dwm_pano.repair_difix_server import read_msg
        hello = read_msg(self.proc.stdout)
        if not hello or not hello.get("ready"):
            raise RuntimeError("difix worker did not start")
        print(f"[repair] difix_ref worker ready {hello['versions']}", flush=True)
        self.res, self.device = tuple(res), device
        self.calls = 0

    @torch.no_grad()
    def __call__(self, image, ref):
        """image / ref (3, h, w) in [0, 1] -> repaired (3, h, w) in [0, 1]."""
        from dwm_pano.repair_difix_server import read_msg, write_msg
        h, w = image.shape[-2:]
        up = lambda x: F.interpolate(x[None].float(), size=self.res, mode="bilinear",
                                     align_corners=False).clamp(0, 1)[0].cpu().numpy().astype("float16")
        write_msg(self.proc.stdin, {"image": up(image), "ref": up(ref)})
        reply = read_msg(self.proc.stdout)
        if reply is None:
            raise RuntimeError("difix worker died")
        self.calls += 1
        out = torch.from_numpy(reply["image"].astype("float32")).to(image.device)[None]
        return F.interpolate(out, size=(h, w), mode="area")[0].clamp(0, 1).to(image.dtype)

    def close(self):
        from dwm_pano.repair_difix_server import write_msg
        try:
            write_msg(self.proc.stdin, {"quit": True})
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def soft_mask(hole, dilate, feather):
    """(V, h, w) bool -> (V, h, w) float weight: holes dilated by `dilate` px, then a box feather of `feather` px."""
    m = hole.float()[:, None]
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    if feather > 0:
        m = F.avg_pool2d(m, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
    return m[:, 0].clamp(0, 1)


def render_cameras(pipe, current, fill, c2w):
    """Six-camera render of current over fill: rgb (6, 3, h, w), depth (6, h, w), alpha (6, h, w)."""
    kw = dict(target_c2w=c2w, target_intrinsics=pipe.nuscenes_render_intrinsics,
              width=pipe.nuscenes_render_width, height=pipe.nuscenes_render_height)
    rgb, depth, alpha = pipe.render_targets(current, **kw)
    if fill is not None and fill.shape[0]:
        rgb_f, depth_f, alpha_f = pipe.render_targets(fill, **kw)
        from dwm_pano.gs_scene_v3 import composite_depth
        depth = composite_depth(depth, alpha, depth_f, alpha_f)
        rgb, alpha = composite(rgb, alpha, rgb_f, alpha_f)
    return rgb.clamp(0, 1), depth, alpha


def stitch_pano(pipe, rgb, depth, alpha):
    """Six cameras -> (3, H, W) stitched pano, holes outside the band push-pull filled like eval_gs_shift."""
    channels = torch.cat([rgb, depth[:, None], alpha[:, None]], dim=1)
    stitched, _ = stitch_camera_views_from_owned_grid(channels, pipe.stitch_sampling_grid, pipe.stitch_owner,
                                                      pipe.stitch_valid)
    a = torch.where(pipe.stitch_valid, stitched[4], torch.zeros_like(stitched[4]))
    return pushpull_fill(stitched[:3].clamp(0, 1), a).clamp(0, 1)


def magnification_cameras(pipe, current, fill, src_cur, src_fill, tgt, c2w):
    """Per-camera alpha-weighted magnification (6, h, w) of current over fill seen from tgt (rig centre)."""
    from dwm_pano.mag_cull import angular_size
    parts = []
    for g, src in ((current, src_cur), (fill, src_fill)):
        if g is None or not g.shape[0]:
            parts.append(None)
            continue
        mag = angular_size(g, tgt) / angular_size(g, src).clamp_min(1e-6)
        vis = g.clone()
        vis[:, 3:6] = torch.stack([(mag / 10).clamp(0, 1), torch.zeros_like(mag), torch.zeros_like(mag)], -1)
        parts.append(vis)
    rgb, _, alpha = render_cameras(pipe, parts[0], parts[1], c2w)
    return rgb[:, 0] * 10.0 / alpha.clamp_min(1e-3)


@torch.no_grad()
def repaired_panos(pipe, fixer, current, fill, c2w, ref_rgb, dilate, feather, src_cur=None, src_fill=None, tgt=None,
                   mag_thr=2.0):
    """Difix repairs of one shifted frame as stitched panos (3, H, W): {"m": holes only, "g": holes + magnified
    content (needs src_cur / src_fill / tgt), "f": full}, plus {"hole": camera hole fraction, "magmask": fraction
    of camera pixels added by the magnification test}.  ref_rgb (6, 3, h, w): the cameras' 0 m renders."""
    rgb, depth, alpha = render_cameras(pipe, current, fill, c2w)
    filled = torch.stack([pushpull_fill(rgb[i], alpha[i]) for i in range(rgb.shape[0])]).clamp(0, 1)
    fixed = torch.stack([fixer(filled[i], ref_rgb[i]) for i in range(rgb.shape[0])])
    hole = alpha < 0.5
    ones = torch.ones_like(alpha)
    out, stats = {}, {"hole": float(hole.float().mean())}
    masks = {"m": hole}
    if tgt is not None:
        magnified = (magnification_cameras(pipe, current, fill, src_cur, src_fill, tgt, c2w) > mag_thr) & ~hole
        masks["g"] = hole | magnified
        stats["magmask"] = float(magnified.float().mean())
    for k, m in masks.items():
        w = soft_mask(m, dilate, feather)[:, None]
        out[k] = stitch_pano(pipe, w * fixed + (1.0 - w) * filled, depth, ones)
    out["f"] = stitch_pano(pipe, fixed, depth, ones)
    return out, stats
