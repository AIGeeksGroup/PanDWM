"""Magnification culling of shifted-view renders (2026-10-30, GS_NEARFIX_PLAN_20261030.md step 1).

Near-object collapse at large lateral shifts (GS_IMPROVEMENT_SURVEY_20261030.md section 1): a pixel-aligned Gaussian
seen from a camera several metres away can cover many times the screen area it covered in its own source view, as a
stretched smear (near content magnified, needles seen side-on, depth-edge curtains).  Here every Gaussian's angular
size is measured from both viewpoints: the largest standard deviation of its covariance across the viewing ray
(the max eigenvalue of the covariance projected on the plane normal to the ray), divided by the distance.  A Gaussian
whose angular size at the target camera exceeds k_mag times the one at its source camera, and is larger than
min_rad at the target, is dropped, so the area becomes a hole for the push-pull fill (or a repair model) instead of a
smear.  Camera centres are approximated by the rig centre (ego origin raised by RIG_HEIGHT_M; the six cameras sit
within ~1 m of it).
"""

from __future__ import annotations

import torch


def quat_to_rot(q):
    """(N, 4) w, x, y, z (normalised here) -> (N, 3, 3)."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1).reshape(-1, 3, 3)


def angular_size(g, centres):
    """Largest across-ray standard deviation over distance (radians) of (N, 14) Gaussians (decoder order: means
    0:3, quaternion 7:11, scales 11:14) seen from centres (N, 3) or (3,)."""
    mu = g[:, 0:3]
    v = mu - centres
    dist = v.norm(dim=-1).clamp_min(1e-3)
    v = v / dist[:, None]
    rot = quat_to_rot(g[:, 7:11])
    cov = rot @ torch.diag_embed(g[:, 11:14] ** 2) @ rot.transpose(1, 2)
    helper = torch.where((v[:, 2:3].abs() < 0.9), v.new_tensor([0.0, 0.0, 1.0]).expand_as(v),
                         v.new_tensor([1.0, 0.0, 0.0]).expand_as(v))
    e1 = torch.nn.functional.normalize(torch.cross(v, helper, dim=-1), dim=-1)
    e2 = torch.cross(v, e1, dim=-1)
    basis = torch.stack([e1, e2], dim=1)                                          # (N, 2, 3)
    c2 = basis @ cov @ basis.transpose(1, 2)                                      # (N, 2, 2)
    half_tr = 0.5 * (c2[:, 0, 0] + c2[:, 1, 1])
    det = c2[:, 0, 0] * c2[:, 1, 1] - c2[:, 0, 1] * c2[:, 1, 0]
    lam = half_tr + (half_tr * half_tr - det).clamp_min(0.0).sqrt()
    return lam.clamp_min(0.0).sqrt() / dist


def magnification_keep(g, src_centres, tgt_centre, k_mag, min_rad):
    """Boolean keep mask over (N, 14) Gaussians: False where the target angular size is > k_mag x the source one and
    > min_rad.  src_centres (N, 3) or (3,); tgt_centre (3,).  Returns (keep, magnification)."""
    if g.shape[0] == 0:
        return torch.ones(0, dtype=torch.bool, device=g.device), g.new_zeros(0)
    a_src = angular_size(g, src_centres).clamp_min(1e-6)
    a_tgt = angular_size(g, tgt_centre)
    mag = a_tgt / a_src
    return ~((mag > k_mag) & (a_tgt > min_rad)), mag
