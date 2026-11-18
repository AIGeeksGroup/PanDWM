"""HD map rendering onto cylindrical pano for 2-pano driving WM.

Loads nuScenes map-expansion-v1.3 JSONs, queries nearby polygons per frame
(within distance threshold from ego), transforms world→ego, and projects each
polygon vertex to pano via the cam-aware forward projection in pano_projection.
Then draws polygon outlines as a series of line segments on the pano canvas
(same as OpenDWM's native pinhole renderer at nuscenes.py:478-567).

Map JSON structure (from /path/to/pandwm_root/data/nuscenes/maps/expansion/{location}.json):
  - node[i]: {token, x, y}     — 2D coord in WORLD / map frame, flat (z=0)
  - polygon[i]: {token, exterior_node_tokens[], holes[{node_tokens}]}
  - drivable_area[i]: {token, polygon_tokens[]}   — multi-polygon region
  - lane[i], ped_crossing[i], walkway[i], road_segment[i]: {polygon_token}
  - stop_line[i], road_divider[i], lane_divider[i]: {line_token} (line segments, not polygons)

Default color table matches OpenDWM MotionDataset.default_hdmap_color_table
(nuscenes.py:105 area).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import ImageDraw
from pyquaternion import Quaternion

from dwm_pano.datasets.pano_projection import (
    PANO_W, PANO_H,
    draw_line_on_pano_with_cams,
    project_ego_point_to_pano_with_cams,
    _segment_crosses_wrap,
)


DEFAULT_MAP_ROOT = "/path/to/pandwm_root/data/nuscenes/maps/expansion"

# Match OpenDWM defaults (see nuscenes.py default_hdmap_color_table)
DEFAULT_HDMAP_COLORS = {
    "drivable_area":  (0, 128, 128),
    "lane":           (255, 128, 0),
    "ped_crossing":   (255, 255, 0),
    "walkway":        (0, 200, 0),
    "stop_line":      (255, 0, 0),
    "carpark_area":   (128, 0, 128),
}

# Default query radius (meters), matches OpenDWM nuscenes.py:513 default max_distance=65.0
DEFAULT_MAX_DISTANCE = 65.0


class MapCache:
    """Loads + caches nuScenes HD map JSON data. Singleton per-location."""
    _instances: dict = {}

    def __init__(self, location: str, map_root: str = DEFAULT_MAP_ROOT):
        path = os.path.join(map_root, f"{location}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Map JSON not found: {path}")
        with open(path, "r") as f:
            raw = json.load(f)

        # Index nodes by token for O(1) lookup
        self.nodes = {n["token"]: (n["x"], n["y"]) for n in raw.get("node", [])}
        self.polygons = {p["token"]: p for p in raw.get("polygon", [])}

        # Layer → list of polygon-containing records (for query_in_patch)
        self.layers = {}
        for layer in ["drivable_area", "lane", "ped_crossing", "walkway", "road_segment",
                      "carpark_area", "stop_line"]:
            if layer in raw:
                self.layers[layer] = raw[layer]

    @classmethod
    def get(cls, location: str, map_root: str = DEFAULT_MAP_ROOT) -> "MapCache":
        if location not in cls._instances:
            cls._instances[location] = cls(location, map_root)
        return cls._instances[location]

    def polygon_exterior_world_pts(self, polygon_token: str) -> np.ndarray:
        """Return (N, 2) exterior ring coords in world/map frame for a polygon."""
        poly = self.polygons.get(polygon_token)
        if poly is None:
            return np.zeros((0, 2))
        pts = np.array([self.nodes[t] for t in poly["exterior_node_tokens"]
                        if t in self.nodes])
        return pts

    def polygon_centroid_world(self, polygon_token: str) -> tuple:
        """Return (cx, cy) centroid for distance filtering."""
        pts = self.polygon_exterior_world_pts(polygon_token)
        if pts.shape[0] == 0:
            return (0.0, 0.0)
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))


def world_to_ego(pts_world_2d, ego_translation, ego_rotation):
    """Transform (N, 2) or (N, 3) world-frame points to ego frame.
    Input assumed z=0 if 2D. Output is (N, 3)."""
    if pts_world_2d.shape[-1] == 2:
        pts_world = np.hstack([pts_world_2d, np.zeros((pts_world_2d.shape[0], 1))])
    else:
        pts_world = np.asarray(pts_world_2d)
    q_e2g = Quaternion(ego_rotation)
    q_g2e = q_e2g.inverse
    t_e2g = np.asarray(ego_translation)
    shifted = pts_world - t_e2g
    return np.stack([q_g2e.rotate(p) for p in shifted])


def _draw_polygon_outline_on_pano(draw, pts_ego, cam_translations, pen_color, pen_width, closed=True):
    """Draw closed polygon outline with cam-aware projection + wrap handling."""
    n = len(pts_ego)
    if n < 2:
        return
    for i in range(n - 1):
        draw_line_on_pano_with_cams(draw, pts_ego[i], pts_ego[i + 1],
                                    cam_translations, pen_color, pen_width)
    if closed:
        draw_line_on_pano_with_cams(draw, pts_ego[-1], pts_ego[0],
                                    cam_translations, pen_color, pen_width)


def render_hdmap_on_pano(
    info: dict,
    map_cache: MapCache,
    cam_translations: dict,
    color_table: dict = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    pen_width: int = 3,
    mode: str = "outline",  # "outline" (edges only) or "fill" (TODO)
):
    """Render HD map layers onto the full 9600×600 pano image.

    Args:
        info: pkl info dict. Must have `ego2global_translation`, `ego2global_rotation`.
        map_cache: MapCache for info["location"].
        cam_translations: dict name → (3,) sensor2ego translation.
        color_table: layer → (r, g, b). Defaults to OpenDWM-compatible palette.
        max_distance: discard polygons whose centroid is farther than this (m) from ego.
        pen_width: stroke width in full-pano pixels.
        mode: only "outline" supported for now.

    Returns: PIL.Image of 9600×600 with polygons drawn on black background.
    """
    from PIL import Image
    img = Image.new("RGB", (PANO_W, PANO_H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    if color_table is None:
        color_table = DEFAULT_HDMAP_COLORS

    ego_t = info["ego2global_translation"]
    ego_r = info["ego2global_rotation"]

    # Per-layer rendering
    for layer_name, color in color_table.items():
        if layer_name not in map_cache.layers:
            continue
        records = map_cache.layers[layer_name]
        for rec in records:
            # Get polygon token(s)
            if "polygon_token" in rec:
                poly_tokens = [rec["polygon_token"]]
            elif "polygon_tokens" in rec:
                poly_tokens = rec["polygon_tokens"]
            else:
                continue

            for pt in poly_tokens:
                # Distance filter via centroid
                cx, cy = map_cache.polygon_centroid_world(pt)
                # Need ego pose to filter: transform centroid to ego and check
                centroid_world = np.array([[cx, cy]])
                centroid_ego = world_to_ego(centroid_world, ego_t, ego_r)[0]
                if np.sqrt(centroid_ego[0]**2 + centroid_ego[1]**2) > max_distance:
                    continue

                # Polygon exterior
                pts_world = map_cache.polygon_exterior_world_pts(pt)
                if pts_world.shape[0] < 3:
                    continue
                pts_ego = world_to_ego(pts_world, ego_t, ego_r)
                _draw_polygon_outline_on_pano(
                    draw, pts_ego, cam_translations, color, pen_width, closed=True)
    return img
