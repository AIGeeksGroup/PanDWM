"""G1 timing gate for MagicDrive-V2 (2026-10-21): its frame times (meta.json pts_ms, ms from clip start) must be the
LIDAR_TOP times of our clip frames (the band timing) within 1 ms for every clip and scored frame."""
import json
import sys
from pathlib import Path

import cam_band_feats as cbf

run = Path(sys.argv[2])
config, dataset, collate, rows = cbf.load_val150(sys.argv[1])
base = dataset.base_dataset
m = {int(v["test_index"]): int(k) for k, v in json.load(open("val150_scene_map.json")).items()}
worst, bad = 0.0, 0
for row in rows:
    pos = int(row["test_index"])
    w = base.clips[int(row["relaxed_index"])]
    meta = json.loads((run / f"clip{m[pos]:03d}" / "meta.json").read_text())
    pts = meta["pts_ms"] if isinstance(meta["pts_ms"], list) else json.loads(meta["pts_ms"])
    for t in range(3, 35):
        d = abs(pts[t] - (int(w[t]["timestamp"]) - int(w[0]["timestamp"])) / 1000.0)
        worst = max(worst, d)
        bad += d >= 1.0
print(f"MD2_TIMING clips={len(rows)} frames>=1ms: {bad}, worst {worst:.3f} ms -> {'PASS' if bad == 0 else 'FAIL'}")
