"""Train clips with their degraded GS render condition (2026-10-21, D2_PLAN_20261021.md Task 3).

The render files come from PanDWM-gs-aa-ft dwm_pano/d2_make_train.py:
    <render_root>/clip<NNNNN>.pt    {"render": (35, 2, 16, h, w) fp16, "mask": (35, 2, h, w) uint8, "meta": {...}}
    <render_root>/clip<NNNNN>.json  the same meta (read at start-up so the 21 MB files are not opened twice)
meta holds scene_token, anchor_timestamp_us and frame_timestamps_us (35) of the GS-dataset clip.  They are
paired with the video dataset's clips by (scene_token, anchor_timestamp_us); every frame timestamp must agree,
otherwise __getitem__ raises (a one-frame offset would silently train next-frame prediction).

Each item is the base item plus "d2_render" = cat(render, mask / 255) on the channel axis: (35, 2, 17, h, w) fp16.
"""

import json
from pathlib import Path

import torch


class D2RenderDataset(torch.utils.data.Dataset):
    def __init__(self, base, render_root, min_items=1):
        self.base = base
        self.render_root = Path(render_root)
        inner = getattr(base, "base_dataset", base)
        key_to_index = {(audit["scene_token"], int(clip[0]["timestamp"])): i
                        for i, (clip, audit) in enumerate(zip(inner.clips, inner.clip_audit))}
        self.items, unmatched = [], 0
        for meta_path in sorted(self.render_root.glob("clip*.json")):
            pt_path = meta_path.with_suffix(".pt")
            if not pt_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            key = (meta["scene_token"], int(meta["anchor_timestamp_us"]))
            if key not in key_to_index:
                unmatched += 1
                continue
            self.items.append((key_to_index[key], pt_path, [int(x) for x in meta["frame_timestamps_us"]]))
        self.unmatched = unmatched
        if len(self.items) < int(min_items):
            raise RuntimeError(f"{len(self.items)} render files match the base clips under {self.render_root} "
                               f"({unmatched} unmatched); need at least {min_items}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        base_index, path, frame_timestamps = self.items[index]
        item = self.base[base_index]
        got = [int(x) for x in item["frame_timestamps_us"]]
        if got != frame_timestamps:
            raise ValueError(f"{path.name}: frame timestamps differ from base clip {base_index}")
        pack = torch.load(path, map_location="cpu", weights_only=True)
        render, mask = pack["render"], pack["mask"]
        frames = len(frame_timestamps)
        if render.ndim != 5 or render.shape[0] != frames or tuple(mask.shape) != (frames, *render.shape[1:2],
                                                                                  *render.shape[3:]):
            raise ValueError(f"{path.name}: render {tuple(render.shape)} / mask {tuple(mask.shape)} malformed")
        coverage = (mask.to(torch.float16) / 255.0).unsqueeze(2)
        item["d2_render"] = torch.cat([render.to(torch.float16), coverage], dim=2)
        return item
