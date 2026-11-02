"""Extend Test-S / Test-L from 71 to 83 scenes (2026-10-24, user decision).

The 12 added scenes are the validation scenes outside the 71 test scenes that also have a valid 195-frame (12-window)
sequence: the 8 long-video development scenes and the 4 project-page scenes (long_video_hpc3_v2.sbatch, TOK array).
Clips are built exactly as build_tests_manifest.py builds Test-S (frozen eval dataset with fps_stride_tuples [[10, 5]],
clips at 5 / 10 / 15 s after the scene start, 35 frames).  New clips get test_index 213..248 and scene_order 71..82,
so their output folders (clipNNN) never collide with the 213 existing ones.

Writes:
  tests_ext36_manifest_20261024.json   the 36 new clips only (for generation)
  tests249_manifest_20261024.json      the 213 Test-S clips followed by the 36 new ones (for features / analysis)
  testl_ext12_long12_20261024.json     the 12 scenes in the Test-L list format (dev76_index = Dev-149 clip index)
  testl83_long12_20261024.json         the 71 Test-L scenes followed by the 12 new ones
"""
import hashlib
import json
from pathlib import Path

import dwm.common

C = Path(__file__).resolve().parent
STRIDE_S = 5.0
OFFSETS_S = (5.0, 10.0, 15.0)
CLIP = 35
NEW = [  # long_video_hpc3_v2.sbatch: 8 development scenes, then the 4 project-page scenes (10 / 31 / 52 / 72)
    "2f56eb47c64f43df8902d9f88aa8a019", "ed242d80ccb34b139aaf9ab89859332e", "9068766ee9374872a380fe75fcfb299e",
    "ca6e45c25d954dc4af6e12dd7b23454d", "380ff00ec86447e3b986edc8e82ffba7", "85651af9c04945c3a394cf845cb480a6",
    "5521cd85ed0e441f8d23938ed09099dd", "0dae482684ce4cd69a7258f55bc98d73",
    "3363f396bb43405fbdd17d65dc123d4e", "4bb9626f13184da7a67ba1b241c19558", "3a2d9bf6115f40898005d1c1df2b7282",
    "7052d21b95fc4bae8761b8d9524f3e42",
]


def dump(path, obj):
    text = json.dumps(obj, indent=1) + "\n"
    (C / path).write_text(text)
    print(f"WROTE {path} sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}")


def main():
    old = json.loads((C / "tests213_manifest_20261018.json").read_text())
    testl = json.loads((C / "review149_test71_long12_20261016.json").read_text())
    split = json.loads((C / "dev149_nontest_split_20261013.json").read_text())
    dev149 = json.loads((C / "dev149_manifest_20261018.json").read_text())
    test_tokens = set(split["test_scene_tokens"])
    assert len(NEW) == 12 and len(set(NEW)) == 12 and not test_tokens & set(NEW), "new scenes overlap the test scenes"
    dev_index = {r["scene_token"]: r["test_index"] for r in dev149["clips"]}
    missing = [t for t in NEW if t not in dev_index]
    assert not missing, f"not in Dev-149: {missing}"

    config = json.loads((C / "cd_eval_frozen149_hpc3.json").read_text())
    ds = config["validation_dataset"]["base_dataset"]
    assert ds["sequence_length"] == CLIP and ds["fps_stride_tuples"] == [[10, 20]]
    ds["fps_stride_tuples"] = [[10, STRIDE_S]]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    base = dataset.base_dataset
    assert len(dataset) == old["dataset_len"], (len(dataset), old["dataset_len"])
    first_ts = {}
    for clip, audit in zip(base.clips, base.clip_audit):
        first_ts.setdefault(audit["scene_token"], int(clip[0]["timestamp"]))
    rows = []
    for k, token in enumerate(NEW):
        found = {}
        for idx, (clip, audit) in enumerate(zip(base.clips, base.clip_audit)):
            if audit["scene_token"] != token:
                continue
            offset = round((int(clip[0]["timestamp"]) - first_ts[token]) / 1e6 / STRIDE_S) * STRIDE_S
            found[offset] = (idx, clip, audit)
        for off in OFFSETS_S:
            if off not in found:
                raise RuntimeError(f"scene {token} has no valid clip at {off} s (has {sorted(found)})")
            idx, clip, audit = found[off]
            rows.append({
                "test_index": old["count"] + len(rows), "dataset_index": idx, "scene_token": token,
                "scene_order": 71 + k, "offset_s": off,
                "anchor_timestamp_us": int(clip[0]["timestamp"]),
                "last_timestamp_us": int(clip[-1]["timestamp"]),
                "scene_first_timestamp_us": first_ts[token],
                "max_time_error_us": int(audit["max_time_error_us"]),
                "frames": CLIP, "context_frames": 3, "fps": 10,
            })
    note = ("Test-S extension 2026-10-24: the 12 validation scenes outside the 71 test scenes that have a valid 12-window "
            "sequence (8 long-video development scenes + 4 project-page scenes), clips at 5 / 10 / 15 s, built like "
            "tests213_manifest_20261018.json (dataset fps_stride_tuples [[10, 5]]); test_index 213..248")
    dump("tests_ext36_manifest_20261024.json", {"note": note, "dataset_fps_stride_tuples": [[10, STRIDE_S]],
                                                "dataset_len": len(dataset), "count": len(rows), "clips": rows})
    dump("tests249_manifest_20261024.json", {
        "note": "Test-S over 83 scenes (2026-10-24): tests213_manifest_20261018.json clips followed by "
                "tests_ext36_manifest_20261024.json clips",
        "dataset_fps_stride_tuples": [[10, STRIDE_S]], "dataset_len": len(dataset),
        "count": old["count"] + len(rows), "clips": old["clips"] + rows})
    ext = [{"dev76_index": dev_index[t], "scene_token": t} for t in NEW]
    dump("testl_ext12_long12_20261024.json", {
        "note": "Test-L extension 2026-10-24: the 12 added scenes (dev76_index = Dev-149 clip index), 12 windows",
        "windows": 12, "count": 12, "scenes": ext})
    dump("testl83_long12_20261024.json", {
        "note": "Test-L over 83 scenes (2026-10-24): review149_test71_long12_20261016.json scenes followed by the 12 "
                "added ones", "windows": 12, "count": 83, "scenes": testl["scenes"] + ext})
    print("NEW dev76_index", [r["dev76_index"] for r in ext])


if __name__ == "__main__":
    main()
