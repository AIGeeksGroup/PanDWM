"""Count val scenes with a valid clip of 3 + 16 * W frames (fps 10, one clip per scene), split test / non-test."""
import json
import sys

import dwm.common

cfg_path, split_path = sys.argv[1], sys.argv[2]
test = set(json.load(open(split_path))["test_scene_tokens"])
for windows in (2, 4, 6, 8, 10, 12):
    config = json.load(open(cfg_path))
    ds = config["validation_dataset"]["base_dataset"]
    ds["sequence_length"] = 3 + 16 * windows
    ds["fps_stride_tuples"] = [[10, 20]]
    dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    toks = [a["scene_token"] for a in dataset.base_dataset.clip_audit]
    n_test = sum(t in test for t in toks)
    print(f"windows {windows:2d} frames {3 + 16 * windows:3d}: clips {len(toks)} test {n_test} non-test {len(toks) - n_test}",
          flush=True)
    if windows in (6, 8):
        json.dump(sorted(t for t in toks if t not in test), open(f"nontest_tokens_w{windows}.json", "w"))
