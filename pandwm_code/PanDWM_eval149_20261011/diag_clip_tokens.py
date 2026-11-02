"""Eval-149 clip index -> scene token (35-frame clips, fps 10, one per scene), and the Dev-78 clips of given scenes."""
import json
import sys

import dwm.common

cfg_path, split_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
config = json.load(open(cfg_path))
ds = config["validation_dataset"]["base_dataset"]
ds["sequence_length"] = 35
ds["fps_stride_tuples"] = [[10, 20]]
dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
toks = [a["scene_token"] for a in dataset.base_dataset.clip_audit]
long12 = ["2f56eb47c64f43df8902d9f88aa8a019", "ed242d80ccb34b139aaf9ab89859332e", "9068766ee9374872a380fe75fcfb299e",
          "ca6e45c25d954dc4af6e12dd7b23454d", "380ff00ec86447e3b986edc8e82ffba7", "85651af9c04945c3a394cf845cb480a6",
          "5521cd85ed0e441f8d23938ed09099dd", "0dae482684ce4cd69a7258f55bc98d73", "3363f396bb43405fbdd17d65dc123d4e",
          "4bb9626f13184da7a67ba1b241c19558", "3a2d9bf6115f40898005d1c1df2b7282", "7052d21b95fc4bae8761b8d9524f3e42"]
nontest = json.load(open(split_path))["non_test_clip_indices"]
excl = [i for i in nontest if toks[i] in long12]
keep = [i for i in nontest if toks[i] not in long12]
json.dump({"clip_scene_tokens": toks, "long12_clip_indices": excl, "dev66_clip_indices": keep}, open(out, "w"))
print(f"CLIPS {len(toks)}; long-12 scenes in Dev-78: {len(excl)} -> {excl}; pool keeps {len(keep)}")
