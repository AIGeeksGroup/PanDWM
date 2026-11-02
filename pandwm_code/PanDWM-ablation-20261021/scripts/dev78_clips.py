"""Write the Dev-78 clip list (test_index = Dev-149 clip index, manifest order) as a JSON list for flicker_stats.py.
  python3 dev78_clips.py dev78_manifest_20261018.json out.json
"""
import json
import os
import sys

clips = [int(c["test_index"]) for c in json.load(open(sys.argv[1]))["clips"]]
assert len(clips) == 78 and len(set(clips)) == 78, len(clips)
tmp = sys.argv[2] + ".tmp"
json.dump(clips, open(tmp, "w"))
os.replace(tmp, sys.argv[2])
