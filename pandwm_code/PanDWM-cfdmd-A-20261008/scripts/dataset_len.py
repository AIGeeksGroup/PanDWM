"""Print the training dataset length and the per-rank DistributedSampler length for a config (2026-10-13).

  python3 scripts/dataset_len.py <config.json> [world_size=8] [step]
With a step, also prints the --resume-data-offset vendor_snapshot/train.py accepts for it (step % sampler length).
"""
import json
import math
import sys

import dwm.common


def main():
    cfg = json.load(open(sys.argv[1]))
    world = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n = len(dwm.common.create_instance_from_config(cfg["training_dataset"]))
    per_rank = math.ceil(n / world)
    print(f"DATASET_LEN {n} SAMPLER_LEN {per_rank} world={world}", flush=True)
    if len(sys.argv) > 3:
        print(f"RESUME_DATA_OFFSET {int(sys.argv[3]) % per_rank}", flush=True)


if __name__ == "__main__":
    main()
