"""Read every file in a committed real checkpoint using CPU memory only."""
import argparse
import json
from pathlib import Path

import torch
from dwm_pano_cfdmd.checkpoint_io import validate_manifest

parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("--step", type=int, default=5)
args = parser.parse_args()
root = Path(args.output)
manifest = validate_manifest(root, args.step, 8)
checked = []
for relative in manifest["files"]:
    state = torch.load(root / relative, map_location="cpu", weights_only=False)
    if relative.startswith("training_state/"):
        assert state["step"] == args.step
        assert state["world_size"] == 8
        assert state["generator_optimizer"]["state"]
        assert state["critic_optimizer"]["state"]
        assert state["grad_scaler"]
        assert set(state["rng"]) == {"python", "numpy", "torch", "cuda", "generator"}
        if "disc_optimizer" in state:
            assert state["disc_optimizer"]["state"]
        print("READBACK_PASS", relative, "rank", state["rank"], flush=True)
    else:
        assert isinstance(state, dict) and state
        assert all(isinstance(tensor, torch.Tensor) for tensor in state.values())
        print("READBACK_PASS", relative, "keys", len(state), flush=True)
    del state
    checked.append(relative)
result = {"step": args.step, "world_size": 8, "readable_files": checked, "passed": True}
(root / "log" / f"checkpoint_readback_{args.step}.json").write_text(json.dumps(result, indent=2) + "\n")
print("FULL_CHECKPOINT_READBACK_PASS", args.step, len(checked), flush=True)
