"""Check that rank-local Adam snapshots reproduce the next FSDP update."""
import os
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from dwm_pano_cfdmd.checkpoint_io import atomic_save

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.distributed.init_process_group("nccl")
torch.manual_seed(71)
model = FSDP(torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.SiLU(),
                                 torch.nn.Linear(32, 8)).cuda(), device_id=rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
sample = torch.randn(4, 32, device="cuda")


def step():
    optimizer.zero_grad()
    loss = model(sample).square().mean()
    loss.backward()
    optimizer.step()


step()
snapshot = {"parameters": [p.detach().clone() for p in model.parameters()],
            "optimizer": optimizer.state_dict()}
path = Path(os.environ["CAMPAIGN_DIR"]) / "preflight" / f"rank{rank}.pth"
atomic_save(snapshot, path)
step()
reference = [p.detach().clone() for p in model.parameters()]
state = torch.load(path, map_location="cpu", weights_only=False)
with torch.no_grad():
    for parameter, saved in zip(model.parameters(), state["parameters"]):
        parameter.copy_(saved.to(parameter.device))
optimizer.load_state_dict(state["optimizer"])
step()
for parameter, expected in zip(model.parameters(), reference):
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
torch.distributed.barrier()
if rank == 0:
    print("PREFLIGHT_PASS: 8-rank FSDP Adam resume matches uninterrupted update", flush=True)
torch.distributed.destroy_process_group()
