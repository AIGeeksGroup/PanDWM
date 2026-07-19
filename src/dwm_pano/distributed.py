import os
import torch.distributed
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def distributed_save_optimizer_state(model, optimizer, folder, filename):
    if torch.distributed.is_initialized():
        # Drain prior NCCL collectives across all PGs before kicking off
        # the optim save's plan-stage gather. Without this, a preceding
        # 8-rank `get_model_state_dict` (full-mesh all_gather) on the same
        # CUDA stream leaves state that trips "NCCL Error 1: unhandled cuda
        # error" or 10-min watchdog timeout when the optim save then opens
        # collectives on the inner-shard subgroup.
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DDP: rank 0 save
            if torch.distributed.get_rank() == 0:
                state_dict = optimizer.state_dict()
                torch.save(
                    state_dict,
                    os.path.join(folder, "{}.pth".format(filename)))

        elif isinstance(model, FSDP):
            # Zero2, Zero3: all ranks save
            # Hybird Zero2, Zero3: ranks in the 1st group save
            state_dict = torch.distributed.checkpoint.state_dict\
                .get_optimizer_state_dict(model, optimizer)
            # Compute the set of ranks responsible for save.
            # Official line `model._device_mesh.mesh[0].tolist()` works for 2D
            # mesh (HYBRID_SHARD): mesh[0] = first ROW (shard group of node 0)
            # → list of ranks. But on 1D mesh (FULL_SHARD single-node), mesh[0]
            # is a 0-dim tensor → tolist() returns int, `rank in int` errors.
            # Per OpenDWM's own comment ("Zero2/Zero3: all ranks save; HYBRID:
            # ranks in 1st group save"), the intent for 1D FULL_SHARD is ALL
            # ranks save (the only shard group is everyone). The save call is
            # a NCCL collective on `process_group=device_mesh.get_group(-1)`,
            # so only-rank-0 hangs the other 3 forever (NCCL watchdog kills at
            # 10 min). Earlier attempt with `[mesh[0].item()]` caused exactly
            # that hang.
            _mesh = model._device_mesh.mesh
            if _mesh.ndim == 0:
                _save_ranks = [int(_mesh.item())]
            elif _mesh.ndim == 1:
                # FULL_SHARD: the whole 1D mesh is one shard group → all save
                _save_ranks = [int(r) for r in _mesh.tolist()]
            else:
                # HYBRID_SHARD 2D: first row only
                _save_ranks = [int(r) for r in _mesh[0].tolist()]
            if model._device_mesh is None or (
                torch.distributed.get_rank() in _save_ranks
            ):
                torch.distributed.checkpoint.state_dict_saver.save(
                    state_dict, checkpoint_id=os.path.join(folder, filename),
                    process_group=(
                        None if model._device_mesh is None
                        else model._device_mesh.get_group(mesh_dim=-1)
                    ))

        else:
            raise Exception(
                "Unsupported distribution framework to save the optimizer "
                "state.")

    else:
        state_dict = optimizer.state_dict()
        torch.save(state_dict, os.path.join(folder, "{}.pth".format(filename)))


def distributed_load_optimizer_state(model, optimizer, folder, filename):
    # No-op silently if neither the FSDP distcp dir nor the DDP `.pth` exist.
    # Caller (e.g., DMD pipeline) may have opted out of optim save — Adam
    # momentum rebuilds in ~100 steps after resume.
    fsdp_path = os.path.join(folder, filename)
    ddp_path = os.path.join(folder, "{}.pth".format(filename))
    if not os.path.exists(fsdp_path) and not os.path.exists(ddp_path):
        return

    if torch.distributed.is_initialized():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DDP: all ranks load the same
            state_dict = torch.load(
                ddp_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state_dict)
        elif isinstance(model, FSDP):
            # Zero2, Zero3: all ranks load
            # Hybird Zero2, Zero3: ranks in the 1st group load
            state_dict = torch.distributed.checkpoint.state_dict\
                .get_optimizer_state_dict(model, optimizer)
            torch.distributed.checkpoint.state_dict_loader.load(
                state_dict, checkpoint_id=fsdp_path,
                planner=torch.distributed.checkpoint.DefaultLoadPlanner(
                    allow_partial_load=True))

    else:
        state_dict = torch.load(
            ddp_path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state_dict)
