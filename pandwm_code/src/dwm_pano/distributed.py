import os
import torch.distributed
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def distributed_save_optimizer_state(model, optimizer, folder, filename):
    if torch.distributed.is_initialized():
        # Drain pending NCCL collectives on all process groups before the
        # optimizer save's plan-stage gather; otherwise a preceding full-mesh
        # all_gather on the same CUDA stream can cause an NCCL error or a
        # watchdog timeout when the save opens collectives on the inner-shard
        # subgroup.
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
            # Compute the set of ranks responsible for the save.
            # `model._device_mesh.mesh[0].tolist()` works for a 2D mesh
            # (HYBRID_SHARD): mesh[0] is the first row (shard group of node 0).
            # On a 1D mesh (FULL_SHARD, single node) mesh[0] is a 0-dim tensor
            # and `rank in int` errors. For 1D FULL_SHARD all ranks save, since
            # the save is a NCCL collective on
            # `process_group=device_mesh.get_group(-1)` and saving on rank 0
            # only would hang the other ranks.
            _mesh = model._device_mesh.mesh
            if _mesh.ndim == 0:
                _save_ranks = [int(_mesh.item())]
            elif _mesh.ndim == 1:
                # FULL_SHARD: the whole 1D mesh is one shard group, so all ranks save
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
    # The caller (e.g., the DMD pipeline) may have opted out of the optimizer
    # save; Adam momentum rebuilds quickly after resume.
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
