"""Plan A: v4 training with full rank-local state and safe pause boundaries."""
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import dwm.distributed as dwm_distributed

from .checkpoint_io import atomic_json, atomic_save, validate_manifest
from .pipeline_v4 import PanoContextForcedDMDv4


class TrainingPaused(Exception):
    pass


class PanoContextForcedDMDResumable(PanoContextForcedDMDv4):
    def __init__(self, *args, checkpoint_steps=50, checkpoint_seconds=1800,
                 target_steps=2000, **kwargs):
        self._output = Path(kwargs["output_path"])
        self._rank = torch.distributed.get_rank()
        self._world = torch.distributed.get_world_size()
        resume = kwargs.get("resume_from")
        if resume is not None:
            validate_manifest(self._output, resume, self._world)
        # Parent loaders use distcp; this run uses rank-local Adam state instead.
        loader = dwm_distributed.distributed_load_optimizer_state
        if resume is not None:
            dwm_distributed.distributed_load_optimizer_state = lambda *a, **k: None
        try:
            super().__init__(*args, **kwargs)
        finally:
            dwm_distributed.distributed_load_optimizer_state = loader
        self._checkpoint_steps = checkpoint_steps
        self._checkpoint_seconds = checkpoint_seconds
        self._target_steps = target_steps
        self._last_saved = resume
        self._last_save_time = time.monotonic()
        self._pending_rng = None
        if resume is not None:
            state = torch.load(self._output / "training_state" / str(resume) /
                               f"rank{self._rank:03d}.pth",
                               map_location="cpu", weights_only=False)
            if state["rank"] != self._rank or state["world_size"] != self._world:
                raise ValueError("Rank-local optimizer checkpoint does not match")
            self.optimizer.load_state_dict(state["generator_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            if self.use_gan:
                self.disc_optimizer.load_state_dict(state["disc_optimizer"])
            self.grad_scaler.load_state_dict(state["grad_scaler"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            for name, value in state["cached_metrics"].items():
                setattr(self, name, value)
            self._pending_rng = state["rng"]
            del state
            if self.should_save:
                print(f"FULL_RESUME step={resume}: optimizers/scaler/RNG restored", flush=True)

    def _restore_rng(self, rng):
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)
        self.generator.set_state(rng["generator"])

    def restore_pending_rng(self):
        if self._pending_rng is not None:
            self._restore_rng(self._pending_rng)
            self._pending_rng = None

    def train_step(self, batch, global_step):
        self.restore_pending_rng()
        super().train_step(batch, global_step)
        step = global_step + 1
        pause_file = os.environ.get("PAUSE_FILE", "")
        request = torch.tensor(
            [int(bool(pause_file) and Path(pause_file).exists()),
             int(time.monotonic() - self._last_save_time >= self._checkpoint_seconds)],
            device=self.device)
        torch.distributed.all_reduce(request, op=torch.distributed.ReduceOp.MAX)
        pause, elapsed = request.tolist()
        if step == 5 or step % self._checkpoint_steps == 0 or elapsed or pause or step >= self._target_steps:
            self.save_checkpoint(str(self._output), step, upload=False)
        if self.should_save and (step % 10 == 0 or pause or step == self._target_steps):
            atomic_json({"step": step, "target_steps": self._target_steps,
                         "checkpoint_step": self._last_saved,
                         "status": "paused" if pause else "completed" if step >= self._target_steps else "training",
                         "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                         "metrics": self.loss_report_list[-1] if self.loss_report_list else {}},
                        self._output / "log" / "progress.json")
        if pause:
            if self.should_save:
                print(f"PAUSED_SAFELY step={step}; campaign will not start the next arm", flush=True)
            raise TrainingPaused()

    def save_checkpoint(self, output_path, steps, upload=False, keep_last_n=3):
        if steps == self._last_saved:
            return
        root = Path(output_path)
        stage = root / "checkpoints" / f".staging_{steps}"
        if self.should_save:
            stage.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        previous = os.environ.get("CKPT_OUT_DIR")
        os.environ["CKPT_OUT_DIR"] = str(stage)
        try:
            super().save_checkpoint(output_path, steps, upload=False)
        finally:
            if previous is None:
                os.environ.pop("CKPT_OUT_DIR", None)
            else:
                os.environ["CKPT_OUT_DIR"] = previous
        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state(self.device),
               "generator": self.generator.get_state()}
        state = {"step": steps, "rank": self._rank, "world_size": self._world,
                 "generator_optimizer": self.optimizer.state_dict(),
                 "critic_optimizer": self.critic_optimizer.state_dict(),
                 "grad_scaler": self.grad_scaler.state_dict(),
                 "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                 "rng": rng,
                 "cached_metrics": {n: getattr(self, n) for n in
                                    ("_last_gen_loss", "_last_dmd_grad_norm", "_last_adv")}}
        if self.use_gan:
            state["disc_optimizer"] = self.disc_optimizer.state_dict()
        atomic_save(state, stage / "training_state" / f"rank{self._rank:03d}.pth")
        del state
        torch.distributed.barrier()
        if self.should_save:
            destination = root / "training_state" / str(steps)
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                if (root / "checkpoints" / f"{steps}.complete.json").exists():
                    raise RuntimeError("Refusing to overwrite a committed checkpoint")
                shutil.rmtree(destination)
            os.replace(stage / "training_state", destination)
            files = list(destination.glob("*.pth"))
            for source in stage.glob("*.pth"):
                target = root / "checkpoints" / source.name
                os.replace(source, target)
                files.append(target)
            manifest = {"step": steps, "world_size": self._world,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "files": {str(p.relative_to(root)): p.stat().st_size for p in files}}
            expected = self._world + 2 + int(self.ema_state is not None) + int(self.use_gan)
            if len(files) != expected:
                raise RuntimeError("Checkpoint is missing model or rank state files")
            atomic_json(manifest, root / "checkpoints" / f"{steps}.complete.json")
            atomic_json(manifest, root / "log" / "latest_checkpoint.json")
            stage.rmdir()
            complete = sorted((int(p.name.split(".")[0]) for p in
                               (root / "checkpoints").glob("*.complete.json")))
            for old_step in complete[:-3]:
                old_manifest = validate_manifest(root, old_step, self._world)
                (root / "checkpoints" / f"{old_step}.complete.json").unlink()
                for relative in old_manifest["files"]:
                    (root / relative).unlink()
                shutil.rmtree(root / "training_state" / str(old_step))
            print(f"FULL_CHECKPOINT_COMMITTED step={steps} files={len(files)}", flush=True)
        torch.distributed.barrier()
        self._last_saved = steps
        self._last_save_time = time.monotonic()
