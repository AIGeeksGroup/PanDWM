"""train_resumable.py with a __main__ guard, so spawn-started dataloader workers can import it safely.

Spawned workers re-import the launching script as __mp_main__.  train_resumable.py runs the trainer at import
time, so every worker would start a second trainer; here only module-level settings the workers also need
(PIL's truncated-image tolerance) run at import.  Otherwise identical to train_resumable.py.
"""
import os
import runpy
import signal
import sys
from pathlib import Path

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def request_pause(signum, frame):
    Path(os.environ["PAUSE_FILE"]).touch()


def main():
    from arm_policy import record_skip, skip_reason

    skipped_output = skip_reason(sys.argv[1:])
    if skipped_output is not None:
        record_skip(skipped_output)
        print("ARM_SKIPPED_NOGAN: compare GAN training with original B0", flush=True)
        sys.exit(0)

    import torch
    from dwm_pano_cfdmd.pipeline_resumable import TrainingPaused

    signal.signal(signal.SIGUSR1, request_pause)
    trainer = Path(__file__).resolve().parents[1] / "vendor_snapshot" / "train.py"
    sys.argv[0] = str(trainer)
    try:
        runpy.run_path(str(trainer), run_name="__main__")
    except TrainingPaused:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        sys.exit(0)


if __name__ == "__main__":
    main()
