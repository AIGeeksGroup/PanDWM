"""Use the frozen trainer snapshot and exit cleanly on a committed pause."""
import os
import runpy
import signal
import sys
from pathlib import Path

from arm_policy import record_skip, skip_reason

# Active Slurm jobs retain their submitted launcher. Enforce the updated
# campaign decision here before allocating models or loading training data.
skipped_output = skip_reason(sys.argv[1:])
if skipped_output is not None:
    record_skip(skipped_output)
    print("ARM_SKIPPED_NOGAN: compare GAN training with original B0", flush=True)
    sys.exit(0)

import torch
from PIL import ImageFile
from dwm_pano_cfdmd.pipeline_resumable import TrainingPaused

ImageFile.LOAD_TRUNCATED_IMAGES = True


def request_pause(signum, frame):
    Path(os.environ["PAUSE_FILE"]).touch()


signal.signal(signal.SIGUSR1, request_pause)
trainer = Path(__file__).resolve().parents[1] / "vendor_snapshot" / "train.py"
sys.argv[0] = str(trainer)
try:
    runpy.run_path(str(trainer), run_name="__main__")
except TrainingPaused:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.exit(0)
