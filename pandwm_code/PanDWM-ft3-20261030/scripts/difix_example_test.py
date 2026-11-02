"""2026-10-30 check of the difix_ref worker (repair_difix.DifixRepair) on Difix3D's own example input / reference:
writes the repaired image and reports the mean change, so the pinned-version worker is validated before use."""
import sys
import time

import numpy as np
import torch
from PIL import Image

from dwm_pano.repair_difix import DifixRepair

model, src, pkgs, out = sys.argv[1:5]
load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
img, ref = load(f"{src}/../assets/example_input.png"), load(f"{src}/../assets/example_ref.png")
fixer = DifixRepair(model, src, "cuda", res=tuple(img.shape[-2:]), pkgs_dir=pkgs)
t0 = time.time()
res = fixer(img, ref)
t1 = time.time()
res = fixer(img, ref)
print(f"[difix-test] input {tuple(img.shape)} first {t1 - t0:.2f}s second {time.time() - t1:.2f}s "
      f"mean |out-in| {float((res - img).abs().mean()):.4f} out range {float(res.min()):.3f}-{float(res.max()):.3f}")
Image.fromarray((res.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out)
fixer.close()
