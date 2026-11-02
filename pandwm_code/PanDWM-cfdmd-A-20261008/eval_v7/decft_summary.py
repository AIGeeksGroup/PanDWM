"""Summarise decoder fine-tuning runs (2026-10-12):  python decft_summary.py run2 run3a run3b
Per run: latest step and the mean over the last 10 log lines of psnr_gain / sharp_gain (raw and EMA decoder vs the
original decoder on the same batches; sharp_gain > 1 means sharper than the original)."""
import json
import sys
from pathlib import Path

W = Path("/path/to/pandwm_root/pandwm_runs/decoder_ft_20261012")
for run in sys.argv[1:]:
    f = W / run / "train_log.jsonl"
    if not f.is_file():
        print(f"{run}: no log yet")
        continue
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    last = rows[-10:]
    mean = lambda k: sum(r[k] for r in last if k in r) / max(1, sum(k in r for r in last))
    text = (f"{run}: step {rows[-1]['step']} psnr_gain {mean('psnr_gain'):+.3f} sharp_gain {mean('sharp_gain'):.3f} "
            f"d_loss {mean('d_loss'):.3f} d_weight {mean('d_weight'):.3f}")
    if "psnr_gain_ema" in rows[-1]:
        text += f" | ema psnr_gain {mean('psnr_gain_ema'):+.3f} sharp_gain {mean('sharp_gain_ema'):.3f}"
    print(text)
