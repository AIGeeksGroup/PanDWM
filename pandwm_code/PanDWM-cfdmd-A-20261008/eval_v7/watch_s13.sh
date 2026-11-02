#!/usr/bin/env bash
# One idempotent step of the inference-only step-schedule test (2026-10-12): H 2250 with denoising_step_list
# [1000, 625, 250, 42] (arm v83h2250s13, jobs in EVAL_OUT/.submitted) against the same checkpoint with the training
# schedule (v83h2250, symlinked from the v8.3 eval).  Runs on the HPC3 login node; call it repeatedly:
#   1) generation done -> launch flicker_diag.py once (setsid, all fds redirected; marker .flicker);
#   2) fin1 report written and flicker_diag finished -> print FID / FVD of both arms and the flicker summary, DONE.
set -uo pipefail
R=/path/to/pandwm_root/pandwm_runs
E=$R/eval149_hpc3_s13_20261012
C=/path/to/pandwm_root/pandwm_code/PanDWM_eval149_20261011
PY=/path/to/pandwm_root/pandwm_venv/bin/python
read -r G F < "$E/.submitted"
g=$(ls "$E"/v83h2250s13/clip*/summary.json 2>/dev/null | wc -l)
if [ "$g" = 149 ] && [ ! -e "$E/.flicker" ]; then
    touch "$E/.flicker"
    cd "$C"
    PYTHONPATH=$C/vendor:$C/src OMP_NUM_THREADS=8 setsid nohup nice -n 10 "$PY" flicker_diag.py "$E" "$E/flicker_diag.json" \
        --pair v83h2250 v83h2250s13 b0 vaert v83h2250 v83h2250s13 > "$E/flicker_diag.log" 2>&1 < /dev/null &
    echo "FLICKER_LAUNCHED"
fi
st=$(sacct -j "$F" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
fl=0; [ -s "$E/flicker_diag.json" ] && fl=1
ok=0; grep -q METRICS_DONE "$E/metrics/metrics.log" 2>/dev/null && grep -q '"v83h2250s13"' "$E/metrics/report.json" && ok=1
echo "STATUS s13 clips=$g/149 fin1=$st metrics_ok=$ok flicker=$fl"
if [ "$ok" = 1 ] && [ "$fl" = 1 ]; then
    "$PY" - <<PYEOF
import json
base = json.load(open("$R/eval149_hpc3_v83_20261012/metrics/report_v83h2250.json"))["arms"]["v83h2250"]
new = json.load(open("$E/metrics/report.json"))["arms"]["v83h2250s13"]
for name, v in (("v83h2250", base), ("v83h2250s13", new)):
    print(f"{name} FID {v['fid_all']:.2f} FVD {v['fvd_w1']:.1f} / {v['fvd_w2']:.1f} sum {v['fvd_w1'] + v['fvd_w2']:.1f}")
d = json.load(open("$E/flicker_diag.json"))
for a in d["arms"]:
    s = d["summary"][a]
    print(a, "win %.3f bnd %.3f sharp %.3f tertiles %s" % (s["win_full"], s["bnd_full"], s["sharp"],
                                                           " ".join("%.2f" % x for x in s["by_motion"])))
print("pair", json.dumps(d["pair"]))
PYEOF
    echo "DONE s13"
fi
