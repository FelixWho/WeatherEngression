#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-split
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/distribution_shift_testing/split_metrics_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate
python -u - <<'PY'
import numpy as np
from experiments.distribution_shift_testing.features import load_split, decorrelate
ds, tr, te = load_split(split="paper", seed=2026)
def arm_counts(idx):
    c1 = ds.flags["BB_criterion1"][idx]
    c2 = ds.flags["BB_criterion2"][idx]
    wildfire = int(c1.sum())
    clean = int((~(c1 | c2)).sum())
    buffer = len(idx) - wildfire - clean
    return wildfire, clean, buffer
def daterange(idx):
    t = ds.time[idx]
    # MATLAB datenum -> year approx: (datenum-719529)/365.25 + 1970
    yrs = (t - 719529)/365.25 + 1970.0
    return float(yrs.min()), float(yrs.max())
print("TOTAL loaded:", len(ds.x))
for name, idx in [("train", tr), ("test", te)]:
    w,c,b = arm_counts(idx)
    y0,y1 = daterange(idx)
    dec = len(decorrelate(idx, 24))
    print(f"{name:5s} rows={len(idx):6d}  wildfire={w:5d} ({100*w/len(idx):.1f}%)  clean={c:6d} ({100*c/len(idx):.1f}%)  buffer={b:5d}  yr=[{y0:.1f},{y1:.1f}]  indep_episodes(stride24)={dec}")
PY
echo "done"
