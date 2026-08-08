#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-region
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/effect_by_region_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

# All-data vs clean effect for the 3 os1x partial-learner runs, evaluated at three
# covariate sets: wildfire-only, clean-only, and all test data. Effect = all(x) - clean(x).

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate
python -u -c "import torch; print('cuda:', torch.cuda.is_available())"

python -u - <<'PY'
import numpy as np
from experiments.partial_learner.test import load_saved
from experiments.t_learner.test import estimand
CKPT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/partial_learner_grid"
print(f"{'arch':14s} {'eval region':12s} {'ratio':>7s} {'log10':>8s} {'raw(cm^-3)':>10s} {'n':>6s}", flush=True)
print("-"*62, flush=True)
for arch in ["recurrent", "per_timestep", "global_latent"]:
    d = f"{CKPT}/BB_criterion1_{arch}_os1x_2251623"
    eng_all, eng_clean, ds, _, _, te_wf, te_clean = load_saved(d, device="cuda")
    te_all = np.union1d(te_wf, te_clean)
    for region, idx in [("wildfire", te_wf), ("no-wildfire", te_clean), ("all", te_all)]:
        eff = estimand(eng_all, eng_clean, ds.x[idx], 400)
        print(f"{arch:14s} {region:12s} {eff['ratio']:7.3f} {eff['mean_log_effect']:+8.3f} "
              f"{eff['mean_raw_ccn_effect']:+10.1f} {eff['n_periods']:6d}", flush=True)
    print("-"*62, flush=True)
PY
echo "done"
