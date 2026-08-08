#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ate-money
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/t_learner/gen_money_ate_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate
python -u -c "import torch; print('cuda:', torch.cuda.is_available())"
python -u - <<'PY'
import json, numpy as np
from experiments.t_learner.test import load_saved, estimand, plot_money
for arch in ["recurrent","per_timestep","global_latent"]:
    ckpt = json.load(open(f"reports/t_learner_effect/{arch}/metrics.json"))["checkpoint_dir"]
    ewf, eclean, ds, _, _, te_wf, te_clean = load_saved(ckpt, device="cuda")
    te_all = np.union1d(te_wf, te_clean)                 # ATE = effect over ALL test covariates
    eff = estimand(ewf, eclean, ds.x[te_all], 400)
    plot_money(eff["factual"], eff["counterfactual"],
               f"Wildfire vs counterfactual over ALL test weather ({arch}, ATE)",
               f"reports/cummulative_summary/money_{arch}_ate.png")
    print(f"{arch}: ATE ratio {eff['ratio']:.3f} -> saved", flush=True)
PY
echo "done"
