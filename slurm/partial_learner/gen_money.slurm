#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-money
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/gen_money_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Regenerate the 3 partial-learner money plots (all-data vs clean, at oversample 1x)
# for the report, on GPU (login-node CPU sampling is too slow / gets killed).

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available())"

python -u - <<'PY'
from experiments.partial_learner.test import load_saved, plot_money
from experiments.t_learner.test import estimand
CKPT = "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/partial_learner_grid"
for arch in ["recurrent", "per_timestep", "global_latent"]:
    d = f"{CKPT}/BB_criterion1_{arch}_os1x_2251623"
    eng_all, eng_clean, ds, _, _, te_wf, _ = load_saved(d, device="cuda")
    eff = estimand(eng_all, eng_clean, ds.x[te_wf], 400)
    plot_money(eff["factual"], eff["counterfactual"],
               f"All-data vs clean model ({arch}, oversample 1x)",
               f"reports/partial_learner_effect/money_{arch}_os1x.png")
    print(f"{arch}: ratio {eff['ratio']:.2f} -> saved", flush=True)
PY
echo "done -> reports/partial_learner_effect/money_{recurrent,per_timestep,global_latent}_os1x.png"
