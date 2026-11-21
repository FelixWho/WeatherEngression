#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ad-seasonal76-audit
#SBATCH --chdir=/home/wg-fhu/WeatherEngression
#SBATCH --output=slurm/ad_control_spacing_validation/seasonal76_%A_%a.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-6%3

# Exact saved 76-control calendar; synthetic PIT only, no model fitting.
# Five known-uniform cases measure false rejections; two wide cases measure power.
set -euo pipefail
cd /home/wg-fhu/WeatherEngression
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCENARIOS=(iid_uniform moderate_3day strong_3day strong_10day strong_30day wide_iid wide_10day)
python -u -m experiments.validate_ad_control_spacing \
  --inventory /storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663781/gap_15 \
  --control-split /storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2027/3685284/arrival/screening_split.npz \
  --scenario "${SCENARIOS[$SLURM_ARRAY_TASK_ID]}" \
  --repetitions 1000 --bootstrap-replicates 2000 --null-simulations 20000 \
  --seed 2433 --alpha 0.05 \
  --output-dir "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_control_spacing_validation/${SLURM_ARRAY_JOB_ID}/${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
