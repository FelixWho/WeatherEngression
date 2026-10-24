#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ad-thinning-validation
#SBATCH --output=slurm/ad_thinning_validation/run_%j.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Separate synthetic PIT experiment; no model fitting or sensitivity screening.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m experiments.validate_ad_thinning \
  --block-sizes-from /storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/ad_arrival_order_2027/3277702/arrival/round_1/t_calibration.json \
  --gap-days 0 30 60 90 \
  --repetitions "${REPETITIONS:-500}" \
  --null-simulations 20000 \
  --seed "${SEED:-2431}" \
  --output "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_thinning_validation/${SLURM_JOB_ID}/results.json"
