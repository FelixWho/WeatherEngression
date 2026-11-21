#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-control-spacing-counts
#SBATCH --output=slurm/control_split_feasibility/run_%j.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Spaced CONTROL from the full clean pool, not from an existing held-out set.
# No surrounding exclusion zone. No models are trained or AD tests performed.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m experiments.wildfire_sensitivity.split_feasibility \
  --population-from /storage3/fs1/myu/Active/felixhu/weather_checkpoints/recalibration_split_feasibility/3662101 \
  --spaced-role control \
  --gap-days 10 \
  --control-guard-days 0 \
  --counts 25 50 75 100 150 200 250 300 \
  --seeds 2228 2229 2230 \
  --output-dir "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/${SLURM_JOB_ID}"
