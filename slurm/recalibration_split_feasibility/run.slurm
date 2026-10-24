#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-recal-split-counts
#SBATCH --output=slurm/recalibration_split_feasibility/run_%j.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Full-data split inventory only: no fitting, predictive draws, or GPU work.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m experiments.wildfire_sensitivity.split_feasibility \
  --mat-path "${MAT_PATH:-/storage3/fs1/myu/Active/felixhu/weather_data.mat}" \
  --gap-days 10 \
  --counts 25 50 75 100 150 200 250 300 \
  --seeds 2228 2229 2230 \
  --output-dir "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/recalibration_split_feasibility/${SLURM_JOB_ID}"
