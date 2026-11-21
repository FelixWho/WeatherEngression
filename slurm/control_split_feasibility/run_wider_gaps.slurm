#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-control-counts-15-20d
#SBATCH --output=slurm/control_split_feasibility/wider_%A_%a.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --array=0-2

set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GAPS=(10 15 20)
python -u -m experiments.wildfire_sensitivity.split_feasibility \
  --population-from /storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663360 \
  --spaced-role control --gap-days "${GAPS[$SLURM_ARRAY_TASK_ID]}" \
  --control-guard-days 0 --counts 25 35 40 45 50 75 --seeds 2228 \
  --output-dir "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/${SLURM_ARRAY_JOB_ID}/gap_${GAPS[$SLURM_ARRAY_TASK_ID]}"
