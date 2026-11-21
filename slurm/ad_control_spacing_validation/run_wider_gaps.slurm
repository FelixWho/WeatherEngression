#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ad-control-10-15-20d
#SBATCH --output=slurm/ad_control_spacing_validation/wider_%A_%a.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-6

# Paired spacings: common seasonal count 45, plus each gap's spacing-only maximum.
# Synthetic scalar PIT only; no model training or trajectory loading.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SCENARIOS=(iid_uniform moderate_3day strong_3day strong_10day strong_30day wide_iid wide_10day)
python -u -m experiments.validate_ad_control_spacing \
  --inventory \
    /storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663781/gap_10 \
    /storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663781/gap_15 \
    /storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663781/gap_20 \
  --seasonal-count 45 \
  --scenario "${SCENARIOS[$SLURM_ARRAY_TASK_ID]}" \
  --repetitions "${REPETITIONS:-1000}" \
  --bootstrap-replicates 2000 --null-simulations 20000 \
  --seed 2433 --selection-seed 2434 \
  --output-dir "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_control_spacing_validation/${SLURM_ARRAY_JOB_ID}/${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
