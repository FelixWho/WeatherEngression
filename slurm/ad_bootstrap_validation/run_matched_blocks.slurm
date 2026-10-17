#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-ad-null-validation
#SBATCH --output=slurm/ad_bootstrap_validation/matched_blocks_%j.log
#SBATCH -p general-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

# Separate synthetic PIT audit. No neural fitting or sensitivity classifications.
# Match observed block sizes; repeat independent known-null experiments 500 times.
# Keep production's 2,000 bootstrap replicates; reuse p-values at .01/.05/.10.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUTPUT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_bootstrap_validation/${SLURM_JOB_ID}/results.json"
python -u -m experiments.validate_ad_bootstrap \
  --repetitions 500 \
  --bootstrap-replicates 2000 \
  --block-sizes-from "/storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/ad_arrival_order_2027/3277702/arrival/round_1/t_calibration.json" \
  --scenarios iid_uniform moderate_dependent_uniform dependent_uniform across_block_uniform miscalibrated \
  --seed 2430 \
  --output "${OUTPUT}"
