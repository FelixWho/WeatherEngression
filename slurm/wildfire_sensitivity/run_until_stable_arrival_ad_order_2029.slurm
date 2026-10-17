#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-ad-arr-s2029-stable
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_ad_order_2029_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Arrival-only sensitivity screening, repeated until stable.
# Accept counterfactual models using AD on recalibrated clean-control PIT:
# p >= 0.05, 2,000 bootstrap replicates, 30-day blocks, at least 10 blocks.
# Only candidate-order seed varies across these three jobs. Training/data seeds
# remain 2026; the independent AD bootstrap seed stays 2430 for comparison.
# Full trajectory inputs (N, 241, K), scalar arrival targets (N, 1).
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
export CALIBRATION_METRIC=ad
export CANDIDATE_ORDER_SEED=2029
export AD_ALPHA=0.05
export AD_BOOTSTRAP_REPLICATES=2000
export AD_BOOTSTRAP_SEED=2430
export CKPT_DIR="${CKPT_DIR:-/storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/ad_arrival_order_2029/${SLURM_JOB_ID:-local}}"
exec bash slurm/wildfire_sensitivity/run.slurm
