#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-ad-c15-unmatched-s2037-stable
#SBATCH --chdir=/home/wg-fhu/WeatherEngression
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_ad_control15_order_2037_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Arrival-only AD sensitivity screening, repeated until stable.
# Reserve maximum earliest-first 15-day-spaced control from ALL clean rows.
# No control month matching or surrounding exclusion zone; other split rules stay.
# The same spaced controls supply corrected PIT for AD and clean 95% coverage.
# AD remains the null-centered 30-day block bootstrap, alpha .05, 2,000 replicates.
# Only candidate order differs across runs; model/data seeds stay 2026.
set -euo pipefail
export REPO_ROOT=/home/wg-fhu/WeatherEngression
cd "${REPO_ROOT}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
export CALIBRATION_METRIC=ad
export CONTROL_GAP_DAYS=15
export CANDIDATE_ORDER_SEED=2037
export AD_ALPHA=0.05
export AD_BOOTSTRAP_REPLICATES=2000
export AD_BOOTSTRAP_SEED=2430
export CKPT_DIR="/storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/ad_control15_arrival_order_2037/${SLURM_JOB_ID:-local}"
exec bash slurm/wildfire_sensitivity/run.slurm
