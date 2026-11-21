#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-cov90-95-s2032-stable
#SBATCH --chdir=/home/wg-fhu/WeatherEngression
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_coverage90_95_order_2032_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Arrival-only targets, full input histories, and until-stable screening.
# Only candidate order changes; training/data/split seeds remain 2026.
# Original blocked, month-matched split: 2,159 controls; no spacing constraint.
# Coverage gate: central 90% and 95%, each within 3 block standard errors.
# Every job saves its own model checkpoints, mappings, settings, and seeds.
set -euo pipefail
export REPO_ROOT=/home/wg-fhu/WeatherEngression
cd "${REPO_ROOT}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
export CALIBRATION_METRIC=coverage
export CONTROL_GAP_DAYS=0
export CANDIDATE_ORDER_SEED=2032
export AD_ALPHA=0.05
export AD_BOOTSTRAP_REPLICATES=2000
export AD_BOOTSTRAP_SEED=2430
export MAT_PATH=/storage3/fs1/myu/Active/felixhu/weather_data.mat
export CKPT_DIR="/storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity/coverage90_95_arrival_order_2032/${SLURM_JOB_ID:-local}"
exec bash slurm/wildfire_sensitivity/run.slurm

