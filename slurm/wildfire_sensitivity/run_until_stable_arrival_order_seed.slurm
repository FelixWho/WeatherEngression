#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-arr-order
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_order_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Same scalar-arrival training and until-stable screening as the baseline job.
# Only candidate order is shuffled; training and split seeds remain 2026.
# Candidate order is shuffled once, then retained among survivors in later rounds.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
export CANDIDATE_ORDER_SEED="${CANDIDATE_ORDER_SEED:-2027}"
exec bash slurm/wildfire_sensitivity/run.slurm
