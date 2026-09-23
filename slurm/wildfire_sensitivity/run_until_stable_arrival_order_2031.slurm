#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-arr-2031
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_order_2031_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Full input histories, scalar arrival targets, until-stable screening.
# Shuffle candidate order once with seed 2031; training/split seeds stay 2026.
# The shared runner logs all seeds and saves checkpoints separately for each job.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
export CANDIDATE_ORDER_SEED=2031
exec bash slurm/wildfire_sensitivity/run.slurm
