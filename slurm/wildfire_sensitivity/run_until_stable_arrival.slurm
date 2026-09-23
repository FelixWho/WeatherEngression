#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-arrival
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_arrival_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Until-stable sensitivity screening at the observation site (final timestep).
# Recalibration, clean-control 90%/95% checks, and wildfire 95% coverage all
# use only the arrival value of each candidate covariate. Training and validation
# use a single arrival target (N, 1), with full input histories (N, 241, K).
# Uses the shared runner's 3-block-SE calibration bands and 2-percentage-point
# sensitivity margin. Models and arrival-specific mappings are saved under
# ${CKPT_DIR}/arrival (CKPT_DIR defaults to a separate directory for each job).

set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

export SCREENING_MODE=until_stable
export EVALUATION_SCOPE=arrival
exec bash slurm/wildfire_sensitivity/run.slurm
