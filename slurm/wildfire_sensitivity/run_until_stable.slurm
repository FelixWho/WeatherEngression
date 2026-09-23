#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-cal-stable
#SBATCH --output=slurm/wildfire_sensitivity/run_until_stable_calibration_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Wildfire-sensitivity screening, repeated until it settles.
#
# Same job as run.slurm with --screening-mode until_stable: after a pass moves
# covariates into the insensitive set M, sensitive and undetermined candidates
# are re-tested against the now-larger M until a whole round moves nobody.
# Every fit must pass the central 90% and 95% clean-control calibration checks
# before a sensitivity decision; run.slurm logs the active thresholds.
# Each round refits every remaining candidate from scratch; the whole job has
# a four-hour limit, matching the single-pass job.
#
# Note the open question in the README: the result may depend on the order the
# candidates are considered in, which this mode reduces but does not settle.

set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

export SCREENING_MODE=until_stable
exec bash slurm/wildfire_sensitivity/run.slurm
