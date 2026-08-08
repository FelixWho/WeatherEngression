#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-aggregate
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/aggregate_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Evaluate the partial-learner gridsearch and write reports/partial_learner_effect/
# (grid_metrics.json, attenuation_curve.png, money plots). Pass the grid array
# job id, e.g.:  GRID_RUN_ID=2251623 sbatch slurm/partial_learner/aggregate.slurm

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

GRID_RUN_ID="${GRID_RUN_ID:?set GRID_RUN_ID to the grid array job id}"
echo "grid run id: ${GRID_RUN_ID}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.partial_learner.aggregate_grid \
  --grid-run-id "${GRID_RUN_ID}" \
  --n-samples 400 \
  --out-dir reports/partial_learner_effect

echo "done -> reports/partial_learner_effect/"
