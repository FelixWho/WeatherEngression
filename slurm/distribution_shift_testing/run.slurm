#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-shift
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/distribution_shift_testing/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00

# Train-vs-test covariate distribution-shift tests for the ENA back-trajectories:
#   (1) C2ST  -- MLP + linear classifier two-sample test (AUC + permutation importance)
#   (2) kNN   -- test->train vs train->train nearest-neighbor distance (overlap/support)
# The MLP classifier is the "complex model" -> run on GPU.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.distribution_shift_testing.run \
  --device cuda \
  --split paper \
  --seed 2026 \
  --decorrelate-stride 24 \
  --n-repeats 5 \
  --epochs 60 \
  --reference-size 4000 \
  --out-dir reports/distribution_shift_testing

echo "done -> reports/distribution_shift_testing/{results.json,c2st_feature_importance.png,knn_overlap.png}"
