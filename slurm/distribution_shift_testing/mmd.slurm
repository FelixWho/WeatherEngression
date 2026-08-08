#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-mmd
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/distribution_shift_testing/mmd_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Formal two-sample test for train-vs-test covariate shift: Gaussian-kernel MMD
# permutation test (+ energy-distance cross-check). Run on the SAME decorrelated
# episodes as the C2ST/kNN job so the permutation null is valid. GPU handles the
# n x n kernel matrix and the vectorized permutation loop.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.distribution_shift_testing.run_mmd \
  --device cuda \
  --split paper \
  --seed 2026 \
  --decorrelate-stride 24 \
  --n-perm 5000 \
  --out-dir reports/distribution_shift_testing

echo "done -> reports/distribution_shift_testing/{mmd_results.json,mmd_null.png}"
