#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pit-unif
#SBATCH --output=slurm/pit_uniformity/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# PIT uniformity p-values for every cell of the lstm_stochastic_sweep_oos grid.
# One process loops all 48 checkpoints (no array): rebuild the identical paper
# split once, then for each checkpoint resample, form the PIT, and run the
# Anderson-Darling / chi-squared / KS uniformity tests plus the autocorrelation
# (tau, n_eff) diagnostic and BH correction across the grid.
#
# Data args match slurm/pit_charts_stochastic_sweep_oos/run.slurm so the PIT is
# computed on the identical 3220-point test set and reproduces the plotted
# histograms. Writes pit_uniformity.csv / .json into the sweep run dir.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u experiments/pit_uniformity.py \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --prediction-samples 400 \
  --n-bins 20 \
  --n-sim 2000 \
  --device cuda
