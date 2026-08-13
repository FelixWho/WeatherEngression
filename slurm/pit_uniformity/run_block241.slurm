#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pit-blk
#SBATCH --output=slurm/pit_uniformity/run_block241_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Same PIT uniformity sweep as run.slurm, but with the dependence-robust null
# blocked at one full trajectory length (241 h = 10 days). Retained points are
# then >= one airmass window apart, so they share no trajectory history -- the
# maximally conservative independence assumption for this hourly Lagrangian data.
#
# Note: the six alternating test months leave only ~17 independent 10-day windows,
# so this run is low-power by construction: a non-rejection means "too few
# independent windows to distinguish from uniform", not proof of calibration. Read
# it as the conservative bound; the data-driven run (run.slurm) is the other end.
#
# Writes to pit_uniformity_block241h.csv/.json so it does not clobber run.slurm.

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
  --block-hours 241 \
  --out runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep_oos/pit_uniformity_block241h.csv \
  --device cuda
