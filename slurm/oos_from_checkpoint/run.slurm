#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-oos-ckpt
#SBATCH --output=slurm/oos_from_checkpoint/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

cd /home/felixhu/WeatherEngression

# Use the project venv so `python` is the right interpreter with the right
# dependencies (torch, h5py, engression). Per the repo convention, all installs
# live in requirements.txt and are pinned into this venv.
source .venv/bin/activate

# Quick CUDA sanity check before launching. The login node has no GPU, so this
# job MUST land on a general-gpu node for --device cuda to work.
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
nvidia-smi

# Embedding-space OOD calibration from a SAVED checkpoint (no retraining).
# Reloads the fitted LSTM-engression model, rebuilds the identical paper split,
# draws conditional samples, and reports kNN + Mahalanobis coverage-by-distance.
#   --checkpoint                 The fitted model to evaluate. Defaults to the
#                                full-data log10(CCN)/paper run produced by
#                                slurm/real_data_lstm_diagnostic/run.slurm.
#   --prediction-samples 800     Conditional draws per test point for the
#                                quantiles. Match the training run so coverage
#                                numbers are directly comparable; this is the
#                                dominant cost and the part the GPU accelerates.
#   --device cuda                Sample/encode on the allocated GPU.
#
# Writes coverage_vs_distance.png and oos_from_checkpoint.json next to the
# checkpoint (runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm/).
python -u experiments/evaluate_oos_from_checkpoint.py \
  --checkpoint runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm/checkpoint_best.pt \
  --prediction-samples 800 \
  --device cuda
