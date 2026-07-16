#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-lstm-raw-ccn
#SBATCH --output=slurm/real_data_lstm_diagnostic/run_raw_ccn_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

cd /home/felixhu/WeatherEngression

# Use the project venv so `python` is the right interpreter with the right
# dependencies (torch, h5py, engression). Per the repo convention, all installs
# live in requirements.txt and are pinned into this venv.
source .venv/bin/activate

# Quick CUDA sanity check before launching training. The login node has no GPU,
# so this run MUST land on a general-gpu node for --device cuda to work.
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
nvidia-smi

# LSTM-engression diagnostic on the REAL ENA weather data (weather_data.mat).
#   --target ccn                 Predict the scalar CCN target from the full 22-var trajectory
#                                (CCN is not inside X, so there is no leakage).
#   --no-log-ccn                 Train/evaluate on raw CCN concentration instead of log10(CCN).
#   --split paper                Source-paper held-out months:
#                                Jan/Mar/May/Jul/Sep/Nov 2022 are test; all
#                                other months are train.
#   --max-samples all            Load every usable trajectory from weather_data.mat.
#   --train-size all --test-size all
#                                Use the full paper train/test pools, with no row caps.
#   --seq-stride 1               Keep all 241 trajectory time steps.
#   --epochs 120                 Longer schedule than a smoke run, to separate
#                                under-dispersion-from-undertraining from genuine
#                                misspecification when reading the coverage numbers.
#   --hidden-dim 192 --noise-dim 96 --num-layer 3   Same head capacity as the synthetic LSTM runs.
#   --lr 0.003 --weight-decay 0.0
#   --prediction-samples 800     Conditional draws used for quantiles AND the energy score.
#   --checkpoint-every 1         Update checkpoint_latest.pt every epoch and checkpoint_best.pt
#                                whenever training energy loss improves.
#   --device cuda                Train/sample on the allocated GPU.
#   --skip-assertions            Always write artifacts; do not fail the job on loose thresholds.
#
# Writes README.md, metrics.json, and posterior_bands.png under
#   runs/real_data_diagnostics/ena_weather/ccn/paper/lstm/
python -u experiments/real_data_diagnostic.py \
  --target ccn \
  --no-log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --epochs 120 \
  --batch-size 256 \
  --hidden-dim 192 \
  --noise-dim 96 \
  --num-layer 3 \
  --lr 0.003 \
  --weight-decay 0.0 \
  --prediction-samples 800 \
  --checkpoint-every 10 \
  --device cuda \
  --skip-assertions
