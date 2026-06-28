#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-lstm-ccn
#SBATCH --output=slurm/real_data_lstm_diagnostic/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

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
#   --split random               Shuffled in-distribution split (conditional-distribution test).
#   --max-samples 8000           Usable trajectories to load (CCN-finite, fully finite features).
#   --seq-stride 4               Keep every 4th time step: 241 -> ~60 steps. Finer temporal
#                                detail than the CPU smoke run (stride 8), affordable on GPU.
#   --train-size 5000 --test-size 800
#   --epochs 120                 Longer schedule than the 30-epoch CPU run, to separate
#                                under-dispersion-from-undertraining from genuine
#                                misspecification when reading the coverage numbers.
#   --hidden-dim 192 --noise-dim 96 --num-layer 3   Same head capacity as the synthetic LSTM runs.
#   --lr 0.003 --weight-decay 0.0
#   --prediction-samples 800     Conditional draws used for quantiles AND the energy score.
#   --device cuda                Train/sample on the allocated GPU.
#   --skip-assertions            Always write artifacts; do not fail the job on loose thresholds.
#
# Writes README.md, metrics.json, and posterior_bands.png under
#   runs/real_data_diagnostics/ena_weather/ccn/random/lstm/
python -u experiments/real_data_diagnostic.py \
  --target ccn \
  --split random \
  --max-samples 8000 \
  --seq-stride 4 \
  --train-size 5000 \
  --test-size 800 \
  --epochs 120 \
  --batch-size 512 \
  --hidden-dim 192 \
  --noise-dim 96 \
  --num-layer 3 \
  --lr 0.003 \
  --weight-decay 0.0 \
  --prediction-samples 800 \
  --device cuda \
  --skip-assertions
