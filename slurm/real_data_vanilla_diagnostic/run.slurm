#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-vanilla-ccn
#SBATCH --output=slurm/real_data_vanilla_diagnostic/run_%j.log
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

# Built-in engression baseline on the REAL ENA weather data (weather_data.mat).
#   --engression-model vanilla   The PUBLIC engression package's own model: a flat
#                                stochastic MLP generator g(X, eps). It is NOT
#                                sequence-aware, so the (241 x 22) trajectory is
#                                FLATTENED to a 5302-dim feature vector per sample.
#                                (No checkpointing and no embedding-OOD: those are
#                                lstm-only and are auto-disabled for this model.)
#   --target ccn --log-ccn       Predict log10(CCN), matching the paper's target setup.
#   --split paper                Source-paper held-out months (Jan/Mar/May/Jul/Sep/Nov 2022
#                                are test; all other months are train).
#   --max-samples all            Every usable trajectory from weather_data.mat.
#   --seq-stride 1               Keep all 241 time steps (-> 241*22 = 5302 flat features).
#   --train-size all --test-size all   Full paper train/test pools, no caps.
#   --epochs 120                 Same schedule as the lstm paper run for comparability.
#   --hidden-dim 192 --noise-dim 96 --num-layer 3   Same generator capacity as the lstm head.
#   --lr 0.003                   (weight_decay is ignored by the vanilla model.)
#   --prediction-samples 800     Conditional draws for quantiles AND the energy score.
#   --device cuda                Train/sample on the allocated GPU.
#   --skip-assertions            Always write artifacts; do not fail on loose thresholds.
#
# Writes README.md, metrics.json, and posterior_bands.png under
#   runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla/
# This is the flat-MLP baseline to compare against the sequence-native lstm run.
python -u experiments/real_data_diagnostic.py \
  --engression-model vanilla \
  --target ccn \
  --log-ccn \
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
  --prediction-samples 800 \
  --device cuda \
  --skip-assertions
