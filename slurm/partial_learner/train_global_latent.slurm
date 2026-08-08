#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-globallatent
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/train_global_latent_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Partial-learner training (GLOBAL-LATENT-NOISE). Fits the CLEAN arm and the
# ALL-DATA arm (clean + wildfire, EQUAL sample weights -- ~11% wildfire, no
# upweighting) with early stopping; checkpoints to storage3.
# Companions: train_recurrent.slurm, train_per_timestep.slurm.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- architecture config (the only block that differs between the 3 files) ----
ARCH="global_latent"
HEAD_FLAG="--global-latent-noise"
HIDDEN=128
LAYER=3
# ------------------------------------------------------------------------------

WILDFIRE_FLAG="${WILDFIRE_FLAG:-BB_criterion1}"
OVERSAMPLE="${WILDFIRE_OVERSAMPLE:-1}"
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
SAVE_DIR="${CKPT_ROOT}/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/partial_learner/${WILDFIRE_FLAG}_${ARCH}_${SLURM_JOB_ID:-local}"

echo "arch          : ${ARCH} (${HEAD_FLAG}, h${HIDDEN}, l${LAYER})"
echo "job id        : ${SLURM_JOB_ID:-local}"
echo "wildfire flag : ${WILDFIRE_FLAG}"
echo "oversample    : ${OVERSAMPLE}x"
echo "checkpoints   -> ${SAVE_DIR}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.partial_learner.train \
  --engression-model lstm \
  ${HEAD_FLAG} \
  --target ccn \
  --log-ccn \
  --split paper \
  --seq-stride 1 \
  --max-samples all \
  --train-size all \
  --test-size all \
  --wildfire-flag "${WILDFIRE_FLAG}" \
  --wildfire-oversample "${OVERSAMPLE}" \
  --epochs 100 \
  --early-stop-patience 12 \
  --batch-size 256 \
  --lr 0.003 \
  --hidden-dim "${HIDDEN}" \
  --num-layer "${LAYER}" \
  --noise-dim 96 \
  --device cuda \
  --save-checkpoint-dir "${SAVE_DIR}"

echo "done. checkpoints written to: ${SAVE_DIR}"
ls -la "${SAVE_DIR}"
