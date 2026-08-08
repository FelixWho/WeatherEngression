#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-tl-recurrent
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/t_learner/train_recurrent_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# T-learner training for the RECURRENT architecture (#1 by CRPS in the sweep).
# Fits both arms (wildfire + wildfire-free) WITH early stopping, checkpoints to
# storage3. Companions: train_per_timestep.slurm, train_global_latent.slurm.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- architecture config (the only block that differs between the 3 files) ----
ARCH="recurrent"
HEAD_FLAG="--recurrent-state-noise"
HIDDEN=192
LAYER=5
# ------------------------------------------------------------------------------

WILDFIRE_FLAG="${WILDFIRE_FLAG:-BB_criterion1}"
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
SAVE_DIR="${CKPT_ROOT}/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/t_learner/${WILDFIRE_FLAG}_${ARCH}_${SLURM_JOB_ID:-local}"

echo "arch          : ${ARCH} (${HEAD_FLAG}, h${HIDDEN}, l${LAYER})"
echo "job id        : ${SLURM_JOB_ID:-local}"
echo "wildfire flag : ${WILDFIRE_FLAG}"
echo "checkpoints   -> ${SAVE_DIR}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.t_learner.train \
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
