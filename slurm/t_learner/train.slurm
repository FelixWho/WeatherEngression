#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-tlearner
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/t_learner/train_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Train BOTH T-learner arms on GPU and checkpoint them to storage3:
#   - wildfire (treatment) arm : samples flagged by WILDFIRE_FLAG
#   - wildfire-free (control) arm: samples flagged by NEITHER BB criterion
# One `python -m experiments.t_learner.train` call fits both arms and writes
# checkpoint_{wildfire,no_wildfire}_{best,latest}.pt into SAVE_DIR -- an ABSOLUTE
# storage3 path, so nothing touches the /home quota.
#
# Uses the sweep's best head (recurrent noise, lr 0.003, h192, l5). Set
# WILDFIRE_FLAG=BB_criterion2 for the strong-event contrast; the control arm is
# identical across criteria, so that run only differs in the wildfire arm.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

WILDFIRE_FLAG="${WILDFIRE_FLAG:-BB_criterion1}"
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
# Unique per job (SLURM_JOB_ID), so reruns never overwrite a previous run's files.
SAVE_DIR="${CKPT_ROOT}/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/t_learner/${WILDFIRE_FLAG}_recurrent_${SLURM_JOB_ID:-local}"

echo "job id        : ${SLURM_JOB_ID:-local}"
echo "wildfire flag : ${WILDFIRE_FLAG}"
echo "checkpoints   -> ${SAVE_DIR}"

# Must land on a GPU node for --device cuda.
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
nvidia-smi

# Full paper split, all samples, seq_stride 1 -- matching the sweep that produced
# the winning recurrent config. Fits the treatment arm then the control arm and
# writes checkpoint_{wildfire,no_wildfire}_{best,latest}.pt into SAVE_DIR.
python -u -m experiments.t_learner.train \
  --engression-model lstm \
  --recurrent-state-noise \
  --target ccn \
  --log-ccn \
  --split paper \
  --seq-stride 1 \
  --max-samples all \
  --train-size all \
  --test-size all \
  --wildfire-flag "${WILDFIRE_FLAG}" \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.003 \
  --hidden-dim 192 \
  --num-layer 5 \
  --noise-dim 96 \
  --device cuda \
  --save-checkpoint-dir "${SAVE_DIR}"

echo "done. checkpoints written to: ${SAVE_DIR}"
ls -la "${SAVE_DIR}"
