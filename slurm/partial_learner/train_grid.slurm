#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-grid
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/train_grid_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-8%6

# Partial-learner GRIDSEARCH: top-3 LSTM architectures x 3 wildfire oversamplings.
#   architectures : recurrent (h192,l5), per_timestep (h128,l5), global_latent (h128,l3)
#   oversamplings : 1x (natural ~11% wildfire), 4x (~33%), 8x (~balanced 50/50)
# = 9 tasks. Each fits the CLEAN arm and the ALL-DATA arm (with the given wildfire
# oversample) using early stopping, and checkpoints to its own storage3 dir.
#
# Task i -> arch = i/3, oversample = i%3:
#   0 recurrent 1x | 1 recurrent 4x | 2 recurrent 8x
#   3 per_timestep 1x | 4 per_timestep 4x | 5 per_timestep 8x
#   6 global_latent 1x | 7 global_latent 4x | 8 global_latent 8x

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- grid definition ----
ARCHS=(recurrent per_timestep global_latent)
HEAD_FLAGS=(--recurrent-state-noise --per-timestep-noise --global-latent-noise)
HIDDENS=(192 128 128)
LAYERS=(5 5 3)
OVERSAMPLES=(1 4 8)

nOver=${#OVERSAMPLES[@]}
i=${SLURM_ARRAY_TASK_ID:-0}
arch_idx=$(( i / nOver ))
over_idx=$(( i % nOver ))

ARCH=${ARCHS[$arch_idx]}
HEAD_FLAG=${HEAD_FLAGS[$arch_idx]}
HIDDEN=${HIDDENS[$arch_idx]}
LAYER=${LAYERS[$arch_idx]}
OVERSAMPLE=${OVERSAMPLES[$over_idx]}

WILDFIRE_FLAG="${WILDFIRE_FLAG:-BB_criterion1}"
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
RUN_ID="${SLURM_ARRAY_JOB_ID:-local}"
SAVE_DIR="${CKPT_ROOT}/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/partial_learner_grid/${WILDFIRE_FLAG}_${ARCH}_os${OVERSAMPLE}x_${RUN_ID}"

echo "task ${i}: arch=${ARCH} (${HEAD_FLAG}, h${HIDDEN}, l${LAYER})  oversample=${OVERSAMPLE}x"
echo "checkpoints -> ${SAVE_DIR}"

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
