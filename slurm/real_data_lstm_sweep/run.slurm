#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-lstm-sweep
#SBATCH --output=slurm/real_data_lstm_sweep/run_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%8

# Hyperparameter sweep for the sequence-native LSTM engression heads on the real
# ENA weather data, applying the lessons from the vanilla sweep:
#   - learning rate is the master knob for under-dispersion (try 0.003 vs 0.01),
#   - depth tends to help coverage (3 vs 5 head layers),
#   - width tends to hurt/destabilize (128 vs 192).
# Two heads are swept: `stonet` (package StoNet, noise everywhere) and
# `pre-additive` (Y = g(phi(X)+eta), monotone g). Training uses early stopping on
# the training energy loss (compute saver; not a calibration guard).
#
# Grid = 2 (head) x 2 (lr) x 2 (layers) x 2 (hidden) = 16 -> --array=0-15.
# %8 runs up to 8 in parallel. Update the array range if you change the grid.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- parameter grid ---------------------------------------------------------
VARIANTS=(stonet preadd)
LRS=(0.003 0.01)
LAYERS=(3 5)
HIDDENS=(128 192)

nV=${#VARIANTS[@]}; nLR=${#LRS[@]}; nL=${#LAYERS[@]}; nH=${#HIDDENS[@]}
TOTAL=$(( nV * nLR * nL * nH ))

i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -ge "$TOTAL" ]; then
  echo "task $i >= grid size $TOTAL; nothing to do"; exit 0
fi

# ---- decode the flat index into one config (mixed-radix) --------------------
v=${VARIANTS[$(( i % nV ))]};  i=$(( i / nV ))
lr=${LRS[$(( i % nLR ))]};     i=$(( i / nLR ))
nl=${LAYERS[$(( i % nL ))]};   i=$(( i / nL ))
hd=${HIDDENS[$(( i % nH ))]}

if [ "$v" = "stonet" ]; then HEAD_FLAG="--stonet-head"; else HEAD_FLAG="--pre-additive"; fi

TAG="${v}_lr${lr}_l${nl}_h${hd}"
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep/${TAG}"
echo "[task ${SLURM_ARRAY_TASK_ID}/${TOTAL}] head=${v} ${HEAD_FLAG}  lr=${lr} layers=${nl} hidden=${hd}  -> ${OUT}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available())"

# Full-data paper split; max 100 epochs with early stop (patience 12) to cut
# wasted epochs. --no-oos keeps each sweep point lean (OOS is a separate job);
# checkpoints are always saved per repo policy (AGENTS.md).
python -u experiments/real_data_diagnostic.py \
  --engression-model lstm \
  ${HEAD_FLAG} \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --epochs 100 \
  --early-stop-patience 12 \
  --batch-size 256 \
  --lr "${lr}" \
  --hidden-dim "${hd}" \
  --num-layer "${nl}" \
  --noise-dim 96 \
  --prediction-samples 400 \
  --device cuda \
  --checkpoint-every 25 \
  --no-oos \
  --out-dir "${OUT}" \
  --skip-assertions
