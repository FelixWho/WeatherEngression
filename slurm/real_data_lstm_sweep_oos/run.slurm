#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-lstm-sweep-oos
#SBATCH --output=slurm/real_data_lstm_sweep_oos/run_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%8

# Re-run the SAME stonet + pre-additive LSTM sweep grid, but with the embedding-
# space OOS diagnostic ENABLED (kNN + Mahalanobis distance-from-training, coverage
# stratified by distance). The original sweep used --no-oos and --no-checkpoint,
# so the models were not saved and OOS was never computed -- this job retrains on
# GPU, computes OOS in-process, and ALSO saves checkpoints so future OOD splits
# (event / ccn_tail) can be evaluated via evaluate_oos_from_checkpoint.py without
# retraining.
#
# Writes to a NEW dir (lstm_sweep_oos/) so the original lstm_sweep/ leaderboard is
# left untouched. Grid = 2 (head) x 2 (lr) x 2 (layers) x 2 (hidden) = 16.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- parameter grid (identical to slurm/real_data_lstm_sweep) ----------------
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

v=${VARIANTS[$(( i % nV ))]};  i=$(( i / nV ))
lr=${LRS[$(( i % nLR ))]};     i=$(( i / nLR ))
nl=${LAYERS[$(( i % nL ))]};   i=$(( i / nL ))
hd=${HIDDENS[$(( i % nH ))]}

if [ "$v" = "stonet" ]; then HEAD_FLAG="--stonet-head"; else HEAD_FLAG="--pre-additive"; fi

TAG="${v}_lr${lr}_l${nl}_h${hd}"
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep_oos/${TAG}"
echo "[task ${SLURM_ARRAY_TASK_ID}/${TOTAL}] head=${v} ${HEAD_FLAG}  lr=${lr} layers=${nl} hidden=${hd}  -> ${OUT}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Same training config as the original sweep, but OOS is ON (no --no-oos) and
# checkpoints are saved (no --no-checkpoint). Embedding-OOS uses the LSTM encoder,
# so it works for both the stonet and pre-additive heads.
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
  --oos-reference-size 2000 \
  --oos-bins 10 \
  --checkpoint-every 25 \
  --device cuda \
  --out-dir "${OUT}" \
  --skip-assertions
