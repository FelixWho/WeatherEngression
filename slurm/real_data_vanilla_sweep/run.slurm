#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-vanilla-sweep
#SBATCH --output=slurm/real_data_vanilla_sweep/run_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%8

# Hyperparameter sweep for the PUBLIC engression package's flat MLP model
# (`--engression-model vanilla`) on the real ENA weather data. One SLURM array
# task per grid point; `%8` runs up to 8 in PARALLEL at a time, SLURM queues the
# rest and starts them as GPUs free up. Drop the `%8` for unlimited concurrency,
# or change N to throttle differently. The (241 x 22) trajectory is flattened to
# 5302 features (no LSTM, no checkpoint, no embedding-OOD).
#
# Grid below = 3 (lr) x 2 (hidden) x 2 (layers) x 2 (noise) = 24 configs
# -> #SBATCH --array=0-23. Update the array range if you change the grid.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- parameter grid ---------------------------------------------------------
LRS=(0.001 0.003 0.01)
HIDDENS=(128 256)
LAYERS=(3 5)
NOISES=(64 128)

nLR=${#LRS[@]}; nH=${#HIDDENS[@]}; nL=${#LAYERS[@]}; nN=${#NOISES[@]}
TOTAL=$(( nLR * nH * nL * nN ))

i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -ge "$TOTAL" ]; then
  echo "task $i >= grid size $TOTAL; nothing to do"; exit 0
fi

# ---- decode the flat index into one config (mixed-radix) --------------------
lr=${LRS[$(( i % nLR ))]};      i=$(( i / nLR ))
hd=${HIDDENS[$(( i % nH ))]};   i=$(( i / nH ))
nl=${LAYERS[$(( i % nL ))]};    i=$(( i / nL ))
nd=${NOISES[$(( i % nN ))]}

TAG="lr${lr}_h${hd}_l${nl}_n${nd}"
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla_sweep/${TAG}"
echo "[task ${SLURM_ARRAY_TASK_ID}/${TOTAL}] ${TAG}  -> ${OUT}"

python -u -c "import torch; print('cuda:', torch.cuda.is_available())"

# Full-data paper split; modest epochs/samples to keep each sweep point cheap.
python -u experiments/real_data_diagnostic.py \
  --engression-model vanilla \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --epochs 80 \
  --batch-size 256 \
  --lr "${lr}" \
  --hidden-dim "${hd}" \
  --num-layer "${nl}" \
  --noise-dim "${nd}" \
  --prediction-samples 400 \
  --device cuda \
  --out-dir "${OUT}" \
  --skip-assertions
