#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-lstm-stoch-sweep
#SBATCH --output=slurm/real_data_lstm_stochastic_sweep/run_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-47%8

# Hyperparameter sweep for the NOISE-IN-THE-LSTM engression variants on the real
# ENA weather data. Unlike the stonet/pre-additive heads (noise in the MLP head),
# these inject fresh noise into the SEQUENCE or the RECURRENCE, then a
# deterministic MLP maps h(X) -> Y. Six variants:
#   - `additive`     : add fresh noise onto the final timestep of x.
#   - `appending`    : append a fresh noise timestep -> LSTM reads (B, S+1, F).
#   - `per_timestep` : concat fresh noise to EVERY timestep -> (B, S, F+k). [StoNet analogue]
#   - `global_latent`: one latent z per sample, broadcast to every timestep. [CVAE form]
#   - `init_state`   : seed the LSTM initial (h0, c0) from noise.
#   - `recurrent`    : per-step noise added to the hidden state (LSTMCell loop). [slowest]
# All are valid energy-trained conditional samplers (like StoNet, not pre-ANM, so
# no extrapolation theorem). Watch the spread term for noise collapse -- single-
# injection variants (additive/appending/init_state) are more collapse-prone than
# the many-injection ones (per_timestep/global_latent/recurrent).
#
# Grid = 6 (variant) x 2 (lr) x 2 (layers) x 2 (hidden) = 48 -> --array=0-47.
# %8 runs up to 8 in parallel. Update the array range if you change the grid.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- parameter grid ---------------------------------------------------------
VARIANTS=(additive appending per_timestep global_latent init_state recurrent)
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

case "$v" in
  additive)      HEAD_FLAG="--additive-noise" ;;
  appending)     HEAD_FLAG="--appending-noise" ;;
  per_timestep)  HEAD_FLAG="--per-timestep-noise" ;;
  global_latent) HEAD_FLAG="--global-latent-noise" ;;
  init_state)    HEAD_FLAG="--stochastic-init-noise" ;;
  recurrent)     HEAD_FLAG="--recurrent-state-noise" ;;
  *) echo "unknown variant $v"; exit 1 ;;
esac

TAG="${v}_lr${lr}_l${nl}_h${hd}"
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep/${TAG}"
echo "[task ${SLURM_ARRAY_TASK_ID}/${TOTAL}] variant=${v} ${HEAD_FLAG}  lr=${lr} layers=${nl} hidden=${hd}  -> ${OUT}"

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
