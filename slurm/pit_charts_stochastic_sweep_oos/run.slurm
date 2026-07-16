#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pit-charts
#SBATCH --output=slurm/pit_charts_stochastic_sweep_oos/run_%A_%a.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-47%8

# Regenerate the shape-calibration charts (PIT histogram, coverage-calibration
# curve, PIT-by-distance) for every config in the lstm_stochastic_sweep_oos grid
# WITHOUT retraining: reload each saved checkpoint, rebuild the identical paper
# split, resample, and write the charts next to the existing posterior_bands.png
# and coverage_vs_distance.png.
#
# Data args below are copied verbatim from the sweep that trained these
# checkpoints (slurm/real_data_lstm_stochastic_sweep_oos/run.slurm) so the split,
# seed (driver default 2026), and sample count match exactly -- otherwise PIT
# would be computed on a different test set. Grid = 6 x 2 x 2 x 2 = 48.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

# ---- same parameter grid / ordering as the training sweep --------------------
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

v=${VARIANTS[$(( i % nV ))]};  i=$(( i / nV ))
lr=${LRS[$(( i % nLR ))]};     i=$(( i / nLR ))
nl=${LAYERS[$(( i % nL ))]};   i=$(( i / nL ))
hd=${HIDDENS[$(( i % nH ))]}

TAG="${v}_lr${lr}_l${nl}_h${hd}"
RUN_DIR="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep_oos/${TAG}"
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
CKPT="${CKPT_ROOT}/${RUN_DIR}/checkpoint_best.pt"

echo "[task ${SLURM_ARRAY_TASK_ID}/${TOTAL}] ${TAG}"
echo "  checkpoint: ${CKPT}"
echo "  out dir   : ${RUN_DIR}"

if [ ! -f "${CKPT}" ]; then
  echo "  MISSING checkpoint, skipping"; exit 0
fi

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Charts write into the LOCAL run dir (--out-dir); the checkpoint is read from
# storage. --prediction-samples 400 matches the sweep so the PIT/coverage match.
python -u experiments/evaluate_oos_from_checkpoint.py \
  --checkpoint "${CKPT}" \
  --out-dir "${RUN_DIR}" \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --prediction-samples 400 \
  --oos-reference-size 2000 \
  --oos-bins 10 \
  --device cuda
