#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-tlearner-test
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/t_learner/test_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Evaluate the trained T-learner arms (companion to train.slurm): reload BOTH
# models (no retraining), sample both at the held-out covariates, and run the
# effect estimation in experiments/t_learner/test.py.
#
# CHECKPOINT_DIR must point at a dir written by train.slurm holding both the
# checkpoints AND the dataset/ pickles (train.py saves the dataset + arm indices
# there). test.py reads those pickles, so no split params are needed and nothing
# can drift from the training run. Override per run, e.g.:
#   CHECKPOINT_DIR=/storage3/.../BB_criterion2_recurrent_12345 sbatch slurm/t_learner/test.slurm

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints"
# Default: the most recent crit1 recurrent run (the verbose one).
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CKPT_ROOT}/runs/real_data_diagnostics/ena_weather/log10_ccn/paper/t_learner/BB_criterion1_recurrent_2236345}"

echo "checkpoint dir : ${CHECKPOINT_DIR}"

if [ ! -f "${CHECKPOINT_DIR}/checkpoint_wildfire_best.pt" ]; then
  echo "ERROR: no checkpoint_wildfire_best.pt in ${CHECKPOINT_DIR}"; exit 1
fi
if [ ! -f "${CHECKPOINT_DIR}/dataset/dataset_obj.pkl" ]; then
  echo "ERROR: no dataset/dataset_obj.pkl in ${CHECKPOINT_DIR} (run predates the pickle dump)"; exit 1
fi

# Must land on a GPU node for --device cuda (fast sampling).
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Dataset + arm indices are read from the pickles under CHECKPOINT_DIR/dataset/,
# so the reloaded arms always match -- no split params to keep in sync.
python -u -m experiments.t_learner.test \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --which best \
  --n-samples 400 \
  --out-dir "${OUT_DIR:-reports/t_learner_effect}" \
  --device cuda

echo "done. charts + metrics.json -> ${OUT_DIR:-reports/t_learner_effect}/"
