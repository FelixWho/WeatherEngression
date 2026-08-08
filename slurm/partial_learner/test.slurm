#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-pl-test
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/partial_learner/test_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Evaluate a partial-learner run: contrast the ALL-DATA arm against the CLEAN arm.
# Set CHECKPOINT_DIR to a dir written by one of the partial_learner train_*.slurm
# files (holds checkpoint_{all,clean}_best.pt + dataset/ pickles), e.g.:
#   CHECKPOINT_DIR=/storage3/.../partial_learner/BB_criterion1_recurrent_12345 \
#     sbatch slurm/partial_learner/test.slurm

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a partial_learner run dir}"

echo "checkpoint dir : ${CHECKPOINT_DIR}"
if [ ! -f "${CHECKPOINT_DIR}/checkpoint_all_best.pt" ]; then
  echo "ERROR: no checkpoint_all_best.pt in ${CHECKPOINT_DIR}"; exit 1
fi
if [ ! -f "${CHECKPOINT_DIR}/dataset/dataset_obj.pkl" ]; then
  echo "ERROR: no dataset/dataset_obj.pkl in ${CHECKPOINT_DIR}"; exit 1
fi

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

python -u -m experiments.partial_learner.test \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --which best \
  --n-samples 400 \
  --out-dir reports/partial_learner_effect \
  --device cuda

echo "done. charts + metrics.json -> reports/partial_learner_effect/"
