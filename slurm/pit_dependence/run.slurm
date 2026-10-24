#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-recal-pit-dependence
#SBATCH --output=slurm/pit_dependence/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Reload saved models; evaluate only their clean recalibration rows. No fitting.
set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"
source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('This evaluation requires CUDA')
print('CUDA:', torch.cuda.get_device_name(0))
PY
CKPT_ROOT=/storage3/fs1/myu/Active/felixhu/weather_checkpoints
python -u -m experiments.wildfire_sensitivity.pit_dependence \
  --run-dirs \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_arrival_order_2027/3277702/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_arrival_order_2028/3277703/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_arrival_order_2029/3277704/arrival" \
  --output-dir "${CKPT_ROOT}/pit_dependence/${SLURM_JOB_ID}" \
  --device cuda \
  --prediction-samples 400 \
  --batch-size 256 \
  --seed 2433
