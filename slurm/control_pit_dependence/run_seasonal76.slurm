#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-seasonal76-pit-compare
#SBATCH --chdir=/home/wg-fhu/WeatherEngression
#SBATCH --output=slurm/control_pit_dependence/seasonal76_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Reload 84 latest arrival models from seven completed, season-matched runs.
# Evaluate their saved 76 controls, then compare with same-calendar simulations.
set -euo pipefail
cd /home/wg-fhu/WeatherEngression
source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('This evaluation requires CUDA')
print('CUDA:', torch.cuda.get_device_name(0))
PY
CKPT_ROOT=/storage3/fs1/myu/Active/felixhu/weather_checkpoints
PIT_OUTPUT="${CKPT_ROOT}/control_pit_dependence/${SLURM_JOB_ID}"
python -u -m experiments.wildfire_sensitivity.pit_dependence \
  --run-dirs \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2027/3685284/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2028/3685285/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2029/3685286/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2030/4036572/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2031/4043606/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2032/4043609/arrival" \
    "${CKPT_ROOT}/wildfire_sensitivity/ad_control15_seasonal_arrival_order_2034/4043684/arrival" \
  --output-dir "${PIT_OUTPUT}" \
  --role control --gap-days 15 30 60 90 \
  --device cuda --prediction-samples 400 --batch-size 256 --seed 2433
python -u -m experiments.wildfire_sensitivity.compare_pit_dependence \
  --pit-dir "${PIT_OUTPUT}" --output-dir "${PIT_OUTPUT}/synthetic_comparison" \
  --repetitions 1000 --seed 2435
