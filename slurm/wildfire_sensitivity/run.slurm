#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-wfsens-cal
#SBATCH --output=slurm/wildfire_sensitivity/run_calibration_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Wildfire-sensitivity screening, single pass over the sensitive covariates.
#
# Pools clean paper-train and paper-test rows, then creates separate training,
# validation, recalibration, and control sets. The best validation checkpoint is
# wrapped with pooled quantile corrections fitted on the recalibration rows. It must
# pass central 90% and 95% clean-control coverage checks (within 3 block SE).
# Failed checks leave the candidate undetermined and out of the input set M.
# Accepted models compare central 95% wildfire coverage against the clean control;
# a drop greater than two percentage points is sensitive, otherwise insensitive.
# Calibration checks are enabled by default in the Python entry point.
#
# Four-hour limit for the serial loop: one full LSTM fit per candidate,
# twelve candidates in the first round. Companion:
# run_until_stable.slurm re-tests the survivors until a round moves nobody.

set -euo pipefail
cd "${REPO_ROOT:-/home/wg-fhu/WeatherEngression}"

# Report a missing environment before attempting to activate it.
test -f .venv/bin/activate || {
  echo "no .venv in $(pwd) -- create one (python -m venv .venv && .venv/bin/pip install -r requirements.txt)" >&2
  echo "or point REPO_ROOT at a checkout that has one." >&2
  exit 1
}
source .venv/bin/activate

SCREENING_MODE="${SCREENING_MODE:-single_pass}"
EVALUATION_SCOPE="${EVALUATION_SCOPE:-trajectory}"
ORDER_ARGS=()
if [[ -n "${CANDIDATE_ORDER_SEED:-}" ]]; then
  ORDER_ARGS=(--candidate-order-seed "${CANDIDATE_ORDER_SEED}")
fi
MAT_PATH="${MAT_PATH:-/storage3/fs1/myu/Active/felixhu/weather_data.mat}"
# Job-scoped so a rerun cannot overwrite a previous run's models (same pattern as
# slurm/t_learner). Override CKPT_DIR to pin an exact path.
CKPT_ROOT="/storage3/fs1/myu/Active/felixhu/weather_checkpoints/wildfire_sensitivity"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${SLURM_JOB_ID:-local}}"

echo "job id        : ${SLURM_JOB_ID:-local}"
echo "screening     : ${SCREENING_MODE}"
echo "scope         : ${EVALUATION_SCOPE}"
echo "order seed    : ${CANDIDATE_ORDER_SEED:-original order}"
echo "data          : ${MAT_PATH}"
echo "checkpoints  -> ${CKPT_DIR}"

# Fail now rather than after the model has been built.
test -r "${MAT_PATH}" || { echo "MAT file not readable: ${MAT_PATH}" >&2; exit 1; }

python -u - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("This SLURM job requires CUDA; no GPU is available.")
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())

from experiments.wildfire_sensitivity.main import (
    CALIBRATION_REQUIRED_LEVELS,
    CALIBRATION_TOLERANCE_SE,
)

levels = ", ".join(f"{level:.0%}" for level in CALIBRATION_REQUIRED_LEVELS)
print(f"calibration   : central {levels}; each within {CALIBRATION_TOLERANCE_SE:g} block SE")
print("failed checks : undetermined; excluded from the input set")
print("evaluation    : best validation checkpoint")
print("recalibration : pooled quantile mapping fitted on separate clean blocks")
PY

python -u -m experiments.wildfire_sensitivity.main \
  --screening-mode "${SCREENING_MODE}" \
  --evaluation-scope "${EVALUATION_SCOPE}" \
  "${ORDER_ARGS[@]}" \
  --mat-path "${MAT_PATH}" \
  --checkpoint-dir "${CKPT_DIR}"

echo "done. checkpoints written to: ${CKPT_DIR}"
ls -la "${CKPT_DIR}"
