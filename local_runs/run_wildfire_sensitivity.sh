#!/usr/bin/env bash
# Train the iterative wildfire-sensitivity counterfactual models locally.
#
# Usage:
#   bash local_runs/run_wildfire_sensitivity.sh
#
# Optional overrides:
#   PYTHON_BIN=.venv/bin/python bash local_runs/run_wildfire_sensitivity.sh
#   ALLOW_CPU=1 bash local_runs/run_wildfire_sensitivity.sh
#   DRY_RUN=1 bash local_runs/run_wildfire_sensitivity.sh
#   SCREENING_MODE=until_stable bash local_runs/run_wildfire_sensitivity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_ROOT="${LOG_ROOT:-local_runs/logs/wildfire_sensitivity}"
ALLOW_CPU="${ALLOW_CPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SCREENING_MODE="${SCREENING_MODE:-single_pass}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "data_generation/weather_data.mat" ]]; then
  echo "MAT file not found: data_generation/weather_data.mat" >&2
  exit 1
fi

DEVICE="$(${PYTHON_BIN} -c 'import torch; print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')"
if [[ "${DRY_RUN}" != "1" && "${DEVICE}" == "cpu" && "${ALLOW_CPU}" != "1" ]]; then
  echo "PyTorch did not detect CUDA or MPS; refusing this long run on CPU." >&2
  echo "Set ALLOW_CPU=1 only if CPU training is intentional." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_STARTED_AT="$(date '+%Y-%m-%d %H:%M:%S %Z')"
LOG_PATH="${LOG_ROOT}/wildfire_sensitivity_${TIMESTAMP}.log"
CMD=("${PYTHON_BIN}" -u -m experiments.wildfire_sensitivity.main
  --screening-mode "${SCREENING_MODE}")

echo "Wildfire-sensitivity counterfactual run"
echo "  started     : ${RUN_STARTED_AT}"
echo "  device      : ${DEVICE}"
echo "  screen mode : ${SCREENING_MODE}"
if [[ "${DEVICE}" == "mps" ]]; then
  echo "  batch size  : 64 (reduced from the PIT sweep's 256 to fit MPS memory)"
  echo "  eval batch  : 64 (wildfire samples are streamed to avoid MPS OOM)"
else
  echo "  batch size  : 256"
  echo "  eval batch  : 1024"
fi
echo "  data        : data_generation/weather_data.mat"
echo "  checkpoints : weather_checkpoints/wildfire_sensitivity"
echo "  log         : ${LOG_PATH}"
echo "  progress    : one candidate model at a time"
printf '  command     : '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

echo "Done at $(date '+%Y-%m-%d %H:%M:%S %Z'). Checkpoints are in weather_checkpoints/wildfire_sensitivity."
