#!/usr/bin/env bash
# Ablate the global-latent epsilon dimension on a local Apple GPU (PyTorch MPS).
#
# Usage:
#   bash local_runs/epsilon_dim_ablation.sh
#   bash local_runs/epsilon_dim_ablation.sh 1 8 32 96
#
# Optional overrides:
#   MAT_PATH=data_generation/weather_data.mat EPOCHS=60 BATCH_SIZE=64 \
#     bash local_runs/epsilon_dim_ablation.sh
#   VERBOSE=0 bash local_runs/epsilon_dim_ablation.sh  # hide per-epoch loss
#   DRY_RUN=1 bash local_runs/epsilon_dim_ablation.sh 1 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MAT_PATH="${MAT_PATH:-data_generation/weather_data.mat}"
OUT_ROOT="${OUT_ROOT:-runs/real_data_diagnostics/ena_weather/log10_ccn/paper/epsilon_dim_ablation}"
LOG_ROOT="${LOG_ROOT:-local_runs/logs/epsilon_dim_ablation}"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PREDICTION_SAMPLES="${PREDICTION_SAMPLES:-400}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
SEED="${SEED:-2026}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-1}"

# Keep the winning global-latent architecture fixed; only epsilon dimension changes.
LR="${LR:-0.003}"
NUM_LAYER="${NUM_LAYER:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"

if [[ "$#" -gt 0 ]]; then
  EPSILON_DIMS=("$@")
else
  EPSILON_DIMS=(1 4 8 16 32 64 96)
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MAT_PATH}" ]]; then
  echo "MAT file not found: ${MAT_PATH}" >&2
  exit 1
fi

for dim in "${EPSILON_DIMS[@]}"; do
  if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Epsilon dimensions must be positive integers; got: ${dim}" >&2
    exit 1
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  MPS_AVAILABLE="$("${PYTHON_BIN}" -c 'import torch; print(int(torch.backends.mps.is_available()))')"
  if [[ "${MPS_AVAILABLE}" != "1" ]]; then
    echo "PyTorch MPS is unavailable in ${PYTHON_BIN}." >&2
    echo "Check with: ${PYTHON_BIN} -c 'import torch; print(torch.backends.mps.is_available())'" >&2
    exit 1
  fi
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

echo "Global-latent epsilon-dimension ablation"
echo "  dimensions : ${EPSILON_DIMS[*]}"
echo "  data       : ${MAT_PATH}"
echo "  device     : mps"
echo "  architecture: layers=${NUM_LAYER}, hidden=${HIDDEN_DIM}, lr=${LR}"
echo "  epochs     : ${EPOCHS} (early stopping patience=${EARLY_STOP_PATIENCE})"
echo "  batch size : ${BATCH_SIZE}"
echo "  pred samples: ${PREDICTION_SAMPLES}"
echo "  seed       : ${SEED}"
echo "  outputs    : ${OUT_ROOT}"
echo "  logs       : ${LOG_ROOT}"
echo "  progress   : $([[ "${VERBOSE}" == "1" ]] && echo 'per epoch' || echo 'summary only')"
echo

TOTAL_DIMS="${#EPSILON_DIMS[@]}"
RUN_NUMBER=0
ABLATION_START="${SECONDS}"

for dim in "${EPSILON_DIMS[@]}"; do
  RUN_NUMBER=$((RUN_NUMBER + 1))
  TAG="global_latent_eps${dim}"
  OUT_DIR="${OUT_ROOT}/${TAG}"
  LOG_PATH="${LOG_ROOT}/${TAG}.log"

  if [[ "${FORCE}" != "1" && -f "${OUT_DIR}/metrics.json" && -f "${OUT_DIR}/checkpoint_best.pt" ]]; then
    echo "[${RUN_NUMBER}/${TOTAL_DIMS}] SKIP epsilon=${dim}: completed artifacts already exist"
    continue
  fi

  CMD=(
    "${PYTHON_BIN}" -u experiments/real_data_diagnostic.py
    --mat-path "${MAT_PATH}"
    --engression-model lstm
    --global-latent-noise
    --target ccn
    --log-ccn
    --split paper
    --max-samples all
    --seq-stride 1
    --train-size all
    --test-size all
    --epochs "${EPOCHS}"
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LR}"
    --hidden-dim "${HIDDEN_DIM}"
    --num-layer "${NUM_LAYER}"
    --noise-dim "${dim}"
    --prediction-samples "${PREDICTION_SAMPLES}"
    --seed "${SEED}"
    --no-oos
    --checkpoint-every "${CHECKPOINT_EVERY}"
    --device mps
    --out-dir "${OUT_DIR}"
    --skip-assertions
  )

  if [[ "${VERBOSE}" == "1" ]]; then
    CMD+=(--verbose)
  fi

  RUN_START="${SECONDS}"
  echo "[${RUN_NUMBER}/${TOTAL_DIMS}] START epsilon=${dim} at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  output: ${OUT_DIR}"
  echo "  log   : ${LOG_PATH}"
  echo "  Loading data can be quiet; per-epoch losses will appear once training starts."
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  %q' "${CMD[@]}"
    printf '\n'
  else
    "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
    RUN_ELAPSED=$((SECONDS - RUN_START))
    printf '[%s/%s] DONE epsilon=%s in %dm %02ds\n' \
      "${RUN_NUMBER}" "${TOTAL_DIMS}" "${dim}" \
      "$((RUN_ELAPSED / 60))" "$((RUN_ELAPSED % 60))"
    echo "  metrics: ${OUT_DIR}/metrics.json"
    echo "  best checkpoint: ${OUT_DIR}/checkpoint_best.pt"
  fi
  echo
done

ABLATION_ELAPSED=$((SECONDS - ABLATION_START))
printf 'Ablation finished in %dm %02ds.\n' \
  "$((ABLATION_ELAPSED / 60))" "$((ABLATION_ELAPSED % 60))"
echo "Compare metrics.json files under ${OUT_ROOT}."
