#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-vanilla-chart
#SBATCH --output=slurm/regen_vanilla_chart/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
nvidia-smi

# Regenerate the vanilla (flat) best-config in-sample chart with the FIXED,
# model-tagged plotting title. This is the full-data vanilla run that OOMs / stalls
# on the login node -- it belongs on a GPU node. Config = the best-calibrated point
# from the vanilla sweep (lr 0.01, hidden 128, 5 layers, noise 64), cov ~0.907.
#
# Writes posterior_bands.png (now titled "[vanilla (flat)]") to
#   runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla_best/
# then copies it into the report as reports/fig_vanilla_is.png.
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla_best"

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
  --lr 0.01 \
  --hidden-dim 128 \
  --num-layer 5 \
  --noise-dim 64 \
  --prediction-samples 400 \
  --device cuda \
  --no-oos \
  --out-dir "${OUT}" \
  --skip-assertions

# Drop the relabeled chart into the report and rebuild the PDF (if a TeX engine
# is available on the node).
cp "${OUT}/posterior_bands.png" reports/fig_vanilla_is.png
echo "copied ${OUT}/posterior_bands.png -> reports/fig_vanilla_is.png"
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -output-directory=reports reports/real_data_engression_summary.tex >/dev/null 2>&1 || true
  rm -f reports/real_data_engression_summary.{aux,log,out}
  echo "rebuilt reports/real_data_engression_summary.pdf"
else
  echo "no pdflatex on node; run pdflatex on the login node to rebuild the PDF"
fi
