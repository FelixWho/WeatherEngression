#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-default-charts
#SBATCH --output=slurm/regen_default_charts/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
nvidia-smi

# Regenerate the LSTM + DEFAULT head best run (the 0.888 config that was overwritten
# when the StoNet run reused the .../lstm/ folder). Default head = LSTM engression
# with neither --stonet-head nor --pre-additive: one noise vector concatenated to
# h(X), plain MLP. OOS is ON (the LSTM encoder gives an embedding), so this produces
# BOTH the in-sample posterior bands and the coverage-vs-distance chart, correctly
# titled "[LSTM (default head)]" by the fixed plotting code. Writes to a DISTINCT
# folder so it does not clobber the StoNet run.
OUT="runs/real_data_diagnostics/ena_weather/log10_ccn/paper/default_head"

python -u experiments/real_data_diagnostic.py \
  --engression-model lstm \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --seq-stride 1 \
  --train-size all \
  --test-size all \
  --epochs 120 \
  --batch-size 256 \
  --hidden-dim 192 \
  --noise-dim 96 \
  --num-layer 3 \
  --lr 0.003 \
  --weight-decay 0.0 \
  --prediction-samples 800 \
  --checkpoint-every 25 \
  --device cuda \
  --out-dir "${OUT}" \
  --skip-assertions

# Drop both charts into the report (the \IfFileExists slots pick them up) and rebuild.
cp "${OUT}/posterior_bands.png"      reports/fig_default_is.png
cp "${OUT}/coverage_vs_distance.png" reports/fig_default_oos.png
echo "copied default-head charts -> reports/fig_default_is.png, reports/fig_default_oos.png"
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -output-directory=reports reports/real_data_engression_summary.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -output-directory=reports reports/real_data_engression_summary.tex >/dev/null 2>&1 || true
  rm -f reports/real_data_engression_summary.{aux,log,out}
  echo "rebuilt reports/real_data_engression_summary.pdf"
else
  echo "no pdflatex on node; rebuild the PDF on the login node afterward"
fi
