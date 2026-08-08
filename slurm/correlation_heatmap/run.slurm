#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-corr
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/correlation_heatmap/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00

# Pearson + Spearman correlation heatmap of all 22 ENA channels + target log10(CCN).
# CPU-only work, but general-gpu mounts storage3 reliably (general-cpu is flaky).

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -m experiments.correlation_heatmap \
  --max-samples 20000 \
  --seq-stride 1 \
  --seed 2026 \
  --dpi 200 \
  --out reports/correlation_heatmap/correlation_heatmap.png

echo "done -> reports/correlation_heatmap/correlation_heatmap.{png,pdf}"
