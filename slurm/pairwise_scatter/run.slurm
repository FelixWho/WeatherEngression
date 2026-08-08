#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-splom
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/pairwise_scatter/run_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00

# Pairwise scatter matrix (SPLOM) of all 22 ENA channels + the target log10(CCN).
# Plotting is CPU-only, but general-cpu nodes intermittently fail to mount storage3
# (where weather_data.mat lives), so we run on general-gpu, which mounts it reliably.

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -m experiments.pairwise_scatter \
  --max-samples 6000 \
  --seq-stride 1 \
  --seed 2026 \
  --dpi 300 \
  --panel-inches 2.0 \
  --out reports/pairwise_scatter/pairwise_scatter.pdf

echo "done -> reports/pairwise_scatter/pairwise_scatter.pdf"
