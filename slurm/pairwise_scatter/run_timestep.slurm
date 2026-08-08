#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-splom-ts
#SBATCH --output=/home/felixhu/WeatherEngression/slurm/pairwise_scatter/run_timestep_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00

# Per-timestep scatter matrix: every (sample, timestep) is its own point. With 241
# timesteps per sample we use a smaller sample subset (400 -> ~96k points/panel) and
# low alpha. general-gpu for reliable storage3 access (plotting is CPU-only).

set -euo pipefail
cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -m experiments.pairwise_scatter \
  --per-timestep \
  --max-samples 400 \
  --seq-stride 1 \
  --seed 2026 \
  --alpha 0.02 \
  --point-size 0.4 \
  --dpi 300 \
  --panel-inches 2.0 \
  --out reports/pairwise_scatter/pairwise_scatter_timestep.pdf

echo "done -> reports/pairwise_scatter/pairwise_scatter_timestep.pdf"
