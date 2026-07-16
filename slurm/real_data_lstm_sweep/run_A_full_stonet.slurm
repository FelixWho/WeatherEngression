#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-sw-A_full-stonet
#SBATCH --output=slurm/real_data_lstm_sweep/A_full_stonet_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Sweep cell: param-set A_full, head 'stonet'.
python -u experiments/real_data_diagnostic.py \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --train-size all \
  --test-size all \
  --seq-stride 1 \
  --epochs 120 \
  --batch-size 256 \
  --hidden-dim 192 \
  --noise-dim 96 \
  --num-layer 3 \
  --stonet-head \
  --lr 0.003 \
  --weight-decay 0.0 \
  --prediction-samples 800 \
  --checkpoint-every 10 \
  --device cuda \
  --skip-assertions \
  --out-dir runs/real_data_diagnostics/log10_ccn/paper/sweep/A_full_stonet
