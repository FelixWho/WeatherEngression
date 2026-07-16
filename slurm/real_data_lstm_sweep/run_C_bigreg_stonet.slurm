#!/bin/bash
#SBATCH -A compute2-myu
#SBATCH --job-name=weng-sw-C_bigreg-stonet
#SBATCH --output=slurm/real_data_lstm_sweep/C_bigreg_stonet_%j.log
#SBATCH -p general-gpu
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00

set -euo pipefail

cd /home/felixhu/WeatherEngression
source .venv/bin/activate

python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Sweep cell: param-set C_bigreg, head 'stonet'.
python -u experiments/real_data_diagnostic.py \
  --target ccn \
  --log-ccn \
  --split paper \
  --max-samples all \
  --train-size all \
  --test-size all \
  --seq-stride 4 \
  --epochs 150 \
  --batch-size 256 \
  --hidden-dim 256 \
  --noise-dim 128 \
  --num-layer 3 \
  --stonet-head \
  --lr 0.003 \
  --weight-decay 0.0001 \
  --prediction-samples 800 \
  --checkpoint-every 10 \
  --device cuda \
  --skip-assertions \
  --out-dir runs/real_data_diagnostics/log10_ccn/paper/sweep/C_bigreg_stonet
