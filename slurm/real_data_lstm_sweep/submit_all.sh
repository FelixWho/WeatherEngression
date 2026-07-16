#!/bin/bash
# Submit every sweep cell.
set -euo pipefail

sbatch slurm/real_data_lstm_sweep/run_A_full_stonet.slurm
sbatch slurm/real_data_lstm_sweep/run_A_full_preadd.slurm
sbatch slurm/real_data_lstm_sweep/run_B_short_stonet.slurm
sbatch slurm/real_data_lstm_sweep/run_B_short_preadd.slurm
sbatch slurm/real_data_lstm_sweep/run_C_bigreg_stonet.slurm
sbatch slurm/real_data_lstm_sweep/run_C_bigreg_preadd.slurm
sbatch slurm/real_data_lstm_sweep/run_D_lowlr_stonet.slurm
sbatch slurm/real_data_lstm_sweep/run_D_lowlr_preadd.slurm
