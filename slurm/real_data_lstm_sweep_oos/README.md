# SLURM: LSTM Sweep + Embedding-OOS Diagnostic (GPU)

Re-runs the **same** stonet + pre-additive LSTM sweep grid as
`slurm/real_data_lstm_sweep/`, but with the **embedding-space OOS diagnostic
enabled** (kNN + Mahalanobis distance-from-training in the LSTM's `h(X)` space,
with 90% coverage stratified by distance) and **checkpoints saved**.

Why a re-run: the original sweep used `--no-oos --no-checkpoint`, so no models
were saved and OOS was never computed. There is no way to get OOS without the
trained models, so this retrains on GPU (computing OOS in-process), and saves
checkpoints so future OOD splits can be evaluated without retraining.

Outputs go to a **new** dir so the original leaderboard is untouched:

```text
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep_oos/<head>_lr..._l..._h.../
  metrics.json            now includes an "embedding_oos" section
  posterior_bands.png
  coverage_vs_distance.png  kNN + Mahalanobis coverage vs distance bin
  checkpoint_best.pt  checkpoint_latest.pt
```

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_lstm_sweep_oos/run.slurm
```

`--array=0-15%8` → 16 configs, up to 8 on GPUs at once.

## Aggregate

```bash
# OOS summary: overall coverage + closest-vs-farthest distance bin (degradation)
.venv/bin/python slurm/real_data_lstm_sweep_oos/aggregate_oos.py

# the usual calibration / CRPS leaderboard also works on these runs:
.venv/bin/python slurm/real_data_lstm_sweep/aggregate.py \
    --sweep-dir runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep_oos
```

## Notes

- Account `compute2-myu`; `general-gpu`; one GPU per task.
- The `paper` split is in-distribution, so coverage-vs-distance is expected to be
  fairly **flat** here. To see distance actually drive miscalibration, reuse the
  saved checkpoints with `experiments/evaluate_oos_from_checkpoint.py --split
  event` or `--split ccn_tail`.
- Embedding-OOS uses the LSTM encoder, so it works for both heads (stonet and
  pre-additive) identically.
