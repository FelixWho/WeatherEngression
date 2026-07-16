# SLURM: Noise-in-the-LSTM Sweep — OOS (GPU)

Re-runs the **6-variant** noise-in-the-LSTM grid (see
`slurm/real_data_lstm_stochastic_sweep/`) with the **embedding-space OOS
diagnostic enabled** and **checkpoints saved**.

## Why a re-run (not a reload)

The original sweep used `--no-oos --no-checkpoint` for lean runs, so the trained
models were never saved and OOS was never computed. OOS needs a trained model's
`encode()` embedding, so there is nothing to reload — this job **retrains** each
config on GPU, computes OOS in-process, and this time **saves checkpoints** (so
future OOD splits — `event` / `ccn_tail` — can be evaluated from the checkpoint
without retraining again). It writes to a **new** folder so the original
`lstm_stochastic_sweep/` leaderboard is preserved.

All six variants keep `encode()` deterministic (noise is injected only inside
`forward`, never `encode`), so the kNN / Mahalanobis distance-from-training
embedding is well defined for every one.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_lstm_stochastic_sweep_oos/run.slurm
```

Grid = `6 variants × 2 lr × 2 layers × 2 hidden = 48` → `--array=0-47%8`. Writes:

```
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep_oos/<tag>/
  ├── metrics.json            # now includes "embedding_oos"
  ├── posterior_bands.png
  ├── coverage_vs_distance.png   # kNN + Mahalanobis coverage-vs-distance
  ├── checkpoint_latest.pt
  └── checkpoint_best.pt
```

> Note: retraining re-computes the base metrics (CRPS, coverage). GPU
> nondeterminism means they will be very close to, but not bit-identical with, the
> `lstm_stochastic_sweep/` numbers — the ranking should hold. `recurrent` is the
> slow variant (241-step Python loop + BPTT), hence the 16h walltime.

## OOS leaderboard

After the array finishes:

```bash
.venv/bin/python slurm/real_data_lstm_stochastic_sweep_oos/aggregate_oos.py
```

Reports per config: overall 90% coverage, and coverage in the CLOSEST vs FARTHEST
distance bin for kNN and Mahalanobis. A large `drop` (near − far) means calibration
degrades away from the training feature space — the informative OOS signal.
