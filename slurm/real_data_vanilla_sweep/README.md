# SLURM: Vanilla (built-in) Engression Hyperparameter Sweep

A SLURM **array** job sweeping hyperparameters for the public `engression`
package's flat MLP model (`--engression-model vanilla`) on the real ENA weather
data. No LSTM: each `(241, 22)` trajectory is flattened to 5302 features.

## Grid (24 configs)

| Param | Values |
|---|---|
| `--lr` | 0.001, 0.003, 0.01 |
| `--hidden-dim` | 128, 256 |
| `--num-layer` | 3, 5 |
| `--noise-dim` | 64, 128 |

`3 x 2 x 2 x 2 = 24` → `#SBATCH --array=0-23%8`. All 24 tasks are submitted at once
and **run in parallel, up to 8 concurrently** (the `%8` throttle); SLURM queues the
rest and starts them as GPUs free up. Drop `%8` for unlimited concurrency, or change
N to throttle differently. The script decodes `SLURM_ARRAY_TASK_ID` into one config;
**if you edit the grid, update the `--array` range to match the new product.**

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_vanilla_sweep/run.slurm
```

Per-task logs: `run_<arrayjobid>_<taskid>.log`. Each task writes its run to:

```text
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla_sweep/<lr..._h..._l..._n...>/
  metrics.json  README.md  posterior_bands.png
```

## Aggregate

After the array finishes, rank the configs:

```bash
.venv/bin/python slurm/real_data_vanilla_sweep/aggregate.py
```

Prints a leaderboard sorted by calibration error `|coverage_90 - 0.90|`, with
energy score (CRPS), median abs error, and interval width.

## Notes

- Account `compute2-myu`; `general-gpu`; one GPU per task.
- Each task uses the full paper split, 80 epochs, 400 prediction samples
  (lighter than the single-run slurms, to keep 24 points affordable).
- This sweeps the flat baseline; the sequence-native counterpart is the `lstm`
  diagnostic. Swap `vanilla` for `regularized`/`adamw` to sweep weight-decayed
  flat models.
