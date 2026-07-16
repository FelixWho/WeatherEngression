# SLURM: LSTM Engression Head Sweep (stonet + pre-additive)

A SLURM **array** job sweeping the two sequence-native LSTM engression heads on
the real ENA weather data, applying the lessons from the vanilla sweep
(under-dispersion is the failure mode; lr is the master knob; depth helps, width
hurts). Training uses early stopping on the **training** energy loss (a compute
saver, not a calibration guard — validation-based selection is future work).

## Grid (16 configs)

| Param | Values |
|---|---|
| head | `stonet` (`--stonet-head`), `preadd` (`--pre-additive`) |
| `--lr` | 0.003, 0.01 |
| `--num-layer` | 3, 5 |
| `--hidden-dim` | 128, 192 |

`2 x 2 x 2 x 2 = 16` → `#SBATCH --array=0-15%8` (up to 8 in parallel). `noise_dim`
is fixed at 96. **If you edit the grid, update the `--array` range.**

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_lstm_sweep/run.slurm
```

Per-task logs: `run_<arrayjobid>_<taskid>.log`. Each task writes its run to:

```text
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep/<head>_lr..._l..._h.../
  metrics.json  README.md  posterior_bands.png
```

## Aggregate

```bash
.venv/bin/python slurm/real_data_lstm_sweep/aggregate.py              # sort by calibration
.venv/bin/python slurm/real_data_lstm_sweep/aggregate.py --sort-by energy   # sort by CRPS
```

## Notes

- Account `compute2-myu`; `general-gpu`; one GPU per task.
- Each task: full paper split, max 100 epochs with early stop (patience 12),
  400 prediction samples, no checkpoint, no embedding-OOD (kept lean).
- The two heads write to distinct folders (`stonet_*` / `preadd_*`); they would
  otherwise collide in the shared `.../lstm/` dir.
- Compare against the flat baseline in `slurm/real_data_vanilla_sweep/` and the
  single `slurm/real_data_lstm_diagnostic/` run.
