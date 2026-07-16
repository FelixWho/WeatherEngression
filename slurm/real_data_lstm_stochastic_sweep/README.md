# SLURM: Noise-in-the-LSTM Sweep (GPU)

Grid search for the **noise-in-the-LSTM** engression generators on the real ENA
weather data (`weather_data.mat`), analogous to `slurm/real_data_lstm_sweep/`
(which sweeps the stonet / pre-additive *head* variants). These inject the noise
into the **sequence** or the **recurrence** before/inside encoding, then a
deterministic MLP maps `h(X) -> Y`. Six variants:

| variant | flag | where the noise goes |
|---|---|---|
| `additive` | `--additive-noise` | added onto the final timestep of `x` |
| `appending` | `--appending-noise` | a fresh noise timestep appended → LSTM reads `(B, S+1, F)` |
| `per_timestep` | `--per-timestep-noise` | fresh noise concatenated to **every** timestep → `(B, S, F+k)` (StoNet analogue) |
| `global_latent` | `--global-latent-noise` | one latent `z` per sample, broadcast to every timestep (CVAE form) |
| `init_state` | `--stochastic-init-noise` | the LSTM initial `(h0, c0)` seeded from noise |
| `recurrent` | `--recurrent-state-noise` | per-step noise added to the hidden state (`LSTMCell` loop; slowest) |

All are legitimate energy-trained conditional samplers `g(X, ε)`. Like StoNet
they are **not** pre-ANM, so they carry the in-distribution consistency guarantee
but **no** extrapolation theorem. **Single-injection** variants
(`additive`/`appending`/`init_state`) are more collapse-prone than the
**many-injection** ones (`per_timestep`/`global_latent`/`recurrent`), so watch the
spread term in the logs — if it collapses at high lr, that is the expected failure
mode.

## Grid

`6 variants × 2 lr (0.003, 0.01) × 2 head-layers (3, 5) × 2 hidden (128, 192) = 48`
runs → `--array=0-47%8` (up to 8 concurrent). Same lr/layers/hidden axes as the
head sweep so all families are comparable.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_lstm_stochastic_sweep/run.slurm
```

Each task trains full-data paper split, 100 epochs with early stopping
(patience 12), `--device cuda`, `--no-checkpoint --no-oos` for lean runs, and
writes to a distinct folder:

```
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep/<variant>_lr<lr>_l<layers>_h<hidden>/
```

## Leaderboard

After the array finishes:

```bash
.venv/bin/python slurm/real_data_lstm_stochastic_sweep/aggregate.py                 # sort by calibration
.venv/bin/python slurm/real_data_lstm_stochastic_sweep/aggregate.py --sort-by energy # sort by CRPS
```

To compare against the head variants, point `aggregate.py` at the other sweep:

```bash
.venv/bin/python slurm/real_data_lstm_stochastic_sweep/aggregate.py \
  --sweep-dir runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_sweep
```

## Notes

- OOS is off here (lean sweep). The stochastic encoders keep `encode()`
  deterministic (noise is only injected inside `forward`, not `encode`), so the
  kNN / Mahalanobis diagnostics still work if you rerun a winning config without
  `--no-oos`.
- Single GPU per task; the login node CPU thrashes on LSTM training (thread
  oversubscription), so always run these on `general-gpu`, never in the foreground.
