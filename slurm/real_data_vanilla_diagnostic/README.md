# SLURM: Built-in (vanilla) Engression on ENA Weather Data

Batch job that fits the **public `engression` package's own model** — the flat
stochastic MLP generator `g(X, eps)` — on the real ENA weather data
(`weather_data.mat`), as a baseline for the sequence-native `lstm` run.

Because the vanilla model is not sequence-aware, each `(241, 22)` trajectory is
**flattened** to a 5302-dim feature vector. Checkpointing and the embedding-space
OOD diagnostic are `lstm`-only and are automatically disabled for this model.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_vanilla_diagnostic/run.slurm
```

The job log is written next to this file as `run_<jobid>.log`.

## Outputs

`experiments/real_data_diagnostic.py --engression-model vanilla` writes:

```text
runs/real_data_diagnostics/ena_weather/log10_ccn/paper/vanilla/
  README.md            run command + metric summary
  metrics.json         energy score, coverage, widths, median error
  posterior_bands.png  predicted bands vs realized log10(CCN)
```

## Notes

- Account `compute2-myu`; partition `general-gpu`; one GPU.
- The matching sequence-native run is `slurm/real_data_lstm_diagnostic/run.slurm`;
  this folder is the flat-MLP baseline to compare against it.
- Swap `--engression-model vanilla` for `regularized` or `adamw` to add weight
  decay (also flat models). All dependencies come from the repo `.venv`.
