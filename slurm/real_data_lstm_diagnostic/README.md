# SLURM: Real-Data LSTM Engression Diagnostic

Batch job that fits the `lstm` engression variant on the **real** ENA weather
data (`weather_data.mat`) and scores the predictive distribution against
held-out realized targets.

The login node has no GPU, so this runs on a `general-gpu` node with
`--device cuda`. It is the GPU counterpart to the ~24-minute CPU run.

## Submit

```bash
cd /home/felixhu/WeatherEngression
sbatch slurm/real_data_lstm_diagnostic/run.slurm
```

The job log is written next to this file as `run_<jobid>.log`.

## Outputs

`experiments/real_data_diagnostic.py` writes the run under:

```text
runs/real_data_diagnostics/ena_weather/ccn/random/lstm/
  README.md            run command + metric summary
  metrics.json         energy score, coverage, widths, median error
  posterior_bands.png  predicted bands vs realized CCN
```

## Notes

- Account is `myu`; partition `general-gpu`; one GPU.
- Edit the `python experiments/real_data_diagnostic.py ...` invocation to change
  the target/split (e.g. `--target T`, `--split event --event-flag dust`,
  `--split chronological`) or the training length.
- All dependencies come from the repo `.venv`; new packages belong in
  `requirements.txt`, not installed ad hoc.
