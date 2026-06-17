# Engression Diagnostic Runs

This folder contains runs that fit the public Python `engression` package to
synthetic weather-like data and compare predicted conditional quantiles against
the known generator truth.

## Create A Run

```bash
python experiments/engression_diagnostic.py \
  --model narx_student_t \
  --split in-support
```

The default output path is:

```text
runs/engression_diagnostics/<model>/<split>/<engression-model>/
```

Each run folder contains:

- `posterior_bands.png`: true versus engression-predicted 90% intervals and
  medians.
- `metrics.json`: scalar metrics from the run.
- `README.md`: a human-readable run card with the command, parameters, metrics,
  and split interpretation.

## Split Types

`in-support` randomly splits rows and keeps held-out rows whose scalar summary
\(\phi(X)\) lies inside the central training range. This is useful for checking
whether engression can learn \(P(Y \mid X=x)\) on held-out but familiar inputs.

`right-extrapolation`, `left-extrapolation`, and `two-sided-extrapolation` hold
out rows outside the training range in \(\phi(X)\)-space. These are controlled
covariate-support tests, not chronological forecasting splits.

## Caveat

The published Python package receives flattened lag windows. A row with
original sequence shape \((L+1,d)\) becomes a vector of length \((L+1)d\). This
is convenient for the package, but it means the default model is a stochastic
MLP rather than an LSTM or other sequence model.
