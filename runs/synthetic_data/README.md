# Synthetic Data Runs

This folder contains visual checks of the synthetic weather-like data
generators. Each subfolder corresponds to one model, or to the `all_models`
gallery.

## Create A Run

```bash
python plot_synthetic_data.py --model narx_gaussian
```

The default output path is:

```text
runs/synthetic_data/<model>/target_timeseries.png
```

For a gallery across all generators:

```bash
python plot_synthetic_data.py --model all
```

which writes to:

```text
runs/synthetic_data/all_models/target_timeseries.png
```

## What The Plot Shows

The main panel shows one sampled target series \(Y_t\), the generator's true
conditional median, and the generator's true 90% interval:

\[
[q_{0.05}(X_t), q_{0.95}(X_t)].
\]

The optional diagnostic panel is model-specific. Examples include conditional
volatility for NARX models, mixture weights for the regime mixture, wet-event
probability for the hurdle-lognormal model, and latent \(\phi(X)\) for the
pre-additive model.
