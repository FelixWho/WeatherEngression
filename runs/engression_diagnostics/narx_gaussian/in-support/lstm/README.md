# Engression Diagnostic Run: NARX Gaussian / in-support / lstm

This folder contains one engression diagnostic run on synthetic
weather-like data with known conditional quantiles.

## Command

```bash
python experiments/engression_diagnostic.py --model narx_gaussian --engression-model lstm --stonet-head --split in-support --epochs 80 --prediction-samples 500
```

## Outputs

- `posterior_bands.png`: true versus engression-predicted posterior bands.
- `metrics.json`: scalar metrics printed by the run.

## Main Metrics

| Metric | Value |
|---|---:|
| mean quantile MAE | `0.155489` |
| q05 MAE | `0.136581` |
| q50 MAE | `0.144882` |
| q95 MAE | `0.185005` |
| predicted 90% interval coverage | `0.894000` |
| true 90% interval coverage on realized y | `0.900000` |
| mean predicted 90% interval width | `1.244416` |
| mean true 90% interval width | `1.177974` |
| width ratio | `1.056404` |

## Out-Of-Support Diagnostics

| Method | Count | Fraction |
|---|---:|---:|
| `scalar_projection` | `0` | `0.000000` |
| `marginal_range` | `11` | `0.011000` |
| `marginal_quantile` | `295` | `0.295000` |
| `knn_distance` | `46` | `0.046000` |

## Run Parameters

| Parameter | Value |
|---|---:|
| data model | `narx_gaussian` |
| engression model | `lstm` |
| split | `in-support` |
| dimension | `6` |
| window | `12` |
| num_samples | `10000` |
| train_size | `8000` |
| test_size | `1000` |
| epochs | `80` |
| batch_size | `512` |
| hidden_dim | `192` |
| noise_dim | `96` |
| num_layer | `3` |
| learning_rate | `0.003` |
| weight_decay | `0.0` |
| prediction_samples | `500` |
| seed | `2026` |

## Split Interpretation

The default in-support split keeps held-out rows whose scalar summary \(\phi(X)\) lies inside the central training range. This checks conditional distribution learning, not chronological forecasting.

The model sees flattened lag windows, so the training matrix has shape
`(8000, 13, 6)` and the test matrix has shape
`(1000, 13, 6)`.
