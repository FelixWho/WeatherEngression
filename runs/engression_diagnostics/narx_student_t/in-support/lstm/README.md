# Engression Diagnostic Run: NARX Student-t / in-support / lstm

This folder contains one engression diagnostic run on synthetic
weather-like data with known conditional quantiles.

## Command

```bash
python experiments/engression_diagnostic.py --engression-model lstm --model narx_student_t -d 12 --num-samples 3000 --train-size 2400 --test-size 300 --window 12 --epochs 50 --batch-size 512 --hidden-dim 128 --noise-dim 64 --prediction-samples 300 --skip-assertions
```

## Outputs

- `posterior_bands.png`: true versus engression-predicted posterior bands.
- `metrics.json`: scalar metrics printed by the run.

## Main Metrics

| Metric | Value |
|---|---:|
| mean quantile MAE | `0.143217` |
| q05 MAE | `0.158447` |
| q50 MAE | `0.099018` |
| q95 MAE | `0.172188` |
| predicted 90% interval coverage | `0.850000` |
| true 90% interval coverage on realized y | `0.900000` |
| mean predicted 90% interval width | `1.219741` |
| mean true 90% interval width | `1.327434` |
| width ratio | `0.918872` |

## Out-Of-Support Diagnostics

| Method | Count | Fraction |
|---|---:|---:|
| `scalar_projection` | `0` | `0.000000` |
| `marginal_range` | `35` | `0.116667` |
| `marginal_quantile` | `224` | `0.746667` |
| `knn_distance` | `17` | `0.056667` |

## Run Parameters

| Parameter | Value |
|---|---:|
| data model | `narx_student_t` |
| engression model | `lstm` |
| split | `in-support` |
| dimension | `12` |
| window | `12` |
| num_samples | `3000` |
| train_size | `2400` |
| test_size | `300` |
| epochs | `50` |
| batch_size | `512` |
| hidden_dim | `128` |
| noise_dim | `64` |
| num_layer | `3` |
| learning_rate | `0.003` |
| weight_decay | `0.0` |
| prediction_samples | `300` |
| seed | `2026` |

## Split Interpretation

The default in-support split keeps held-out rows whose scalar summary \(\phi(X)\) lies inside the central training range. This checks conditional distribution learning, not chronological forecasting.

The model sees flattened lag windows, so the training matrix has shape
`(2400, 13, 12)` and the test matrix has shape
`(300, 13, 12)`.
