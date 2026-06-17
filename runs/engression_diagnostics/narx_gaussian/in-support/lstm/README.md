# Engression Diagnostic Run: NARX Gaussian / in-support / lstm

This folder contains one engression diagnostic run on synthetic
weather-like data with known conditional quantiles.

## Command

```bash
python experiments/engression_diagnostic.py --model narx_gaussian --engression-model lstm -d 6 --num-samples 5000 --train-size 4000 --test-size 500 --window 12 --epochs 40 --batch-size 512 --hidden-dim 192 --noise-dim 96 --lr 0.003 --weight-decay 0 --prediction-samples 800 --skip-assertions
```

## Outputs

- `posterior_bands.png`: true versus engression-predicted posterior bands.
- `metrics.json`: scalar metrics printed by the run.

## Main Metrics

| Metric | Value |
|---|---:|
| mean quantile MAE | `0.136805` |
| q05 MAE | `0.147266` |
| q50 MAE | `0.093161` |
| q95 MAE | `0.169989` |
| predicted 90% interval coverage | `0.890000` |
| true 90% interval coverage on realized y | `0.892000` |
| mean predicted 90% interval width | `1.486413` |
| mean true 90% interval width | `1.333020` |
| width ratio | `1.115072` |

## Out-Of-Support Diagnostics

| Method | Count | Fraction |
|---|---:|---:|
| `scalar_projection` | `0` | `0.000000` |
| `marginal_range` | `12` | `0.024000` |
| `marginal_quantile` | `151` | `0.302000` |
| `knn_distance` | `30` | `0.060000` |

## Run Parameters

| Parameter | Value |
|---|---:|
| data model | `narx_gaussian` |
| engression model | `lstm` |
| split | `in-support` |
| dimension | `6` |
| window | `12` |
| num_samples | `5000` |
| train_size | `4000` |
| test_size | `500` |
| epochs | `40` |
| batch_size | `512` |
| hidden_dim | `192` |
| noise_dim | `96` |
| num_layer | `3` |
| learning_rate | `0.003` |
| weight_decay | `0.0` |
| prediction_samples | `800` |
| seed | `2026` |

## Split Interpretation

The default in-support split keeps held-out rows whose scalar summary \(\phi(X)\) lies inside the central training range. This checks conditional distribution learning, not chronological forecasting.

The model sees flattened lag windows, so the training matrix has shape
`(4000, 13, 6)` and the test matrix has shape
`(500, 13, 6)`.
