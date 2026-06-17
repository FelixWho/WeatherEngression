# Engression Diagnostic Run: Preadditive / in-support / vanilla

This folder contains one engression diagnostic run on synthetic
weather-like data with known conditional quantiles.

## Command

```bash
python experiments/engression_diagnostic.py --model preadditive --engression-model vanilla -d 12 --num-samples 5000 --train-size 4000 --test-size 500 --window 12 --epochs 120 --batch-size 512 --hidden-dim 192 --noise-dim 96 --lr 0.003 --prediction-samples 800 --skip-assertions
```

## Outputs

- `posterior_bands.png`: true versus engression-predicted posterior bands.
- `metrics.json`: scalar metrics printed by the run.

## Main Metrics

| Metric | Value |
|---|---:|
| mean quantile MAE | `1.161257` |
| q05 MAE | `1.214387` |
| q50 MAE | `0.989102` |
| q95 MAE | `1.280281` |
| predicted 90% interval coverage | `0.706000` |
| true 90% interval coverage on realized y | `0.902000` |
| mean predicted 90% interval width | `4.344786` |
| mean true 90% interval width | `5.815728` |
| width ratio | `0.747075` |

## Out-Of-Support Diagnostics

| Method | Count | Fraction |
|---|---:|---:|
| `scalar_projection` | `0` | `0.000000` |
| `marginal_range` | `24` | `0.048000` |
| `marginal_quantile` | `328` | `0.656000` |
| `knn_distance` | `23` | `0.046000` |

## Run Parameters

| Parameter | Value |
|---|---:|
| data model | `preadditive` |
| engression model | `vanilla` |
| split | `in-support` |
| dimension | `12` |
| window | `12` |
| num_samples | `5000` |
| train_size | `4000` |
| test_size | `500` |
| epochs | `120` |
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
`(4000, 156)` and the test matrix has shape
`(500, 156)`.
