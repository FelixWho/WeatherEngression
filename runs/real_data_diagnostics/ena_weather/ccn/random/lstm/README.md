# Real-Data Engression Run: ENA CCN / random / lstm

This folder contains one LSTM-engression run on the **real** ENA
weather data (`weather_data.mat`). The true conditional law is
unknown, so the predictive distribution is scored against held-out
realized targets rather than against generator truth.

## Command

```bash
python experiments/real_data_diagnostic.py --target ccn --split random --max-samples 8000 --seq-stride 4 --train-size 5000 --test-size 800 --epochs 120 --batch-size 512 --hidden-dim 192 --noise-dim 96 --num-layer 3 --lr 0.003 --weight-decay 0.0 --prediction-samples 800 --device cuda --skip-assertions
```

## Outputs

- `posterior_bands.png`: predicted conditional bands with realized targets overlaid.
- `metrics.json`: scalar metrics printed by the run.

## Main Metrics

| Metric | Value |
|---|---:|
| energy score (CRPS, lower better) | `43.015511` |
| median abs error | `48.136920` |
| 90% interval coverage | `0.298750` |
| 50% interval coverage | `0.111250` |
| mean 90% interval width | `35.642734` |
| mean 50% interval width | `14.567799` |

Nominal coverage is 0.90 and 0.50; closer is better-calibrated.

## Run Parameters

| Parameter | Value |
|---|---:|
| data | `/storage3/fs1/myu/Active/felixhu/weather_data.mat` |
| target | `CCN` |
| split | `random` |
| event_flag | `n/a` |
| max_samples | `8000` |
| seq_stride | `4` |
| sequence shape (train) | `(5000, 61, 22)` |
| train_size | `5000` |
| test_size | `800` |
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

A shuffled in-distribution split: held-out rows are drawn from the same airmass population as training. This checks conditional distribution learning, not forecasting or shift.

The `lstm` variant keeps lag windows unflattened, so the training
tensor has shape `(5000, 61, 22)`
`(n, sequence_length, n_features)`.
