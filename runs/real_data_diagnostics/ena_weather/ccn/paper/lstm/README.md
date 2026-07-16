# Real-Data Engression Run: ENA CCN / paper / lstm

This folder contains one LSTM-engression run on the **real** ENA
weather data (`weather_data.mat`). The true conditional law is
unknown, so the predictive distribution is scored against held-out
realized targets rather than against generator truth.

## Command

```bash
python experiments/real_data_diagnostic.py --target ccn --no-log-ccn --split paper --max-samples all --seq-stride 1 --train-size all --test-size all --epochs 120 --batch-size 256 --hidden-dim 192 --noise-dim 96 --num-layer 3 --lr 0.003 --weight-decay 0.0 --prediction-samples 800 --checkpoint-every 10 --device cuda --skip-assertions
```

## Outputs

- `posterior_bands.png`: predicted conditional bands with realized targets overlaid.
- `metrics.json`: scalar metrics printed by the run.
- `checkpoint_latest.pt`: latest epoch model, optimizer state, and standardization stats.
- `checkpoint_best.pt`: best training-energy-loss checkpoint.

## Main Metrics

| Metric | Value |
|---|---:|
| energy score (CRPS, lower better) | `47.574367` |
| median abs error | `56.156111` |
| 90% interval coverage | `0.384783` |
| 50% interval coverage | `0.154969` |
| mean 90% interval width | `63.155978` |
| mean 50% interval width | `25.327935` |

Nominal coverage is 0.90 and 0.50; closer is better-calibrated.

## Run Parameters

| Parameter | Value |
|---|---:|
| data | `/storage3/fs1/myu/Active/felixhu/weather_data.mat` |
| target | `CCN` |
| log_ccn | `False` |
| split | `paper` |
| event_flag | `n/a` |
| max_samples | `all` |
| seq_stride | `1` |
| sequence shape (train) | `(57646, 241, 22)` |
| train_size arg | `all` |
| test_size arg | `all` |
| train rows used | `57646` |
| test rows used | `3220` |
| epochs | `120` |
| batch_size | `256` |
| hidden_dim | `192` |
| noise_dim | `96` |
| num_layer | `3` |
| learning_rate | `0.003` |
| weight_decay | `0.0` |
| prediction_samples | `800` |
| checkpointing | `True` |
| checkpoint_every | `10` |
| seed | `2026` |

## Split Interpretation

The test pool follows the source paper: the held-out months are January, March, May, July, September, and November 2022. Training rows are drawn from all other months. When provided as positive integers, the CLI train/test size arguments cap how many rows are used from each pool; `all` uses the full pools.

The `lstm` variant keeps lag windows unflattened, so the training
tensor has shape `(57646, 241, 22)`
`(n, sequence_length, n_features)`.
