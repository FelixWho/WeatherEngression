# Real-Data Engression Run: ENA log10(CCN) / paper / lstm

This folder contains one LSTM-engression run on the **real** ENA
weather data (`weather_data.mat`). The true conditional law is
unknown, so the predictive distribution is scored against held-out
realized targets rather than against generator truth.

## Command

```bash
python experiments/real_data_diagnostic.py --engression-model lstm --appending-noise --target ccn --log-ccn --split paper --max-samples all --seq-stride 1 --train-size all --test-size all --epochs 100 --early-stop-patience 12 --batch-size 256 --lr 0.01 --hidden-dim 128 --num-layer 5 --noise-dim 96 --prediction-samples 400 --oos-reference-size 2000 --oos-bins 10 --checkpoint-every 25 --device cuda --out-dir runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep_oos/appending_lr0.01_l5_h128 --skip-assertions
```

## Outputs

- `posterior_bands.png`: predicted conditional bands with realized targets overlaid.
- `metrics.json`: scalar metrics printed by the run.
- `coverage_vs_distance.png`: 90% coverage stratified by kNN and Mahalanobis distance from training in the LSTM embedding space.
- `checkpoint_latest.pt`: latest epoch model, optimizer state, and standardization stats.
- `checkpoint_best.pt`: best training-energy-loss checkpoint.

## Main Metrics

| Metric | Value |
|---|---:|
| energy score (CRPS, lower better) | `0.133025` |
| median abs error | `0.182834` |
| 90% interval coverage | `0.815217` |
| 50% interval coverage | `0.403416` |
| mean 90% interval width | `0.636224` |
| mean 50% interval width | `0.237589` |

Nominal coverage is 0.90 and 0.50; closer is better-calibrated.

## Run Parameters

| Parameter | Value |
|---|---:|
| data | `/storage3/fs1/myu/Active/felixhu/weather_data.mat` |
| target | `log10(CCN)` |
| log_ccn | `True` |
| split | `paper` |
| event_flag | `n/a` |
| ccn_tail_quantile | `n/a` |
| max_samples | `all` |
| seq_stride | `1` |
| sequence shape (train) | `(57646, 241, 22)` |
| train_size arg | `all` |
| test_size arg | `all` |
| train rows used | `57646` |
| test rows used | `3220` |
| epochs | `100` |
| batch_size | `256` |
| hidden_dim | `128` |
| noise_dim | `96` |
| num_layer | `5` |
| learning_rate | `0.01` |
| weight_decay | `0.0` |
| prediction_samples | `400` |
| checkpointing | `True` |
| checkpoint_every | `25` |
| seed | `2026` |

## Split Interpretation

The test pool follows the source paper: the held-out months are January, March, May, July, September, and November 2022. Training rows are drawn from all other months. When provided as positive integers, the CLI train/test size arguments cap how many rows are used from each pool; `all` uses the full pools.

The `lstm` engression model was fit; the training
tensor has shape `(57646, 241, 22)`. The `lstm` variant keeps
lag windows unflattened `(n, seq_len, n_features)`; the flat variants (`vanilla`, `regularized`, `adamw`) take the flattened `(n, seq_len * n_features)`.
