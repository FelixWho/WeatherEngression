# LSTM Engression Sweep: narx_student_t / in-support / d=12

This run sweeps only ordinary optimization knobs for the local LSTM
engression model: epoch count and classic Adam `weight_decay`.

## Command

```bash
python engression_modifications/lstm_sweep.py --model narx_student_t --split in-support -d 12 --num-samples 5000 --train-size 4000 --test-size 500 --window 12 --epochs-list 40,60,80,120 --weight-decays 0,0.003 --hidden-dim 192 --noise-dim 96 --batch-size 512 --lr 0.003 --prediction-samples 300
```

## Best By Heuristic Calibration Score

| Metric | Value |
|---|---:|
| epochs | `40` |
| weight decay | `0.0` |
| mean quantile MAE | `0.141264` |
| predicted 90% coverage | `0.884000` |
| true 90% coverage | `0.920000` |
| predicted 90% width | `1.239518` |
| true 90% width | `1.295003` |
| width ratio | `0.957154` |
| calibration score | `0.198687` |

The heuristic score is:

\[
\text{mean quantile MAE}
+ |\widehat{c}_{90}-c_{90}|
+ 0.5\,|\widehat{w}_{90}/w_{90}-1|.
\]

Use the raw `results.csv` columns for final judgment.
