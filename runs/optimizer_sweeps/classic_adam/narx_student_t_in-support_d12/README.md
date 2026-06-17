# Classic Adam Sweep: narx_student_t / in-support / d=12

This run sweeps classic `torch.optim.Adam(..., weight_decay=...)`
while keeping the engression stochastic MLP architecture fixed.

## Command

```bash
python engression_modifications/classic_adam_sweep.py
```

## Outputs

- `results.csv`: sweep table sorted by the heuristic calibration score.
- `results.json`: command metadata and all scalar results.

## Best By Heuristic Calibration Score

| Metric | Value |
|---|---:|
| learning rate | `0.003` |
| weight decay | `0.003` |
| mean quantile MAE | `0.157488` |
| predicted 90% coverage | `0.866000` |
| true 90% coverage | `0.920000` |
| predicted 90% width | `1.304359` |
| true 90% width | `1.295003` |
| width ratio | `1.007224` |
| calibration score | `0.215100` |

The heuristic score is:

\[
\text{mean quantile MAE}
+ |\widehat{c}_{90}-c_{90}|
+ 0.5\,|\widehat{w}_{90}/w_{90}-1|.
\]

It is only a selection aid. The raw columns in `results.csv` are the
important record.
