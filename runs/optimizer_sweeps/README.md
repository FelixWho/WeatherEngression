# Optimizer Sweeps

This folder contains optimizer-only sweeps for engression diagnostics.

The first sweep type is classic Adam weight decay:

```bash
python engression_modifications/classic_adam_sweep.py
```

The default run varies learning rate and classic Adam `weight_decay` while
keeping the stochastic MLP architecture fixed. Each run folder contains:

- `results.csv`: sorted scalar results for all parameter combinations.
- `results.json`: command metadata and all scalar results.
- `README.md`: a short run card with the best setting by a heuristic score.

The main quantities to watch are:

\[
\text{mean quantile MAE}, \qquad
\widehat{c}_{90}, \qquad
\widehat{w}_{90}/w_{90}.
\]

Here \(\widehat{c}_{90}\) is predicted 90% interval coverage and
\(\widehat{w}_{90}/w_{90}\) is predicted interval width divided by true
interval width.
