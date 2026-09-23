# Testing for Sensitive Variables

## Motivation

We aim to understand which variables are mediators of wildfire effects. For example, wildfire may impact the temperature covariate, which in turn impacts CCN count.

Once we have the set of insensitive versus sensitive covariates, $M, M'$ respectively, we can predict the values for $M'$ in the absence of wildfires. This is the **counterfactual**.

## Stategy

We start with an insensitive set determined by weather experts (Shengqian, Wang), $M$. The remaining covariates, $M'$ are assumed to be wildfire-sensitive until we have evidence otherwise.

Imagine $M = \{A, B, C\}$, $M' = \{D, E, F\}$. One by one consider $D, E, F$ to see if they should be moved into $M$. How?

Train a counterfactual model $D = f(M, \varepsilon)$ on clean trajectories, then ask how
often the observed $D$ falls inside its predicted interval. Low coverage during
wildfire periods suggests $D$ moves with smoke.

Coverage alone cannot say that, though, because it also drops when the model is
under-dispersed or when wildfire periods are simply a different season (84% arrive
Jun-Sep). So the comparison is against a **clean control**: trajectories the model
never trained on, drawn to match the wildfire group's month distribution. This
reduces seasonal differences but does not guarantee equal prediction error in the
two groups or establish a causal effect. $D$ is classified sensitive when wildfire coverage falls
more than `SENSITIVITY_MARGIN` below the control's.

The screening combines the original paper split's clean training and test rows.
It then creates four separate sets: training, validation, recalibration, and
clean control. The control is held out first, recalibration rows are held out
from the remainder, then the remaining rows are split for training and validation.
Validation controls early stopping
and selects the checkpoint with the lowest validation energy loss. That checkpoint
is reloaded before recalibration. Recalibration and control rows are separately
month-matched to the wildfire group. Wildfire outcomes are used only for evaluation.

[`RecalibratedEngressor`](../../engression_modifications/recalibrated_engressor.py)
wraps the fitted model and stores a nominal-to-adjusted quantile-level mapping.
It takes empirical quantiles of the raw model's randomized PIT values on the
recalibration set, pooling all trajectories and selected timesteps into one mapping per
candidate. This is empirical quantile mapping, not isotonic regression. For a
nominal quantile $q$, the adjusted probability is $\hat R^{-1}(q)$, where $\hat R$
is the pooled empirical PIT CDF. The wrapper then evaluates the base model at
that adjusted quantile level.

Both clean-control and wildfire coverage use the same fixed mapping. Coverage
is counted directly against adjusted numerical interval endpoints, including
for the acceptance checks. PIT histograms and moments remain explicitly labeled
**raw-model** diagnostics; this wrapper does not recalibrate joint samples.
Mappings are saved as JSON alongside each round's base-model checkpoints, and
`screening_split.npz` records the dataset row indices used for each role.

`--evaluation-scope trajectory` (the default) trains and evaluates all 241 target
timesteps. `--evaluation-scope arrival` trains and evaluates only the final array
position, following the ENA loader's endpoint convention. Inputs still contain
the full history `(N, 241, K)`, but training and validation targets are `(N, 1)`,
so the model has `out_dim=1`. Recalibration, clean-control checks (including block
standard errors and raw PIT diagnostics), and wildfire coverage use that same
scalar target. Arrival models and quantile mappings are fitted from scratch;
old trajectory-output checkpoints are not reused.
Arrival runs save artifacts under an `arrival/` subdirectory of the checkpoint
directory; the scope is also logged and stored in `screening_split.npz`.
Shape checks reject a trajectory-output model in arrival mode, mismatched
predictions and targets, and nonfinite predictions. Month matching stops if any
wildfire month has no clean counterpart. See [the implementation audit](AUDIT.md)
for checks and remaining statistical assumptions.

Before comparing wildfire coverage, a model must pass the clean-control central
90% and 95% coverage checks. Each coverage must lie within 3 block standard
errors of its nominal level. The standard errors use the same observation
weighting as pooled coverage and include every block. The 50% and 75% checks and
PIT summaries are still reported, but do not determine acceptance.

If either required check fails, or there are too few blocks to estimate
uncertainty, the candidate is **undetermined** and stays out of the insensitive
input set. Accepted models use the existing sensitivity rule: a central 95%
coverage drop greater than two percentage points. In `until_stable` mode,
undetermined candidates are retried alongside sensitive candidates when the
input set grows; the run stops when no candidate is added in a round.

The control lives in [`control.py`](control.py). Two details the ENA data forces:
trajectories are hourly and overlap by 240 of their 241 hours, so the holdout is
split off in contiguous blocks of arrival time rather than at random, and each
holdout block discards its leading 241 hours. Fit rows immediately after a holdout
block are also purged so their histories cannot reach backwards into the holdout.



## Running

On the cluster (the screening trains one LSTM per candidate, so it needs a GPU):

```bash
sbatch slurm/wildfire_sensitivity/run_until_stable.slurm
```

For arrival-only training and evaluation:

```bash
sbatch slurm/wildfire_sensitivity/run_until_stable_arrival.slurm
```

This dedicated job uses a four-hour GPU allocation and writes
`run_until_stable_arrival_<jobid>.log`. It trains on full input histories with
single arrival-value targets and evaluates those arrival values.

`run_until_stable.slurm` is the same job with `--screening-mode until_stable`.
Both read the `.mat` from storage3 and write checkpoints there; override with
`MAT_PATH`, `CKPT_DIR`, or `REPO_ROOT`. The scripts in `local_runs/` run the same
screening on a local GPU or Apple silicon.

To compare a different candidate order while keeping training and split seeds
fixed, submit `slurm/wildfire_sensitivity/run_until_stable_arrival_order_seed.slurm`.
It sets `CANDIDATE_ORDER_SEED=2027` (overridable), which passes
`--candidate-order-seed` to the entry point. Candidates are shuffled once; later
rounds preserve their relative order. The default without this option retains
the original ordering. Each run logs the order and saves `screening_order.json`
alongside its split file and checkpoints.

Every run prints all seeds before loading data and includes them in
`screening_order.json`: the sweep/data/training seed, paper-split seed, clean
holdout and month-matching seeds, validation seed, PIT base seed, and candidate
order seed (`null` means original order). It also records the PIT batch seed rule,
batch sizes, and predictive draw count. Predictive draws continue the Torch RNG
stream after training; they do not use an independently reset sampling seed.
