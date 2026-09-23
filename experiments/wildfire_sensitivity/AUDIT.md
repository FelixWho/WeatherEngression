# Sensitivity implementation audit — 2026-09-19

Scope: both target modes in `main.py`, the shared LSTM fitting and checkpoint
code, quantile mapping, data splits, coverage, and until-stable classification.
Job 3110691 is the corrected arrival run; it was still pending during this audit.
No new real-data results are claimed here.

## Target contract

| Stage | Trajectory | Arrival |
|---|---|---|
| Conditioning inputs | `(N, 241, K)` | `(N, 241, K)` |
| Training and validation target | `(N, 241)` | `(N, 1)` |
| Model output dimension | 241 | 1 |
| Predictive draws | `(N, 241, S)` | `(N, 1, S)` |
| Values contributing to calibration/coverage | `N * 241` | `N` |

The candidate variable is excluded from the conditioning inputs. Both modes fit
on clean training rows, standardize using training moments, select the checkpoint
with minimum validation energy loss, and reload it before calibration. Arrival
uses the final raw array position. Evaluation now rejects incompatible model or
target dimensions instead of silently slicing a trajectory model's last output.

The loader treats the last row as the endpoint. A raw trajectory was checked to
have shape `(241, 22)`. The supplied MAT file contains arrival times but no
per-timestep timestamps or coordinates; raw chronological ordering cannot be
independently established from that metadata. Confirmation from the data producer
or original preparation code remains outstanding.

## Calculations checked

- Training, validation, recalibration, and clean-control rows are disjoint;
  sequential block splits guard both temporal boundaries. Inputs retain their
  full history in arrival mode, so the temporal guards remain necessary.
- Recalibration uses its own clean rows. PIT values are flattened over selected
  target coordinates and used to fit one probability mapping per candidate.
  The mapping is fixed for clean-control and wildfire evaluation.
- Central 90% intervals use mapped 5% and 95% endpoints; central 95% intervals
  use mapped 2.5% and 97.5% endpoints. One-sided coverage is diagnostic only.
- Coverage counts numerical endpoint comparisons, divided by the total number
  of target values. A short final batch has its actual weight.
- For block counts `n_b`, coverage `p_b`, total count `N`, and pooled coverage
  `p`, the standard error is
  \[
  \widehat{\mathrm{SE}}(p)=
  \frac{\sqrt{\frac{B}{B-1}\sum_b[n_b(p_b-p)]^2}}{N}.
  \]
  Counts include all selected coordinates: one per row for arrival, 241 per row
  for trajectory. Unequal block sizes are retained. Fewer than two blocks fails
  the gate; both required coverages must satisfy the 3-SE rule.
- Failed gates leave candidates undetermined. Accepted candidates are sensitive
  only when control 95% coverage minus wildfire 95% coverage exceeds .02.
  A numerical tolerance of `1e-12` prevents float roundoff at exactly .02.
- Until-stable screening retries sensitive/undetermined candidates when the input
  set grows. Already accepted candidates remain in the input set.
- Runtime checks reject mismatched dimensions, nonfinite samples/targets, and
  invalid control times. Missing clean-reference months now stop the run.

## Validation evidence

28 unit tests passed across `test_wildfire_calibration`,
`test_recalibrated_engressor`, and `test_sensitivity_audit`. Cases include both
target modes; hand-calculated central and one-sided coverage; unequal blocks and
batches; asymmetric adjusted endpoints; mapping save/reload; held-out synthetic
calibration; actual small LSTM forward/sample and checkpoint round trips with
1 and 241 outputs; correct output units; validation-target forwarding;
best-checkpoint selection; split separation; and the sensitivity boundary.
Tests perform no network fitting or full-data loading. Shell syntax checks passed
for the shared, until-stable, and arrival SLURM scripts. The queued job was checked
for `until_stable`/`arrival` dispatch, `compute2-myu`, one GPU, and four hours.

## Limits of the statistical interpretation

- Trajectory coverage is pooled marginal coverage, not simultaneous coverage of
  an entire path or calibration conditional on every input. Arrival coverage is
  pooled over arrival observations, also not conditional calibration.
- The 3-SE gate is an approximate rule assuming approximately independent time
  blocks. It conditions on the fitted model/mapping and does not include their
  refitting uncertainty. Wide bands can accept practically poor coverage.
- Recalibration is approximate with 400 predictive draws. Randomized-rank PIT
  inversion and interpolated sample quantiles are not exact inverses. Even for a
  perfect Uniform(0,1) forecast with an identity mapping, expected central
  coverage is `level * 399/401`: about 89.55% and 94.53%. No change to this
  established method or draw count was made during the audit.
- The .02 sensitivity margin is a chosen effect-size cutoff, not a significance
  test or a multiple-testing correction. Labels concern lower wildfire coverage;
  other distributional changes may be missed. Month matching is approximate due
  to integer rounding and does not remove every distribution shift.
- In arrival mode, an admitted variable has passed an arrival-value test only,
  but its full history becomes an input. That does not establish that its earlier
  history is wildfire-insensitive. Input-set growth also uses previous evaluation
  decisions, so held-out rows are not an untouched final validation of the entire
  adaptive screening procedure. Order dependence remains possible.

These are limits of the current design, not resolved by matching tensor shapes.
