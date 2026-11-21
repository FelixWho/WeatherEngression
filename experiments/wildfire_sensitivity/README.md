# Testing for Sensitive Variables

## Read the code in this order

The question is whether a covariate falls outside its predicted interval more
often during wildfire conditions than on clean observations. We first require
the counterfactual model to pass a calibration check on clean control data.

Start at [`main.main()`](main.py). Its candidate loop follows four steps:

1. Fit on **training** rows and reload the best **validation** checkpoint.
2. Use **recalibration** rows to fit the quantile correction.
3. Check the corrected model on **control** rows. A failed check leaves the
   variable undetermined and skips wildfire evaluation.
4. Compare wildfire and control 95% coverage. A drop above two percentage points
   means sensitive; otherwise add the variable to the conditioning inputs.

Until-stable mode repeats the remaining candidates only when the input set grew.
An added variable can therefore affect later candidates in the same round.

| File | Responsibility |
| --- | --- |
| [`main.py`](main.py) | CLI, run records, candidate order, screening rounds, classifications |
| [`settings.py`](settings.py) | Fixed seeds, split fractions, batch sizes, calibration levels, thresholds, starting inputs |
| [`control.py`](control.py) | `split_screening_data()` builds the four clean sets; block guards and month matching live below it |
| [`counterfactual.py`](counterfactual.py) | Arrival/trajectory target construction, training, best-checkpoint reload, recalibration fitting |
| [`evaluation.py`](evaluation.py) | Control calibration checks and wildfire coverage; numerical coverage diagnostics are separate from formatting/plots |

The production split still pools the original clean training/test data and reserves
**control → recalibration → validation**, leaving the rest for training. Each role
is shared across candidates. The CLI, seeds, sampling order, checkpoint names,
JSON/NPZ output fields, and decision rules are unchanged by this refactor.
The established training/evaluation helpers remain importable through `main.py`;
their implementations now live in the modules above.

The clarity refactor was checked against the pre-refactor implementation:
65 existing tests passed; 32 calibration cases and both recalibration target
modes produced identical values, logs, artifacts, and random-generator states;
16 complete mocked runs matched in both scopes, both gates, and both screening
modes. All four production split arrays also matched saved real-data indices.
An independent agent review found no actionable issues. These checks preserve
implementation behavior; they do not validate the statistical assumptions.

[`pit_dependence.py`](pit_dependence.py) and
[`split_feasibility.py`](split_feasibility.py) are separate diagnostics. In
particular, their comparisons do not alter production settings. The default split
and AD gate are unchanged; spaced control is now an explicit option described below.
The dated `HANDOFF.md`
is historical; use this README and current code for the active pipeline.

## Optional spaced control for sensitivity runs

`--control-gap-days 15` reserves control observations from the **entire eligible
clean pool**, taking the earliest observation and then the earliest at least
15 days after each selection. This maximizes the count under spacing alone.
It does not thin the old control set, match control to wildfire months, or
exclude nearby rows from fitting. Control rows themselves are removed before
recalibration and validation are split, so all roles remain disjoint.
Recalibration stays month-matched and its existing block guards remain, as do
the validation guards. The default, `--control-gap-days 0`, retains the original
blocked, month-matched control split exactly.

The same spaced control supplies **corrected PIT for AD and clean 95% coverage**.
The AD bootstrap, rejection threshold, quantile correction, and sensitivity rule
are unchanged. Thus these runs investigate a different clean reference population;
seasonal composition and remaining dependence still affect interpretation.

The runner was checked against the saved inventory: at 15 days, every role's row
indices match `control_split_feasibility/3663781/gap_15/earliest_first_maximum.npz`.

| Training | Validation | Recalibration | Control | Wildfire evaluation |
| ---: | ---: | ---: | ---: | ---: |
| 33,247 | 3,868 | 1,754 | 201 | 6,786 |

The controls occupy 107 populated 30-day AD blocks. Of 54,026 eligible clean rows,
14,956 are excluded by the remaining split guards and recalibration month matching.
Spacing and counts do not establish reliable AD p-values.

Three launchers, `slurm/wildfire_sensitivity/run_until_stable_arrival_ad_control15_order_{2027,2028,2029}.slurm`,
use arrival targets, AD, until-stable screening, and four-hour GPU limits.
Only candidate order varies across runs; data/training seeds stay 2026 and the
local AD bootstrap seed stays 2430. The shared runner accepts `CONTROL_GAP_DAYS`.
Checkpoints go into separate `ad_control15_arrival_order_<seed>/<job>/arrival/`
directories. `screening_split.npz` stores the selected rows; `screening_split.json`
stores split settings, counts, and seeds. Split settings and counts are also
included in `screening_order.json` and every candidate's calibration JSON.

Submitted on 2026-10-25: jobs **3685219 / 3685220 / 3685221**, corresponding to
order seeds **2027 / 2028 / 2029**. The user selected the 201-point control
configuration without seasonal matching. All 79 small tests passed before
submission, including both target scopes, both calibration gates, split disjointness,
default-split preservation, and the three launcher configurations. Test mocks were
updated to target the training/evaluation helpers currently defined in `main.py`.

Six additional runs without control month matching were submitted on 2026-10-29,
using candidate-order seeds 2035–2040. They keep the same 201 controls, 15-day
spacing, arrival targets, AD settings, until-stable screening, and four-hour GPU
limits. The new launchers explicitly select this checkout and separate
`ad_control15_arrival_order_<seed>/<job>/` checkpoint directories, so they cannot
inherit the seasonal trial's checkout or output path.

| Order seed | Job ID |
|---|---|
| 2035 | 4226425 |
| 2036 | 4226426 |
| 2037 | 4226427 |
| 2038 | 4226428 |
| 2039 | 4226429 |
| 2040 | 4226430 |

All six returned job IDs and were verified pending in `squeue` after submission.
Launcher shell syntax and settings were checked; the existing pipeline code was
unchanged and matched the source snapshot from the original 201-control runs.

On 2026-10-30, accounting confirmed seeds 2036–2040 completed in 53–68 minutes.
Seed 2035's job 4226425 timed out after four hours on c2-gpu-010: its log stopped
before split/training output and it wrote no checkpoint directory. The same
launcher was resubmitted as **4246525**, excluding that node for this retry;
the scientific settings and four-hour limit are unchanged.

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

With the default `--calibration-metric coverage`, a model must pass the clean-control central
90% and 95% coverage checks. Each coverage must lie within 3 block standard
errors of its nominal level. The standard errors use the same observation
weighting as pooled coverage and include every block. The 50% and 75% checks and
PIT summaries are still reported, but do not determine acceptance.

Alternatively, `--calibration-metric ad` uses an Anderson–Darling test of the
**recalibrated** marginal PIT on those same held-out clean control rows. It rejects
when the approximate block-bootstrap p-value is below `--ad-alpha` (default 0.05).
The 90%/95% coverage checks remain diagnostics in this mode; they do not also have
to pass. Both modes keep the same 95% control-versus-wildfire sensitivity rule.

AD uses 30-day arrival blocks, at least 10 populated blocks, and 2,000 bootstrap
replicates. All outputs of each trajectory stay together, including in trajectory
mode. Blocks are resampled with replacement; their actual sizes preserve pooled
observation weighting. The reference statistic is
$N^*\int_0^1(\widehat F^*(u)-\widehat F(u))^2/[u(1-u)]\,du$,
compared with observed $A^2$ against uniformity. Centering on the original empirical
CDF estimates sampling variation rather than resampling the observed departure
from uniformity. Integration is exact between clipped PIT breakpoints. P-values
use $(1+\#\{A^{2*}\ge A^2\})/(B_{\mathrm{boot}}+1)$.

This is an approximate check conditional on the fitted model and mapping. It
assumes representative, approximately independent blocks, does not resample the
recalibration fit, and does not establish conditional or joint calibration.
Predictive PIT values are also Monte Carlo approximations with 400 draws. Before
relying on AD classifications, inspect synthetic rejection rates from
`slurm/ad_bootstrap_validation/run.slurm`: it includes IID and dependent uniform
PIT, unequal blocks, an independently fitted recalibration map, and a
miscalibrated alternative. This validation job is separate from screening.

The simpler time-spacing alternative is tested separately in
`experiments/validate_ad_thinning.py`, submitted with
`sbatch slurm/ad_thinning_validation/run.slurm`. It retains the earliest observation,
then the next at least 30, 60, or 90 days later, and compares its AD score with
IID-uniform simulations of the same retained sample size. A zero-gap baseline
retains all observations. Selection uses timestamps only; there is one p-value
per selected sample, with no averaging or selection among subset p-values.
The experiment records false-rejection rates, power, uncertainty intervals,
retained indices/counts, and seeds. It reuses the same synthetic dependence
cases and observed block sizes as the bootstrap audit, but places synthetic
blocks 30 days apart rather than using the actual ENA calendar. The IID reference
is cached by sample size for efficiency. This experiment does not change the
production calibration gate or establish that any spacing ensures independence.

The follow-up audit (job `3571312`, seed 2432, 5,000 repetitions per case) found
false-rejection rates of 4.9–5.4% at 30-day spacing when dependence was confined
to independent blocks. With dependence across blocks, false rejections were
22.3%, 11.5%, and 8.0% at 30/60/90 days. Those spacings retained 33/17/11 values;
detection of the chosen miscalibrated alternative fell from 95.4% to 12.2% to
2.4%. These are synthetic results, not a selection of an ENA spacing. Full
settings, p-values, and uncertainty intervals are in
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_thinning_validation/3571312/results.json`.
The initial 500-repetition audit is retained under job `3571302`.

For real-data spacing diagnostics, submit `sbatch slurm/pit_dependence/run.slurm`.
It reloads the latest saved model for each candidate in the three arrival AD
until-stable runs and samples only their saved clean recalibration rows. The full
input histories and original conditioning-feature order are preserved. It saves
raw/corrected PIT with timestamps and row IDs, correlations at elapsed-time lags
of 1/3/7/10/14/21/30/60/90 days (±0.5-day bins), and correlations between adjacent
retained observations after thinning at each minimum gap. Tail diagnostics use
PIT ≤0.05 and ≥0.95. Outputs include pair counts, tail-event counts, retained
sample counts, actual gap ranges, a heatmap, and seeds/checkpoint paths in a
manifest. Pairs overlap, so the script does not attach IID confidence intervals
or automatically declare independence. These are development diagnostics on the
rows used to fit the recalibration map, not independent calibration validation;
no control outcomes are evaluated and no production spacing or gate changes.

For dependence on the **held-out control rows used by AD**, submit
`sbatch slurm/control_pit_dependence/run.slurm` (job `3664159`). The shared
diagnostic now accepts `--role control` and explicit `--gap-days`; the original
recalibration default remains available. This job reloads the same 36 latest
arrival checkpoints from the three until-stable AD runs and their existing
recalibration mappings. It evaluates each model on its original 2,159 clean
control rows, checking disjointness from training, validation, recalibration,
and wildfire. Mapping size is checked against the 1,350 recalibration rows,
independently of the number of control observations. No model or mapping is fitted.

It measures corrected/raw PIT rank and Pearson correlations, plus lower/upper
5% event correlations, at elapsed lags 1/3/7/10/15/20/30/60/90 days (±0.5-day
bins). The adjacent-observation view also reports the actual gaps and counts
remaining after thinning these saved controls. Raw PIT, corrected PIT, timestamps,
row IDs, checkpoints, seeds, pair counts, and unique-row counts are retained under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_pit_dependence/3664159/`.
The GPU job loads trajectories once and has a two-hour limit. Eight small tests
pass, including an end-to-end control selection check with different control
and recalibration counts. Control outcomes were previously used by screening;
this is a descriptive dependence diagnostic, not a new independent validation
set or proof that any gap makes observations independent.

For the **76-control seasonal runs**, job **4229123** was submitted on 2026-10-29
with `slurm/control_pit_dependence/run_seasonal76.slurm`. It reloads the latest
12 candidate models from each of the seven completed order seeds
2027–2032 and 2034 (84 models), retaining their saved feature orders and mappings.
The existing diagnostic samples 400 fresh draws on each model's saved controls;
no model or recalibration mapping is fitted.

`compare_pit_dependence.py` then compares corrected-PIT Spearman correlations
with the five known-uniform scenarios from the AD audit, using the exact same
dates and pairs. At 15/30/60/90 days (±0.5 days), there are 43/34/15/9 pairs.
Each scenario has 1,000 simulations, seed 2435. Its 2.5th/50th/97.5th percentiles
are saved alongside each model's observed correlation and pair counts. These
are pointwise simulated ranges, not confidence intervals for true dependence
or a fitted decay time. The 84 models share observations; they are not 84
independent datasets. Sparse pairs and PIT ties are reported, not filtered away.

The job requests one GPU for two hours on `compute2-myu`. PIT arrays, inference
seeds, and the model manifest are saved in
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_pit_dependence/4229123/`;
`synthetic_comparison/` contains `comparison.csv`, `settings.json`, and the
simulated correlations. All 20 dependence/comparison/audit tests passed.

Job **4229123** completed successfully in **3m07s**, confirmed by accounting on
2026-10-30. Its manifest contains all 84 models and the synthetic comparison
finished and saved its CSV, settings, and reference arrays.

To measure a different split **before training**, submit
`sbatch slurm/recalibration_split_feasibility/run.slurm`. This diagnostic selects
recalibration points directly from the **full eligible clean pool**, at least
10 days apart. It compares requested counts 25/50/75/100/150/200/250/300 with
three local selection seeds. Random greedy selection uses integer month quotas
matching the wildfire month mix; any shortfall is reported without relaxing the
spacing. It removes other clean arrivals within 241 hours on either side of
each selection, then applies the existing blocked control and validation splits
and control month matching. It also reproduces the current split and computes
the earliest-first maximum under spacing alone (without month quotas or a
requirement to leave training data). Outputs account separately for each role,
guard exclusions, month-matching discards, and any unassigned rows if a split
cannot finish. JSON/CSV counts and all split indices are saved under
`weather_checkpoints/recalibration_split_feasibility/<jobid>/`. This CPU job
loads the data once, fits no models, and does not change production splitting.

To repeat the comparison without the exclusion zone around recalibration
points, use `sbatch slurm/recalibration_split_feasibility/run_no_recalibration_guard.slurm`.
It passes `--recalibration-guard-days 0` while retaining the 10-day spacing
between recalibration points, disjoint row assignments, and the existing
control/validation block guards. It reuses the full eligible population saved
by job `3662101` via `--population-from`, avoiding a second trajectory load.
The current baseline always retains its original splitting rules.

Job `3662101` completed this inventory in 2m45s. All four baseline index arrays
matched the saved until-stable run `3277702` exactly. The full clean pool has
54,026 rows; earliest-first retains 296 at a 10-day minimum gap. Guarding those
296 excludes all other 53,730 rows. With 25 requested, seasonally allocated
recalibration points, the three selections left 24,052–26,058 training,
2,696–3,130 validation, and 441–1,465 control rows; recalibration guards alone
excluded 8,397–8,794. With 50 points these ranges were 21,102–21,984 training,
2,037–2,838 validation, 76–562 control, and 14,305–15,559 guard exclusions.
At 75 requested points and above, every tested allocation failed control month
matching (missing wildfire months in its control pool); requests of 100 or more
also had greedy selection shortfalls. This is evidence about this allocation
procedure, not proof that no other split is feasible. Detailed counts and all
indices are in
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/recalibration_split_feasibility/3662101/`.

The comparison without recalibration exclusion zones (job `3662970`, 14s)
reused exactly the same population and recalibration selections. All proposed
splits completed, with no rows excluded by a recalibration guard. At the
spacing-only maximum of 296 recalibration points, 31,143 rows remained for
training, 3,575 for validation, and 2,136 for control; 16,876 were excluded by
the remaining block guards and control month matching. This maximum does not
enforce month quotas on recalibration. For a seasonally allocated example,
75 recalibration points with selection seed 2228 left 31,283 training,
3,589 validation, and 2,154 control rows. Across the three seeds, the 75-point
counts were 31,283–31,290 training, 3,588–3,592 validation, and 2,145–2,154 control.
Requests above 75 retained the previous greedy selection shortfalls; removing
the exclusion zone fixes the downstream allocation failures, not those
shortfalls. Full counts and indices are under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/recalibration_split_feasibility/3662970/`.
Neither inventory changes production splitting or validates calibration.

The follow-up comparison targets **control**, whose PIT values are used by AD,
rather than recalibration, which fits the correction. Submit
`sbatch slurm/control_split_feasibility/run.slurm` to select control rows from
the full clean pool at least 10 days apart, without excluding nearby rows from
other roles. Recalibration and validation then use their existing blocked splits;
recalibration remains month-matched and is not thinned. This is a count diagnostic
only; production splitting and the AD test are unchanged.

Job `3663360` completed in 14s using the cached population. Counts were:

| Clean-data role | Current split | 75 spaced control rows | Maximum 296 spaced control rows |
| --- | ---: | ---: | ---: |
| Training | 23,564 | 33,343 | 33,187 |
| Validation | 3,110 | 3,877 | 3,864 |
| Recalibration | 1,350 | 1,750 | 1,750 |
| Control | 2,159 | 75 | 296 |
| Excluded | 23,843 | 14,981 | 14,929 |

Each column accounts for all 54,026 clean rows; 6,786 wildfire rows are separate.
The 75-row example uses selection seed 2228 and rounded wildfire-month quotas;
all three seeds achieved this count. The 296-row maximum uses earliest-first
selection without month quotas. Requests of 100–300 with month quotas retained
91–132 rows depending on requested count and seed; these greedy shortfalls do
not establish the maximum under seasonal matching. Remaining exclusions come
from recalibration/validation block guards and recalibration month matching.
Counts alone do not establish PIT independence or reliable AD p-values.
All settings, seeds, counts, and split indices are saved under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/control_split_feasibility/3663360/`.

To test whether this control spacing helps AD, use the separate audit
`experiments/validate_ad_control_spacing.py`, submitted through
`sbatch slurm/ad_control_spacing_validation/run.slurm` (array `3663625`). It uses
the saved real observation dates and compares current control (2,159 rows),
75 spaced rows, and the spacing-only maximum (296 rows). Each spaced selection
has a random, unspaced comparator with exactly the same count in each month.
Selections are fixed before simulation; each trial uses the same synthetic PIT
process across all five selections. No trajectories, models, or outcomes are
loaded, and production screening is unchanged.

The synthetic process has exactly uniform marginal PIT under the null. A
stationary Gaussian component has correlation $\exp(-|\Delta t|/\tau)$; adding
independent Gaussian noise gives latent covariance
$\rho\exp(-|\Delta t|/\tau)$ between distinct observations. Applying the normal
CDF gives uniform PIT. The five null cases are independent values, moderate
dependence ($\rho=0.3$, $\tau=3$ days), and strong dependence ($\rho=0.8$,
$\tau=3,10,30$ days). Two alternatives use $0.1+0.8U$ (independent and strong
10-day dependence) to assess detection of overly wide predictions. These are
stress-test assumptions, not dependence estimates fitted to ENA data.

Each scenario runs 1,000 trials. It calls the unchanged production block AD
bootstrap (30-day blocks, 2,000 replicates, minimum 10 blocks, rejection below
0.05) and separately reports the IID AD reference (20,000 null simulations per
sample size). Inspect false-rejection rates and 95% binomial intervals under
the null, and detection rates under the alternatives. The 75/296 comparisons
do not by themselves pick a production control count. This audit excludes
recalibration-fit uncertainty and finite predictive sampling error.

Simulation seed 2433, comparator-selection seed 2434, seed derivations, input
indices, all p-values, invalid reasons, and CSV summaries are saved under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_control_spacing_validation/3663625/<scenario>/`.
Partial results are saved every 100 trials. The array runs at most three CPU
tasks at once, with a one-hour limit per task.

All seven tasks completed successfully in 1m55s–2m02s each. For the **current
block-bootstrap AD procedure**, the false-rejection percentages were:

| Synthetic dependence | Current 2,159 | Random 75 | Spaced 75 | Spaced 296 |
| --- | ---: | ---: | ---: | ---: |
| Independent | 7.3% | 6.4% | 5.4% | 6.5% |
| Moderate, 3-day decay | 11.1% | 6.4% | 6.1% | 4.8% |
| Strong, 3-day decay | 13.8% | 7.2% | 5.3% | 5.8% |
| Strong, 10-day decay | 15.6% | 9.4% | 7.8% | 7.5% |
| Strong, 30-day decay | 21.3% | 14.3% | 13.8% | 20.4% |

The target is 5%. Spacing performs well in the short-dependence cases, but does
not resolve the longer-dependence failures. For spaced 75, the 95% intervals
are 6.2–9.6% in the strong 10-day case and 11.7–16.1% in the strong 30-day case.
Random 75 also improves over the current selection, so changes in count and
calendar spread contribute to the improvement. The 296-row maximum remains
unmatched to wildfire seasonality and is not a chosen production split.

Detection of the specified overly wide alternative was 92.9% with independent
PIT and 88.7% with strong 10-day dependence for spaced 75, and 100% in both
cases for spaced 296. The separate IID test on spaced 75 had false-rejection
rates 4.7/5.2/4.6/8.7/19.6% in the five table scenarios, and detection rates
63.9/66.9%. It also fails to control false rejections under longer dependence.
All 70,000 p-values were valid, selection arrays matched across tasks, and
comparator month counts matched their spaced partners. Full intervals and
both methods' results are in the per-scenario JSON/CSV files. These simulations
do not justify treating 10-day spacing as a complete fix for the real-data AD
gate; no production split or calibration method was changed by this audit.

The 15/20-day follow-up uses `slurm/control_split_feasibility/run_wider_gaps.slurm`
for counts and `slurm/ad_control_spacing_validation/run_wider_gaps.slurm` for AD.
The final count inventory is `3663781`: with the same seasonal selection seed
2228, all gaps complete 45 points, but 20 days retains only 48 of 50 requested.
Requests for 75 retain 75/65/55 at 10/15/20 days. These are greedy-selection
outcomes, not maximum counts under seasonal quotas. Under spacing alone,
the maximum counts are 296/201/153.

Audit array `3663788` pairs all three gaps in each of the same seven scenarios,
with 1,000 trials per scenario and the same AD methods, settings, and seeds.
It compares a fixed seasonal count of **45 at each gap**, each gap's maximum,
and equal-count/month random comparators, keeping one current-control baseline.
This reruns the 10-day baseline in the joint comparison; adding observation
dates changes the individual simulated realizations relative to `3663625`.
Each trial now shares one realization across all gaps. The common seasonal
count was chosen from timestamp-only feasibility checks, before any new PIT
results. No production split, AD gate, or fitted model changes.

The audit accepts multiple `--inventory` paths and a `--seasonal-count`; its
original single-inventory, 75-point default remains available. It rejects
incomplete seasonal selections, mismatched source calendars or current controls,
and duplicate gaps. Seven small audit tests pass, including paired PIT reuse
and multiple-inventory checks. Results, indices, and seed rules are saved under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_control_spacing_validation/3663788/<scenario>/`.

All seven follow-up tasks completed in 4m48s–5m09s. With **45 seasonal points
at every gap**, the current block-bootstrap AD false-rejection rates were:

| Synthetic dependence | 10-day spacing | 15-day spacing | 20-day spacing |
| --- | ---: | ---: | ---: |
| Independent | 5.9% | 5.8% | 4.7% |
| Moderate, 3-day decay | 5.8% | 5.9% | 5.3% |
| Strong, 3-day decay | 6.2% | 4.9% | 4.5% |
| Strong, 10-day decay | 6.9% | 7.7% | 6.9% |
| Strong, 30-day decay | 12.0% | 12.0% | 11.0% |

The target remains 5%. Wider gaps do not resolve the longer-dependence problem:
the 20-day, strong 30-day case has a 95% interval of 9.1–13.1%. At the respective
maximum counts 296/201/153, the strong 30-day false-rejection rates are
20.7/18.9/17.4%, while the strong 10-day rates are 8.3/8.0/6.8%. These maximum
comparisons change both count and spacing, and do not enforce wildfire seasonality.

For the specified overly wide alternative, block AD detection at 45 points is
59.0/57.7/56.8% under independence and 57.6/57.5/59.7% with strong 10-day dependence
(ordered 10/15/20-day spacing). All maximum-size selections detect it in 100% of
trials. The separate IID-reference test also over-rejects under strong 30-day
dependence: 15.1/14.2/10.9% at 45 points and 37.9/27.0/22.0% at the maxima. Its
detection rates at 45 points are only 15.7–20.5% across these alternatives.
Thus 15/20-day spacing is not a general solution in the simulated conditions.
All 182,000 p-values were valid; selection arrays matched across scenarios,
and the three seasonal random comparators used exactly the same rows.

On 2026-10-29, array **4227288** was submitted with
`slurm/ad_control_spacing_validation/run_seasonal76.slurm` to audit the exact
**76 season-matched controls** used by sensitivity run **3685284** (order seed
2027). The audit's new `--control-split` option reads those saved rows directly;
the cached calendar supplies their dates. No new selection is made. The controls
retain their 15-day minimum spacing and 51 populated 30-day blocks.

The seven existing scenarios each run 1,000 trials: five known-uniform cases
measure false rejections and two overly wide cases measure detection. Production
AD uses 2,000 bootstrap replicates and alpha 0.05; audit seed 2433 and per-trial
seed rules are saved. Read the `block_bootstrap` summary rows for the production
test; `iid_reference` remains a separate diagnostic. Dependence strengths are
assumed scenarios, not fitted estimates. This audit uses exact synthetic PIT,
without model fitting, predictive sampling error, or recalibration fitting.

The CPU array runs at most three tasks at once, with 20 minutes per task.
Results, rejection-rate intervals, and the exact row IDs/dates are saved under
`/storage3/fs1/myu/Active/felixhu/weather_checkpoints/ad_control_spacing_validation/4227288/<scenario>/`.
All 27 audit/AD tests and a tiny CLI check passed; the production AD source is
identical to the seasonal checkout's. The sensitivity pipeline was not changed.

The wrapper now saves all sorted calibration PIT values in versioned JSON. Its
linear empirical quantile interpolation preserves existing adjusted boundaries;
the generalized inverse provides the corrected CDF. For ties, CDF/rank interval
bounds are transformed **before** PIT randomization. This adds marginal CDF/PIT
evaluation, not a corrected joint sampler. Old sparse quantile JSON files still
work for coverage, but AD requires refitting their full mapping; it never falls
back to raw PIT. The reusable model-facing API is
`experiments.calibration.evaluate_ad(model, x, y, time_days, evaluation_scope=...)`;
precomputed draws can use `model.pit_from_samples(...)` and `ad_block_bootstrap(...)`.

If the selected check fails, is invalid, or there are too few blocks to estimate
uncertainty, the candidate is **undetermined** and stays out of the insensitive
input set. Accepted models use the existing sensitivity rule: a central 95%
coverage drop greater than two percentage points. In `until_stable` mode,
undetermined candidates are retried alongside sensitive candidates when the
input set grows; the run stops when no candidate is added in a round.

The production split lives in [`control.py`](control.py). Nearby arrivals have
overlapping history dates and may be dependent. The current conservative rule
uses contiguous arrival-time blocks, discards each holdout block's leading
241 hours, and purges nearby fit rows after the block. This is an experimental
design choice; overlapping dates do not establish identical measurements or
prove that a full history-length exclusion is necessary.



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

To select AD for an arrival-only until-stable run:

```bash
sbatch --export=ALL,CALIBRATION_METRIC=ad slurm/wildfire_sensitivity/run_until_stable_arrival.slurm
```

The shared runner also accepts `AD_ALPHA`, `AD_BOOTSTRAP_REPLICATES`, and
`AD_BOOTSTRAP_SEED` (defaults 0.05, 2000, 2430). The corresponding Python options
are `--ad-alpha`, `--ad-bootstrap-replicates`, and `--ad-bootstrap-seed`.
The bootstrap has its own local RNG and reuses existing prediction draws, so it
does not advance the training or predictive sampling streams. Metric settings
are saved in `screening_order.json`; each round also saves
`<candidate>_calibration.json` with settings, seeds, measurements, acceptance, and
reason. Invalid numerical fields are JSON `null`.

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

## Additional AD and coverage order runs: submitted 2026-10-30

The user requested six more runs with the latest season-matched AD design and
ten more with the original 90%/95% coverage design. All use arrival-only targets,
until-stable screening, unchanged training/split seeds, and four-hour single-GPU
allocations on `compute2-myu`.

| Calibration check | Additional runs | Candidate-order seeds | Clean control |
|---|---:|---|---|
| AD, alpha 0.05 | 6 | 2041–2046 | 76 season-matched rows, at least 15 days apart |
| Central 90% and 95%, each within 3 block SE | 10 | 2032–2041 | Original 2,159-row blocked, season-matched control; no spacing constraint |

AD runs use `runs/seasonal_control_trial` and its existing
`run_until_stable_arrival_ad_control15_seasonal_order_<seed>.slurm` naming.
Coverage runs use this checkout and new
`run_until_stable_arrival_coverage90_95_order_<seed>.slurm` launchers.
Checkpoints are isolated by metric, order seed, and job ID. Experiment source
code is unchanged. After cancellation, all 16 jobs were resubmitted with these
original configurations at the user's request: AD jobs 4247550–4247555 and
coverage jobs 4247556–4247565. The two groups intentionally retain their different
historical control splits. The temporary coverage jobs using 15-day spacing were
canceled and are not part of this experiment.

The [submission manifest](../../slurm/wildfire_sensitivity/additional_ad_and_coverage_2026-10-30.json)
records the current 16 job IDs, launchers, log/checkpoint paths, settings, and
source hashes; canceled attempts remain under `superseded_runs`.
All launchers passed shell syntax and intercepted-environment checks before
submission. These additions bring the planned totals to 20 season-matched AD
runs and 16 coverage runs, counting the completed original-order coverage run.
