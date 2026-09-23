# Handoff: wildfire-sensitivity screening

Written 2026-09-16. Covers the recent work on `experiments/wildfire_sensitivity/`.
Read [PROGRESS.md](../../PROGRESS.md) first for where this sits in the project, and
[README.md](README.md) for what the screening is trying to do.

## What the screening does

For each candidate covariate `c`, train a generative model `c = f(M, eps)` on clean
(no-wildfire) trajectories, where `M` is the set of covariates currently believed to
be wildfire-insensitive. Then check how often the observed `c` falls inside the
model's 95% interval. If wildfire-period values fall outside more often than clean
held-out values do, smoke is moving `c`, and `c` stays out of `M`.

## What changed recently

1. **A clean control** ([control.py](control.py), new file). Wildfire coverage used
   to be compared against the nominal 95%, which conflates a real smoke effect with
   model miscalibration and with the fact that 84% of wildfire trajectories arrive
   Jun-Sep. It is now compared against coverage on held-out clean trajectories,
   month-matched to the wildfire group, so both nuisances affect the two numbers
   equally. The split is by contiguous 30-day blocks of arrival time, not at random,
   because trajectories an hour apart share 240 of their 241 hours; each holdout
   block also drops its leading 241 hours so no retained history reaches into a
   training block.
2. **Validation-based early stopping** ([training.py](../../engression_modifications/lstm/training.py)).
   `fit_lstm_engression` now takes optional `x_val`/`y_val`; when given, early
   stopping and best-checkpoint selection track held-out energy loss instead of
   training loss. Omitting them preserves the old behaviour, so no other caller
   changed. The screening carves the validation set out of the training rows with
   the same time-blocked splitter, kept separate from the control.
3. **`print_calibration`** in [main.py](main.py). Prints central coverage and
   below-quantile rates at 50/75/90/95, PIT mean and variance, a text PIT histogram
   for the SLURM log, and a PNG per candidate next to the checkpoints. It also now
   decides pass/fail per level (see next section). It only prints; it does not
   change how variables are classified.
4. **SLURM jobs** under `slurm/wildfire_sensitivity/`: `run.slurm` (single pass) and
   `run_until_stable.slurm`. Checkpoints go to a job-id-scoped directory so reruns
   cannot overwrite earlier models.

## The calibration criterion (implemented, never run)

A level passes when its coverage lands within 2.5 standard errors of the level. The
standard error is measured from the data, not chosen: group the control rows into
month-long blocks, compute coverage separately inside each block, take the spread of
those numbers divided by the square root of the block count. Blocks rather than rows
are the unit because neighbouring trajectories carry nowhere near a full row of
information. 2.5 rather than 2 because four levels are checked per candidate, which
keeps the false-failure rate near 5% overall.

Tested against fake models with known answers: a calibrated model passed 20/20 random
trials; models 10% too narrow, half as wide, and twice as wide failed every time.

**This has not been run on real data.** No run so far printed the bands. Expect
bands of roughly half a point to two points, widest at the 50% level.

## Results so far

Three completed runs, all single-GPU.

| job | mode | early stop | time | sensitive |
| --- | --- | --- | --- | --- |
| 3022866 | single pass | training loss | 1:14:37 | CO, LWP |
| 3026432 | single pass | validation | 1:12:43 | CO, LWP, PBLH |
| 3058025 | until stable (3 rounds) | validation | 1:52:30 | CO, LWP |

Logs are in `slurm/wildfire_sensitivity/`. The report at
[reports/wildfire_sensitivity/summary.tex](../../reports/wildfire_sensitivity/summary.tex)
covers **run 3022866 only** and is now out of date.

Key numbers from the earlier comparison of 3022866 vs 3026432: the correlation
between a variable's detected drop and its model's control coverage fell from -0.87
to -0.15, so the screening is no longer mostly reporting which model happened to get
sharp intervals. Correlation between the drop and expert opinion stayed near zero
(-0.05 then -0.07).

In the until-stable run, PBLH was sensitive in round 1 (drop 3.1) and insensitive in
round 2 (drop 0.8) once `M` had grown. Round 3 repeated round 2, so it stopped. This
partly answers the order-dependence question in the README: for PBLH the answer did
depend on what was already in `M`.

## Open problems, roughly in priority order

1. **`M` probably leaks smoke.** The final `M` contains SWGDN, T, CFLOW, CFMID, RH,
   PREC and LWC, which are exactly the variables a weather expert expects to be
   wildfire-sensitive. If a smoke-affected covariate is an input, the model can
   reconstruct a smoke-affected candidate during wildfire periods and no gap appears.
   Once one sensitive variable is wrongly admitted it hides the next. PBLH's round-2
   flip is this mechanism visibly operating. **Suggested test:** rerun with `M` frozen
   at the expert's original ten, never adding anything.
2. **Coverage pools all 241 timesteps.** Smoke is entrained over the continent and
   matters near arrival, so a concentrated effect is averaged against ~200 unaffected
   ocean hours. This survived both calibration regimes: SWGDN showed nothing at 98.0%
   control coverage and nothing again at 88.7%. **Suggested test:** recompute the gap
   using only the final day before arrival, reusing saved checkpoints so nothing is
   retrained.
3. **CO is probably miscalibrated.** In round 3 its control coverage was 83.7% at the
   90% level and its PIT mean was 0.42, so its intervals are too narrow and its
   predictions sit high. A model that over-predicts clean CO makes wildfire CO look
   low, which inflates the drop. CO is the study's headline result, so this matters.
   Under the criterion above it would very likely be called undetermined.
4. **The 2-point sensitivity margin is still hand-picked.** `SENSITIVITY_MARGIN` in
   main.py. The same block-spread idea used for calibration would give a data-driven
   threshold for the drop as well.
5. **The "undetermined" label is not implemented.** The plan is a three-way result:
   sensitive, insensitive, undetermined (when the counterfactual is miscalibrated).
   `print_calibration` returns a `calibrated` flag but the loop ignores it. An
   undetermined variable should stay OUT of `M`, since using a possibly
   smoke-affected covariate as an input leaks smoke into the counterfactual.

## Practical notes

- **Never train on the login node.** Use `sbatch`; account is `compute2-myu`, the only
  one this user has. See the Running Jobs section of [AGENTS.md](../../AGENTS.md).
- The venv is Python **3.9**, so `X | None` type hints fail at import. Use
  `Optional[X]`. This crashed a queued job once.
- The data is at `/storage3/fs1/myu/Active/felixhu/weather_data.mat`, not in the repo.
  Checkpoints live on storage3 too; `/home` has a small quota.
- Existing slurm files under `slurm/` other than `wildfire_sensitivity/` hardcode
  `/home/felixhu/WeatherEngression`, which is not where this checkout lives.
- LaTeX is installed per-user via TinyTeX at `~/.TinyTeX`, on PATH already. Add
  packages with `tlmgr install <pkg>`.
- Nothing is committed. `git status` shows the full set of modified and new files.

## Tests

Numpy-only test scripts for the splitting and calibration logic were written in the
session scratch directory, not the repo. They check that train/validation/control are
disjoint and time-separated, that month matching reproduces the wildfire month mix,
and that the calibration bands behave on models with known answers. Worth
reconstructing under a proper test directory if this work continues.
