# Engression Smoke Test Notes

This folder contains smoke tests for fitting the public `engression` package to
synthetic weather-like data with known conditional quantiles.

The main script is:

```bash
python smoke_tests/engression_synthetic_weather.py
```

It trains engression on synthetic pairs \((X_t,Y_t)\), asks the fitted model for
sample-based estimates of \(q_{0.05}(x)\), \(q_{0.50}(x)\), and \(q_{0.95}(x)\),
then compares those against the generator's saved true quantiles.

## What A Row Means

Each supervised row is a lag-window input and one target:

\[
X_t = (W_{t-L}, \ldots, W_t),
\qquad
Y_t.
\]

Nearby rows are temporally dependent because their lag windows overlap. For
example, \(X_t\) and \(X_{t+1}\) share almost all of their weather history.

## Current In-Support Split

The default split is `--split in-support`.

This is a random supervised split, not a chronological forecasting split:

1. Generate `num-samples` rows.
2. Randomly shuffle row indices.
3. Use the first `train-size` rows for training.
4. Pick held-out rows from the remaining rows.
5. Keep only held-out rows whose scalar summary \(\phi(X)\) lies inside the
   central training range.

Concretely, if

\[
a = \operatorname{quantile}_{0.02}(\phi(X_{\text{train}})),
\qquad
b = \operatorname{quantile}_{0.98}(\phi(X_{\text{train}})),
\]

then test rows must satisfy:

\[
a \le \phi(X_{\text{test}}) \le b.
\]

This split is useful for a basic conditional-distribution check:

\[
\text{Can engression learn } P(Y \mid X=x)
\text{ for held-out } x \text{ inside observed support?}
\]

It is not a time-series forecasting evaluation.

## Why Random Splitting Is Limited

Because the data are time series, random row splitting can leak temporal
structure. Training might contain \(X_t\) while testing contains \(X_{t+1}\),
and those two inputs share most of the same lag window.

That means random splitting can make results look better than a real forecasting
setup. A forecasting split should be chronological:

\[
\text{train} = \{(X_t,Y_t): t \le T\},
\]

\[
\text{test} = \{(X_t,Y_t): t > T\}.
\]

An even cleaner version should include a gap of length at least \(L\), the
lookback window, so the first test windows do not reuse weather states from the
training period.

## Current Extrapolation Split

The script also supports:

```bash
--split right-extrapolation
--split left-extrapolation
--split two-sided-extrapolation
```

These splits are based on the scalar summary \(\phi(X)\), not on chronological
time.

For `right-extrapolation`, rows are sorted by \(\phi(X)\). Training uses the
lowest `train-size` rows, and testing uses the next `test-size` rows to the
right:

\[
\underbrace{\text{train}}_{\text{lower } \phi(X)}
\quad
\underbrace{\text{test}}_{\text{higher } \phi(X)}.
\]

For `left-extrapolation`, training uses the highest \(\phi(X)\) rows and testing
uses rows immediately to the left.

For `two-sided-extrapolation`, training uses a middle block of \(\phi(X)\), and
testing uses both tails.

These modes test covariate-support extrapolation in \(\phi(X)\)-space:

\[
\phi(X_{\text{test}}) \notin
[\min \phi(X_{\text{train}}), \max \phi(X_{\text{train}})].
\]

They do not prove full high-dimensional out-of-support extrapolation in the
flattened lag-window \(X\).

## How The Charts Are Ordered

For `in-support`, the test rows are sorted by original sample index before
plotting. This is chronological among the selected held-out rows, but it is not
continuous time because the rows were randomly selected first.

For extrapolation splits, the x-axis is \(\phi(X)\), not time. The gray shaded
region shows the training \(\phi(X)\) support, and the plotted test points lie
outside that region for extrapolation modes.

## Pre-Additive Versus Post-Additive Targets

Only the `preadditive` generator follows the engression paper's pre-additive
structure:

\[
Y = g(\phi(X) + \eta).
\]

The NARX generators are post-additive or output-noise models. For example:

\[
Y \mid X=x = \mu(x) + \sigma(x)Z,
\qquad
Z \sim \mathcal{N}(0,1).
\]

So the NARX models are useful robustness tests for distributional regression,
but they are not direct demonstrations of the paper's pre-additive
extrapolation theory.

## Stronger Out-Of-Support Checks

The current extrapolation modes only ensure test rows are outside the training
range of \(\phi(X)\). Stronger checks could include:

- chronological train/test splits with a lag-window gap;
- marginal feature checks, where at least one feature summary lies outside the
  training range;
- distance-based empirical support checks, such as nearest-neighbor distance in
  standardized flattened \(X\);
- synthetic regime construction, where train and test data are generated from
  deliberately separated weather regimes.

The cleanest future setup is probably:

1. Use chronological splitting for forecasting realism.
2. Report whether test \(X_t\) is outside training support according to
   \(\phi(X)\), marginal ranges, and/or kNN distance.
3. Keep the current \(\phi(X)\)-based split for controlled covariate
   extrapolation experiments.
