# Data Generation

This folder contains synthetic weather-like time-series generators.

The immediate goal is to create supervised pairs

$$
(X_t,Y_t),
$$

where $X_t$ is a lag window of weather-like covariates and the true conditional distribution

$$
P(Y_t \mid X_t=x)
$$

is known by construction.

For `narx_garch`, the known law is the one-step forecasting law conditioned on the simulated volatility state, so it is more precisely $P(Y_t \mid X_t,\mathcal{F}_{t-1})$.

The generators share one weather-like covariate process across several target models. The covariates have autocorrelation, cross-variable interactions, daily cycles, seasonal cycles, intermittent precipitation, and bounded/positive transformed variables.

## File Layout

- `common.py`: shared configuration, covariate simulation, lag-window construction, and helper math.
- `preadditive.py`: theory-friendly pre-additive target.
- `narx_gaussian.py`: nonlinear heteroskedastic NARX-style target.
- `narx_student_t.py`: NARX-style target with heavy-tailed Student-t noise.
- `narx_garch.py`: NARX-style target with volatility clustering.
- `regime_mixture.py`: X-dependent Gaussian regime mixture.
- `hurdle_lognormal.py`: zero-inflated precipitation-like target.
- `synthetic_weather.py`: CLI, model dispatcher, and `reference_conditional_samples(...)`.

## Target Models

### `preadditive`

This is the theory-friendly baseline:

1. nonlinear autocorrelated covariates with daily and seasonal cycles;
2. lag windows $X_t=(W_{t-L},\ldots,W_t)$;
3. a pre-additive target

   $$
   Y_t=g(\phi(X_t)+\eta_t).
   $$

When $g$ is monotone, quantiles are available as

$$
Q_\alpha(Y_t \mid X_t=x)
=
g(\phi(x)+Q_\alpha(\eta)).
$$

This is useful as a sanity check, but it is intentionally favorable to the engression paper's extrapolation assumptions.

### `narx_gaussian`

This is a nonlinear heteroskedastic NARX-style target:

$$
Y_t \mid X_t=x \sim \mathcal{N}(\mu(x),\sigma^2(x)).
$$

The mean $\mu(x)$ depends on lagged radiation, precipitation, humidity, pressure tendency, wind, and a nonlinear latent weather index. The scale $\sigma(x)$ also depends on weather conditions. This is a realistic vertical-noise conditional law, not a pre-additive noise model.

Reference quantiles are analytic:

$$
Q_\alpha(Y_t\mid X_t=x)=\mu(x)+\sigma(x)\Phi^{-1}(\alpha).
$$

### `narx_student_t`

This keeps the same nonlinear NARX mean structure, but replaces Gaussian noise with a heavy-tailed Student-t innovation:

$$
Y_t \mid X_t=x
=
\mu(x)+s(x)T_\nu,
\qquad
\nu=5.
$$

The scale $s(x)$ changes with weather conditions, while the fixed $\nu=5$ gives occasional large shocks. The 5%, 50%, and 95% reference quantiles use precomputed $t_5$ quantiles, so no SciPy dependency is needed.

### `narx_garch`

This keeps the nonlinear NARX mean, but lets the conditional variance cluster over time:

$$
Y_t \mid X_t,\mathcal{F}_{t-1}
\sim
\mathcal{N}(\mu(X_t),\sigma_t^2),
$$

with

$$
\sigma_t^2
=
\omega
+ \alpha e_{t-1}^2
+ \beta\sigma_{t-1}^2
+ \gamma s^2(X_t).
$$

Here $s(X_t)$ is the weather-driven base scale and $e_{t-1}$ is the previous generated residual. This model is useful because the variance is not just a memoryless function of $X_t$; it also remembers recent shocks. The saved `sigma` field is the realized one-step conditional scale, so reference conditional samples are still known for each generated row.

### `regime_mixture`

This target mimics clear/photochemical, wet-removal, and transported/polluted regimes:

$$
P(Y_t\mid X_t=x)
=
\sum_{r=1}^3 \pi_r(x)\,
\mathcal{N}(\mu_r(x),\sigma_r^2(x)).
$$

The regime is sampled but not given to the model as a separate label. The exact mixture weights, means, and scales are saved in the dataset, and reference quantiles are computed numerically from the known mixture CDF. This tests whether engression can represent multimodal or skewed conditional distributions.

### `hurdle_lognormal`

This is a precipitation-like zero-inflated target:

$$
Y_t =
\begin{cases}
0, & \text{with probability } 1-p_{\mathrm{wet}}(x),\\
\mathrm{LogNormal}(m(x),s^2(x)), & \text{with probability } p_{\mathrm{wet}}(x).
\end{cases}
$$

It has a point mass at zero and a positive heavy-tailed component. Reference quantiles are analytic via the hurdle probability and lognormal quantile.

## Usage

Print one generated dataset to stdout as JSON Lines:

```bash
python data_generation/synthetic_weather.py --model narx_gaussian
```

Save generated observations to an `.npz` file:

```bash
python data_generation/synthetic_weather.py --model narx_gaussian --out data/synthetic_narx_gaussian_weather.npz
```

Available models:

```text
preadditive
narx_gaussian
narx_student_t
narx_garch
regime_mixture
hurdle_lognormal
```

The stdout format starts with one metadata JSON object, then prints one generated observation per line. When saving to `.npz`, each output file contains `X`, `y`, `phi`, `feature_names`, and reference quantiles `q05`, `q50`, and `q95`. Some models include additional parameters such as `mu`, `sigma`, `scale`, `base_sigma`, `mixture_weights`, or `p_wet` so the full conditional law can be reconstructed.

For distributional validation, `synthetic_weather.py` also exposes:

```python
reference_conditional_samples(dataset, n_samples=256, seed=0)
```

This returns known samples from the true conditional law for each row in a generated dataset.
