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

The generator in `synthetic_weather.py` shares one weather-like covariate process across several target models. The covariates have autocorrelation, cross-variable interactions, daily cycles, seasonal cycles, intermittent precipitation, and bounded/positive transformed variables.

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

Generate one dataset:

```bash
python data_generation/synthetic_weather.py --model narx_gaussian
```

Available models:

```text
preadditive
narx_gaussian
regime_mixture
hurdle_lognormal
```

Each output `.npz` contains `X`, `y`, `phi`, `feature_names`, and reference quantiles `q05`, `q50`, and `q95`. Some models include additional parameters such as `mu`, `sigma`, `mixture_weights`, or `p_wet` so the full conditional law can be reconstructed.

For distributional validation, `synthetic_weather.py` also exposes:

```python
reference_conditional_samples(dataset, n_samples=256, seed=0)
```

This returns known samples from the true conditional law for each row in a generated dataset.
