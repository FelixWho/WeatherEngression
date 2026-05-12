# WeatherEngression

WeatherEngression is a research sandbox for applying **engression** to weather-like time-series forecasting. The project connects two papers in `resources/`:

- `resources/engression.pdf`: engression as neural distributional regression, with extrapolation theory under pre-additive noise assumptions.
- `resources/weather.pdf`: Lagrangian time-series machine learning for aerosol / cloud condensation nuclei prediction from airmass history.

The working research question is:

> Can an engression-style model learn calibrated conditional distributions `P(Y | X=x)` for weather-like trajectory histories, and can synthetic data with known `P(Y | X=x)` validate whether the learned distribution matches the data-generating process?

## Project Structure

```text
WeatherEngression/
├── resources/
│   ├── engression.pdf
│   └── weather.pdf
├── data_generation/
│   ├── README.md
│   └── synthetic_weather.py
├── engression_model/
│   ├── README.md
│   ├── energy.py
│   └── networks.py
├── AGENTS.md
└── README.md
```

## Quickstart

Install the minimal Python dependencies:

```bash
pip install -r requirements.txt
```

Generate a small synthetic weather dataset:

```bash
python data_generation/synthetic_weather.py --model narx_gaussian
```

Available models are `preadditive`, `narx_gaussian`, `regime_mixture`, and `hurdle_lognormal`. The generated file contains lag windows `X`, targets `y`, the latent index `phi`, and reference quantiles `q05`, `q50`, and `q95`.

For validation against the full known law, use `reference_conditional_samples(...)` from `data_generation/synthetic_weather.py`.

## Core Intuition

Standard regression learns a point summary such as:

```math
\mathbb{E}[Y \mid X=x].
```

Engression learns a conditional sampler:

```math
\widehat{Y}=g_\theta(X,\varepsilon),
```

where `epsilon` is artificial noise, usually sampled from a simple distribution. For fixed `x`, repeated samples:

```math
g_\theta(x,\varepsilon_1),\ldots,g_\theta(x,\varepsilon_m)
```

are interpreted as draws from:

```math
\widehat{P}(Y \mid X=x).
```

The training loss is based on the energy score:

```math
\mathcal{L}(\theta)
=
\mathbb{E}\left[
\|Y-g_\theta(X,\varepsilon)\|
-
\frac{1}{2}
\|g_\theta(X,\varepsilon)-g_\theta(X,\varepsilon')\|
\right].
```

The first term pulls generated samples toward observations. The second term rewards conditional spread and discourages collapse to a point predictor.

## Why Synthetic Data First?

For real weather data, the true conditional distribution `P(Y | X=x)` is unknown. Synthetic data lets us define the truth exactly.

The target simulators use weather-like covariates `W_t`, lag windows:

```math
X_t=(W_{t-L},\ldots,W_t),
```

and several known conditional laws. The theory-friendly baseline is a pre-additive target:

```math
Y_t = g(\phi(X_t)+\eta_t).
```

If `g` is monotone and `eta_t` has known distribution, then the true conditional quantiles are known:

```math
Q_\alpha(Y_t \mid X_t=x)
=
g\!\left(\phi(x)+Q_\alpha(\eta)\right).
```

This makes it possible to test whether engression recovers calibrated quantiles, prediction intervals, and full conditional samples.

The less friendly models are intentionally not pre-ANM:

- `narx_gaussian`: nonlinear heteroskedastic Gaussian `Y | X=x`.
- `regime_mixture`: X-dependent mixture of Gaussian weather regimes.
- `hurdle_lognormal`: zero-inflated precipitation-like target with a lognormal positive tail.

These are designed to test whether engression learns useful conditional distributions even when the extrapolation assumptions from the engression paper are not literally true.

## Important Assumptions

The engression paper has two distinct layers:

1. **Distributional regression:** energy loss can identify `P(Y | X=x)` in the correctly specified population setting.
2. **Extrapolation theory:** stronger claims require structural assumptions, especially a pre-additive form.

For example:

```math
Y=g(X+\eta).
```

These stronger claims also require assumptions such as monotonicity and regularity of `g`.

The practical neural model in this repo should therefore be treated as:

> A probabilistic conditional distribution model first.

and only cautiously as:

> A theorem-backed extrapolation model.

## Initial Validation Plan

1. Generate synthetic weather-like lag-window data with known `P(Y | X=x)`.
2. Train an engression model using the energy loss.
3. Compare generated conditional samples against the known data-generating distribution.
4. Evaluate CRPS / energy score, quantile calibration, interval coverage, and tail behavior.
5. Stress-test under distribution shift by holding out ranges of the synthetic covariate support.
