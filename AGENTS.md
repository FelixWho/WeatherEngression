# Agent Context

This repo is dedicated to a research project on engression for weather-like time-series prediction.

## Source Papers

The relevant papers are in `resources/`:

- `resources/engression.pdf`: introduces engression, a neural distributional regression method trained with energy loss. Important ideas: conditional sampler $g_\theta(X,\varepsilon)$, energy score, pre-additive noise, extrapolation under monotonicity/regularity assumptions.
- `resources/weather.pdf`: proposes a Lagrangian time-series ML framework for predicting cloud condensation nuclei from airmass trajectory histories. Important ideas: each sample is a multivariate time series along an airmass trajectory; the target is aerosol concentration at arrival.

## Current Research Framing

The project should be framed as:

> Probabilistic Lagrangian / weather-like forecasting with engression-style distributional regression.

Avoid overstating the extrapolation theorem. The safest claim is that engression can learn calibrated predictive conditional distributions $P(Y \mid X=x)$. Strong extrapolation claims require the synthetic or real data-generating process to satisfy additional pre-additive and monotonicity assumptions.

## Mental Model To Preserve

In pre-additive noise,

$$
Y=g(X+\eta).
$$

The observed $x$ is not itself corrupted. Instead, $x$ is the anchor, $\eta$ is a latent horizontal perturbation, and applying $g$ turns horizontal uncertainty into vertical conditional spread:

$$
x+\eta \mapsto g(x+\eta) \sim Y \mid X=x.
$$

Engression uses artificial base noise $\varepsilon$, then learns or biases toward a transformation

$$
\eta=h_\theta(\varepsilon).
$$

## Repo Organization

- `data_generation/`: synthetic weather-like time-series simulators with known conditional laws.
- `engression_model/`: model/loss code for engression-style conditional generators.
- `resources/`: PDFs only; do not edit or overwrite paper PDFs.

## Synthetic Models

`data_generation/synthetic_weather.py` currently supports:

- `preadditive`: the theory-friendly baseline $Y=g(\phi(X)+\eta)$.
- `narx_gaussian`: nonlinear heteroskedastic Gaussian $Y\mid X=x$.
- `regime_mixture`: X-dependent mixture of Gaussian weather regimes.
- `hurdle_lognormal`: zero-inflated precipitation-like target.

Use the non-preadditive models to test robustness when the engression paper's structural assumptions are violated but the true conditional law is still known.

## Implementation Principles

- Keep synthetic data generators explicit enough that the true conditional distribution is available analytically or by a known sampler.
- Prefer simple, inspectable data-generating processes before adding realism.
- Keep model code separate from data generation.
- Use LaTeX math in documentation.
- For validation, prioritize calibration and distributional metrics over only RMSE:
  - quantile calibration;
  - prediction interval coverage;
  - CRPS / energy score;
  - tail-bin performance;
  - generated sample diagnostics.
