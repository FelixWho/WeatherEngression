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

## Storage

Large data lives outside the repo. The main storage directory for this project is:

```text
/storage3/fs1/myu/Active
```

The real Eastern North Atlantic (ENA) weather dataset is a MATLAB v7.3 (HDF5) file at:

```text
/storage3/fs1/myu/Active/felixhu/weather_data.mat
```

Read it with `h5py` (the file is HDF5 under the hood). Each sample is an airmass back-trajectory of weather variables with a scalar cloud-condensation-nuclei (CCN) target.

## Synthetic Models

Synthetic weather types are split into one module per target law:

- `data_generation/preadditive.py`: the theory-friendly baseline $Y=g(\phi(X)+\eta)$.
- `data_generation/narx_gaussian.py`: nonlinear heteroskedastic Gaussian $Y\mid X=x$.
- `data_generation/narx_student_t.py`: nonlinear heteroskedastic Student-t target with heavy-tailed shocks.
- `data_generation/narx_garch.py`: nonlinear NARX mean with GARCH-style volatility clustering. Its saved one-step conditional law is $Y_t\mid X_t,\mathcal{F}_{t-1}$ because the variance remembers previous generated residuals.
- `data_generation/regime_mixture.py`: X-dependent mixture of Gaussian weather regimes.
- `data_generation/hurdle_lognormal.py`: zero-inflated precipitation-like target.

`data_generation/common.py` owns shared covariate simulation and lag-window utilities. `data_generation/synthetic_weather.py` should stay thin: CLI, model dispatch, and true-law reference sampling.

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
