"""Synthetic weather-like time series with known conditional distributions.

The generators share the same autocorrelated, seasonal weather covariates W_t
and lag-window input X_t = (W_{t-L}, ..., W_t). They differ in the target law
P(Y_t | X_t=x).

Available target models:
    preadditive:
        Friendly to engression theory: Y = g(phi(X) + eta).
    narx_gaussian:
        Nonlinear lagged mean with heteroskedastic Gaussian noise.
    regime_mixture:
        Weather-regime mixture of Gaussians with X-dependent mixture weights.
    hurdle_lognormal:
        Zero-inflated precipitation-like target with a lognormal positive tail.

Each dataset includes reference quantiles q05, q50, and q95 for validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np


FEATURE_NAMES = (
    "temperature",
    "humidity",
    "pressure",
    "radiation",
    "precipitation",
    "wind_speed",
)

MODEL_NAMES = (
    "preadditive",
    "narx_gaussian",
    "regime_mixture",
    "hurdle_lognormal",
)

REFERENCE_QUANTILES = (0.05, 0.50, 0.95)


@dataclass(frozen=True)
class SyntheticWeatherConfig:
    n_steps: int = 10_000
    window: int = 24
    seed: int = 7
    noise_scale: float = 0.35
    post_noise_scale: float = 0.0
    seasonal_period: int = 24 * 365


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def normal_ppf(alpha: float) -> float:
    return NormalDist().inv_cdf(alpha)


def normal_cdf_array(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def monotone_response(z: np.ndarray) -> np.ndarray:
    """A nonlinear strictly increasing response function."""
    return z + 0.15 * z**3


def generate_covariates(config: SyntheticWeatherConfig) -> np.ndarray:
    """Generate weather-like covariates with autocorrelation and cycles.

    Returns:
        Array with shape ``(config.n_steps, 6)`` in the order of
        ``FEATURE_NAMES``.
    """
    rng = np.random.default_rng(config.seed)
    d = len(FEATURE_NAMES)
    state = np.zeros((config.n_steps, d), dtype=np.float64)

    ar = np.array(
        [
            [0.88, 0.05, 0.02, 0.00, 0.00, 0.03],
            [0.04, 0.82, -0.04, 0.04, 0.08, 0.00],
            [0.02, -0.03, 0.90, 0.00, -0.06, -0.02],
            [0.03, -0.08, 0.00, 0.70, -0.05, 0.00],
            [0.00, 0.10, -0.05, -0.02, 0.65, 0.04],
            [0.03, 0.00, -0.08, 0.00, 0.03, 0.80],
        ],
        dtype=np.float64,
    )
    nonlinear = np.array([0.08, -0.04, 0.03, 0.05, 0.06, -0.03])

    for t in range(1, config.n_steps):
        daily = np.sin(2 * np.pi * t / 24.0)
        daily_daylight = max(0.0, daily)
        seasonal = np.sin(2 * np.pi * t / config.seasonal_period)
        forcing = np.array(
            [
                0.9 * seasonal + 0.25 * daily,
                -0.25 * seasonal - 0.15 * daily_daylight,
                -0.2 * seasonal,
                1.3 * daily_daylight * (1.0 + 0.25 * seasonal),
                0.15 * seasonal,
                0.25 * np.cos(2 * np.pi * t / (24.0 * 5.0)),
            ],
            dtype=np.float64,
        )
        innovation = rng.normal(0.0, [0.25, 0.20, 0.15, 0.20, 0.25, 0.20])
        state[t] = ar @ state[t - 1] + nonlinear * np.tanh(state[t - 1]) + forcing + innovation

    covariates = state.copy()

    # Make selected variables weather-like while keeping the array numeric and
    # standardized enough for modeling.
    covariates[:, 1] = sigmoid(covariates[:, 1])  # humidity in (0, 1)
    covariates[:, 3] = np.maximum(0.0, covariates[:, 3])  # radiation

    precip_prob = sigmoid(1.6 * state[:, 1] - 0.7 * state[:, 2] - 0.3 * state[:, 3])
    precip_amount = rng.gamma(shape=1.4, scale=0.7, size=config.n_steps)
    precip_event = rng.binomial(1, precip_prob)
    covariates[:, 4] = precip_event * precip_amount

    covariates[:, 5] = np.log1p(np.exp(covariates[:, 5]))  # positive wind speed

    return covariates


def make_lag_windows(covariates: np.ndarray, window: int) -> np.ndarray:
    """Create lag windows with shape ``(n-window, window+1, n_features)``."""
    return np.stack([covariates[i - window : i + 1] for i in range(window, len(covariates))])


def phi_from_windows(x: np.ndarray) -> np.ndarray:
    """Map each lag window to a scalar latent weather index."""
    n_lags = x.shape[1]
    lag_age = np.arange(n_lags - 1, -1, -1, dtype=np.float64)
    lag_weights = np.exp(-lag_age / 8.0)
    lag_weights = lag_weights / lag_weights.sum()

    feature_weights = np.array([0.45, -0.35, 0.20, 0.50, -0.65, 0.25], dtype=np.float64)
    return np.einsum("nlf,l,f->n", x, lag_weights, feature_weights)


def lagged_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Summaries used by the target models.

    The covariate window has shape ``(n, n_lags, n_features)``. The returned
    summaries mimic physically meaningful quantities such as recent conditions,
    exponentially weighted history, and short-term changes.
    """
    n_lags = x.shape[1]
    current = x[:, -1, :]
    lag_6h = x[:, max(0, n_lags - 7), :]
    lag_24h = x[:, max(0, n_lags - 25), :]

    lag_age = np.arange(n_lags - 1, -1, -1, dtype=np.float64)
    recent_weights = np.exp(-lag_age / 6.0)
    recent_weights = recent_weights / recent_weights.sum()
    long_weights = np.exp(-lag_age / 24.0)
    long_weights = long_weights / long_weights.sum()

    recent = np.einsum("nlf,l->nf", x, recent_weights)
    long = np.einsum("nlf,l->nf", x, long_weights)
    tendency = current - lag_6h
    daily_change = current - lag_24h

    return {
        "current": current,
        "lag_6h": lag_6h,
        "lag_24h": lag_24h,
        "recent": recent,
        "long": long,
        "tendency": tendency,
        "daily_change": daily_change,
    }


def preadditive_quantile(phi: np.ndarray, alpha: float, noise_scale: float) -> np.ndarray:
    """Analytic quantile for Y = g(phi(X) + eta), eta ~ Normal(0, noise_scale)."""
    normal_quantile = normal_ppf(alpha)
    return monotone_response(phi + noise_scale * normal_quantile)


def preadditive_conditional_samples(
    phi: np.ndarray,
    n_samples: int,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    """Known conditional sampler with shape ``(n_observations, n_samples)``."""
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, noise_scale, size=(len(phi), n_samples))
    return monotone_response(phi[:, None] + eta)


def narx_gaussian_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear lagged mean and X-dependent scale.

    This is not a pre-ANM target. It is a conventional conditional distribution
    with vertical, heteroskedastic noise:

        Y | X=x ~ Normal(mu(x), sigma(x)^2).
    """
    f = lagged_features(x)
    phi = phi_from_windows(x)
    current = f["current"]
    recent = f["recent"]
    long = f["long"]
    tendency = f["tendency"]

    temp, humidity, pressure, radiation, precip, wind = range(len(FEATURE_NAMES))

    mu = (
        0.55 * np.sin(phi)
        + 0.18 * phi**2
        + 0.35 * np.tanh(recent[:, radiation])
        - 0.45 * np.sqrt(np.maximum(recent[:, precip], 0.0))
        + 0.25 * current[:, humidity] * np.log1p(current[:, wind])
        - 0.18 * tendency[:, pressure]
        + 0.12 * np.sin(long[:, temp])
    )
    sigma = (
        0.12
        + 0.50
        * sigmoid(
            -0.4
            + 0.9 * recent[:, precip]
            + 0.8 * np.abs(tendency[:, pressure])
            - 0.5 * current[:, radiation]
            + 0.35 * current[:, wind]
        )
    )
    return mu, sigma


def regime_mixture_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X-dependent three-regime Gaussian mixture.

    The regimes are clear/photochemical, wet-removal, and transported/polluted.
    The regime itself is not observed by the model; instead, the exact
    conditional law is the mixture with weights p_r(x).
    """
    f = lagged_features(x)
    phi = phi_from_windows(x)
    current = f["current"]
    recent = f["recent"]
    long = f["long"]
    tendency = f["tendency"]

    temp, humidity, pressure, radiation, precip, wind = range(len(FEATURE_NAMES))

    logits = np.column_stack(
        [
            0.8 * recent[:, radiation] - 0.8 * recent[:, precip] - 0.3 * current[:, humidity],
            1.2 * recent[:, precip] + 0.6 * current[:, humidity] - 0.4 * recent[:, radiation],
            0.7 * current[:, wind] + 0.5 * np.abs(tendency[:, pressure]) + 0.2 * long[:, temp],
        ]
    )
    weights = softmax(logits)

    base = 0.35 * np.sin(phi) + 0.15 * phi
    means = np.column_stack(
        [
            base + 0.75 + 0.25 * np.tanh(recent[:, radiation]),
            base - 0.85 - 0.35 * np.sqrt(np.maximum(recent[:, precip], 0.0)),
            base + 0.15 + 0.45 * np.log1p(current[:, wind]),
        ]
    )
    sigmas = np.column_stack(
        [
            0.18 + 0.10 * sigmoid(current[:, radiation]),
            0.28 + 0.20 * sigmoid(recent[:, precip]),
            0.35 + 0.15 * sigmoid(np.abs(tendency[:, pressure])),
        ]
    )
    return weights, means, sigmas


def mixture_cdf(y: np.ndarray, weights: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    z = (y[:, None] - means) / sigmas
    return (weights * normal_cdf_array(z)).sum(axis=1)


def mixture_quantile(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
    n_iter: int = 80,
) -> np.ndarray:
    lower = means.min(axis=1) - 8.0 * sigmas.max(axis=1)
    upper = means.max(axis=1) + 8.0 * sigmas.max(axis=1)
    for _ in range(n_iter):
        mid = 0.5 * (lower + upper)
        cdf = mixture_cdf(mid, weights, means, sigmas)
        lower = np.where(cdf < alpha, mid, lower)
        upper = np.where(cdf >= alpha, mid, upper)
    return 0.5 * (lower + upper)


def hurdle_lognormal_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-inflated precipitation-like conditional law.

    With probability 1 - p_wet(x), Y = 0. Otherwise,

        log(Y) | wet, X=x ~ Normal(log_mu(x), log_sigma(x)^2).
    """
    f = lagged_features(x)
    current = f["current"]
    recent = f["recent"]
    tendency = f["tendency"]

    humidity, pressure, radiation, precip, wind = 1, 2, 3, 4, 5

    p_wet = sigmoid(
        -1.0
        + 2.4 * current[:, humidity]
        + 0.9 * recent[:, precip]
        - 0.7 * current[:, radiation]
        - 0.5 * tendency[:, pressure]
    )
    log_mu = (
        -0.35
        + 0.9 * current[:, humidity]
        + 0.45 * np.log1p(recent[:, precip])
        + 0.25 * current[:, wind]
        - 0.25 * current[:, radiation]
    )
    log_sigma = 0.25 + 0.45 * sigmoid(0.8 * recent[:, precip] + 0.5 * current[:, wind])
    return p_wet, log_mu, log_sigma


def hurdle_lognormal_quantile(
    p_wet: np.ndarray,
    log_mu: np.ndarray,
    log_sigma: np.ndarray,
    alpha: float,
) -> np.ndarray:
    dry_mass = 1.0 - p_wet
    q = np.zeros_like(p_wet)
    wet_mask = alpha > dry_mass
    if np.any(wet_mask):
        tail_alpha = (alpha - dry_mass[wet_mask]) / p_wet[wet_mask]
        tail_alpha = np.clip(tail_alpha, 1e-8, 1.0 - 1e-8)
        tail_z = np.array([normal_ppf(float(a)) for a in tail_alpha], dtype=np.float64)
        q[wet_mask] = np.exp(log_mu[wet_mask] + log_sigma[wet_mask] * tail_z)
    return q


def base_dataset(config: SyntheticWeatherConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariates = generate_covariates(config)
    x = make_lag_windows(covariates, config.window)
    phi = phi_from_windows(x)
    return covariates, x, phi


def with_common_fields(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    extras: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {
        "X": x.astype(np.float32),
        "y": y.astype(np.float32),
        "phi": phi.astype(np.float32),
        "model": np.array(model),
        "feature_names": np.array(FEATURE_NAMES),
    }
    result.update(extras)
    return result


def generate_preadditive_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate a supervised dataset with known pre-additive conditional law."""
    rng = np.random.default_rng(config.seed + 1)
    _, x, phi = base_dataset(config)

    eta = rng.normal(0.0, config.noise_scale, size=len(phi))
    y = monotone_response(phi + eta)
    if config.post_noise_scale > 0:
        y = y + rng.normal(0.0, config.post_noise_scale, size=len(phi))

    extras = {
        f"q{int(alpha * 100):02d}": preadditive_quantile(phi, alpha, config.noise_scale).astype(np.float32)
        for alpha in REFERENCE_QUANTILES
    }
    extras["noise_scale"] = np.array(config.noise_scale, dtype=np.float32)
    return with_common_fields("preadditive", x, y, phi, extras)


def generate_narx_gaussian_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate a heteroskedastic nonlinear NARX-style conditional Gaussian."""
    rng = np.random.default_rng(config.seed + 2)
    _, x, phi = base_dataset(config)
    mu, sigma = narx_gaussian_parameters(x)
    y = rng.normal(mu, sigma)
    extras = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }
    for alpha in REFERENCE_QUANTILES:
        z = normal_ppf(alpha)
        extras[f"q{int(alpha * 100):02d}"] = (mu + sigma * z).astype(np.float32)
    return with_common_fields("narx_gaussian", x, y, phi, extras)


def generate_regime_mixture_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate an X-dependent Gaussian mixture target."""
    rng = np.random.default_rng(config.seed + 3)
    _, x, phi = base_dataset(config)
    weights, means, sigmas = regime_mixture_parameters(x)
    regimes = np.array([rng.choice(3, p=w) for w in weights], dtype=np.int64)
    y = rng.normal(means[np.arange(len(means)), regimes], sigmas[np.arange(len(sigmas)), regimes])
    extras = {
        "mixture_weights": weights.astype(np.float32),
        "mixture_means": means.astype(np.float32),
        "mixture_sigmas": sigmas.astype(np.float32),
        "sampled_regime": regimes.astype(np.int64),
    }
    for alpha in REFERENCE_QUANTILES:
        extras[f"q{int(alpha * 100):02d}"] = mixture_quantile(weights, means, sigmas, alpha).astype(np.float32)
    return with_common_fields("regime_mixture", x, y, phi, extras)


def generate_hurdle_lognormal_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate a zero-inflated precipitation-like target."""
    rng = np.random.default_rng(config.seed + 4)
    _, x, phi = base_dataset(config)
    p_wet, log_mu, log_sigma = hurdle_lognormal_parameters(x)
    wet = rng.binomial(1, p_wet).astype(bool)
    y = np.zeros_like(p_wet)
    y[wet] = rng.lognormal(mean=log_mu[wet], sigma=log_sigma[wet])
    extras = {
        "p_wet": p_wet.astype(np.float32),
        "log_mu": log_mu.astype(np.float32),
        "log_sigma": log_sigma.astype(np.float32),
        "wet_event": wet.astype(np.int8),
    }
    for alpha in REFERENCE_QUANTILES:
        extras[f"q{int(alpha * 100):02d}"] = hurdle_lognormal_quantile(
            p_wet,
            log_mu,
            log_sigma,
            alpha,
        ).astype(np.float32)
    return with_common_fields("hurdle_lognormal", x, y, phi, extras)


def generate_dataset(model: str, config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    generators = {
        "preadditive": generate_preadditive_dataset,
        "narx_gaussian": generate_narx_gaussian_dataset,
        "regime_mixture": generate_regime_mixture_dataset,
        "hurdle_lognormal": generate_hurdle_lognormal_dataset,
    }
    return generators[model](config)


def reference_conditional_samples(
    dataset: dict[str, np.ndarray],
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample from the known conditional law saved in a generated dataset.

    Args:
        dataset: A dictionary returned by one of the ``generate_*_dataset``
            functions, or a dict-like object from ``np.load``.
        n_samples: Number of conditional samples per observation.
        seed: Random seed for reproducibility.

    Returns:
        Array with shape ``(n_observations, n_samples)``.
    """
    rng = np.random.default_rng(seed)
    model = str(np.asarray(dataset["model"]))
    n = len(dataset["y"])

    if model == "preadditive":
        phi = np.asarray(dataset["phi"], dtype=np.float64)
        noise_scale = float(np.asarray(dataset["noise_scale"]))
        eta = rng.normal(0.0, noise_scale, size=(n, n_samples))
        return monotone_response(phi[:, None] + eta)

    if model == "narx_gaussian":
        mu = np.asarray(dataset["mu"], dtype=np.float64)
        sigma = np.asarray(dataset["sigma"], dtype=np.float64)
        return rng.normal(mu[:, None], sigma[:, None], size=(n, n_samples))

    if model == "regime_mixture":
        weights = np.asarray(dataset["mixture_weights"], dtype=np.float64)
        weights = weights / weights.sum(axis=1, keepdims=True)
        means = np.asarray(dataset["mixture_means"], dtype=np.float64)
        sigmas = np.asarray(dataset["mixture_sigmas"], dtype=np.float64)
        draws = np.empty((n, n_samples), dtype=np.float64)
        for i in range(n):
            regimes = rng.choice(weights.shape[1], size=n_samples, p=weights[i])
            draws[i] = rng.normal(means[i, regimes], sigmas[i, regimes])
        return draws

    if model == "hurdle_lognormal":
        p_wet = np.asarray(dataset["p_wet"], dtype=np.float64)
        log_mu = np.asarray(dataset["log_mu"], dtype=np.float64)
        log_sigma = np.asarray(dataset["log_sigma"], dtype=np.float64)
        wet = rng.binomial(1, p_wet[:, None], size=(n, n_samples)).astype(bool)
        draws = np.zeros((n, n_samples), dtype=np.float64)
        draws[wet] = rng.lognormal(
            mean=np.repeat(log_mu, n_samples).reshape(n, n_samples)[wet],
            sigma=np.repeat(log_sigma, n_samples).reshape(n, n_samples)[wet],
        )
        return draws

    raise ValueError(f"unknown model in dataset: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="narx_gaussian")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    args = parser.parse_args()

    config = SyntheticWeatherConfig(
        n_steps=args.n_steps,
        window=args.window,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    out = args.out or Path(f"data/synthetic_{args.model}_weather.npz")
    dataset = generate_dataset(args.model, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dataset)
    print(
        f"wrote {out} with model={args.model}, "
        f"X={dataset['X'].shape}, y={dataset['y'].shape}"
    )


if __name__ == "__main__":
    main()
