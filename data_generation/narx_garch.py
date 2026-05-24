"""Non-preadditive NARX-style target with GARCH conditional volatility."""

from __future__ import annotations

import numpy as np

from .common import (
    REFERENCE_QUANTILES,
    SyntheticWeatherConfig,
    normal_ppf,
    weather_history,
    with_common_fields,
)
from .narx_gaussian import narx_gaussian_parameters


GARCH_OMEGA = 0.018
GARCH_ALPHA = 0.18
GARCH_BETA = 0.68
GARCH_WEATHER_WEIGHT = 0.28


def simulate_garch_volatility(
    base_sigma: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GARCH-style residuals and their conditional scales.

    ``base_sigma`` is the weather-driven scale coming from the NARX Gaussian
    formula. The recursion adds volatility clustering:

    ``variance[t] = omega + alpha * residual[t-1]^2 + beta * variance[t-1]
    + weather_weight * base_sigma[t]^2``.

    Therefore the one-step conditional law is known after the previous
    residual has been simulated, but the scale is no longer a pure function of
    the current lag window ``X_t``. It depends on the generated history state.
    """

    n = len(base_sigma)
    residual = np.empty(n, dtype=np.float64)
    variance = np.empty(n, dtype=np.float64)

    long_run_variance = (
        GARCH_OMEGA + GARCH_WEATHER_WEIGHT * float(np.mean(base_sigma**2))
    ) / max(1.0 - GARCH_ALPHA - GARCH_BETA, 1e-6)
    variance[0] = max(long_run_variance, 1e-6)
    residual[0] = np.sqrt(variance[0]) * rng.normal()

    for t in range(1, n):
        variance[t] = (
            GARCH_OMEGA
            + GARCH_ALPHA * residual[t - 1] ** 2
            + GARCH_BETA * variance[t - 1]
            + GARCH_WEATHER_WEIGHT * base_sigma[t] ** 2
        )
        variance[t] = max(variance[t], 1e-6)
        residual[t] = np.sqrt(variance[t]) * rng.normal()

    sigma = np.sqrt(variance)
    return sigma, residual


def generate_narx_garch_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate samples from a NARX mean with GARCH volatility clustering.

    Conditional on the simulated history state, the target has the known law:

    ``Y_t | X_t, F_{t-1} ~ Normal(mu(X_t), sigma_t^2)``.

    This is deliberately less friendly than the static NARX Gaussian model
    because the conditional variance remembers past shocks. It is useful for
    asking whether a distributional model handles clustered uncertainty.
    """

    rng = np.random.default_rng(config.seed + 6)
    x, phi = weather_history(config)
    mu, base_sigma = narx_gaussian_parameters(x)
    sigma, residual = simulate_garch_volatility(base_sigma, rng)
    y = mu + residual
    extras = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "base_sigma": base_sigma.astype(np.float32),
        "garch_residual": residual.astype(np.float32),
        "garch_variance": (sigma**2).astype(np.float32),
        "garch_omega": np.array(GARCH_OMEGA, dtype=np.float32),
        "garch_alpha": np.array(GARCH_ALPHA, dtype=np.float32),
        "garch_beta": np.array(GARCH_BETA, dtype=np.float32),
        "garch_weather_weight": np.array(GARCH_WEATHER_WEIGHT, dtype=np.float32),
    }
    for alpha in REFERENCE_QUANTILES:
        extras[f"q{int(alpha * 100):02d}"] = (
            mu + sigma * normal_ppf(alpha)
        ).astype(np.float32)
    return with_common_fields("narx_garch", x, y, phi, extras)
