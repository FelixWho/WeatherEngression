"""Non-preadditive heteroskedastic NARX-style Gaussian target."""

from __future__ import annotations

import numpy as np

from .common import (
    HUMIDITY,
    PRECIPITATION,
    PRESSURE,
    RADIATION,
    REFERENCE_QUANTILES,
    SyntheticWeatherConfig,
    TEMPERATURE,
    WIND_SPEED,
    lagged_features,
    normal_ppf,
    phi_from_windows,
    sigmoid,
    weather_history,
    with_common_fields,
)


def narx_gaussian_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``mu(x)`` and ``sigma(x)`` for a conditional Gaussian.

    NARX means "nonlinear autoregressive with exogenous inputs". Here that just
    means the target depends nonlinearly on recent weather history. For each
    input window ``x``, this function returns parameters for
    ``Y | X=x ~ Normal(mu(x), sigma(x)^2)``.

    ``mu`` controls the center of the conditional distribution. ``sigma`` is the
    weather-dependent standard deviation, so the amount of uncertainty changes
    with the input conditions.
    """

    f = lagged_features(x)
    phi = phi_from_windows(x)
    current = f["current"]
    recent = f["recent"]
    long = f["long"]
    tendency = f["tendency"]

    # Handcoded conditional-mean formula.
    #
    # The coefficients below define the synthetic relationship between weather
    # history and the center of Y | X=x. They are chosen to mix smooth nonlinear
    # effects (sin, tanh, square), recent weather summaries, and tendencies. The
    # goal is not meteorological realism from data fitting; the goal is a known,
    # inspectable conditional distribution that is harder than a linear model.
    mu = (
        0.55 * np.sin(phi)  # nonlinear latent-weather effect
        + 0.18 * phi**2  # makes large absolute phi values matter more
        + 0.35 * np.tanh(recent[:, RADIATION])  # bright recent history raises mean
        - 0.45 * np.sqrt(np.maximum(recent[:, PRECIPITATION], 0.0))  # wet history lowers mean
        + 0.25 * current[:, HUMIDITY] * np.log1p(current[:, WIND_SPEED])  # humid/windy interaction
        - 0.18 * tendency[:, PRESSURE]  # pressure changes shift mean
        + 0.12 * np.sin(long[:, TEMPERATURE])  # slow temperature-history effect
    )
    # Handcoded conditional-noise formula.
    #
    # sigma is forced positive by starting at 0.12 and adding a sigmoid-bounded
    # term. The 0.50 multiplier controls how much heteroskedasticity is possible.
    # The inner coefficients say wet, unstable, windy conditions increase
    # uncertainty, while high radiation decreases it in this synthetic setup.
    sigma = (
        0.12
        + 0.50
        * sigmoid(
            -0.4
            + 0.9 * recent[:, PRECIPITATION]  # wet recent history increases uncertainty
            + 0.8 * np.abs(tendency[:, PRESSURE])  # pressure instability increases uncertainty
            - 0.5 * current[:, RADIATION]  # bright conditions reduce uncertainty
            + 0.35 * current[:, WIND_SPEED]  # wind increases uncertainty
        )
    )
    return mu, sigma


def generate_narx_gaussian_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate samples from the known NARX-style Gaussian law.

    The sampled target is ``y ~ Normal(mu(X), sigma(X)^2)``. Because this
    distribution is Gaussian, exact reference quantiles are available as
    ``mu + sigma * normal_ppf(alpha)``.
    """

    rng = np.random.default_rng(config.seed + 2)
    x, phi = weather_history(config)
    mu, sigma = narx_gaussian_parameters(x)
    y = rng.normal(mu, sigma)
    extras = {
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }
    for alpha in REFERENCE_QUANTILES:
        extras[f"q{int(alpha * 100):02d}"] = (mu + sigma * normal_ppf(alpha)).astype(np.float32)
    return with_common_fields("narx_gaussian", x, y, phi, extras)
