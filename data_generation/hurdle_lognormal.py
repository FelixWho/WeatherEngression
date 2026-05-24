"""Non-preadditive zero-inflated precipitation-like lognormal target."""

from __future__ import annotations

import numpy as np

from .common import (
    HUMIDITY,
    PRECIPITATION,
    PRESSURE,
    RADIATION,
    REFERENCE_QUANTILES,
    SyntheticWeatherConfig,
    WIND_SPEED,
    lagged_features,
    normal_ppf,
    sigmoid,
    weather_history,
    with_common_fields,
)


def hurdle_lognormal_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the known parameters for a zero-inflated target.

    ``p_wet`` is the probability that the observation is positive. With
    probability ``1 - p_wet``, the target is exactly zero. If the observation is
    positive, then ``log(Y)`` is normal with mean ``log_mu`` and standard
    deviation ``log_sigma``.
    """

    f = lagged_features(x)
    current = f["current"]
    recent = f["recent"]
    tendency = f["tendency"]

    # Handcoded wet-event probability.
    #
    # The intercept -1.0 makes dry observations common by default. Humidity and
    # recent precipitation raise the chance of a positive value; radiation and
    # rising pressure reduce it. The sigmoid turns this score into p_wet in
    # (0, 1).
    p_wet = sigmoid(
        -1.0
        + 2.4 * current[:, HUMIDITY]  # humidity strongly increases wet probability
        + 0.9 * recent[:, PRECIPITATION]  # recent rain increases wet probability
        - 0.7 * current[:, RADIATION]  # bright conditions reduce wet probability
        - 0.5 * tendency[:, PRESSURE]  # pressure tendency reduces wet probability here
    )
    # Handcoded mean of log(Y) conditional on the event being wet. Because this
    # is on the log scale, additive effects here become multiplicative effects
    # on the positive target Y.
    log_mu = (
        -0.35
        + 0.9 * current[:, HUMIDITY]  # humidity increases positive amount
        + 0.45 * np.log1p(recent[:, PRECIPITATION])  # recent rain increases positive amount
        + 0.25 * current[:, WIND_SPEED]  # wind modestly increases positive amount
        - 0.25 * current[:, RADIATION]  # radiation reduces positive amount
    )
    # Handcoded positive-tail uncertainty. The 0.25 baseline prevents the
    # positive tail from becoming deterministic; the 0.45 term lets wet/windy
    # conditions produce a heavier positive tail.
    log_sigma = 0.25 + 0.45 * sigmoid(0.8 * recent[:, PRECIPITATION] + 0.5 * current[:, WIND_SPEED])
    return p_wet, log_mu, log_sigma


def hurdle_lognormal_quantile(
    p_wet: np.ndarray,
    log_mu: np.ndarray,
    log_sigma: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Analytic quantile for the zero-inflated lognormal target.

    The distribution has a point mass at zero. If the requested quantile falls
    inside that dry mass, the answer is exactly zero. Otherwise, the quantile is
    inside the positive lognormal tail. In that case, the code rescales
    ``alpha`` to a tail probability and uses ``normal_ppf`` because a lognormal
    variable is built by exponentiating a normal variable.
    """

    dry_mass = 1.0 - p_wet
    q = np.zeros_like(p_wet)
    wet_mask = alpha > dry_mass
    if np.any(wet_mask):
        tail_alpha = (alpha - dry_mass[wet_mask]) / p_wet[wet_mask]
        # Keep the inverse-normal call away from exactly 0 or 1, where the
        # theoretical normal quantile would be infinite.
        tail_alpha = np.clip(tail_alpha, 1e-8, 1.0 - 1e-8)
        tail_z = np.array([normal_ppf(float(a)) for a in tail_alpha], dtype=np.float64)
        q[wet_mask] = np.exp(log_mu[wet_mask] + log_sigma[wet_mask] * tail_z)
    return q


def generate_hurdle_lognormal_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate samples from the zero-inflated lognormal conditional law."""

    rng = np.random.default_rng(config.seed + 4)
    x, phi = weather_history(config)
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
