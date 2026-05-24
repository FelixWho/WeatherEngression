"""Non-preadditive X-dependent Gaussian mixture target."""

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
    normal_cdf_array,
    phi_from_windows,
    sigmoid,
    softmax,
    weather_history,
    with_common_fields,
)


def regime_mixture_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the known mixture parameters for ``Y | X=x``.

    For each observation, this builds a three-component Gaussian mixture:

    - ``weights`` are regime probabilities and each row sums to 1;
    - ``means`` are the three regime-specific conditional means;
    - ``sigmas`` are the three regime-specific standard deviations.

    The generated dataset samples one hidden regime, but these parameters
    describe the full conditional distribution, not just the sampled outcome.
    """

    f = lagged_features(x)
    phi = phi_from_windows(x)
    current = f["current"]
    recent = f["recent"]
    long = f["long"]
    tendency = f["tendency"]

    # Handcoded regime-score formulas.
    #
    # These logits are converted to probabilities by softmax. The three columns
    # encode synthetic regimes:
    # 1. bright/dry;
    # 2. wet/humid;
    # 3. windy/pressure-change/transport-like.
    # Larger coefficients make a weather feature more influential in assigning a
    # sample to that hidden regime.
    logits = np.column_stack(
        [
            0.8 * recent[:, RADIATION]  # bright history favors clear/dry regime
            - 0.8 * recent[:, PRECIPITATION]  # rain suppresses clear/dry regime
            - 0.3 * current[:, HUMIDITY],  # humidity suppresses clear/dry regime
            1.2 * recent[:, PRECIPITATION]  # rain strongly favors wet regime
            + 0.6 * current[:, HUMIDITY]  # humidity favors wet regime
            - 0.4 * recent[:, RADIATION],  # radiation suppresses wet regime
            0.7 * current[:, WIND_SPEED]  # wind favors transported regime
            + 0.5 * np.abs(tendency[:, PRESSURE])  # pressure instability favors transported regime
            + 0.2 * long[:, TEMPERATURE],  # slow temperature signal mildly favors transported regime
        ]
    )
    weights = softmax(logits)

    # Shared mean baseline used by all regimes. The hardcoded 0.35 and 0.15 make
    # the latent index phi affect all regimes, but still allow regime-specific
    # offsets below to dominate the modality.
    base = 0.35 * np.sin(phi) + 0.15 * phi

    # Handcoded regime-specific means. The offsets 0.75, -0.85, and 0.15 create
    # separated modes, while the smaller weather terms make each mode move with
    # current/recent conditions.
    means = np.column_stack(
        [
            base + 0.75 + 0.25 * np.tanh(recent[:, RADIATION]),  # higher clear/dry mode
            base - 0.85 - 0.35 * np.sqrt(np.maximum(recent[:, PRECIPITATION], 0.0)),  # lower wet-removal mode
            base + 0.15 + 0.45 * np.log1p(current[:, WIND_SPEED]),  # wind-driven transported mode
        ]
    )
    # Handcoded regime-specific standard deviations. These keep each component
    # positive and let wet or unstable conditions widen the corresponding
    # Gaussian component.
    sigmas = np.column_stack(
        [
            0.18 + 0.10 * sigmoid(current[:, RADIATION]),  # clear regime is relatively tight
            0.28 + 0.20 * sigmoid(recent[:, PRECIPITATION]),  # wet regime widens with rain
            0.35 + 0.15 * sigmoid(np.abs(tendency[:, PRESSURE])),  # transported regime is broad
        ]
    )
    return weights, means, sigmas


def mixture_cdf(y: np.ndarray, weights: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """Evaluate the CDF of the three-component Gaussian mixture.

    For each row, this asks: under the known mixture distribution for that
    input, what is ``P(Y <= y)``? The candidate ``y`` is standardized separately
    for each Gaussian component, converted to a component CDF with
    ``normal_cdf_array``, then averaged using the mixture weights.
    """

    z = (y[:, None] - means) / sigmas
    return (weights * normal_cdf_array(z)).sum(axis=1)


def mixture_quantile(
    weights: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    alpha: float,
    n_iter: int = 80,
) -> np.ndarray:
    """Numerically invert the mixture CDF to get a reference quantile.

    A Gaussian mixture does not have a simple closed-form quantile formula.
    Instead, this uses binary search. For each observation, it keeps a lower and
    upper candidate value for the desired quantile, evaluates the mixture CDF at
    the midpoint, and shrinks the interval until it finds the value where
    ``P(Y <= q_alpha)`` is approximately ``alpha``.
    """

    # The search interval is deliberately much wider than the component means:
    # 8 standard deviations on either side is effectively all Gaussian mass for
    # this synthetic use case. n_iter=80 then makes binary-search error tiny.
    lower = means.min(axis=1) - 8.0 * sigmas.max(axis=1)
    upper = means.max(axis=1) + 8.0 * sigmas.max(axis=1)
    for _ in range(n_iter):
        mid = 0.5 * (lower + upper)
        cdf = mixture_cdf(mid, weights, means, sigmas)
        lower = np.where(cdf < alpha, mid, lower)
        upper = np.where(cdf >= alpha, mid, upper)
    return 0.5 * (lower + upper)


def generate_regime_mixture_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate samples from the known X-dependent Gaussian mixture.

    Each row first samples an unobserved regime according to ``weights[x]`` and
    then samples ``Y`` from that regime's Gaussian distribution. The dataset also
    stores the full mixture parameters, so downstream validation can reconstruct
    the true ``Y | X=x`` distribution.
    """

    rng = np.random.default_rng(config.seed + 3)
    x, phi = weather_history(config)
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
