"""Shared utilities for synthetic weather-like time-series generators."""

from __future__ import annotations

from dataclasses import dataclass
import math
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

(
    TEMPERATURE,
    HUMIDITY,
    PRESSURE,
    RADIATION,
    PRECIPITATION,
    WIND_SPEED,
) = range(len(FEATURE_NAMES))
REFERENCE_QUANTILES = (0.05, 0.50, 0.95)


@dataclass(frozen=True)
class SyntheticWeatherConfig:
    """Small bundle of knobs shared by all synthetic generators.

    ``n_steps`` is the length of the raw weather time series before lag windows
    are made. ``window`` is the number of previous time steps kept in each
    input, so the final ``X`` has ``window + 1`` rows per sample: the previous
    ``window`` states plus the current state. ``noise_scale`` controls the
    latent noise size for the pre-additive target.
    """

    n_steps: int = 10_000
    window: int = 24
    seed: int = 7
    noise_scale: float = 0.35
    seasonal_period: int = 24 * 365


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Map real-valued scores to numbers between 0 and 1.

    This is used whenever a synthetic rule needs a probability or bounded
    weather variable. For example, a large positive score becomes a probability
    near 1, and a large negative score becomes a probability near 0.
    """

    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn one row of arbitrary regime scores into regime probabilities.

    ``logits`` has shape ``(n_observations, n_regimes)``. Each row can contain
    any real numbers. The result has the same shape, but every row is positive
    and sums to 1. In the regime-mixture generator, this gives
    ``pi_1(x), pi_2(x), pi_3(x)``.
    """

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def normal_ppf(alpha: float) -> float:
    """Return the standard-normal quantile for probability ``alpha``.

    PPF means "percent point function", which is another name for inverse CDF.
    If ``Z`` is a standard normal random variable, this returns the number ``z``
    such that ``P(Z <= z) = alpha``.

    Examples:
    - ``normal_ppf(0.50)`` is approximately 0.
    - ``normal_ppf(0.95)`` is approximately 1.645.

    The generators use this to convert known distribution parameters into true
    reference quantiles like ``q05``, ``q50``, and ``q95``.
    """

    return NormalDist().inv_cdf(alpha)


def normal_cdf_array(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF.

    If ``Z`` is standard normal, this returns ``P(Z <= x)`` elementwise for an
    array of ``x`` values. The result has the same shape as the input.

    In the mixture generator, a candidate target value ``y`` is standardized as
    ``z = (y - mean) / sigma`` for each Gaussian component. Calling this helper
    gives the probability that each component is less than or equal to ``y``.
    Those component probabilities are then weighted and added to get the full
    mixture CDF.
    """

    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def generate_covariates(config: SyntheticWeatherConfig) -> np.ndarray:
    """Generate the raw weather-like time series ``W_t``.

    This function creates one multivariate time series with columns ordered by
    ``FEATURE_NAMES``. The process is synthetic, but it tries to preserve a few
    weather-like properties:

    - autoregression: the next state depends on the previous state;
    - cross-variable dependence: variables influence one another through ``ar``;
    - daily and seasonal cycles: radiation and temperature have periodic
      forcing;
    - constrained variables: humidity is squashed to ``(0, 1)``, radiation and
      wind are nonnegative, and precipitation is intermittent.

    The returned array has shape ``(n_steps, n_features)``.
    """

    rng = np.random.default_rng(config.seed)
    d = len(FEATURE_NAMES)
    state = np.zeros((config.n_steps, d), dtype=np.float64)

    # Hand-tuned autoregression matrix for the synthetic weather state.
    #
    # Rows say how yesterday/previous-step variables affect today's variable.
    # Diagonal values near 1 make each variable persistent over time. Off-diagonal
    # values create weak cross-weather effects, for example humidity and
    # precipitation nudging each other, or pressure and wind moving together.
    # These numbers are not estimated from data; they are deliberately chosen to
    # create nontrivial but stable weather-like time series.
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
    # Small nonlinear self-feedback terms. Applying tanh keeps these bounded, so
    # they add curvature without letting the synthetic process explode.
    nonlinear = np.array([0.08, -0.04, 0.03, 0.05, 0.06, -0.03])

    for t in range(1, config.n_steps):
        daily = np.sin(2 * np.pi * t / 24.0)
        daily_daylight = max(0.0, daily)
        seasonal = np.sin(2 * np.pi * t / config.seasonal_period)
        # Seasonal/daily forcing terms. These are hardcoded to make the raw
        # process feel weather-like: temperature tracks season + day, radiation
        # rises only during positive "daylight", and wind has a slower cycle.
        forcing = np.array(
            [
                0.9 * seasonal + 0.25 * daily,  # temperature
                -0.25 * seasonal - 0.15 * daily_daylight,  # humidity
                -0.2 * seasonal,  # pressure
                1.3 * daily_daylight * (1.0 + 0.25 * seasonal),  # radiation
                0.15 * seasonal,  # precipitation tendency before intermittency
                0.25 * np.cos(2 * np.pi * t / (24.0 * 5.0)),  # wind speed
            ],
            dtype=np.float64,
        )
        # Per-feature innovation scales. Larger values make the corresponding
        # weather variable jumpier from one step to the next.
        innovation = rng.normal(
            0.0,
            [
                0.25,  # temperature
                0.20,  # humidity
                0.15,  # pressure
                0.20,  # radiation
                0.25,  # precipitation
                0.20,  # wind speed
            ],
        )
        state[t] = ar @ state[t - 1] + nonlinear * np.tanh(state[t - 1]) + forcing + innovation

    covariates = state.copy()
    covariates[:, HUMIDITY] = sigmoid(covariates[:, HUMIDITY])
    covariates[:, RADIATION] = np.maximum(0.0, covariates[:, RADIATION])

    # Precipitation occurrence is another handcoded synthetic rule: more latent
    # humidity increases rain probability, while higher pressure and radiation
    # decrease it. The coefficients control how strongly each variable pushes
    # the Bernoulli probability after the sigmoid.
    precip_prob = sigmoid(
        1.6 * state[:, HUMIDITY]
        - 0.7 * state[:, PRESSURE]
        - 0.3 * state[:, RADIATION]
    )
    precip_amount = rng.gamma(shape=1.4, scale=0.7, size=config.n_steps)
    precip_event = rng.binomial(1, precip_prob)
    covariates[:, PRECIPITATION] = precip_event * precip_amount
    covariates[:, WIND_SPEED] = np.log1p(np.exp(covariates[:, WIND_SPEED]))
    return covariates


def make_lag_windows(covariates: np.ndarray, window: int) -> np.ndarray:
    """Convert the raw time series into supervised inputs.

    A sample at time ``t`` receives the history ``(W_{t-window}, ..., W_t)``.
    Therefore each row of the returned array is a small time series, not a flat
    feature vector. The output shape is
    ``(n_steps - window, window + 1, n_features)``.
    """

    return np.stack([covariates[i - window : i + 1] for i in range(window, len(covariates))])


def phi_from_windows(x: np.ndarray) -> np.ndarray:
    """Compress each weather-history window into one latent index ``phi``.

    ``x`` has shape ``(n_observations, n_lags, n_features)``. This function
    computes a weighted average over both time lags and weather variables:

    - recent lags receive larger weights than old lags;
    - feature weights decide which variables increase or decrease ``phi``.

    The output has shape ``(n_observations,)``. It is a synthetic hidden signal,
    not a real meteorological quantity. The pre-additive target uses it as the
    deterministic anchor in ``Y = g(phi(X) + eta)``.

    This is also the bridge from this repo's multivariate time-series input to
    the simpler scalar-index setup used by the pre-additive example. The model
    input is still the full lag window ``X``; ``phi`` is saved so we can inspect
    the known data-generating process.
    """

    n_lags = x.shape[1]
    lag_age = np.arange(n_lags - 1, -1, -1, dtype=np.float64)
    # The 8.0 decay length says roughly how quickly old weather history loses
    # influence on phi. Smaller values would make phi depend almost entirely on
    # current weather; larger values would make older lags matter more.
    lag_weights = np.exp(-lag_age / 8.0)
    lag_weights = lag_weights / lag_weights.sum()

    # Handcoded feature weights for the latent weather index phi.
    #
    # Positive weights make a feature increase phi; negative weights make it
    # decrease phi. These encode the synthetic story that warm, bright, windy
    # conditions raise the latent index, while humid/wet conditions lower it.
    # They are not learned and are not meant to be meteorologically authoritative.
    feature_weights = np.array(
        [
            0.45,   # temperature: warmer history raises phi
            -0.35,  # humidity: humid history lowers phi
            0.20,   # pressure: higher pressure mildly raises phi
            0.50,   # radiation: brighter history raises phi
            -0.65,  # precipitation: wetter history strongly lowers phi
            0.25,   # wind speed: windier history mildly raises phi
        ],
        dtype=np.float64,
    )
    return np.einsum("nlf,l,f->n", x, lag_weights, feature_weights)


def lagged_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Create readable summaries of each lag window for target formulas.

    The target models could index directly into the 3D lag tensor, but that
    makes the equations hard to read. This helper exposes common summaries:

    - ``current``: the most recent weather state ``W_t``;
    - ``recent``: an exponentially weighted short-history average;
    - ``long``: an exponentially weighted longer-history average;
    - ``tendency``: current state minus the roughly 6-hour-old state.

    Each returned array has shape ``(n_observations, n_features)``.
    """

    n_lags = x.shape[1]
    current = x[:, -1, :]
    lag_6h = x[:, max(0, n_lags - 7), :]

    lag_age = np.arange(n_lags - 1, -1, -1, dtype=np.float64)
    # Two hardcoded history summaries used by the target models. The 6.0 decay
    # gives a short-memory recent average; the 24.0 decay gives a slower
    # long-memory average. These are synthetic modeling choices, not fitted
    # parameters.
    recent_weights = np.exp(-lag_age / 6.0)
    recent_weights = recent_weights / recent_weights.sum()
    long_weights = np.exp(-lag_age / 24.0)
    long_weights = long_weights / long_weights.sum()

    return {
        "current": current,
        "recent": np.einsum("nlf,l->nf", x, recent_weights),
        "long": np.einsum("nlf,l->nf", x, long_weights),
        "tendency": current - lag_6h,
    }


def weather_history(config: SyntheticWeatherConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build the two shared inputs every target model needs.

    This is the common front half of every synthetic dataset:

    1. simulate raw weather covariates ``W_t``;
    2. turn them into lag-window inputs ``X``;
    3. compute the latent scalar index ``phi(X)``.

    Target-specific files then decide how to generate ``Y`` from ``X`` and
    ``phi``.
    """

    covariates = generate_covariates(config)
    x = make_lag_windows(covariates, config.window)
    phi = phi_from_windows(x)
    return x, phi


def with_common_fields(
    model: str,
    x: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    extras: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Assemble the fields shared by every saved or printed dataset.

    ``extras`` contains model-specific truth such as ``mu`` and ``sigma`` for a
    Gaussian target, mixture weights for the regime target, or reference
    quantiles. Keeping the common fields here makes every generator return the
    same basic contract: ``X``, ``y``, ``phi``, ``model``, and ``feature_names``.
    """

    result = {
        "X": x.astype(np.float32),
        "y": y.astype(np.float32),
        "phi": phi.astype(np.float32),
        "model": np.array(model),
        "feature_names": np.array(FEATURE_NAMES),
    }
    result.update(extras)
    return result
