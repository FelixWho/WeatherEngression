"""Theory-friendly pre-additive target model."""

from __future__ import annotations

import numpy as np

from .common import (
    REFERENCE_QUANTILES,
    SyntheticWeatherConfig,
    normal_ppf,
    weather_history,
    with_common_fields,
)


def monotone_response(z: np.ndarray) -> np.ndarray:
    """Map the latent noisy index into the observed target scale.

    The pre-additive generator first forms ``z = phi(X) + eta``. This function
    then turns that latent value into ``Y``. It is strictly increasing, so known
    quantiles of ``eta`` can be pushed through this same function to get known
    quantiles of ``Y | X``.
    """

    # The 0.15 cubic coefficient is a handcoded curvature strength. If it were
    # 0, the target would be almost linear in phi + eta. Increasing it makes the
    # tails and nonlinear effects stronger while preserving monotonicity.
    return z + 0.15 * z**3


def preadditive_quantile(phi: np.ndarray, alpha: float, noise_scale: float) -> np.ndarray:
    """Analytic conditional quantile for the pre-additive target.

    For this generator, ``Y = g(phi(X) + eta)`` and ``eta`` is normal with mean
    0 and standard deviation ``noise_scale``. For a fixed input ``X=x``,
    ``phi(x)`` is fixed, so the only randomness is ``eta``.

    ``normal_ppf(alpha)`` gives the alpha-quantile of a standard normal. After
    scaling it by ``noise_scale`` and shifting by ``phi``, passing through
    ``monotone_response`` gives the exact alpha-quantile of ``Y | X=x``.
    """

    return monotone_response(phi + noise_scale * normal_ppf(alpha))


def generate_preadditive_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate observations from ``Y = g(phi(X) + eta)``.

    This is the theory-friendly synthetic case. It stores both the sampled
    target ``y`` and exact reference quantiles, so later model checks can compare
    engression's learned conditional distribution against known truth.
    """

    rng = np.random.default_rng(config.seed + 1)
    x, phi = weather_history(config)

    eta = rng.normal(0.0, config.noise_scale, size=len(phi))
    y = monotone_response(phi + eta)

    extras = {
        f"q{int(alpha * 100):02d}": preadditive_quantile(phi, alpha, config.noise_scale).astype(np.float32)
        for alpha in REFERENCE_QUANTILES
    }
    extras["noise_scale"] = np.array(config.noise_scale, dtype=np.float32)
    return with_common_fields("preadditive", x, y, phi, extras)
