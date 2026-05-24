"""Non-preadditive NARX-style target with Student-t innovations."""

from __future__ import annotations

import numpy as np

from .common import (
    REFERENCE_QUANTILES,
    SyntheticWeatherConfig,
    weather_history,
    with_common_fields,
)
from .narx_gaussian import narx_gaussian_parameters


STUDENT_T_DF = 5.0
STANDARD_T_QUANTILES = {
    0.05: -2.0150483733330233,
    0.50: 0.0,
    0.95: 2.0150483733330233,
}


def narx_student_t_parameters(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``mu(x)``, Student-t scale, and degrees of freedom.

    This target reuses the NARX Gaussian conditional mean and weather-dependent
    scale shape, but swaps the Gaussian innovation for a heavy-tailed Student-t
    innovation:

    ``Y | X=x = mu(x) + scale(x) * T_df``.

    The Student-t ``scale`` is not the standard deviation. For ``df=5``, the
    standard deviation is ``scale * sqrt(5 / 3)``. This makes the model useful
    for testing whether an engression fit can reproduce rare large deviations,
    not just ordinary heteroskedasticity.
    """

    mu, gaussian_sigma = narx_gaussian_parameters(x)
    scale = 0.80 * gaussian_sigma
    return mu, scale, STUDENT_T_DF


def student_t_quantile(alpha: float) -> float:
    """Return a precomputed standard Student-t quantile for ``df=5``.

    The repo intentionally avoids a SciPy dependency for these lightweight
    generators. Since we only save the common 5%, 50%, and 95% reference
    quantiles, hardcoding the corresponding ``t_5`` values keeps the true
    conditional quantiles analytic and inspectable.
    """

    return STANDARD_T_QUANTILES[alpha]


def generate_narx_student_t_dataset(config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    """Generate samples from a known heavy-tailed NARX-style conditional law."""

    rng = np.random.default_rng(config.seed + 5)
    x, phi = weather_history(config)
    mu, scale, df = narx_student_t_parameters(x)
    y = mu + scale * rng.standard_t(df, size=len(mu))
    extras = {
        "mu": mu.astype(np.float32),
        "scale": scale.astype(np.float32),
        "df": np.array(df, dtype=np.float32),
    }
    for alpha in REFERENCE_QUANTILES:
        extras[f"q{int(alpha * 100):02d}"] = (
            mu + scale * student_t_quantile(alpha)
        ).astype(np.float32)
    return with_common_fields("narx_student_t", x, y, phi, extras)
