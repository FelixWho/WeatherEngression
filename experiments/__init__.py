"""Shared experiment utilities for WeatherEngression."""

from .pipeline import (
    EngressionFitConfig,
    ExperimentData,
    ExperimentResult,
    PredictionConfig,
    SplitConfig,
    SyntheticDataConfig,
    evaluate_experiment_model,
    fit_experiment_model,
    prepare_experiment_data,
    run_engression_experiment,
    set_reproducible_seeds,
)
from .oos import (
    OOSConfig,
    OOSDiagnostics,
    compute_oos_diagnostics,
    knn_distance_oos,
    mahalanobis_distance_oos,
    mahalanobis_distances,
    marginal_quantile_oos,
    marginal_range_oos,
    scalar_projection_oos,
)

__all__ = [
    "EngressionFitConfig",
    "ExperimentData",
    "ExperimentResult",
    "OOSConfig",
    "OOSDiagnostics",
    "PredictionConfig",
    "SplitConfig",
    "SyntheticDataConfig",
    "compute_oos_diagnostics",
    "evaluate_experiment_model",
    "fit_experiment_model",
    "knn_distance_oos",
    "mahalanobis_distance_oos",
    "mahalanobis_distances",
    "marginal_quantile_oos",
    "marginal_range_oos",
    "prepare_experiment_data",
    "run_engression_experiment",
    "scalar_projection_oos",
    "set_reproducible_seeds",
]
