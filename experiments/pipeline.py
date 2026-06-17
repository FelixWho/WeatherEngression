"""Central experiment pipeline for data generation, fitting, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from engression_modifications import EngressionModelSpec, get_engression_model
from generate_data import (
    BASE_X_DIMENSION,
    DEFAULT_SEASONAL_PERIOD,
    generate_supervised_dataset,
)

from .constants import QUANTILE_KEYS
from .data import flattened_tensors_from_dataset, tensors_from_dataset
from .metrics import quantile_diagnostic_metrics
from .oos import OOSConfig, OOSDiagnostics, compute_oos_diagnostics
from .predictions import predict_quantiles
from .splits import select_split


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for generating one synthetic supervised dataset."""

    data_model: str = "narx_student_t"
    num_samples: int = 10_000
    x_dimension: int = BASE_X_DIMENSION
    window: int = 12
    seed: int = 2026
    noise_scale: float = 0.35
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for selecting train/test rows from a generated dataset."""

    split: str = "in-support"
    train_size: int = 8_000
    test_size: int = 1_000
    seed_offset: int = 17


@dataclass(frozen=True)
class EngressionFitConfig:
    """Configuration shared by importable engression model variants."""

    engression_model: str | EngressionModelSpec = "vanilla"
    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    add_bn: bool = False
    lr: float = 0.003
    weight_decay: float = 0.0
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    device: str | torch.device = "cpu"
    verbose: bool = False


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for sample-based conditional quantile prediction."""

    sample_size: int = 2_000
    seed_offset: int = 101


@dataclass(frozen=True)
class ExperimentData:
    """Generated data plus train/test tensors and truth fields."""

    dataset: dict[str, np.ndarray]
    train_idx: np.ndarray
    test_idx: np.ndarray
    phi_support: tuple[float, float]
    train_phi: np.ndarray
    test_phi: np.ndarray
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_train_flat: torch.Tensor
    x_test_flat: torch.Tensor
    y_test_np: np.ndarray
    true_quantiles: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    """Full output of one generated-data engression experiment."""

    data: ExperimentData
    engressor: object
    predicted_quantiles: np.ndarray
    metrics: dict[str, object]
    oos: OOSDiagnostics


def set_reproducible_seeds(seed: int) -> None:
    """Set NumPy and Torch seeds for a deterministic CPU-friendly run."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def prepare_experiment_data(
    data_config: SyntheticDataConfig,
    split_config: SplitConfig,
    input_kind: str = "flat",
) -> ExperimentData:
    """Generate data, select rows, and build tensors for model fitting."""

    dataset = generate_supervised_dataset(
        model=data_config.data_model,
        num_samples=data_config.num_samples,
        x_dimension=data_config.x_dimension,
        window=data_config.window,
        seed=data_config.seed,
        noise_scale=data_config.noise_scale,
        seasonal_period=data_config.seasonal_period,
    )
    phi = np.asarray(dataset["phi"], dtype=np.float32)
    train_idx, test_idx, support = select_split(
        phi=phi,
        train_size=split_config.train_size,
        test_size=split_config.test_size,
        seed=data_config.seed + split_config.seed_offset,
        split=split_config.split,
    )
    x_train, y_train = tensors_from_dataset(dataset, train_idx, input_kind=input_kind)
    x_test, y_test = tensors_from_dataset(dataset, test_idx, input_kind=input_kind)
    x_train_flat, _ = flattened_tensors_from_dataset(dataset, train_idx)
    x_test_flat, _ = flattened_tensors_from_dataset(dataset, test_idx)
    true_quantiles = np.column_stack(
        [np.asarray(dataset[key], dtype=np.float32)[test_idx] for key in QUANTILE_KEYS]
    )
    return ExperimentData(
        dataset=dataset,
        train_idx=train_idx,
        test_idx=test_idx,
        phi_support=support,
        train_phi=phi[train_idx],
        test_phi=phi[test_idx],
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_train_flat=x_train_flat,
        x_test_flat=x_test_flat,
        y_test_np=y_test.detach().cpu().numpy().reshape(-1),
        true_quantiles=true_quantiles,
    )


def fit_experiment_model(
    data: ExperimentData,
    fit_config: EngressionFitConfig,
) -> object:
    """Fit one importable engression model variant on prepared data."""

    overrides = {
        "num_layer": fit_config.num_layer,
        "hidden_dim": fit_config.hidden_dim,
        "noise_dim": fit_config.noise_dim,
        "add_bn": fit_config.add_bn,
        "lr": fit_config.lr,
        "num_epochs": fit_config.num_epochs,
        "batch_size": fit_config.batch_size,
        "standardize": fit_config.standardize,
        "device": fit_config.device,
        "verbose": fit_config.verbose,
    }
    model_spec = get_engression_model(fit_config.engression_model)
    if model_spec.name in {"regularized", "adamw", "lstm"}:
        overrides["weight_decay"] = fit_config.weight_decay
    return model_spec.fit(
        data.x_train,
        data.y_train,
        **overrides,
    )


def evaluate_experiment_model(
    data: ExperimentData,
    engressor: object,
    data_config: SyntheticDataConfig,
    split_config: SplitConfig,
    fit_config: EngressionFitConfig,
    prediction_config: PredictionConfig,
    oos_config: OOSConfig | None = None,
) -> tuple[np.ndarray, dict[str, object], OOSDiagnostics]:
    """Predict conditional quantiles and compute diagnostics."""

    torch.manual_seed(data_config.seed + prediction_config.seed_offset)
    predicted_quantiles = predict_quantiles(
        engressor=engressor,
        x_test=data.x_test,
        sample_size=prediction_config.sample_size,
    )
    metrics = quantile_diagnostic_metrics(
        predicted_quantiles=predicted_quantiles,
        true_quantiles=data.true_quantiles,
        y_test=data.y_test_np,
    )
    outside_support = int(
        np.sum((data.test_phi < data.phi_support[0]) | (data.test_phi > data.phi_support[1]))
    )
    oos = compute_oos_diagnostics(
        x_train=data.x_train_flat,
        x_test=data.x_test_flat,
        train_phi=data.train_phi,
        test_phi=data.test_phi,
        config=oos_config,
        seed=data_config.seed,
    )
    model_name = get_engression_model(fit_config.engression_model).name
    metrics.update(
        {
            "data_model": data_config.data_model,
            "engression_model": model_name,
            "split": split_config.split,
            "train_rows": int(len(data.train_idx)),
            "test_rows": int(len(data.test_idx)),
            "test_rows_outside_train_phi_support": outside_support,
            "train_phi_support": [float(data.phi_support[0]), float(data.phi_support[1])],
            "test_phi_range": [float(np.min(data.test_phi)), float(np.max(data.test_phi))],
            "x_train_shape": list(data.x_train.shape),
            "x_test_shape": list(data.x_test.shape),
            "x_train_flat_shape": list(data.x_train_flat.shape),
            "x_test_flat_shape": list(data.x_test_flat.shape),
            "oos": oos.summary,
        }
    )
    return predicted_quantiles, metrics, oos


def run_engression_experiment(
    data_config: SyntheticDataConfig,
    split_config: SplitConfig,
    fit_config: EngressionFitConfig,
    prediction_config: PredictionConfig,
    oos_config: OOSConfig | None = None,
) -> ExperimentResult:
    """Generate data, fit a selected model, and evaluate conditional quantiles."""

    set_reproducible_seeds(data_config.seed)
    model_spec = get_engression_model(fit_config.engression_model)
    data = prepare_experiment_data(data_config, split_config, input_kind=model_spec.input_kind)
    engressor = fit_experiment_model(data, fit_config)
    predicted_quantiles, metrics, oos = evaluate_experiment_model(
        data=data,
        engressor=engressor,
        data_config=data_config,
        split_config=split_config,
        fit_config=fit_config,
        prediction_config=prediction_config,
        oos_config=oos_config,
    )
    return ExperimentResult(
        data=data,
        engressor=engressor,
        predicted_quantiles=predicted_quantiles,
        metrics=metrics,
        oos=oos,
    )
