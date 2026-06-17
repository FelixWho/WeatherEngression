"""Registry for importable engression model variants."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import torch

from .adamw_engression import AdamWEngressionConfig, fit_adamw_engression
from .lstm_engression import LSTMEngressionConfig, fit_lstm_engression
from .regularized_engression import RegularizedEngressionConfig, fit_regularized_engression
from .vanilla_engression import VanillaEngressionConfig, fit_vanilla_engression


ConfigFactory = Callable[..., object]
FitFunction = Callable[[torch.Tensor, torch.Tensor, object], object]


class EngressionModelSpec:
    """Small callable wrapper around a fitted engression model variant."""

    def __init__(
        self,
        name: str,
        config_factory: ConfigFactory,
        fit_function: FitFunction,
        description: str,
        input_kind: str = "flat",
    ) -> None:
        self.name = name
        self.config_factory = config_factory
        self.fit_function = fit_function
        self.description = description
        self.input_kind = input_kind

    def config(self, **overrides: Any) -> object:
        """Create a config object for this model variant."""

        return self.config_factory(**overrides)

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        config: object | None = None,
        **overrides: Any,
    ) -> object:
        """Fit this model variant.

        Examples
        --------
        ```python
        from engression_modifications import regularized

        engressor = regularized.fit(
            x_train,
            y_train,
            lr=0.003,
            weight_decay=0.003,
        )
        ```
        """

        if config is None:
            config = self.config(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        return self.fit_function(x, y, config)

    def __repr__(self) -> str:
        return f"EngressionModelSpec(name={self.name!r})"


vanilla = EngressionModelSpec(
    name="vanilla",
    config_factory=VanillaEngressionConfig,
    fit_function=fit_vanilla_engression,
    description="Unmodified public-package engression baseline.",
)

regularized = EngressionModelSpec(
    name="regularized",
    config_factory=RegularizedEngressionConfig,
    fit_function=fit_regularized_engression,
    description="Public-package engression with classic Adam weight decay.",
)

adamw = EngressionModelSpec(
    name="adamw",
    config_factory=AdamWEngressionConfig,
    fit_function=fit_adamw_engression,
    description="Public-package engression with decoupled AdamW weight decay.",
)

lstm = EngressionModelSpec(
    name="lstm",
    config_factory=LSTMEngressionConfig,
    fit_function=fit_lstm_engression,
    description="Sequence-native LSTM encoder with stochastic engression head.",
    input_kind="sequence",
)

MODEL_REGISTRY: dict[str, EngressionModelSpec] = {
    vanilla.name: vanilla,
    regularized.name: regularized,
    adamw.name: adamw,
    lstm.name: lstm,
    "classic_adam": regularized,
    "adam_weight_decay": regularized,
}


def get_engression_model(model: str | EngressionModelSpec) -> EngressionModelSpec:
    """Return a model spec by name, or pass through an existing spec."""

    if isinstance(model, EngressionModelSpec):
        return model
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown engression model {model!r}; choose one of: {valid}") from exc


def fit_engression_model(
    model: str | EngressionModelSpec,
    x: torch.Tensor,
    y: torch.Tensor,
    config: object | None = None,
    **overrides: Any,
) -> object:
    """Fit an engression model variant by name or model spec."""

    return get_engression_model(model).fit(x, y, config=config, **overrides)
