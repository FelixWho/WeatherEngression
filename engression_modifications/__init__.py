"""Extensions and wrappers around the public Python engression package."""

from .adamw_engression import AdamWEngressionConfig, fit_adamw_engression
from .lstm import LSTMEngressionConfig, fit_lstm_engression
from .regularized_engression import RegularizedEngressionConfig, fit_regularized_engression
from .registry import (
    MODEL_REGISTRY,
    EngressionModelSpec,
    adamw,
    fit_engression_model,
    get_engression_model,
    lstm,
    regularized,
    vanilla,
)
from .vanilla_engression import VanillaEngressionConfig, fit_vanilla_engression

__all__ = [
    "AdamWEngressionConfig",
    "EngressionModelSpec",
    "LSTMEngressionConfig",
    "MODEL_REGISTRY",
    "RegularizedEngressionConfig",
    "VanillaEngressionConfig",
    "adamw",
    "fit_adamw_engression",
    "fit_engression_model",
    "fit_lstm_engression",
    "fit_regularized_engression",
    "fit_vanilla_engression",
    "get_engression_model",
    "lstm",
    "regularized",
    "vanilla",
]
