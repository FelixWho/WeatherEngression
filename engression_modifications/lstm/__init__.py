"""Sequence-native LSTM engression model.

The public ``engression`` package flattens every predictor tensor before it
reaches the stochastic MLP. This subpackage keeps the energy-loss idea, but
replaces the flat generator with an LSTM encoder followed by a stochastic head.

The pieces are split by concern:
  - ``config``       : ``LSTMEngressionConfig`` (all knobs)
  - ``monotone``     : ``PositiveLinear`` / ``MonotonePositiveMLP`` (pre-additive g)
  - ``lstm_backbone``: the shared ``DeterministicLSTMEncoder`` building block
  - ``generators``   : one generator class per head + ``build_lstm_model`` (the model)
  - ``engressor``    : ``LSTMEngressor`` (the fitted wrapper, separate from the model)
  - ``preprocessing``-- input validation + standardization
  - ``checkpoint``   : save/reload
  - ``training``     : ``fit_lstm_engression`` (the training loop)

Import the public names straight from this package:
``from engression_modifications.lstm import LSTMEngressionConfig, fit_lstm_engression``.
"""

from __future__ import annotations

# Configure matplotlib for headless use BEFORE importing the ``engression``
# package (some of its modules pull matplotlib at import time). Must run before
# the submodule imports below.
import os
from pathlib import Path
import tempfile

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from .checkpoint import load_lstm_engressor_checkpoint
from .config import LSTMEngressionConfig
from .lstm_backbone import DeterministicLSTMEncoder
from .engressor import LSTMEngressor
from .generators import (
    DefaultHeadGenerator,
    GlobalLatentLSTMGenerator,
    LSTMGenerator,
    PerTimestepNoiseLSTMGenerator,
    PreAdditiveGenerator,
    RecurrentStateNoiseLSTMGenerator,
    StoNetHeadGenerator,
    StochasticAdditiveLSTMGenerator,
    StochasticAppendingLSTMGenerator,
    StochasticInitStateLSTMGenerator,
    build_lstm_model,
)
from .monotone import MonotonePositiveMLP, PositiveLinear
from .preprocessing import standardization_stats, validate_sequence_x, validate_y
from .training import fit_lstm_engression

__all__ = [
    "DefaultHeadGenerator",
    "DeterministicLSTMEncoder",
    "GlobalLatentLSTMGenerator",
    "LSTMEngressionConfig",
    "LSTMEngressor",
    "LSTMGenerator",
    "MonotonePositiveMLP",
    "PerTimestepNoiseLSTMGenerator",
    "PositiveLinear",
    "PreAdditiveGenerator",
    "RecurrentStateNoiseLSTMGenerator",
    "StoNetHeadGenerator",
    "StochasticAdditiveLSTMGenerator",
    "StochasticAppendingLSTMGenerator",
    "StochasticInitStateLSTMGenerator",
    "build_lstm_model",
    "fit_lstm_engression",
    "load_lstm_engressor_checkpoint",
    "standardization_stats",
    "validate_sequence_x",
    "validate_y",
]
