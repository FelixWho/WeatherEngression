"""Checkpoint serialization and reload for LSTM engression.

Saves everything needed to reconstruct a fitted ``LSTMEngressor``: the config
(minus the un-serializable ``model`` override), the network weights, optimizer
state, and standardization moments. Reload rebuilds the network via
``build_lstm_model`` and rewraps it.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .config import LSTMEngressionConfig
from .engressor import LSTMEngressor
from .generators import build_lstm_model


def _to_cpu(value: object) -> object:
    """Recursively move checkpoint tensors to CPU before serialization."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: LSTMEngressionConfig,
    epoch: int,
    train_loss: float,
    input_dim: int,
    out_dim: int,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    val_loss: float | None = None,
) -> dict[str, object]:
    """Build a reloadable checkpoint payload.

    ``val_loss`` is the held-out energy loss when the fit was given a validation
    set, and None otherwise. It is recorded for provenance only; reload ignores it.
    """

    # Serialize the config field-by-field, dropping ``model``: a live nn.Module
    # cannot round-trip through the JSON-ish checkpoint config and its weights are
    # already saved in model_state_dict. Custom architectures are rebuilt by
    # passing the same model/factory back to load_lstm_engressor_checkpoint.
    config_dict = {f.name: getattr(config, f.name) for f in fields(config) if f.name != "model"}
    config_dict["device"] = str(config_dict["device"])
    return {
        "model_type": "lstm_engression",
        "epoch": int(epoch),
        "train_energy_loss": float(train_loss),
        "val_energy_loss": None if val_loss is None else float(val_loss),
        "input_dim": int(input_dim),
        "out_dim": int(out_dim),
        "config": config_dict,
        "model_state_dict": _to_cpu(model.state_dict()),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "y_mean": y_mean.detach().cpu(),
        "y_std": y_std.detach().cpu(),
    }


def _migrate_legacy_state_dict(state_dict: dict[str, object]) -> dict[str, object]:
    """Remap a pre-refactor checkpoint onto the current module layout.

    The old monolithic generator held the LSTM directly at ``lstm.*``; the
    encoder is now a submodule, so those weights live at ``encoder.lstm.*``. All
    three heads only ever changed by this one prefix, so a single rename covers
    every legacy checkpoint. Non-``lstm`` keys (head, to_index, g_monotone, ...)
    are unchanged.
    """

    return {
        (f"encoder.{key}" if key.startswith("lstm.") else key): value
        for key, value in state_dict.items()
    }


def _write_checkpoint(path: str, payload: dict[str, object]) -> None:
    """Write a checkpoint, creating the parent directory if needed."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_lstm_engressor_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
    model: "nn.Module | Callable[[int, int, LSTMEngressionConfig], nn.Module] | None" = None,
) -> LSTMEngressor:
    """Reload a fitted ``LSTMEngressor`` checkpoint.

    ``model`` mirrors ``LSTMEngressionConfig.model``: for a checkpoint saved from
    a custom architecture, pass the same ``nn.Module`` (freshly constructed with
    matching dims) or factory so the weights can be loaded back. Built-in heads
    reload with ``model=None``.
    """

    load_device = torch.device(device or "cpu")
    checkpoint = torch.load(path, map_location=load_device)
    config_dict = dict(checkpoint["config"])
    config_dict["device"] = str(load_device)
    config = LSTMEngressionConfig(**config_dict)
    if model is not None:
        config = replace(config, model=model)
    net = build_lstm_model(
        config,
        input_dim=int(checkpoint["input_dim"]),
        out_dim=int(checkpoint["out_dim"]),
    ).to(load_device)
    state_dict = checkpoint["model_state_dict"]
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        # Fall back to the legacy (pre-encoder-refactor) key layout.
        net.load_state_dict(_migrate_legacy_state_dict(state_dict))
    net.eval()
    return LSTMEngressor(
        model=net,
        config=config,
        x_mean=checkpoint["x_mean"],
        x_std=checkpoint["x_std"],
        y_mean=checkpoint["y_mean"],
        y_std=checkpoint["y_std"],
    )
