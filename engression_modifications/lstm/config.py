"""Configuration for the sequence-native LSTM engression model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMEngressionConfig:
    """Configuration for a sequence-native engression model."""

    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    lstm_hidden_dim: int | None = None
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    add_bn: bool = False
    # Custom generator override. When set, this REPLACES the built-in head
    # selection below, so you can plug in any architecture that implements
    # ``nn.Module``. Two forms are accepted:
    #   - an ``nn.Module`` instance, used as-is (you are responsible for its
    #     input/output dims matching the data);
    #   - a factory ``callable(input_dim, out_dim, config) -> nn.Module``, called
    #     at fit time so the module can size itself to the data.
    # Leave as None to use the built-in default / StoNet / pre-additive heads
    # chosen by the flags below. Not serialized into checkpoints; pass the same
    # model/factory to ``load_lstm_engressor_checkpoint`` to reload a custom model.
    model: "nn.Module | Callable[[int, int, LSTMEngressionConfig], nn.Module] | None" = None
    # Pre-additive (engression-paper) mode: Y = g(phi(X) + eta) with g monotone.
    # When False, the original concat-noise head is used.
    pre_additive: bool = False
    index_dim: int | None = None
    monotone_hidden_dim: int | None = None
    monotone_num_layer: int = 2
    # Loose mode: feed h(X) into the package's own StoNet (noise at input + every
    # layer, unconstrained, so monotonicity is hoped for rather than forced). The match
    # to the upstream engression model, but sequence-native.
    stonet_head: bool = False
    noise_all_layer: bool = True
    resblock: bool = False
    # Noise-in-the-encoder heads: inject fresh noise into the SEQUENCE (not the MLP
    # head) before encoding, then a deterministic head maps h(X) -> Y. Mutually
    # exclusive with each other and with stonet_head / pre_additive.
    appending_noise: bool = False   # append a fresh noise timestep -> LSTM reads (B, S+1, F)
    additive_noise: bool = False    # add fresh noise onto the final timestep of x
    # Recurrence-noise heads: inject noise into the LSTM computation itself (input
    # stream, initial state, or per-step hidden state) rather than one timestep.
    # Mutually exclusive with each other and with the flags above.
    per_timestep_noise: bool = False    # concat fresh noise to EVERY timestep -> (B, S, F+noise_dim)
    global_latent_noise: bool = False   # one latent z per sample, broadcast to all timesteps
    stochastic_init_noise: bool = False # seed (h0, c0) from noise; input unchanged
    recurrent_state_noise: bool = False # per-step noise added to the hidden state (LSTMCell loop)
    beta: float = 1.0
    lr: float = 0.003
    weight_decay: float = 0.0
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    grad_clip: float | None = 5.0
    device: str | torch.device = "cpu"
    verbose: bool = False
    print_every_nepoch: int | None = None
    checkpoint_path: str | None = None
    checkpoint_best_path: str | None = None
    checkpoint_every_nepoch: int | None = None
    # Early stopping on the TRAINING energy loss: stop if the mean epoch loss has
    # not improved by more than ``early_stop_min_delta`` for ``early_stop_patience``
    # epochs. None disables it. NOTE: this monitors training loss, which rewards
    # sharpness. It saves compute but does not by itself fix under-dispersion; a
    # validation-based criterion is the proper calibration guard (future work).
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-4

    def __post_init__(self) -> None:
        """Reject incompatible built-in generator selections."""

        architecture_flags = {
            "pre_additive": self.pre_additive,
            "stonet_head": self.stonet_head,
            "appending_noise": self.appending_noise,
            "additive_noise": self.additive_noise,
            "per_timestep_noise": self.per_timestep_noise,
            "global_latent_noise": self.global_latent_noise,
            "stochastic_init_noise": self.stochastic_init_noise,
            "recurrent_state_noise": self.recurrent_state_noise,
        }
        enabled = [name for name, value in architecture_flags.items() if value]
        if len(enabled) > 1:
            raise ValueError(
                "select at most one built-in model architecture; "
                f"got {', '.join(enabled)}"
            )
