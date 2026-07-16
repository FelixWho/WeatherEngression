"""The shared LSTM backbone.

The backbone reads one trajectory ``x`` of shape ``(B, S, F)`` and returns its
summary ``h(X)`` of shape ``(B, H)`` -- the fixed-size vector every generator
head conditions on. It lives here in one place and is composed into each
generator (see ``generators.py``).

Dimension key used throughout: B = batch, S = seq_len, F = input_dim (features
per timestep), H = hidden_dim, L = num_layers.
"""

from __future__ import annotations

import torch
from torch import nn


class DeterministicLSTMEncoder(nn.Module):
    """Read a trajectory and return its summary vector h(X).

    ``nn.LSTM`` walks the ``(seq_len, n_features)`` sequence step by step; we keep
    only the final hidden state of the last layer -- a fixed-size summary of the
    whole history. ``output_dim`` is the width of that summary, which the heads
    use to size their input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim          # scalar: width of h(X) the heads consume
        self.lstm = nn.LSTM(
            input_size=input_dim,             # F: features per timestep
            hidden_size=hidden_dim,           # H: width of the hidden/cell state
            num_layers=num_layers,            # L: stacked LSTM layers
            # PyTorch only applies inter-layer dropout when num_layers > 1.
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:                (batch, seq_len, input_dim)   i.e. (B, S, F)
        # self.lstm(x) -> (output, (h_n, c_n)):
        #   output:         (B, S, H)   hidden state at every timestep  -- discarded
        #   h_n = hidden:   (L, B, H)   final hidden state per layer     -- kept
        #   c_n:            (L, B, H)   final cell state per layer        -- discarded
        _, (hidden, _) = self.lstm(x)
        # hidden[-1]: top layer's final hidden state -> (B, H) == (batch, hidden_dim)
        return hidden[-1]
