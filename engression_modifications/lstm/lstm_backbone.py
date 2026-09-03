"""The shared LSTM backbone.

The backbone reads trajectories shaped ``(batch_size, sequence_length,
feature_dim)`` and returns summaries shaped ``(batch_size, hidden_dim)``. This
fixed-size summary is what every generator head conditions on.
"""

from __future__ import annotations

import torch
from torch import nn


class DeterministicLSTMEncoder(nn.Module):
    """Read a trajectory and return its summary vector h(X).

    ``nn.LSTM`` walks the ``(sequence_length, feature_dim)`` sequence step by step; we keep
    only the final hidden state of the last layer, which summarizes the whole
    history at fixed size. ``output_dim`` is the width of that summary, which the heads
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
        self.output_dim = hidden_dim          # scalar width of the summary the heads consume
        self.lstm = nn.LSTM(
            input_size=input_dim,             # feature_dim: covariates at each timestep
            hidden_size=hidden_dim,           # width of each hidden/cell state
            num_layers=num_layers,            # number of stacked LSTM layers
            # PyTorch only applies inter-layer dropout when num_layers > 1.
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, sequence_length, feature_dim)
    ) -> torch.Tensor:
        # x:                (batch_size, sequence_length, feature_dim)
        # self.lstm(x) -> (output, (h_n, c_n)):
        #   output:         (batch_size, sequence_length, hidden_dim) -- discarded
        #   h_n = hidden:   (num_layers, batch_size, hidden_dim)      -- kept
        #   c_n:            (num_layers, batch_size, hidden_dim)      -- discarded
        _, (hidden, _) = self.lstm(x)
        # hidden[-1]: final state from the top layer -> (batch_size, hidden_dim)
        return hidden[-1]
