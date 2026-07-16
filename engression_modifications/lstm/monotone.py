"""Monotone building blocks for the pre-additive (engression-paper) head.

These implement a strictly increasing, twice-differentiable ``g`` -- the map the
pre-ANM extrapolation theory (Assumption A3) needs. Used only by the
pre-additive generator head; nothing else depends on them.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PositiveLinear(nn.Module):
    """Linear layer with non-negative weights (softplus-constrained).

    A composition of these with a monotone activation is non-decreasing in every
    input coordinate, which is what the engression paper's pre-ANM theory needs
    for ``g`` (Assumption A3: strictly monotone, twice differentiable).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        # softplus(-2) ~ 0.13, so weights start small and positive.
        self.weight_raw = nn.Parameter(torch.empty(out_dim, in_dim).normal_(-2.0, 0.1))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.weight_raw), self.bias)


class MonotonePositiveMLP(nn.Module):
    """A strictly increasing, twice-differentiable network ``g``.

    Positive-weight layers + softplus activations give a non-decreasing map; a
    positive linear skip from input to output makes it strictly increasing.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layer: int) -> None:
        super().__init__()
        num_layer = max(1, num_layer)
        dims = [in_dim] + [hidden_dim] * (num_layer - 1) + [out_dim]
        self.layers = nn.ModuleList(
            PositiveLinear(dims[i], dims[i + 1]) for i in range(num_layer)
        )
        self.act = nn.Softplus()
        self.skip = PositiveLinear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if index < len(self.layers) - 1:
                h = self.act(h)
        return h + self.skip(x)
