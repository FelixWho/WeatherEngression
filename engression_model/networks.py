"""Starter neural generators for engression experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


class PreAdditiveEngressionMLP(nn.Module):
    """MLP generator with a pre-additive architectural bias.

    The input window ``x`` is flattened, projected, and added to a projection
    of the noise seed ``eps``:

        z = W_x x + W_eps eps.

    A nonlinear body maps ``z`` to the response, with a linear skip connection
    from ``x`` for a stable baseline extrapolation path.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        noise_dim: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.input_dim = math.prod(input_shape)
        self.noise_dim = noise_dim

        self.x_projection = nn.Linear(self.input_dim, latent_dim)
        self.noise_projection = nn.Linear(noise_dim, latent_dim, bias=False)

        layers: list[nn.Module] = []
        dim = latent_dim
        for _ in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.body = nn.Sequential(*layers)
        self.linear_skip = nn.Linear(self.input_dim, output_dim)

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        z = self.x_projection(x_flat) + self.noise_projection(eps)
        return self.body(z) + self.linear_skip(x_flat)

