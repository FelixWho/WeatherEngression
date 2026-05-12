"""Energy-score loss for engression-style generators."""

from __future__ import annotations

import torch


def energy_loss(y_true: torch.Tensor, y_generated: torch.Tensor) -> torch.Tensor:
    """Compute the empirical energy loss.

    Args:
        y_true: Tensor with shape ``(batch, y_dim)`` or ``(batch,)``.
        y_generated: Tensor with shape ``(batch, n_samples, y_dim)`` or
            ``(batch, n_samples)``.

    Returns:
        Scalar loss approximating

        E[||Y - g(X, eps)|| - 0.5 ||g(X, eps) - g(X, eps')||].
    """
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_generated.ndim == 2:
        y_generated = y_generated.unsqueeze(-1)

    residual_distance = torch.linalg.vector_norm(
        y_true[:, None, :] - y_generated,
        dim=-1,
    ).mean()

    pairwise_distance = torch.cdist(y_generated, y_generated, p=2).mean()
    return residual_distance - 0.5 * pairwise_distance


@torch.no_grad()
def sample_predictive_distribution(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int,
    noise_dim: int,
) -> torch.Tensor:
    """Sample ``g_theta(x, eps)`` repeatedly for each row of ``x``."""
    batch = x.shape[0]
    eps = torch.randn(batch, n_samples, noise_dim, device=x.device, dtype=x.dtype)
    x_rep = x[:, None, ...].expand(batch, n_samples, *x.shape[1:])
    flat_x = x_rep.reshape(batch * n_samples, *x.shape[1:])
    flat_eps = eps.reshape(batch * n_samples, noise_dim)
    y = model(flat_x, flat_eps)
    return y.reshape(batch, n_samples, -1)

