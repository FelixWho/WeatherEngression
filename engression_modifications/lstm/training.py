"""The training loop: fit an LSTM engression model with the energy loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from engression.engression import energy_loss_two_sample

from .checkpoint import _checkpoint_payload, _write_checkpoint
from .config import LSTMEngressionConfig
from .engressor import LSTMEngressor
from .generators import build_lstm_model
from .preprocessing import standardization_stats, validate_sequence_x, validate_y


def fit_lstm_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: LSTMEngressionConfig | None = None,
) -> LSTMEngressor:
    """Fit LSTM engression on unflattened lag-window inputs."""

    if config is None:
        config = LSTMEngressionConfig()
    x = validate_sequence_x(x)
    y = validate_y(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")

    device = torch.device(config.device) if isinstance(config.device, str) else config.device
    x_mean, x_std, y_mean, y_std = standardization_stats(x, y, config.standardize)
    x_train = ((x - x_mean) / x_std if config.standardize else x).to(device)
    y_train = ((y - y_mean) / y_std if config.standardize else y).to(device)

    model = build_lstm_model(config, input_dim=x.shape[2], out_dim=y.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    batch_size = config.batch_size or len(x_train)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    log_every = config.print_every_nepoch or 10
    checkpoint_every = config.checkpoint_every_nepoch
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError("checkpoint_every_nepoch must be >= 1 when provided")

    model.train()
    best_loss = float("inf")
    best_monitor_loss = float("inf")
    epochs_since_improve = 0
    for epoch_idx in range(config.num_epochs):
        total_loss = 0.0
        total_fit = 0.0
        total_spread = 0.0
        total_rows = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_sample1 = model(x_batch)
            y_sample2 = model(x_batch)
            loss = energy_loss_two_sample(
                y_batch,
                y_sample1,
                y_sample2,
                beta=config.beta,
                verbose=False,
            )
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            # Decompose the energy score for monitoring: fit pulls samples toward y,
            # spread rewards conditional variance. A collapsing spread term is the
            # signature of under-dispersion.
            with torch.no_grad():
                fit_term = 0.5 * (
                    (y_batch - y_sample1).norm(dim=1) + (y_batch - y_sample2).norm(dim=1)
                ).mean()
                spread_term = 0.5 * (y_sample1 - y_sample2).norm(dim=1).mean()
            total_loss += float(loss.detach().cpu()) * len(x_batch)
            total_fit += float(fit_term.cpu()) * len(x_batch)
            total_spread += float(spread_term.cpu()) * len(x_batch)
            total_rows += len(x_batch)
        rows = max(1, total_rows)
        mean_loss = total_loss / rows
        mean_fit = total_fit / rows
        mean_spread = total_spread / rows
        epoch = epoch_idx + 1
        checkpoint_due = (
            checkpoint_every is not None
            and config.checkpoint_path is not None
            and (epoch == config.num_epochs or epoch % checkpoint_every == 0)
        )
        if checkpoint_due:
            _write_checkpoint(
                config.checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    train_loss=mean_loss,
                    input_dim=x.shape[2],
                    out_dim=y.shape[1],
                    x_mean=x_mean,
                    x_std=x_std,
                    y_mean=y_mean,
                    y_std=y_std,
                ),
            )
        if config.checkpoint_best_path is not None and mean_loss < best_loss:
            best_loss = mean_loss
            _write_checkpoint(
                config.checkpoint_best_path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    train_loss=mean_loss,
                    input_dim=x.shape[2],
                    out_dim=y.shape[1],
                    x_mean=x_mean,
                    x_std=x_std,
                    y_mean=y_mean,
                    y_std=y_std,
                ),
            )
        if epoch == 1 or epoch % log_every == 0 or epoch == config.num_epochs:
            print(
                f"[epoch {epoch:>4}/{config.num_epochs}] "
                f"energy-loss {mean_loss:.4f} | fit {mean_fit:.4f} | spread {mean_spread:.4f}",
                flush=True,
            )

        if config.early_stop_patience is not None:
            if mean_loss < best_monitor_loss - config.early_stop_min_delta:
                best_monitor_loss = mean_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= config.early_stop_patience:
                    print(
                        f"[epoch {epoch}] early stop: training energy loss has not improved "
                        f"by >{config.early_stop_min_delta} for {config.early_stop_patience} epochs",
                        flush=True,
                    )
                    break

    return LSTMEngressor(
        model=model,
        config=config,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
