"""The training loop: fit an LSTM engression model with the energy loss.

Early stopping monitors the HELD-OUT energy loss when ``fit_lstm_engression`` is
given ``x_val``/``y_val``, and the training loss otherwise. The training loss
rewards sharpness, so stopping on it tends to leave the model under-dispersed;
a validation set is the proper guard. Passing no validation set preserves the
original behavior for callers that have not been updated.
"""

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


def _mean_energy_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    batch_size: int,
) -> float:
    """Mean two-sample energy loss over a held-out set, in eval mode.

    Batched so a large validation set cannot exhaust GPU memory. The two sample
    draws are fresh each call, so this carries some Monte-Carlo noise; averaging
    over the whole set keeps it small relative to the epoch-to-epoch signal.
    """

    was_training = model.training
    model.eval()
    total = 0.0
    rows = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            x_batch = x[start : start + batch_size]
            y_batch = y[start : start + batch_size]
            loss = energy_loss_two_sample(
                y_batch, model(x_batch), model(x_batch), beta=beta, verbose=False
            )
            total += float(loss) * len(x_batch)
            rows += len(x_batch)
    if was_training:
        model.train()
    return total / max(1, rows)


def fit_lstm_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: LSTMEngressionConfig | None = None,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
) -> LSTMEngressor:
    """Fit LSTM engression on unflattened lag-window inputs.

    When ``x_val``/``y_val`` are given, early stopping and best-checkpoint
    selection both track the validation energy loss instead of the training loss.
    The caller owns the split: these rows must not appear in ``x``/``y``, and for
    overlapping trajectory data they must be separated in time, not sampled at
    random.
    """

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

    if (x_val is None) != (y_val is None):
        raise ValueError("pass both x_val and y_val, or neither")
    if x_val is not None:
        x_val = validate_sequence_x(x_val)
        y_val = validate_y(y_val)
        if x_val.shape[0] != y_val.shape[0]:
            raise ValueError("x_val and y_val must have the same number of rows")
        if x_val.shape[2] != x.shape[2] or y_val.shape[1] != y.shape[1]:
            raise ValueError("validation tensors must match the training feature/target dims")
        # Standardize with the TRAINING moments: the validation set must be scored
        # through exactly the transform the model was fitted under.
        x_val = ((x_val - x_mean) / x_std if config.standardize else x_val).to(device)
        y_val = ((y_val - y_mean) / y_std if config.standardize else y_val).to(device)

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
            # Population energy loss = E[ ||Y - g(X, eps)||_2^beta
            #   - 0.5 * ||g(X, eps) - g(X, eps_prime)||_2^beta ],
            # where the expectation is over training (X, Y) and independent
            # noise draws eps and eps_prime.
            # Two-sample estimate = mean over minibatch rows of:
            #   0.5 * ||y - y_sample1||_2^beta
            #   + 0.5 * ||y - y_sample2||_2^beta
            #   - 0.5 * ||y_sample1 - y_sample2||_2^beta.
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
            # no_grad saved computation and memory here.
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

        # What early stopping and best-checkpoint selection watch. Held-out loss
        # when a validation set was supplied, training loss otherwise.
        val_loss = (
            None
            if x_val is None
            else _mean_energy_loss(model, x_val, y_val, config.beta, batch_size)
        )
        monitor_loss = mean_loss if val_loss is None else val_loss
        monitor_name = "training" if val_loss is None else "validation"

        # Save model weight checkpoint
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
                    val_loss=val_loss,
                ),
            )
        if config.checkpoint_best_path is not None and monitor_loss < best_loss:
            best_loss = monitor_loss
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
                    val_loss=val_loss,
                ),
            )

        # Log
        if epoch == 1 or epoch % log_every == 0 or epoch == config.num_epochs:
            val_text = "" if val_loss is None else f" | val {val_loss:.4f}"
            print(
                f"[epoch {epoch:>4}/{config.num_epochs}] "
                f"energy-loss {mean_loss:.4f} | fit {mean_fit:.4f} | spread {mean_spread:.4f}"
                f"{val_text}",
                flush=True,
            )

        # Early stop
        if config.early_stop_patience is not None:
            if monitor_loss < best_monitor_loss - config.early_stop_min_delta:
                best_monitor_loss = monitor_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= config.early_stop_patience:
                    print(
                        f"[epoch {epoch}] early stop: {monitor_name} energy loss has not improved "
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
