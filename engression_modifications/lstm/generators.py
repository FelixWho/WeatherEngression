"""The stochastic generators: one class per head, all sharing a ``DeterministicLSTMEncoder``.

Each generator maps one trajectory + a noise draw to one sample of the target.
They all reuse the same encoder building block (``lstm_backbone.py``) and differ only
in how the head turns h(X) + noise into a sample, so each head is its own
class rather than a branch inside one model. The ``build_lstm_model`` factory at the
bottom is the single place that picks a class from the config flags.

Noise-in-the-HEAD variants (share one deterministic encoder):
  - ``DefaultHeadGenerator``  : noise concatenated ONCE beside h(X), plain ReLU MLP.
  - ``StoNetHeadGenerator``   : package StoNet, noise injected at every layer.
  - ``PreAdditiveGenerator``  : Y = g(phi(X) + eta), monotone g (engression pre-ANM).
  - ``StochasticAppending*``  : append a fresh noise timestep to the sequence.
  - ``StochasticAdditive*``   : add fresh noise onto the final timestep.

Noise-in-the-ENCODER/recurrence variants (build their own recurrent module):
  - ``PerTimestepNoise*``     : fresh noise concatenated to every timestep.
  - ``GlobalLatent*``         : one latent draw per sample, broadcast to every timestep.
  - ``StochasticInitState*``  : noise seeds only the initial hidden and cell states.
  - ``RecurrentStateNoise*``  : fresh noise added to the hidden state at every timestep.

``build_lstm_model`` at the bottom is the single place that maps config flags to a class.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from engression.models import StoNet

from .config import LSTMEngressionConfig
from .lstm_backbone import DeterministicLSTMEncoder
from .monotone import MonotonePositiveMLP


class LSTMGenerator(nn.Module):
    """Base for every LSTM engression generator: a shared encoder + a head.

    Subclasses build their own head in ``__init__`` and implement ``forward``.
    ``encode`` exposes h(X) for the OOS diagnostics and is shared by all heads.
    Calling the model draws fresh noise, so repeated calls on the same x trace
    out the conditional distribution P(Y | X).
    """

    def __init__(self, encoder: DeterministicLSTMEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the trajectory summary h(X); shape (batch, encoder.output_dim)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


def _mlp_head(in_dim: int, out_dim: int, config: LSTMEngressionConfig) -> nn.Sequential:
    """Plain ReLU MLP mapping a feature vector to the target.

    Deterministic on purpose: the stochastic-encoder generators inject all their
    noise into the sequence, so the head draws none of its own.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, config.num_layer)):     # Build the requested number of hidden blocks.
        layers.append(nn.Linear(dim, config.hidden_dim))
        if config.add_bn:                          # Batch normalization is optional and off by default.
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        dim = config.hidden_dim
    layers.append(nn.Linear(dim, out_dim))         # Project the final hidden features to the target.
    return nn.Sequential(*layers)


class StochasticAppendingLSTMGenerator(LSTMGenerator):
    """Noise-in-the-encoder generator: APPEND a fresh noise timestep.

    Before encoding, a random ``(batch_size, 1, feature_dim)`` timestep is appended
    to each sequence. The LSTM therefore reads ``(batch_size, sequence_length + 1,
    feature_dim)`` and its final hidden state
    h(X) becomes a sample. A deterministic MLP head then maps h(X) -> Y; all the
    stochasticity lives in the appended noise, not the head. ``encode`` still runs
    on the RAW sequence, so the OOS embedding stays deterministic.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Draw one noise timestep per trajectory: (batch_size, 1, feature_dim).
        eps = torch.randn(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, eps], dim=1)          # Add it without modifying the original input.
        return self.head(self.encode(x_aug))        # Encode the augmented sequence, then predict the target.


class StochasticAdditiveLSTMGenerator(LSTMGenerator):
    """Noise-in-the-encoder generator: ADD fresh noise onto the final timestep.

    Before encoding, a random ``(batch_size, feature_dim)`` vector is added to the last timestep of
    the sequence (length unchanged), making h(X) a sample. A deterministic MLP
    head maps h(X) -> Y. The addition is out-of-place, so ``x`` is never mutated
    and the two training draws stay independent. ``encode`` runs on the RAW
    sequence, so the OOS embedding stays deterministic.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Draw fresh noise for the final timestep: (batch_size, feature_dim).
        eps = torch.randn(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        noise = torch.zeros_like(x)                 # Keep the noise in a separate tensor.
        noise[:, -1, :] = eps                       # Perturb only the final timestep.
        x_aug = x + noise                           # Keep the caller's input unchanged.
        return self.head(self.encode(x_aug))        # Encode the perturbed sequence, then predict the target.



class PerTimestepNoiseLSTMGenerator(LSTMGenerator):
    """Concatenate fresh noise onto every timestep, the sequence answer to StoNet.

    A ``(batch_size, sequence_length, noise_dim)`` noise block is concatenated to
    the features at every step, so the LSTM reads ``(batch_size, sequence_length,
    feature_dim + noise_dim)``. Many injection points make this the most
    collapse-resistant of the sequence-noise schemes. A deterministic MLP head
    maps h(X) -> Y; ``encode`` zeros the noise channels for a stable OOS embedding.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        encoder = DeterministicLSTMEncoder(
            input_dim=input_dim + config.noise_dim,   # Input covariates plus noise channels.
            hidden_dim=config.lstm_hidden_dim or config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
        super().__init__(encoder)
        self.noise_dim = config.noise_dim
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Draw independent noise for every trajectory, timestep, and noise channel.
        eps = torch.randn(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, eps], dim=2)          # Append noise channels to the covariates.
        return self.head(self.encoder(x_aug))       # Encode the noisy sequence, then predict the target.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Use zero noise channels so the OOS embedding is deterministic.
        zeros = torch.zeros(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        return self.encoder(torch.cat([x, zeros], dim=2))


class GlobalLatentLSTMGenerator(LSTMGenerator):
    """Draw one latent ``z`` per sample and reuse it at every timestep.

    Draw ``z ~ N(0, I_k)`` once per sequence and concat that same ``z`` onto every
    frame, so the LSTM reads ``(batch_size, sequence_length, feature_dim +
    noise_dim)``. Because ``z`` is re-presented at
    every step it cannot be forgotten. This is the clean conditional-generator (CVAE)
    form. Deterministic MLP head; ``encode`` uses ``z = 0``.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        encoder = DeterministicLSTMEncoder(
            input_dim=input_dim + config.noise_dim,   # Input covariates plus latent channels.
            hidden_dim=config.lstm_hidden_dim or config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
        super().__init__(encoder)
        self.noise_dim = config.noise_dim
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(x.shape[0], 1, self.noise_dim, device=x.device, dtype=x.dtype)  # Draw one latent vector per trajectory.
        z = z.expand(-1, x.shape[1], -1)            # Repeat that same vector at every timestep.
        return self.head(self.encoder(torch.cat([x, z], dim=2)))   # Encode covariates and latent channels together.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Use zero latent channels so the OOS embedding is deterministic.
        zeros = torch.zeros(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        return self.encoder(torch.cat([x, zeros], dim=2))


class StochasticInitStateLSTMGenerator(LSTMGenerator):
    """Seed the LSTM's initial hidden and cell states from noise.

    ``z ~ N(0, I_k)`` is mapped to ``(h0, c0)``; the input sequence is unchanged.
    Truly recurrence-level and elegant, but a SINGLE injection at t=0 that a long
    sequence can forget. Deterministic MLP head; ``encode`` uses the default zero
    init, so the OOS embedding stays stable.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        hidden_dim = config.lstm_hidden_dim or config.hidden_dim
        encoder = DeterministicLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
        super().__init__(encoder)
        self.num_layers = config.lstm_num_layers
        self.hidden_dim = hidden_dim
        self.noise_dim = config.noise_dim
        # Map each latent vector to flattened initial hidden and cell states.
        self.z_to_state = nn.Linear(config.noise_dim, 2 * self.num_layers * hidden_dim)
        self.head = _mlp_head(hidden_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = torch.randn(b, self.noise_dim, device=x.device, dtype=x.dtype)   # Draw one latent vector per trajectory.
        h0, c0 = self.z_to_state(z).chunk(2, dim=1)              # Split it into initial hidden and cell states.
        h0 = h0.view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()  # Arrange as PyTorch expects: (num_layers, batch_size, hidden_dim).
        c0 = c0.view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        _, (hidden, _) = self.encoder.lstm(x, (h0, c0))          # Run the LSTM from the noisy initial state.
        return self.head(hidden[-1])                             # Predict from the top layer's final hidden state.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)   # The default zero initialization makes this embedding deterministic.


class RecurrentStateNoiseLSTMGenerator(nn.Module):
    """Add fresh noise to the hidden state at every timestep.

    A hand-rolled single-layer recurrence (``nn.LSTMCell``): after each step the
    hidden state is perturbed, ``h_t <- h_t + softplus(scale) * eps_t`` with fresh
    ``eps_t`` per step, so randomness ACCUMULATES through time (a stochastic-RNN /
    VRNN flavor). Highest ceiling, most expensive (Python loop over timesteps +
    BPTT). Single layer only, so ``lstm_num_layers`` is ignored here. ``encode``
    runs the same loop with the noise scale forced off.

    Not an ``LSTMGenerator`` subclass: the recurrence uses a cell, not the shared
    ``DeterministicLSTMEncoder``.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__()
        hidden_dim = config.lstm_hidden_dim or config.hidden_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)          # Use a single-layer recurrence.
        self.noise_log_scale = nn.Parameter(torch.zeros(hidden_dim))  # Learn one noise standard deviation per hidden unit.
        self.head = _mlp_head(hidden_dim, out_dim, config)

    def _run(self, x: torch.Tensor, noisy: bool) -> torch.Tensor:
        h = x.new_zeros(x.shape[0], self.hidden_dim)   # Start with a zero hidden state.
        c = x.new_zeros(x.shape[0], self.hidden_dim)   # Start with a zero cell state.
        scale = F.softplus(self.noise_log_scale)       # Keep the learned noise scale positive.
        for t in range(x.shape[1]):                    # Unroll the LSTM cell over timesteps.
            h, c = self.cell(x[:, t, :], (h, c))       # Advance one timestep.
            if noisy:
                h = h + scale * torch.randn_like(h)   # Add fresh, accumulating noise without modifying in place.
        return h                                       # Return the final hidden state.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._run(x, noisy=True))    # A noisy recurrence produces one target sample.

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, noisy=False)   # Noise off gives a deterministic embedding.


class DefaultHeadGenerator(LSTMGenerator):
    """Concatenate noise ONCE beside h(X), then a plain ReLU MLP.

    Y_hat = g([h(X), noise]). The noise sits in its own input slots (never added
    to the signal); the MLP is unconstrained.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        self.noise_dim = config.noise_dim
        layers: list[nn.Module] = []
        in_dim = encoder.output_dim + config.noise_dim
        for _ in range(max(1, config.num_layer)):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.add_bn:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)                             # Encode the trajectory into a deterministic summary.
        noise = torch.randn(x.shape[0], self.noise_dim, device=x.device, dtype=x.dtype)  # Draw one noise vector per trajectory.
        return self.head(torch.cat([h, noise], dim=1))  # Feed the summary and noise together into the head.


class StoNetHeadGenerator(LSTMGenerator):
    """Feed h(X) into the package's StoNet (the "loose" upstream generator).

    Plain ReLU MLP, but fresh Gaussian noise is mixed in at the INPUT and at
    every hidden layer. Injecting noise repeatedly gives the network many
    independent chances to turn randomness into output spread, and in practice it is
    the strongest guard against collapsing to a point predictor. No monotonicity is
    imposed, so it carries none of the engression-paper extrapolation theory.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        # Require an input and output StoLayer, even if the config asks for fewer.
        self.head = StoNet(
            in_dim=encoder.output_dim,       # Consume the trajectory summary.
            out_dim=out_dim,                 # Produce the target output.
            num_layer=max(2, config.num_layer),
            hidden_dim=config.hidden_dim,
            noise_dim=config.noise_dim,
            add_bn=config.add_bn,
            resblock=config.resblock,
            noise_all_layer=config.noise_all_layer,
            verbose=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # StoNet creates its own noise internally; it only needs the trajectory summary.
        return self.head(self.encode(x))


class PreAdditiveGenerator(LSTMGenerator):
    """Engression-paper pre-ANM head: Y = g(phi(X) + eta), with g monotone.

    The encoder summary is projected to a low-dim index phi(X); affine-Gaussian
    noise eta is ADDED to it (horizontal jitter); a strictly increasing g squashes
    the sum. This is the only head that carries the paper's extrapolation theory.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        index_dim = config.index_dim or config.noise_dim
        self.index_dim = index_dim
        self.to_index = nn.Linear(encoder.output_dim, index_dim)
        self.noise_log_scale = nn.Parameter(torch.zeros(index_dim))
        self.noise_bias = nn.Parameter(torch.zeros(index_dim))
        self.g_monotone = MonotonePositiveMLP(
            in_dim=index_dim,
            out_dim=out_dim,
            hidden_dim=config.monotone_hidden_dim or config.hidden_dim,
            num_layer=config.monotone_num_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.to_index(self.encode(x))            # Project the trajectory summary to a low-dimensional index.
        eps = torch.randn(x.shape[0], self.index_dim, device=x.device, dtype=x.dtype)
        eta = F.softplus(self.noise_log_scale) * eps + self.noise_bias   # Add learned affine-Gaussian jitter to the index.
        return self.g_monotone(phi + eta)              # Map the perturbed index to the target monotonically.


def build_lstm_model(
    config: LSTMEngressionConfig,
    input_dim: int,
    out_dim: int,
) -> nn.Module:
    """Construct the generator network for a fit.

    ``config.model`` takes precedence over the built-in heads, letting callers
    drop in any ``nn.Module``:

      - an ``nn.Module`` instance is returned as-is (you own the input/output
        dims);
      - a callable is treated as a factory and invoked as
        ``config.model(input_dim, out_dim, config)`` so it can size itself.

    When ``config.model`` is None a shared ``DeterministicLSTMEncoder`` is built and
    wrapped in the generator class selected by the config flags (StoNet /
    pre-additive / default). This factory is the only place that maps flags to a
    class; the generators themselves never branch on the config.
    """

    custom = config.model
    if custom is not None:
        if isinstance(custom, nn.Module):
            return custom
        if callable(custom):
            built = custom(input_dim, out_dim, config)
            if not isinstance(built, nn.Module):
                raise TypeError(
                    "config.model factory must return an nn.Module; "
                    f"got {type(built)!r}"
                )
            return built
        raise TypeError(
            "config.model must be an nn.Module, a factory "
            "callable(input_dim, out_dim, config), or None; "
            f"got {type(custom)!r}"
        )

    # These variants need a specially configured encoder: extra input channels,
    # custom initial states, or a manual recurrent loop. Build them before the
    # ordinary encoder below, which expects only the raw input features.
    if config.per_timestep_noise:
        return PerTimestepNoiseLSTMGenerator(input_dim, out_dim, config)
    if config.global_latent_noise:
        return GlobalLatentLSTMGenerator(input_dim, out_dim, config)
    if config.stochastic_init_noise:
        return StochasticInitStateLSTMGenerator(input_dim, out_dim, config)
    if config.recurrent_state_noise:
        return RecurrentStateNoiseLSTMGenerator(input_dim, out_dim, config)

    # The remaining variants share one deterministic encoder.
    encoder = DeterministicLSTMEncoder(
        input_dim=input_dim,
        hidden_dim=config.lstm_hidden_dim or config.hidden_dim,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    )
    if config.appending_noise:
        return StochasticAppendingLSTMGenerator(encoder, out_dim, config)
    if config.additive_noise:
        return StochasticAdditiveLSTMGenerator(encoder, out_dim, config)
    if config.stonet_head:
        return StoNetHeadGenerator(encoder, out_dim, config)
    if config.pre_additive:
        return PreAdditiveGenerator(encoder, out_dim, config)
    return DefaultHeadGenerator(encoder, out_dim, config)
