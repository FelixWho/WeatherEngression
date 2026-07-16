"""The stochastic generators: one class per head, all sharing a ``DeterministicLSTMEncoder``.

Each generator maps one trajectory + a noise draw to one sample of the target.
They all reuse the same encoder building block (``lstm_backbone.py``) and differ ONLY
in how the head turns h(X) + noise into a sample -- so each head is its own
class, not a branch inside one model. The ``build_lstm_model`` factory at the
bottom is the single place that picks a class from the config flags.

Noise-in-the-HEAD variants (share one deterministic encoder):
  - ``DefaultHeadGenerator``  : noise concatenated ONCE beside h(X), plain ReLU MLP.
  - ``StoNetHeadGenerator``   : package StoNet, noise injected at every layer.
  - ``PreAdditiveGenerator``  : Y = g(phi(X) + eta), monotone g (engression pre-ANM).
  - ``StochasticAppending*``  : append a fresh noise timestep -> (B, S+1, F).
  - ``StochasticAdditive*``   : add fresh noise onto the final timestep.

Noise-in-the-ENCODER/recurrence variants (build their own recurrent module):
  - ``PerTimestepNoise*``     : #1 fresh noise concatenated to EVERY timestep.
  - ``GlobalLatent*``         : #2 one z per sample, same z broadcast to every step (CVAE form).
  - ``StochasticInitState*``  : #3 noise seeds the initial (h0, c0) only.
  - ``RecurrentStateNoise*``  : #4 fresh noise added to the hidden state at every step (accumulates).

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
    for _ in range(max(1, config.num_layer)):     # num_layer hidden blocks
        layers.append(nn.Linear(dim, config.hidden_dim))
        if config.add_bn:                          # optional BatchNorm (off by default)
            layers.append(nn.BatchNorm1d(config.hidden_dim))
        layers.append(nn.ReLU())
        dim = config.hidden_dim
    layers.append(nn.Linear(dim, out_dim))         # final projection to the target
    return nn.Sequential(*layers)


class StochasticAppendingLSTMGenerator(LSTMGenerator):
    """Noise-in-the-encoder generator: APPEND a fresh noise timestep.

    Before encoding, a random ``(B, 1, F)`` "timestep" is concatenated to the end
    of the sequence, so the LSTM reads ``(B, S+1, F)`` and its final hidden state
    h(X) becomes a sample. A deterministic MLP head then maps h(X) -> Y; all the
    stochasticity lives in the appended noise, not the head. ``encode`` still runs
    on the RAW sequence, so the OOS embedding stays deterministic.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) ; eps: (B, 1, F) -- one fresh noise timestep per sample
        eps = torch.randn(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, eps], dim=1)          # (B, S+1, F), out-of-place -- x untouched
        return self.head(self.encode(x_aug))        # encode -> (B, H) -> head -> (B, out_dim)


class StochasticAdditiveLSTMGenerator(LSTMGenerator):
    """Noise-in-the-encoder generator: ADD fresh noise onto the final timestep.

    Before encoding, a random ``(B, F)`` vector is added to the last timestep of
    the sequence (length unchanged), making h(X) a sample. A deterministic MLP
    head maps h(X) -> Y. The addition is out-of-place, so ``x`` is never mutated
    and the two training draws stay independent. ``encode`` runs on the RAW
    sequence, so the OOS embedding stays deterministic.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F) ; eps: (B, F) -- fresh noise on the final timestep only
        eps = torch.randn(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        noise = torch.zeros_like(x)                 # (B, S, F), a private scratch tensor
        noise[:, -1, :] = eps                       # only the last timestep is noised
        x_aug = x + noise                           # out-of-place; x untouched
        return self.head(self.encode(x_aug))        # encode -> (B, H) -> head -> (B, out_dim)



class PerTimestepNoiseLSTMGenerator(LSTMGenerator):
    """#1 -- concat FRESH noise to every timestep (the StoNet analogue for sequences).

    A ``(B, S, k)`` noise block is concatenated to the features at every step, so
    the LSTM reads ``(B, S, F+k)``. Many injection points make this the most
    collapse-resistant of the sequence-noise schemes. A deterministic MLP head
    maps h(X) -> Y; ``encode`` zeros the noise channels for a stable OOS embedding.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        encoder = DeterministicLSTMEncoder(
            input_dim=input_dim + config.noise_dim,   # F + k
            hidden_dim=config.lstm_hidden_dim or config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
        super().__init__(encoder)
        self.noise_dim = config.noise_dim
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fresh, INDEPENDENT noise per timestep -> (B, S, k)
        eps = torch.randn(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, eps], dim=2)          # (B, S, F+k)
        return self.head(self.encoder(x_aug))       # encode -> (B, H) -> head -> (B, out_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # deterministic embedding: zero the noise channels
        zeros = torch.zeros(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        return self.encoder(torch.cat([x, zeros], dim=2))


class GlobalLatentLSTMGenerator(LSTMGenerator):
    """#2 -- one latent ``z`` per sample, broadcast to EVERY timestep.

    Draw ``z ~ N(0, I_k)`` once per sequence and concat the SAME ``z`` onto every
    frame, so the LSTM reads ``(B, S, F+k)``. Because ``z`` is re-presented at
    every step it cannot be forgotten -- the clean conditional-generator (CVAE)
    form. Deterministic MLP head; ``encode`` uses ``z = 0``.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        encoder = DeterministicLSTMEncoder(
            input_dim=input_dim + config.noise_dim,   # F + k
            hidden_dim=config.lstm_hidden_dim or config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
        super().__init__(encoder)
        self.noise_dim = config.noise_dim
        self.head = _mlp_head(encoder.output_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(x.shape[0], 1, self.noise_dim, device=x.device, dtype=x.dtype)  # one draw / sequence
        z = z.expand(-1, x.shape[1], -1)            # same z broadcast to every timestep -> (B, S, k)
        return self.head(self.encoder(torch.cat([x, z], dim=2)))   # LSTM reads (B, S, F+k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # deterministic embedding: z = 0
        zeros = torch.zeros(x.shape[0], x.shape[1], self.noise_dim, device=x.device, dtype=x.dtype)
        return self.encoder(torch.cat([x, zeros], dim=2))


class StochasticInitStateLSTMGenerator(LSTMGenerator):
    """#3 -- seed the LSTM's initial hidden/cell state from noise.

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
        # z -> initial (h0, c0), each flattened to (num_layers * hidden_dim)
        self.z_to_state = nn.Linear(config.noise_dim, 2 * self.num_layers * hidden_dim)
        self.head = _mlp_head(hidden_dim, out_dim, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        z = torch.randn(b, self.noise_dim, device=x.device, dtype=x.dtype)   # one draw / sequence
        h0, c0 = self.z_to_state(z).chunk(2, dim=1)              # split into (h0, c0), each (B, L*H)
        h0 = h0.view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()  # (L, B, H)
        c0 = c0.view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        _, (hidden, _) = self.encoder.lstm(x, (h0, c0))          # run LSTM from the noisy init state
        return self.head(hidden[-1])                             # top-layer final hidden -> head

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)   # default zero init -> deterministic h(X)


class RecurrentStateNoiseLSTMGenerator(nn.Module):
    """#4 -- inject fresh noise into the hidden state at EVERY timestep.

    A hand-rolled single-layer recurrence (``nn.LSTMCell``): after each step the
    hidden state is perturbed, ``h_t <- h_t + softplus(scale) * eps_t`` with fresh
    ``eps_t`` per step, so randomness ACCUMULATES through time (a stochastic-RNN /
    VRNN flavor). Highest ceiling, most expensive (Python loop over timesteps +
    BPTT). Single layer only -- ``lstm_num_layers`` is ignored here. ``encode``
    runs the same loop with the noise scale forced off.

    Not an ``LSTMGenerator`` subclass: the recurrence uses a cell, not the shared
    ``DeterministicLSTMEncoder``.
    """

    def __init__(self, input_dim: int, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__()
        hidden_dim = config.lstm_hidden_dim or config.hidden_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTMCell(input_dim, hidden_dim)          # single-layer recurrence
        self.noise_log_scale = nn.Parameter(torch.zeros(hidden_dim))  # learned per-unit noise std
        self.head = _mlp_head(hidden_dim, out_dim, config)

    def _run(self, x: torch.Tensor, noisy: bool) -> torch.Tensor:
        h = x.new_zeros(x.shape[0], self.hidden_dim)   # (B, H) initial hidden state
        c = x.new_zeros(x.shape[0], self.hidden_dim)   # (B, H) initial cell state
        scale = F.softplus(self.noise_log_scale)       # (H,) positive noise std, constant across t
        for t in range(x.shape[1]):                    # manual unroll over the S timesteps
            h, c = self.cell(x[:, t, :], (h, c))       # one LSTMCell step
            if noisy:
                h = h + scale * torch.randn_like(h)   # perturb h; out-of-place, fresh noise per step, accumulates
        return h                                       # (B, H) final hidden state only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._run(x, noisy=True))    # noisy run -> head -> sample

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, noisy=False)   # noise off -> deterministic embedding


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
        h = self.encode(x)                             # deterministic summary h(X), (B, H)
        noise = torch.randn(x.shape[0], self.noise_dim, device=x.device, dtype=x.dtype)  # one draw beside h
        return self.head(torch.cat([h, noise], dim=1))  # g([h(X), noise]) -> (B, out_dim)


class StoNetHeadGenerator(LSTMGenerator):
    """Feed h(X) into the package's StoNet (the "loose" upstream generator).

    Plain ReLU MLP, but fresh Gaussian noise is mixed in at the INPUT and at
    EVERY hidden layer. Injecting noise repeatedly gives the network many
    independent chances to turn randomness into output spread -- empirically the
    strongest guard against collapsing to a point predictor. No monotonicity is
    imposed, so it carries none of the engression-paper extrapolation theory.
    """

    def __init__(self, encoder: DeterministicLSTMEncoder, out_dim: int, config: LSTMEngressionConfig) -> None:
        super().__init__(encoder)
        # num_layer >= 2 so there is at least an input + output StoLayer.
        self.head = StoNet(
            in_dim=encoder.output_dim,       # consumes the trajectory summary h(X)
            out_dim=out_dim,                 # emits one target value per call
            num_layer=max(2, config.num_layer),
            hidden_dim=config.hidden_dim,
            noise_dim=config.noise_dim,
            add_bn=config.add_bn,
            resblock=config.resblock,
            noise_all_layer=config.noise_all_layer,
            verbose=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # StoNet draws and injects its own noise internally, so just hand it h(X).
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
        phi = self.to_index(self.encode(x))            # h(X) -> low-dim index phi(X)
        eps = torch.randn(x.shape[0], self.index_dim, device=x.device, dtype=x.dtype)
        eta = F.softplus(self.noise_log_scale) * eps + self.noise_bias   # affine-Gaussian jitter on the index
        return self.g_monotone(phi + eta)              # monotone g squashes phi + eta -> Y


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
    pre-additive / default). This factory is the ONLY place that maps flags to a
    class -- the generators themselves never branch on the config.
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

    # Recurrence-noise variants build their OWN recurrent module (they change the
    # LSTM's input width or initial state, or use a cell), so they take input_dim
    # and are dispatched before the shared encoder is constructed.
    if config.per_timestep_noise:
        return PerTimestepNoiseLSTMGenerator(input_dim, out_dim, config)
    if config.global_latent_noise:
        return GlobalLatentLSTMGenerator(input_dim, out_dim, config)
    if config.stochastic_init_noise:
        return StochasticInitStateLSTMGenerator(input_dim, out_dim, config)
    if config.recurrent_state_noise:
        return RecurrentStateNoiseLSTMGenerator(input_dim, out_dim, config)

    # Head-noise variants share one deterministic encoder.
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
