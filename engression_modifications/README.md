# Engression Modifications

This folder is for small, explicit changes around the public Python
`engression` package.

The package's high-level `engression(...)` function creates and trains an
`Engressor` immediately. That is convenient, but it hides the optimizer:

```python
self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
```

So regularization such as Adam weight decay is easiest to add by manually
instantiating `Engressor`, replacing the optimizer, and then calling
`.train(...)`.

## The four report architectures (where each one lives)

The real-data report compares four architectures. Three of them are the same
LSTM encoder with a different *head*, so they share one file and are chosen by a
flag rather than living in separate files:

| Report architecture   | File                    | How to select it                          |
| --------------------- | ----------------------- | ----------------------------------------- |
| Vanilla (flat StoNet) | `vanilla_engression.py` | `--engression-model vanilla`              |
| LSTM + default head   | `lstm_engression.py`    | `--engression-model lstm` (no head flag)  |
| LSTM + StoNet head    | `lstm_engression.py`    | `--engression-model lstm --stonet-head`   |
| LSTM + pre-additive   | `lstm_engression.py`    | `--engression-model lstm --pre-additive`  |

(`regularized_engression.py` and `adamw_engression.py` are the vanilla model
with different optimizers, not separate architectures.)

### Plugging in your own architecture

You are not limited to the four built-in heads. `LSTMEngressionConfig` has a
`model` field: set it and it replaces the built-in head selection, so any
`nn.Module` trained with the energy loss works. Two forms are accepted:

```python
from engression_modifications.lstm_engression import (
    LSTMEngressionConfig, fit_lstm_engression, load_lstm_engressor_checkpoint,
)

# (a) an nn.Module instance -- you own the input/output dims
cfg = LSTMEngressionConfig(model=MyGenerator(in_flat=seq*feat, out_dim=1))

# (b) a factory callable(input_dim, out_dim, config) -> nn.Module, sized at fit time
cfg = LSTMEngressionConfig(model=lambda in_dim, out_dim, c: MyGenerator(seq*in_dim, out_dim))

engressor = fit_lstm_engression(x, y, cfg)   # x is (n, seq_len, n_features)
```

Requirements for a custom model:

- `forward(x)` returns ONE stochastic draw of `y` per row (shape `(n, out_dim)`);
  draw fresh noise internally each call -- that per-call variability *is* the
  predicted conditional distribution.
- Optional: define `encode(x) -> (n, d)` if you want the kNN / Mahalanobis OOS
  diagnostics; otherwise the wrapper falls back to a `.lstm` attribute, and
  errors if neither exists.

The `model` field is **not** saved in checkpoints (only the weights are). To
reload a custom model, pass the same instance/factory back:
`load_lstm_engressor_checkpoint(path, model=factory)`.

## Files

- `vanilla_engression.py`: wrapper for fitting the public package without
  optimizer replacement.
- `regularized_engression.py`: wrapper for fitting the public package with
  classic Adam optimizer-level regularization.
- `adamw_engression.py`: wrapper for fitting the public package with decoupled
  AdamW weight decay.
- `lstm/`: local sequence-native engression model (LSTM encoder + stochastic
  head), split by concern into one file per piece:
  - `config.py`: `LSTMEngressionConfig` (all knobs).
  - `monotone.py`: `PositiveLinear` / `MonotonePositiveMLP` (pre-additive `g`).
  - `lstm_backbone.py`: `DeterministicLSTMEncoder` -- the shared LSTM building
    block (trajectory -> `h(X)`), reused by every generator.
  - `generators.py`: one class per head, all composing `DeterministicLSTMEncoder` --
    `DefaultHeadGenerator`, `StoNetHeadGenerator`, `PreAdditiveGenerator` (base
    `LSTMGenerator`) -- plus `build_lstm_model`, the single factory that picks a
    class from the config flags. The generators never branch on the config
    themselves.
  - `engressor.py`: `LSTMEngressor` -- the fitted **wrapper** (sampling,
    quantiles, `encode`), kept separate from the model.
  - `preprocessing.py`: input validation + standardization.
  - `checkpoint.py`: save / `load_lstm_engressor_checkpoint`.
  - `training.py`: `fit_lstm_engression` (the training loop).
  - `__init__.py` re-exports the public names, so
    `from engression_modifications.lstm import LSTMEngressionConfig,
    fit_lstm_engression` keeps working.
- `lstm_sweep.py`: script for sweeping LSTM epoch count and classic Adam
  weight decay on a fixed synthetic diagnostic task.
- `registry.py`: model registry and common `EngressionModelSpec` interface.
- `classic_adam_sweep.py`: script for sweeping classic Adam learning rates and
  weight decay values on a fixed synthetic diagnostic task.

## Import Pattern

The preferred pattern is to import a model spec and call `.fit(...)`:

```python
from engression_modifications import vanilla, regularized, adamw, lstm

model = regularized
engressor = model.fit(
    x_train,
    y_train,
    lr=0.003,
    weight_decay=0.003,
    num_epochs=120,
)
```

You can also choose by string:

```python
from engression_modifications import fit_engression_model

engressor = fit_engression_model(
    "regularized",
    x_train,
    y_train,
    lr=0.003,
    weight_decay=0.003,
)
```

Supported names are:

```python
vanilla
regularized
adamw
lstm
```

## Scope

The `vanilla`, `regularized`, and `adamw` variants keep the public package's
flat stochastic MLP generator:

\[
\widehat{Y}=g_\theta(X,\varepsilon).
\]

The `lstm` variant is a local implementation. It expects unflattened lag-window
inputs:

\[
X \in \mathbb{R}^{n\times \text{lags}\times d},
\]

encodes each sequence with an LSTM, concatenates fresh noise, and predicts with
a stochastic head:

\[
\widehat{Y}=g_\theta(\operatorname{LSTM}(X),\varepsilon).
\]

## Adam Versus AdamW

`regularized_engression.py` uses:

```python
torch.optim.Adam(..., weight_decay=...)
```

That couples the decay term to Adam's adaptive gradient machinery.

`adamw_engression.py` uses:

```python
torch.optim.AdamW(..., weight_decay=...)
```

That applies decoupled shrinkage:

\[
\theta \leftarrow (1-\eta\lambda)\theta.
\]
