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

## Files

- `vanilla_engression.py`: wrapper for fitting the public package without
  optimizer replacement.
- `regularized_engression.py`: wrapper for fitting the public package with
  classic Adam optimizer-level regularization.
- `adamw_engression.py`: wrapper for fitting the public package with decoupled
  AdamW weight decay.
- `lstm_engression.py`: local sequence-native engression model with an LSTM
  encoder and stochastic MLP head.
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
