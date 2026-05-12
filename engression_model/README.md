# Engression Model

This folder contains model and loss code for engression-style conditional distribution learning.

The model is a conditional generator:

$$
\widehat{Y}=g_\theta(X,\varepsilon).
$$

For each $x$, repeated samples of $\varepsilon$ produce samples from the fitted conditional distribution:

$$
g_\theta(x,\varepsilon_1),\ldots,g_\theta(x,\varepsilon_m)
\sim
\widehat{P}(Y\mid X=x).
$$

The starter architecture in `networks.py` uses a pre-additive bias:

$$
z = W_x x + W_\varepsilon \varepsilon,
$$

followed by an MLP and a linear skip connection from $x$. This is meant to echo the pre-ANM structure from the engression paper, while remaining practical for weather-like lag-window inputs.

Important caveat: the starter MLP does **not** enforce monotonicity. The theory in the engression paper assumes monotone $g$ for some extrapolation results, but this implementation should initially be validated as a distributional regression model.

