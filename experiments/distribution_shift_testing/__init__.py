"""Train-vs-test covariate distribution-shift tests for the ENA back-trajectories.

Two tests, both run on per-channel summary statistics of each (241, 22)
back-trajectory (see ``features.summarize_trajectories``):

* ``c2st``: classifier two-sample test, "are train and test different?" Reports
  held-out MLP/linear AUC plus permutation importance for which channels drifted.
  The verdict is an effect size, not a p-value.
* ``knn_overlap``: "does test still have train support?" Compares test->train
  nearest-neighbor distances against a within-train baseline. This is the
  extrapolation check behind trusting ATT over ATC/ATE.

Both share the same split, features, and decorrelation subsample. Entry point is
``experiments.distribution_shift_testing.run``.
"""
