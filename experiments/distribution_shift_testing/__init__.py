"""Train-vs-test covariate distribution-shift tests for the ENA back-trajectories.

Two complementary tests, both operating on per-channel summary statistics of each
(241, 22) back-trajectory (see ``features.summarize_trajectories``):

* ``c2st``        -- classifier two-sample test: "are train and test different?"
                     Held-out MLP/linear AUC + permutation importance (which channels
                     drifted). Verdict is an effect size (AUC), not a p-value.
* ``knn_overlap`` -- "does test still have train support?" Compares test->train
                     nearest-neighbor distances against a within-train baseline;
                     this is the extrapolation check behind trusting ATT over ATC/ATE.

Both share the same split, features, and decorrelation subsample. Entry point:
``experiments.distribution_shift_testing.run``.
"""
