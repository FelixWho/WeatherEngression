"""kNN overlap test: does the TEST covariate region still have TRAIN support?

C2ST answers "are train and test different?"; this answers the complementary and,
for the downstream causal estimands, more consequential question: "do test points
sit in regions the training data actually covers, or must the model extrapolate?"

For every test summary vector we take its Euclidean distance to the nearest TRAIN
vector, and compare that distribution against the train's own nearest-neighbor
distances (a within-train baseline). If test distances are systematically larger,
the test set lives partly outside the training support -- exactly the extrapolation
risk that makes ATC/ATE less trustworthy than ATT.

Reuses the vetted numpy kNN helpers in ``experiments.oos``.
"""

from __future__ import annotations

import numpy as np

from experiments.oos import choose_reference_indices, nearest_neighbor_distances


def run_knn_overlap(
    feats_train: np.ndarray,
    feats_test: np.ndarray,
    *,
    reference_size: int = 4000,
    batch_size: int = 2048,
    seed: int = 0,
) -> dict:
    """Compare test->train nearest-neighbor distances against train->train.

    ``feats_*`` are already standardized (by train moments). Returns summary stats
    plus the raw distance arrays (for plotting).
    """
    feats_train = np.asarray(feats_train, dtype=np.float32)
    feats_test = np.asarray(feats_test, dtype=np.float32)

    # Fixed reference subset of train (keeps the O(n_ref) distance work bounded and
    # the train/test comparison on equal footing).
    ref_idx = choose_reference_indices(len(feats_train), reference_size, seed)
    reference = feats_train[ref_idx]

    train_nn = nearest_neighbor_distances(
        queries=reference, reference=reference, batch_size=batch_size, exclude_self=True
    )
    test_nn = nearest_neighbor_distances(
        queries=feats_test, reference=reference, batch_size=batch_size, exclude_self=False
    )

    # "Out-of-support" test fraction = share of test points farther than the 95th
    # percentile of the within-train nearest-neighbor distance.
    thr95 = float(np.quantile(train_nn, 0.95))
    thr99 = float(np.quantile(train_nn, 0.99))
    qs = [0.5, 0.9, 0.95, 0.99]
    return {
        "reference_size": int(len(reference)),
        "n_test_rows": int(len(feats_test)),
        "train_nn_median": float(np.median(train_nn)),
        "test_nn_median": float(np.median(test_nn)),
        "median_ratio": float(np.median(test_nn) / max(np.median(train_nn), 1e-12)),
        "train_nn_quantiles": {str(q): float(np.quantile(train_nn, q)) for q in qs},
        "test_nn_quantiles": {str(q): float(np.quantile(test_nn, q)) for q in qs},
        "frac_test_beyond_train_p95": float(np.mean(test_nn > thr95)),
        "frac_test_beyond_train_p99": float(np.mean(test_nn > thr99)),
        # raw arrays for the histogram (popped before JSON serialization)
        "_train_nn": train_nn,
        "_test_nn": test_nn,
    }
