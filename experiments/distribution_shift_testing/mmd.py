"""MMD permutation test: is the train/test difference statistically real?

C2ST and kNN both give effect sizes. This one gives a p-value, for
H0: train and test summary vectors come from the same distribution.

The statistic is the unbiased squared MMD with a Gaussian kernel, bandwidth set once
by the median heuristic. For the null we pool everything, re-split at random into the
original group sizes a few thousand times, and see where the observed value lands.
A characteristic kernel means MMD is zero only when the two distributions match, so
this picks up differences in spread and shape, not just in means.

Energy distance comes along as a cross-check, since it is the same statistic with a
different kernel and reuses the permutation machinery.

The caveat that matters: the permutation null assumes independent samples, so run this
on the stride-subsampled episodes. On raw hourly rows the p-value is garbage.

The n x n kernel matrix and the permutations run on GPU.
"""

from __future__ import annotations

import numpy as np
import torch


def median_heuristic_gamma(z: torch.Tensor) -> float:
    """RBF bandwidth via the median heuristic: gamma = 1 / median(pairwise sq. dist).

    Computed once on the pooled sample and then held FIXED across all permutations
    (a per-permutation bandwidth would invalidate the test).
    """
    d2 = torch.cdist(z, z) ** 2
    n = z.shape[0]
    offdiag = d2[~torch.eye(n, dtype=torch.bool, device=z.device)]
    med = torch.median(offdiag)
    return float(1.0 / torch.clamp(med, min=1e-12))


def _quadratic_forms(M: torch.Tensor, U: torch.Tensor):
    """For each row u of U (an indicator over pooled points), return the three block
    sums (u^T M u, v^T M v, u^T M v) with v = 1 - u. M is symmetric (kernel or dist).

    Vectorized over all P permutations at once: UM = U @ M is (P, n), everything else
    is a row-wise reduction. This is the whole cost of the permutation loop.
    """
    UM = U @ M                       # (P, n) = each row is u^T M
    col_sums = M.sum(dim=0)          # (n,)   = 1^T M  (column sums of M)
    V = 1.0 - U
    uMu = (UM * U).sum(dim=1)
    uMv = (UM * V).sum(dim=1)
    # v^T M v = (1-u)^T M (1-u); (1-u)^T M row = col_sums - u^T M
    vM = col_sums.unsqueeze(0) - UM
    vMv = (vM * V).sum(dim=1)
    return uMu, vMv, uMv


def _mmd2_unbiased(uKu, vKv, uKv, m, n):
    """Unbiased MMD^2 from block sums. RBF diagonal is 1, so subtract m and n."""
    return (uKu - m) / (m * (m - 1)) + (vKv - n) / (n * (n - 1)) - 2.0 * uKv / (m * n)


def _energy_distance(uDu, vDv, uDv, m, n):
    """Unbiased energy distance from block sums of the distance matrix (diag = 0)."""
    return 2.0 * uDv / (m * n) - uDu / (m * (m - 1)) - vDv / (n * (n - 1))


def _make_permutation_indicators(n_total, m, n_perm, device, rng):
    """(P, n) indicator matrix: each row marks a random size-m subset as group X."""
    U = torch.zeros((n_perm, n_total), dtype=torch.float32, device=device)
    for p in range(n_perm):
        pick = rng.choice(n_total, size=m, replace=False)
        U[p, pick] = 1.0
    return U


def run_mmd_test(
    feats_train: np.ndarray,
    feats_test: np.ndarray,
    *,
    device: str = "cuda",
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    """Gaussian-kernel MMD permutation test (+ energy-distance cross-check).

    ``feats_*`` are already standardized (train moments). Returns the observed
    statistics, permutation p-values, standardized effect sizes, and the null
    distributions (arrays, popped before JSON).
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    m = len(feats_train)
    n = len(feats_test)
    n_total = m + n

    # Pooled points: first m rows are the (observed) train group, rest are test.
    z = torch.as_tensor(
        np.concatenate([feats_train, feats_test], axis=0), dtype=torch.float32, device=dev
    )
    gamma = median_heuristic_gamma(z)

    d2 = torch.cdist(z, z) ** 2
    K = torch.exp(-gamma * d2)      # Gaussian kernel matrix (RBF), diag = 1
    D = torch.sqrt(torch.clamp(d2, min=0.0))  # Euclidean distance matrix, diag = 0

    # Observed split indicator (first m = train).
    u_obs = torch.zeros((1, n_total), dtype=torch.float32, device=dev)
    u_obs[0, :m] = 1.0
    uKu, vKv, uKv = _quadratic_forms(K, u_obs)
    mmd2_obs = float(_mmd2_unbiased(uKu, vKv, uKv, m, n)[0])
    uDu, vDv, uDv = _quadratic_forms(D, u_obs)
    energy_obs = float(_energy_distance(uDu, vDv, uDv, m, n)[0])

    # Permutation null (same fixed kernel/bandwidth).
    rng = np.random.default_rng(seed)
    U = _make_permutation_indicators(n_total, m, n_perm, dev, rng)
    uKu, vKv, uKv = _quadratic_forms(K, U)
    mmd2_null = _mmd2_unbiased(uKu, vKv, uKv, m, n).cpu().numpy()
    uDu, vDv, uDv = _quadratic_forms(D, U)
    energy_null = _energy_distance(uDu, vDv, uDv, m, n).cpu().numpy()

    def _pvalue(obs, null):  # +1 smoothing => never reports exactly 0
        return float((1 + np.sum(null >= obs)) / (n_perm + 1))

    def _zscore(obs, null):
        return float((obs - null.mean()) / (null.std() + 1e-12))

    return {
        "n_train_episodes": m,
        "n_test_episodes": n,
        "n_features": int(z.shape[1]),
        "n_permutations": n_perm,
        "rbf_gamma_median_heuristic": gamma,
        "mmd2_observed": mmd2_obs,
        "mmd2_pvalue": _pvalue(mmd2_obs, mmd2_null),
        "mmd2_null_mean": float(mmd2_null.mean()),
        "mmd2_null_std": float(mmd2_null.std()),
        "mmd2_zscore": _zscore(mmd2_obs, mmd2_null),
        "energy_observed": energy_obs,
        "energy_pvalue": _pvalue(energy_obs, energy_null),
        "energy_zscore": _zscore(energy_obs, energy_null),
        "_mmd2_null": mmd2_null,
        "_mmd2_obs": mmd2_obs,
    }
