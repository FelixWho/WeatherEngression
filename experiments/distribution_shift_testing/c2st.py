"""Classifier two-sample test (C2ST) for train-vs-test covariate shift.

Label every TRAIN summary vector 0 and every TEST summary vector 1, then train a
classifier to tell them apart on a held-out split. The verdict is the held-out
**AUC**:

    AUC ~ 0.5  ->  the two covariate distributions are indistinguishable (no shift)
    AUC  > 0.5  ->  shift; permutation importance says WHICH channels/stats drifted

We report AUC (a prevalence-independent effect size) rather than a p-value. After
decorrelation only ~10^2 independent episodes remain, so a p-value here would be
meaningless. The MLP runs alongside a linear (logistic) probe: a wide gap between
the two flags *nonlinear* shift.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), tie-corrected. labels in {0, 1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks within ties so tied scores don't bias the statistic
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank = (start + csum + 1) / 2.0
    ranks = avg_rank[inv]
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 2, p_drop: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p_drop)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_probe(x_tr, y_tr, in_dim, *, linear, device, epochs, lr, batch_size, pos_weight, seed):
    """Fit one probe (linear or MLP). Returns a callable scoring an (n, d) numpy array."""
    torch.manual_seed(seed)
    model = (nn.Linear(in_dim, 1) if linear else _MLP(in_dim)).to(device)
    fwd = (lambda xb: model(xb).squeeze(-1)) if linear else (lambda xb: model(xb))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    x_tr = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    n = len(x_tr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            opt.zero_grad()
            loss_fn(fwd(x_tr[b]), y_tr[b]).backward()
            opt.step()

    def score(x_eval: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
            return fwd(xt).cpu().numpy()

    return score


def run_c2st(
    feats_train: np.ndarray,
    feats_test: np.ndarray,
    feature_names: list[str],
    *,
    device: str = "cuda",
    n_repeats: int = 5,
    val_frac: float = 0.3,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    """Full C2ST: MLP + linear AUC over ``n_repeats`` splits, plus MLP permutation importance.

    ``feats_*`` are already standardized (by train moments). Returns a dict with
    per-probe AUC mean/std and a ranked feature-importance table.
    """
    x = np.concatenate([feats_train, feats_test], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(feats_train)), np.ones(len(feats_test))]).astype(np.float32)
    in_dim = x.shape[1]
    # class imbalance (many more train rows than test): up-weight the positive class
    pos_weight = float(len(feats_train) / max(1, len(feats_test)))

    mlp_aucs, lin_aucs = [], []
    imp_sum = np.zeros(in_dim, dtype=np.float64)  # permutation importance, MLP only

    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        # stratified split so both classes appear in train and val
        va_mask = np.zeros(len(y), dtype=bool)
        for cls in (0.0, 1.0):
            cls_idx = np.flatnonzero(y == cls)
            n_va = int(round(val_frac * len(cls_idx)))
            va_mask[rng.choice(cls_idx, size=n_va, replace=False)] = True
        x_tr, y_tr = x[~va_mask], y[~va_mask]
        x_va, y_va = x[va_mask], y[va_mask]

        mlp_score = _fit_probe(
            x_tr, y_tr, in_dim, linear=False, device=device,
            epochs=epochs, lr=lr, batch_size=batch_size, pos_weight=pos_weight, seed=seed + r,
        )
        lin_score = _fit_probe(
            x_tr, y_tr, in_dim, linear=True, device=device,
            epochs=epochs, lr=lr, batch_size=batch_size, pos_weight=pos_weight, seed=seed + r,
        )
        base_auc = auc_score(mlp_score(x_va), y_va)
        mlp_aucs.append(base_auc)
        lin_aucs.append(auc_score(lin_score(x_va), y_va))

        # permutation importance: reuse the TRAINED MLP, shuffle one val column at a
        # time, measure the AUC drop. Bigger drop => that feature drove the split.
        for j in range(in_dim):
            saved = x_va[:, j].copy()
            x_va[:, j] = rng.permutation(saved)
            imp_sum[j] += base_auc - auc_score(mlp_score(x_va), y_va)
            x_va[:, j] = saved

    imp = imp_sum / n_repeats
    order = np.argsort(imp)[::-1]
    importance_table = [
        {"feature": feature_names[j], "auc_drop": float(imp[j])} for j in order
    ]
    return {
        "n_features": in_dim,
        "n_train_rows": int(len(feats_train)),
        "n_test_rows": int(len(feats_test)),
        "mlp_auc_mean": float(np.mean(mlp_aucs)),
        "mlp_auc_std": float(np.std(mlp_aucs)),
        "linear_auc_mean": float(np.mean(lin_aucs)),
        "linear_auc_std": float(np.std(lin_aucs)),
        "mlp_auc_per_repeat": [float(a) for a in mlp_aucs],
        "importance_table": importance_table,
    }
