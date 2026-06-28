"""Loader for the real Eastern North Atlantic (ENA) weather dataset.

This module is the real-data counterpart to the synthetic generators in this
package. The synthetic generators define ``P(Y | X=x)`` exactly; the ENA data
does not, so here we only build the supervised ``(X, y)`` tensors a
sequence-native model needs and leave evaluation to held-out realized targets.

The source is a MATLAB v7.3 (HDF5) file with these variables:

- ``X``: a ``1 x N`` cell array; each cell is a ``(T, d)`` airmass back-trajectory
  (``T`` time steps of ``d`` weather variables arriving at the ENA site).
- ``CCN``: a ``1 x N`` row of cloud-condensation-nuclei targets (the science
  target; contains NaN where no valid measurement exists).
- ``var_name``: a ``1 x d`` cell of variable names.
- ``time_ENA``: a ``7 x N`` array; rows are ``[year, month, day, hour, minute,
  second, datenum]``.
- ``period_flag``: a struct of ``1 x N`` boolean event masks (``dust``,
  ``BB_criterion1``, ``BB_criterion2`` biomass-burning periods).

The supervised problem keeps the engression contract: ``X`` is an unflattened
sequence ``(n, sequence_length, d)`` and ``y`` is a scalar target per sequence,
matching the ``lstm`` engression variant's ``input_kind="sequence"``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

DEFAULT_MAT_PATH = "/storage3/fs1/myu/Active/felixhu/weather_data.mat"

# The natural science target. It is a dedicated scalar per trajectory rather than
# a column inside the trajectory, so predicting it never leaks the answer into X.
CCN_TARGET = "ccn"

# The 0-based row of ``time_ENA`` holding the MATLAB datenum (for ordering in time).
_TIME_DATENUM_ROW = 6


@dataclass(frozen=True)
class ENADataset:
    """One supervised ENA dataset ready for sequence-native engression."""

    x: np.ndarray  # (n, sequence_length, n_features) float32
    y: np.ndarray  # (n,) float32
    feature_names: tuple[str, ...]
    target_name: str
    target_slug: str
    time: np.ndarray  # (n,) float64 MATLAB datenum, ordering key
    flags: dict[str, np.ndarray]  # name -> (n,) bool event mask
    source_index: np.ndarray  # (n,) original column index in the .mat file

    @property
    def sequence_length(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[2])

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a plain dict view (X, y) for the experiment plumbing."""

        return {"X": self.x, "y": self.y}


def _require_h5py():
    """Import h5py with an actionable error if the venv is missing it."""

    try:
        import h5py  # noqa: WPS433 (local import keeps non-loader code lightweight)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Reading the ENA .mat (MATLAB v7.3 / HDF5) needs h5py. "
            "It is listed in requirements.txt; run `.venv/bin/pip install -r requirements.txt`."
        ) from exc
    return h5py


def _decode_matlab_string(file, ref) -> str:
    """Decode a MATLAB char array (referenced by ``ref``) into a Python string."""

    codes = np.asarray(file[ref][()]).flatten()
    return "".join(chr(int(code)) for code in codes)


def slugify(name: str) -> str:
    """Make a filesystem-safe slug from a variable name like ``PBLH^(1/5)``."""

    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return slug or "feature"


def _resolve_target(target: str, feature_names: tuple[str, ...]) -> tuple[str, int | None]:
    """Return ``(display_name, column_or_None)`` for a requested target.

    ``column_or_None`` is ``None`` for the dedicated CCN target, or the 0-based
    feature column for a trajectory variable (predicted one step ahead).
    """

    key = str(target).strip()
    if key.lower() == CCN_TARGET:
        return "CCN", None
    if key.isdigit():
        col = int(key)
        if not 0 <= col < len(feature_names):
            raise ValueError(f"target column {col} out of range 0..{len(feature_names) - 1}")
        return feature_names[col], col
    lowered = [name.lower() for name in feature_names]
    if key.lower() in lowered:
        col = lowered.index(key.lower())
        return feature_names[col], col
    raise ValueError(
        f"unknown target {target!r}; use 'ccn', a column index, or one of: "
        + ", ".join(feature_names)
    )


def load_ena_supervised_dataset(
    mat_path: str = DEFAULT_MAT_PATH,
    target: str = CCN_TARGET,
    max_samples: int | None = 6_000,
    seq_stride: int = 1,
    seed: int = 2026,
) -> ENADataset:
    """Load the ENA data as supervised sequences for the ``lstm`` engression model.

    Parameters
    ----------
    mat_path:
        Path to the MATLAB v7.3 file.
    target:
        ``"ccn"`` (default) predicts the scalar CCN target from the full
        trajectory. A variable name or column index predicts that variable's
        value at the final time step from the preceding history (one step ahead),
        so the predicted step is never shown to the model.
    max_samples:
        Cap on usable samples kept (random, seeded). ``None`` keeps all usable
        samples. Trajectories with non-finite features or a non-finite target are
        always skipped.
    seq_stride:
        Keep every ``seq_stride``-th time step to shorten long trajectories for a
        CPU-friendly LSTM. ``1`` keeps the full sequence.
    seed:
        Seed for the subsample shuffle.
    """

    if seq_stride < 1:
        raise ValueError("seq_stride must be >= 1")

    h5py = _require_h5py()
    rng = np.random.default_rng(seed)

    with h5py.File(mat_path, "r") as file:
        var_name = file["var_name"]
        feature_names = tuple(
            _decode_matlab_string(file, var_name[0, i]) for i in range(var_name.shape[1])
        )
        target_name, target_col = _resolve_target(target, feature_names)

        cells = file["X"]
        n_total = cells.shape[1]
        ccn = np.asarray(file["CCN"][0], dtype=np.float64) if target_col is None else None
        datenum = np.asarray(file["time_ENA"][_TIME_DATENUM_ROW], dtype=np.float64)

        flag_group = file["period_flag"]
        flag_arrays = {
            name: np.asarray(flag_group[name][0], dtype=np.float64) > 0.5
            for name in flag_group.keys()
        }

        order = rng.permutation(n_total)
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        kept_index: list[int] = []

        for i in order:
            if target_col is None and not np.isfinite(ccn[i]):
                continue
            traj = np.asarray(file[cells[0, i]][()], dtype=np.float32)  # (T, d)
            if not np.isfinite(traj).all():
                continue

            if target_col is None:
                sequence = traj[::seq_stride]
                target_value = float(ccn[i])
            else:
                # One step ahead: predict the final step from earlier history.
                target_value = float(traj[-1, target_col])
                sequence = traj[:-1][::seq_stride]

            x_rows.append(sequence)
            y_rows.append(target_value)
            kept_index.append(int(i))
            if max_samples is not None and len(kept_index) >= max_samples:
                break

    if not x_rows:
        raise ValueError("no usable samples found (all targets/trajectories non-finite)")

    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    kept = np.asarray(kept_index, dtype=np.int64)
    return ENADataset(
        x=x,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
        target_slug=slugify(target_name),
        time=datenum[kept],
        flags={name: mask[kept] for name, mask in flag_arrays.items()},
        source_index=kept,
    )


def select_real_split(
    dataset: ENADataset,
    split: str,
    train_size: int,
    test_size: int,
    seed: int,
    event_flag: str = "dust",
) -> tuple[np.ndarray, np.ndarray]:
    """Pick train/test rows under a chosen evaluation regime.

    - ``random``: a shuffled in-distribution split.
    - ``chronological``: earliest rows train, later rows test (forecast the future).
    - ``event``: hold out a meteorological event (``event_flag``) as the test set,
      train on the remaining clean periods -- a real distribution-shift stress test
      analogous to the synthetic out-of-support splits.
    """

    n = len(dataset.y)
    rng = np.random.default_rng(seed)

    if split == "random":
        perm = rng.permutation(n)
        train_idx = perm[:train_size]
        test_idx = perm[train_size : train_size + test_size]
    elif split == "chronological":
        order = np.argsort(dataset.time, kind="stable")
        train_idx = order[:train_size]
        test_idx = order[train_size : train_size + test_size]
    elif split == "event":
        if event_flag not in dataset.flags:
            valid = ", ".join(sorted(dataset.flags))
            raise ValueError(f"unknown event flag {event_flag!r}; choose one of: {valid}")
        mask = dataset.flags[event_flag]
        event_rows = rng.permutation(np.flatnonzero(mask))
        clean_rows = rng.permutation(np.flatnonzero(~mask))
        test_idx = event_rows[:test_size]
        train_idx = clean_rows[:train_size]
    else:
        raise ValueError(
            f"unknown split {split!r}; choose 'random', 'chronological', or 'event'"
        )

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"split {split!r} produced empty train/test "
            f"(train={len(train_idx)}, test={len(test_idx)}); "
            "lower --train-size/--test-size or raise --max-samples"
        )
    return train_idx, test_idx
