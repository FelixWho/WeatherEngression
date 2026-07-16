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

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

import numpy as np

DEFAULT_MAT_PATH = "/storage3/fs1/myu/Active/felixhu/weather_data.mat"

# The natural science target. It is a dedicated scalar per trajectory rather than
# a column inside the trajectory, so predicting it never leaks the answer into X.
CCN_TARGET = "ccn"

# The 0-based row of ``time_ENA`` holding the MATLAB datenum (for ordering in time).
_TIME_DATENUM_ROW = 6

# The held-out test months used by the source paper: six whole months in 2022,
# chosen to cover all seasons.
PAPER_TEST_YEAR = 2022
PAPER_TEST_MONTHS = (1, 3, 5, 7, 9, 11)


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
    target_transform: str = "none"

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


def parse_count_or_all(value: str) -> int | None:
    """Parse a positive row count, or ``all``/``none`` for no cap."""

    normalized = value.strip().lower()
    if normalized in {"all", "none", "null", "unlimited"}:
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer or 'all', got {value!r}"
        ) from exc
    if count < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer or 'all', got {value!r}"
        )
    return count


def format_count_or_all(value: int | None) -> str:
    """Format an optional row cap for CLI output and run docs."""

    return "all" if value is None else str(value)


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


def _matlab_datenum_to_datetime(value: float) -> datetime:
    """Convert a MATLAB datenum scalar to a Python ``datetime``."""

    ordinal = int(value)
    fractional_day = float(value) % 1
    return datetime.fromordinal(ordinal) + timedelta(days=fractional_day) - timedelta(days=366)


def _paper_test_month_mask(time: np.ndarray) -> np.ndarray:
    """Return rows in the source paper's held-out 2022 test months."""

    test_months = set(PAPER_TEST_MONTHS)
    mask = np.zeros(len(time), dtype=bool)
    for row, value in enumerate(time):
        if not np.isfinite(value):
            continue
        timestamp = _matlab_datenum_to_datetime(float(value))
        mask[row] = timestamp.year == PAPER_TEST_YEAR and timestamp.month in test_months
    return mask


def load_ena_supervised_dataset(
    mat_path: str = DEFAULT_MAT_PATH,
    target: str = CCN_TARGET,
    max_samples: int | None = None,
    seq_stride: int = 1,
    seed: int = 2026,
    log_ccn: bool = False,
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
    log_ccn:
        If ``True`` and ``target="ccn"``, use ``log10(CCN)`` as the target,
        matching the source paper's target preprocessing. Non-positive CCN
        values are skipped because their log is undefined.
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
        if log_ccn and target_col is not None:
            raise ValueError("--log-ccn only applies when target='ccn'")
        target_transform = "log10" if log_ccn and target_col is None else "none"
        if target_transform == "log10":
            target_name = "log10(CCN)"

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
            if target_transform == "log10" and ccn[i] <= 0:
                continue
            traj = np.asarray(file[cells[0, i]][()], dtype=np.float32)  # (T, d)
            if not np.isfinite(traj).all():
                continue

            if target_col is None:
                sequence = traj[::seq_stride]
                target_value = float(ccn[i])
                if target_transform == "log10":
                    target_value = float(np.log10(target_value))
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
        target_transform=target_transform,
    )


CCN_TARGET_SLUGS = ("ccn", "log10_ccn")


def select_real_split(
    dataset: ENADataset,
    split: str,
    train_size: int | None,
    test_size: int | None,
    seed: int,
    event_flag: str = "dust",
    ccn_tail_quantile: float = 0.80,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick train/test rows under a chosen evaluation regime.

    - ``random``: a shuffled in-distribution split.
    - ``chronological``: earliest rows train, later rows test (forecast the future).
    - ``paper``: the source paper's held-out months (Jan, Mar, May, Jul, Sep,
      and Nov 2022) are the test pool; all other months are the train pool.
    - ``event``: hold out a meteorological event (``event_flag``) as the test set,
      train on the remaining clean periods -- a real distribution-shift stress test
      analogous to the synthetic out-of-support splits.
    - ``ccn_tail``: train on the lower CCN range (below the ``ccn_tail_quantile``
      quantile) and test on the held-out high-CCN tail. Because the target never
      reaches these values in training, this is a genuine *extrapolation* test of
      whether engression pushes its conditional distribution beyond the observed
      target support. Requires a CCN target. ``log10`` is monotone in CCN, so the
      tail of ``y`` is the tail of CCN under either transform.
    """

    n = len(dataset.y)
    rng = np.random.default_rng(seed)

    if split == "random":
        if train_size is None and test_size is None:
            raise ValueError("random split needs --train-size, --test-size, or both")
        perm = rng.permutation(n)
        resolved_train_size = train_size if train_size is not None else n - int(test_size)
        resolved_test_size = test_size if test_size is not None else n - resolved_train_size
        if resolved_train_size < 1 or resolved_test_size < 1:
            raise ValueError(
                "random split produced a non-positive train/test size; "
                "lower --train-size or --test-size"
            )
        train_idx = perm[:resolved_train_size]
        test_idx = perm[resolved_train_size : resolved_train_size + resolved_test_size]
    elif split == "chronological":
        if train_size is None and test_size is None:
            raise ValueError("chronological split needs --train-size, --test-size, or both")
        order = np.argsort(dataset.time, kind="stable")
        resolved_train_size = train_size if train_size is not None else n - int(test_size)
        resolved_test_size = test_size if test_size is not None else n - resolved_train_size
        if resolved_train_size < 1 or resolved_test_size < 1:
            raise ValueError(
                "chronological split produced a non-positive train/test size; "
                "lower --train-size or --test-size"
            )
        train_idx = order[:resolved_train_size]
        test_idx = order[resolved_train_size : resolved_train_size + resolved_test_size]
    elif split == "paper":
        paper_test = _paper_test_month_mask(dataset.time)
        train_pool = np.flatnonzero(~paper_test)
        test_pool = np.flatnonzero(paper_test)
        train_idx = rng.permutation(train_pool)
        test_idx = rng.permutation(test_pool)
        if train_size is not None:
            train_idx = train_idx[:train_size]
        if test_size is not None:
            test_idx = test_idx[:test_size]
    elif split == "event":
        if event_flag not in dataset.flags:
            valid = ", ".join(sorted(dataset.flags))
            raise ValueError(f"unknown event flag {event_flag!r}; choose one of: {valid}")
        mask = dataset.flags[event_flag]
        test_idx = rng.permutation(np.flatnonzero(mask))
        train_idx = rng.permutation(np.flatnonzero(~mask))
        if train_size is not None:
            train_idx = train_idx[:train_size]
        if test_size is not None:
            test_idx = test_idx[:test_size]
    elif split == "ccn_tail":
        if dataset.target_slug not in CCN_TARGET_SLUGS:
            raise ValueError(
                f"ccn_tail split needs a CCN target, got {dataset.target_name!r}; "
                "use --target ccn"
            )
        if not 0.0 < ccn_tail_quantile < 1.0:
            raise ValueError(f"ccn_tail_quantile must be in (0, 1), got {ccn_tail_quantile}")
        # Train on the low/mid CCN range; hold out the high-CCN tail. log10 is
        # monotone, so thresholding y is the same as thresholding raw CCN.
        threshold = float(np.quantile(dataset.y, ccn_tail_quantile))
        train_idx = rng.permutation(np.flatnonzero(dataset.y <= threshold))
        test_idx = rng.permutation(np.flatnonzero(dataset.y > threshold))
        if train_size is not None:
            train_idx = train_idx[:train_size]
        if test_size is not None:
            test_idx = test_idx[:test_size]
    else:
        raise ValueError(
            f"unknown split {split!r}; choose 'random', 'chronological', "
            "'paper', 'event', or 'ccn_tail'"
        )

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"split {split!r} produced empty train/test "
            f"(train={len(train_idx)}, test={len(test_idx)}); "
            "check the split pools, lower explicit size caps, or use --max-samples all"
        )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Manual inspection CLI
#
# ``python data_generation/ena_weather.py`` walks every stage of the ingestion
# pipeline and prints what it produced, so each part can be eyeballed for
# correctness: variable decoding, shapes/dtypes, NaN filtering, target
# construction (including the one-step-ahead, leakage-free column targets),
# time and event-flag fields, and the split regimes' invariants. It also
# re-reads a few raw cells straight from the .mat and asserts the stored
# ``(X, y)`` match a from-scratch recomputation -- an end-to-end correctness
# check, not just a smoke test.
# ---------------------------------------------------------------------------


def _matlab_datenum_to_iso(datenum: float) -> str:
    """Convert a MATLAB datenum to an ISO date string (best effort)."""

    try:
        dt = datetime.fromordinal(int(datenum)) + timedelta(days=float(datenum) % 1) - timedelta(days=366)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError):  # pragma: no cover - defensive
        return f"datenum={datenum:.3f}"


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def _verify_against_raw(dataset: ENADataset, mat_path: str, seq_stride: int, n_check: int) -> bool:
    """Re-read raw cells for the first ``n_check`` kept samples and compare.

    Recomputes ``(sequence, y)`` from the raw trajectory exactly as the loader
    should have, and asserts it matches the stored arrays. This is the strongest
    correctness check: it ties the produced tensors back to the source file.
    """

    h5py = _require_h5py()
    target_col = None
    ccn_target = dataset.target_name == "CCN" or dataset.target_transform == "log10"
    if not ccn_target:
        target_col = dataset.feature_names.index(dataset.target_name)

    all_ok = True
    with h5py.File(mat_path, "r") as file:
        cells = file["X"]
        ccn = np.asarray(file["CCN"][0], dtype=np.float64) if target_col is None else None
        for j in range(min(n_check, len(dataset.y))):
            src = int(dataset.source_index[j])
            traj = np.asarray(file[cells[0, src]][()], dtype=np.float32)
            if target_col is None:
                seq_expected = traj[::seq_stride]
                y_expected = float(ccn[src])
                if dataset.target_transform == "log10":
                    y_expected = float(np.log10(y_expected))
            else:
                y_expected = float(traj[-1, target_col])
                seq_expected = traj[:-1][::seq_stride]
            # The loader stores X and y as float32, so compare at float32
            # precision (an exact float64 == would fail on rounding alone).
            x_ok = np.array_equal(seq_expected, dataset.x[j])
            y_ok = np.float32(y_expected) == dataset.y[j]
            all_ok &= _check(
                f"row {j} (source #{src})",
                x_ok and y_ok,
                f"X match={x_ok}, y match={y_ok} (y={dataset.y[j]:.4g})",
            )
    return all_ok


def _build_inspect_parser() -> argparse.ArgumentParser:
    """Build the parser for the manual inspection CLI."""

    parser = argparse.ArgumentParser(
        description="Inspect and verify the ENA data ingestion pipeline stage by stage."
    )
    parser.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    parser.add_argument(
        "--target",
        type=str,
        default=CCN_TARGET,
        help="'ccn', a variable name, or a column index (one-step-ahead for columns).",
    )
    ccn_transform = parser.add_mutually_exclusive_group()
    ccn_transform.add_argument(
        "--log-ccn",
        dest="log_ccn",
        action="store_true",
        default=False,
        help="Use log10(CCN) when target='ccn'.",
    )
    ccn_transform.add_argument(
        "--no-log-ccn",
        dest="log_ccn",
        action="store_false",
        help="Use raw CCN when target='ccn'.",
    )
    parser.add_argument(
        "--max-samples",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for every usable sample.",
    )
    parser.add_argument("--seq-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split",
        choices=("random", "chronological", "paper", "event", "ccn_tail"),
        default="paper",
    )
    parser.add_argument("--event-flag", type=str, default="dust")
    parser.add_argument("--ccn-tail-quantile", type=float, default=0.80)
    parser.add_argument(
        "--train-size",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for the full train pool.",
    )
    parser.add_argument(
        "--test-size",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for the full test pool.",
    )
    parser.add_argument("--verify-raw", type=int, default=5, help="Raw cells to re-read and assert.")
    parser.add_argument("--preview-rows", type=int, default=5, help="Trajectory rows to print.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Walk the ingestion pipeline and print/verify each stage."""

    args = _build_inspect_parser().parse_args(argv)
    all_ok = True

    _section("1. LOAD")
    print(f"  mat_path : {args.mat_path}")
    print(
        f"  target   : {args.target}   "
        f"log_ccn={args.log_ccn}   "
        f"max_samples={format_count_or_all(args.max_samples)}   seq_stride={args.seq_stride}"
    )
    dataset = load_ena_supervised_dataset(
        mat_path=args.mat_path,
        target=args.target,
        max_samples=args.max_samples,
        seq_stride=args.seq_stride,
        seed=args.seed,
        log_ccn=args.log_ccn,
    )

    _section("2. VARIABLES & SHAPES")
    print(f"  feature_names ({len(dataset.feature_names)}): {list(dataset.feature_names)}")
    print(f"  target_name   : {dataset.target_name}   (slug: {dataset.target_slug})")
    print(f"  usable samples: {len(dataset.y)}")
    print(f"  sequence_len  : {dataset.sequence_length}   n_features: {dataset.n_features}")
    print(f"  X: shape={dataset.x.shape} dtype={dataset.x.dtype}")
    print(f"  y: shape={dataset.y.shape} dtype={dataset.y.dtype}")
    all_ok &= _check("X is 3-D (n, seq_len, n_features)", dataset.x.ndim == 3)
    all_ok &= _check("y is 1-D and aligned with X", dataset.y.ndim == 1 and len(dataset.y) == len(dataset.x))
    all_ok &= _check(
        "source_index/time/flags aligned with X",
        len(dataset.source_index) == len(dataset.x)
        and len(dataset.time) == len(dataset.x)
        and all(len(m) == len(dataset.x) for m in dataset.flags.values()),
    )

    _section("3. FINITENESS (loader must drop NaN/Inf in X and y)")
    all_ok &= _check("all X finite", bool(np.isfinite(dataset.x).all()))
    all_ok &= _check("all y finite", bool(np.isfinite(dataset.y).all()))

    _section("4. TARGET STATS")
    y = dataset.y
    print(f"  min={y.min():.4g}  median={np.median(y):.4g}  mean={y.mean():.4g}  "
          f"max={y.max():.4g}  std={y.std():.4g}")
    if dataset.target_name != "CCN" and dataset.target_transform == "none":
        print("  (column target: y is the FINAL trajectory step; X is the preceding history)")

    _section("5. TIME FIELD")
    t = dataset.time
    print(f"  datenum range: {t.min():.3f} .. {t.max():.3f}")
    print(f"  as dates     : {_matlab_datenum_to_iso(t.min())}  ..  {_matlab_datenum_to_iso(t.max())}")

    _section("6. EVENT FLAGS")
    for name, mask in sorted(dataset.flags.items()):
        print(f"  {name:14s}: {int(mask.sum()):5d} / {len(mask)}  ({mask.mean():.3f})")

    _section(f"7. RAW-SOURCE VERIFICATION (re-read {args.verify_raw} cells from the .mat)")
    all_ok &= _verify_against_raw(dataset, args.mat_path, args.seq_stride, args.verify_raw)

    _section(f"8. SPLIT: {args.split}")
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=args.split,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed + 17,
        event_flag=args.event_flag,
        ccn_tail_quantile=args.ccn_tail_quantile,
    )
    print(f"  train rows: {len(train_idx)}   test rows: {len(test_idx)}")
    disjoint = len(np.intersect1d(train_idx, test_idx)) == 0
    all_ok &= _check("train/test index sets are disjoint", disjoint)
    if args.split == "chronological":
        train_end = dataset.time[train_idx].max()
        test_start = dataset.time[test_idx].min()
        all_ok &= _check(
            "train precedes test in time", train_end <= test_start,
            f"max(train_time)={train_end:.2f} <= min(test_time)={test_start:.2f}",
        )
    elif args.split == "paper":
        paper_test = _paper_test_month_mask(dataset.time)
        all_ok &= _check("all test rows are in paper-held-out months", bool(paper_test[test_idx].all()))
        all_ok &= _check("all train rows are outside paper-held-out months", bool((~paper_test[train_idx]).all()))
    elif args.split == "event":
        mask = dataset.flags[args.event_flag]
        all_ok &= _check(f"all test rows are in '{args.event_flag}' event", bool(mask[test_idx].all()))
        all_ok &= _check(f"all train rows are clean (not '{args.event_flag}')", bool((~mask[train_idx]).all()))
    elif args.split == "ccn_tail":
        threshold = float(np.quantile(dataset.y, args.ccn_tail_quantile))
        all_ok &= _check(
            f"all test rows are above the {args.ccn_tail_quantile:.0%} CCN threshold",
            bool((dataset.y[test_idx] > threshold).all()),
            f"threshold y={threshold:.4g}, "
            f"train max={dataset.y[train_idx].max():.4g}, test min={dataset.y[test_idx].min():.4g}",
        )
        all_ok &= _check(
            "all train rows are at or below the threshold (no tail leakage)",
            bool((dataset.y[train_idx] <= threshold).all()),
        )

    _section(f"9. PREVIEW (first sample, first {args.preview_rows} of {dataset.sequence_length} steps)")
    print(f"  source #{int(dataset.source_index[0])}  y={dataset.y[0]:.4g}")
    with np.printoptions(precision=3, suppress=True, linewidth=140):
        print(dataset.x[0, : args.preview_rows])

    _section("SUMMARY")
    print(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- see [FAIL] above'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
