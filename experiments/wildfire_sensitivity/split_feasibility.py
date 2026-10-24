"""Measure the cost of selecting spaced recalibration rows from ALL clean data.

No models are fitted. Reproduce the current split, then select recalibration
points first, optionally purge nearby arrivals, and reuse the existing control/
validation splitters. Run the real-data inventory through SLURM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_generation.ena_weather import DEFAULT_MAT_PATH, _matlab_datenum_to_datetime
from experiments.wildfire_sensitivity.control import (
    _months, month_matched_subsample, split_clean_holdout,
)


def month_quotas(reference_months, count):
    """Largest-remainder allocation of a fixed count to the reference months."""
    weights = np.bincount(reference_months, minlength=13)[1:].astype(float)
    if count < 0 or weights.sum() == 0:
        raise ValueError("require a nonnegative count and nonempty reference months")
    expected = count * weights / weights.sum()
    quotas = np.floor(expected).astype(int)
    order = np.argsort(-(expected - quotas), kind="stable")
    quotas[order[:count - quotas.sum()]] += 1
    return np.r_[0, quotas]


def select_spaced(clean, times, months, quotas, gap_days, seed):
    """Seeded random greedy selection from the full pool, with month quotas.

    Return the achievable selection without relaxing spacing or month quotas.
    A shortfall describes this greedy selection, not a proof of infeasibility.
    """
    if not np.isfinite(gap_days) or gap_days <= 0:
        raise ValueError("gap_days must be positive and finite")
    selected = []
    selected_times = []
    remaining = np.array(quotas, copy=True)
    for row in np.random.default_rng(seed).permutation(clean):
        month = months[row]
        if remaining[month] <= 0:
            continue
        if selected_times and np.min(np.abs(times[row] - np.asarray(selected_times))) < gap_days:
            continue
        selected.append(row)
        selected_times.append(times[row])
        remaining[month] -= 1
        if remaining.sum() == 0:
            break
    return np.asarray(selected, dtype=np.int64)


def earliest_spaced(clean, times, gap_days):
    """Maximum count under spacing alone, without month quotas."""
    if not np.isfinite(gap_days) or gap_days <= 0:
        raise ValueError("gap_days must be positive and finite")
    selected = []
    for row in clean[np.argsort(times[clean], kind="stable")]:
        if not selected or times[row] - times[selected[-1]] >= gap_days:
            selected.append(row)
    return np.asarray(selected, dtype=np.int64)


def purge_near_selected(clean, selected, times, guard_days):
    """Exclude nearby arrivals; zero removes only the selected rows themselves."""
    if not np.isfinite(guard_days) or guard_days < 0:
        raise ValueError("guard_days must be nonnegative and finite")
    remaining = np.setdiff1d(clean, selected)
    excluded = np.zeros(len(remaining), dtype=bool)
    for value in times[selected]:
        excluded |= np.abs(times[remaining] - value) < guard_days
    return remaining[~excluded], remaining[excluded]


def split_inventory(clean, wildfire, times, window_hours, settings, recalibration=None):
    """Existing split (None), or reserve provided recalibration rows FIRST.

    Every clean row is accounted for, including guards, month-matching discards,
    and unassigned rows if the existing splitter cannot finish.
    """
    subsets = {}
    counts = {}
    status, reason = "ok", ""
    recalibration_guard_days = settings.get("recalibration_guard_days", window_hours / 24)

    def holdout(pool, role, fraction, seed):
        if not len(pool):
            raise ValueError(f"no eligible observations remain to reserve {role}")
        fit, held = split_clean_holdout(
            pool, times, window_hours=window_hours, holdout_fraction=fraction,
            block_days=settings["block_days"], seed=seed,
        )
        subsets[role + "_guard"] = np.setdiff1d(pool, np.r_[fit, held])
        if role in ("control", "recalibration"):
            counts[role + "_pool"] = len(held)
            try:
                used = month_matched_subsample(held, wildfire, times, seed=seed)
            except ValueError:
                subsets[role + "_unmatched_pool"] = held
                raise
            subsets[role + "_month_discard"] = np.setdiff1d(held, used)
        else:
            used = held
        subsets[role] = used
        return fit

    try:
        remaining = clean
        if recalibration is not None:
            subsets["recalibration"] = recalibration
            remaining, subsets["recalibration_guard"] = purge_near_selected(
                clean, recalibration, times, recalibration_guard_days,
            )
            counts["after_recalibration_guard"] = len(remaining)
        remaining = holdout(remaining, "control", settings["control_fraction"],
                            settings["control_seed"])
        if recalibration is None:
            remaining = holdout(remaining, "recalibration", settings["recalibration_fraction"],
                                settings["recalibration_seed"])
        remaining = holdout(remaining, "validation", settings["validation_fraction"],
                            settings["validation_seed"])
        subsets["train"] = remaining
    except ValueError as exc:
        status, reason = "invalid", str(exc)

    assigned = np.concatenate(list(subsets.values())) if subsets else np.array([], dtype=int)
    if len(assigned) != len(np.unique(assigned)):
        raise AssertionError("split categories overlap")
    subsets["unassigned"] = np.setdiff1d(clean, assigned)
    if sum(map(len, subsets.values())) != len(clean):
        raise AssertionError("not all clean observations accounted for")
    if status == "ok" and len(subsets["unassigned"]):
        raise AssertionError("completed split left unassigned observations")

    counts.update({name: len(rows) for name, rows in subsets.items()})
    counts.update(status=status, reason=reason)
    counts["excluded_total"] = sum(
        len(rows) for name, rows in subsets.items()
        if name not in ("train", "validation", "recalibration", "control", "unassigned")
    )
    # Row disjointness was checked above. Validate only the requested temporal
    # guards; the current baseline always retains its original four-role guards.
    roles = [name for name in ("train", "validation", "recalibration", "control")
             if name in subsets]
    for i, left in enumerate(roles):
        a = np.sort(times[subsets[left]])
        for right in roles[i + 1:]:
            required_gap = (recalibration_guard_days
                            if recalibration is not None and "recalibration" in (left, right)
                            else window_hours / 24)
            if required_gap == 0:
                continue
            b = np.sort(times[subsets[right]])
            if not a.size or not b.size:
                continue
            positions = np.searchsorted(a, b)
            nearest = np.minimum(
                abs(b - a[np.minimum(positions, len(a) - 1)]),
                abs(b - a[np.maximum(positions - 1, 0)]),
            )
            if nearest.min() < required_gap - 1e-8:
                raise AssertionError(f"trajectory guard failed between {left} and {right}")
    return counts, subsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", default=DEFAULT_MAT_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-days", type=float, default=10.)
    parser.add_argument("--recalibration-guard-days", type=float, default=None,
                        help="Exclusion on either side of each recalibration point; "
                        "default: trajectory history length. Zero removes this exclusion only.")
    parser.add_argument("--population-from", type=Path,
                        help="Reuse population.npz and dataset metadata from a previous inventory.")
    parser.add_argument("--counts", type=int, nargs="+", default=[25, 50, 75, 100, 150, 200, 250, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2228, 2229, 2230])
    args = parser.parse_args()
    if any(n < 1 for n in args.counts) or any(seed < 0 for seed in args.seeds):
        parser.error("counts must be positive and seeds nonnegative")
    if not np.isfinite(args.gap_days) or args.gap_days <= 0:
        parser.error("gap-days must be positive and finite")
    if args.recalibration_guard_days is not None and (
        not np.isfinite(args.recalibration_guard_days) or args.recalibration_guard_days < 0
    ):
        parser.error("recalibration-guard-days must be nonnegative and finite")

    # Reuse the production loader and constants so the eligible population and
    # baseline counts match screening exactly. Load trajectories once in SLURM.
    from experiments.wildfire_arms import load_wildfire_arms
    from experiments.wildfire_sensitivity.settings import (
        HOLDOUT_BLOCK_DAYS, HOLDOUT_FRACTION, RECALIBRATION_FRACTION,
        SEQ_STRIDE, SWEEP_SEED, VALIDATION_FRACTION,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.population_from is not None:
        if args.population_from.resolve() == args.output_dir.resolve():
            parser.error("output-dir must not overwrite the source inventory")
        original = json.loads((args.population_from / "results.json").read_text())
        if original["settings"]["dataset_seed"] != SWEEP_SEED:
            raise ValueError("cached population has a different dataset seed")
        with np.load(args.population_from / "population.npz") as saved:
            clean, wildfire = saved["clean"], saved["wildfire"]
            times, source_index = saved["time_days"], saved["source_index"]
        window_hours = original["settings"]["window_hours"]
        args.mat_path = original["mat_path"]
        print(f"Reusing full eligible population from {args.population_from}; no model fitting", flush=True)
    else:
        print("Loading full eligible dataset once; no model fitting", flush=True)
        dataset, arms = load_wildfire_arms(
            mat_path=args.mat_path, target="ccn", log_ccn=True, split="paper",
            seq_stride=SEQ_STRIDE, max_samples=None, train_size=None, test_size=None,
            seed=SWEEP_SEED, wildfire_flag="BB_criterion1",
        )
        clean = np.sort(np.r_[arms.train_clean, arms.test_clean])
        wildfire = np.r_[arms.train_wildfire, arms.test_wildfire]
        times, source_index = dataset.time, dataset.source_index
        window_hours = dataset.x.shape[1] * SEQ_STRIDE
        del dataset
    if not np.isfinite(times).all():
        raise ValueError("dataset contains nonfinite timestamps")
    months = _months(times)
    settings = dict(
        dataset_seed=SWEEP_SEED, control_seed=SWEEP_SEED,
        recalibration_seed=SWEEP_SEED + 202, validation_seed=SWEEP_SEED + 101,
        control_fraction=HOLDOUT_FRACTION, recalibration_fraction=RECALIBRATION_FRACTION,
        validation_fraction=VALIDATION_FRACTION, block_days=HOLDOUT_BLOCK_DAYS,
        window_hours=window_hours, gap_days=args.gap_days,
        recalibration_guard_days=(window_hours / 24 if args.recalibration_guard_days is None
                                  else args.recalibration_guard_days),
    )
    maximum = earliest_spaced(clean, times, args.gap_days)
    np.savez_compressed(args.output_dir / "population.npz", time_days=times,
                        source_index=source_index, clean=clean, wildfire=wildfire,
                        earliest_spaced=maximum)
    print(f"Full clean pool: {len(clean)}; maximum {args.gap_days:g}-day-spaced count: {len(maximum)}",
          flush=True)
    results = dict(
        settings=settings, mat_path=args.mat_path, clean_count=len(clean),
        population_from=str(args.population_from) if args.population_from else None,
        wildfire_count=len(wildfire), earliest_first_maximum=len(maximum),
        date_start=str(_matlab_datenum_to_datetime(times[clean].min())),
        date_end=str(_matlab_datenum_to_datetime(times[clean].max())),
        selection="random greedy from all clean rows; largest-remainder wildfire month quotas",
        notes=["Counts are a split-feasibility diagnostic, not calibration or sensitivity results.",
               "The earliest-first maximum ignores month quotas and the need for other sets.",
               "Greedy shortfalls do not establish the maximum under month quotas.",
               "Control and validation reuse production blocked splits; control remains month-matched.",
               "Control/validation guards and the current baseline use the full trajectory history.",
               "Proposed splits use recalibration_guard_days around recalibration points; zero permits nearby rows in other sets."],
        rows=[],
    )

    def record(name, selected=None, seed=None, requested=None, quotas=None):
        counts, subsets = split_inventory(clean, wildfire, times, window_hours, settings, selected)
        recal = subsets.get("recalibration", np.array([], dtype=int))
        sorted_times = np.sort(times[recal])
        row = dict(scenario=name, selection_seed=seed, requested=requested,
                   selection_complete=None if requested is None else len(recal) == requested,
                   **counts,
                   min_recalibration_gap_days=float(np.diff(sorted_times).min()) if len(recal) > 1 else None,
                   recalibration_month_counts=np.bincount(months[recal], minlength=13)[1:].tolist(),
                   recalibration_year_counts=dict(zip(*np.unique(
                       [str(_matlab_datenum_to_datetime(t).year) for t in sorted_times], return_counts=True))),
                   requested_month_counts=None if quotas is None else quotas[1:].tolist())
        row["recalibration_year_counts"] = {k: int(v) for k, v in row["recalibration_year_counts"].items()}
        if selected is not None and len(recal) > 1 and row["min_recalibration_gap_days"] < args.gap_days:
            raise AssertionError("recalibration spacing violated")
        np.savez_compressed(args.output_dir / f"{name}.npz", **subsets)
        results["rows"].append(row)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        keys = sorted({k for r in results["rows"] for k in r if not isinstance(r[k], (list, dict))})
        with (args.output_dir / "counts.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results["rows"])
        print(f"{name}: recal={len(recal)}, recal_guard={row.get('recalibration_guard')}, "
              f"train={row.get('train')}, val={row.get('validation')}, control={row.get('control')}, "
              f"status={row['status']} {row['reason']}", flush=True)

    record("current")
    for count in args.counts:
        quotas = month_quotas(months[wildfire], count)
        for seed in args.seeds:
            selected = select_spaced(clean, times, months, quotas, args.gap_days, seed)
            record(f"n{count}_seed{seed}", selected, seed, count, quotas)
    record("earliest_first_maximum", maximum)
    print(f"Complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
