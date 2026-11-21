"""Audit AD on synthetic scalar PIT at the actual clean observation dates.

Audit a saved screening control set, or compare spacing inventories with
equally sized, month-matched random selections. No models or recalibration
mappings are fitted. The production block-bootstrap test and the simpler
IID reference are reported separately.
Run research repetitions through SLURM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

from experiments.calibration import ad_block_bootstrap, validate_ad_settings
from experiments.validate_ad_thinning import iid_ad_pvalue, iid_ad_reference
from experiments.wildfire_sensitivity.control import _months, time_blocks


# rho is the shared fraction of latent Gaussian variance, not PIT correlation.
# tau is the e-folding time: correlation decays continuously with elapsed days.
SCENARIOS = {
    "iid_uniform": dict(rho=0., tau_days=3., miscalibrated=False),
    "moderate_3day": dict(rho=.3, tau_days=3., miscalibrated=False),
    "strong_3day": dict(rho=.8, tau_days=3., miscalibrated=False),
    "strong_10day": dict(rho=.8, tau_days=10., miscalibrated=False),
    "strong_30day": dict(rho=.8, tau_days=30., miscalibrated=False),
    "wide_iid": dict(rho=0., tau_days=3., miscalibrated=True),
    "wide_10day": dict(rho=.8, tau_days=10., miscalibrated=True),
}


def simulate_pit(times, rng, *, rho, tau_days, miscalibrated=False):
    """Uniform marginals with stationary dependence on an irregular calendar.

    A stationary Gaussian process L has correlation exp(-|dt|/tau). Set
    Z=sqrt(rho)*L + sqrt(1-rho)*independent_normal, then U=Phi(Z).
    Every U is exactly uniform in the null cases, including at duplicate dates.
    The alternative 0.1+0.8*U represents intervals that are too wide.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        raise ValueError("require a nonempty vector of finite times")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted")
    if not np.isfinite(rho) or not 0 <= rho <= 1 or not np.isfinite(tau_days) or tau_days <= 0:
        raise ValueError("require rho in [0, 1] and positive finite tau_days")
    latent, individual = rng.normal(size=(2, len(times)))
    decay = np.exp(-np.diff(times) / tau_days)
    innovation_sd = np.sqrt(-np.expm1(-2 * np.diff(times) / tau_days))
    for i, (a, sd) in enumerate(zip(decay, innovation_sd), start=1):
        latent[i] = a * latent[i - 1] + sd * latent[i]
    pit = norm.cdf(np.sqrt(rho) * latent + np.sqrt(1 - rho) * individual)
    return .1 + .8 * pit if miscalibrated else pit


def matched_random_rows(clean, months, selected, seed):
    """Same number in each calendar month as selected, with no spacing rule."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(months[selected], minlength=13)
    return np.concatenate([
        rng.choice(clean[months[clean] == month], size=int(counts[month]), replace=False)
        for month in range(1, 13)
    ])


def load_selections(directory, selection_seed, seasonal_count=75):
    """Use saved dates/row IDs only; never load trajectories or outcomes."""
    if seasonal_count < 2:
        raise ValueError("require at least two seasonal control points")
    metadata = json.loads((directory / "results.json").read_text())
    if metadata["settings"].get("spaced_role") != "control":
        raise ValueError("source inventory must space control, not recalibration")
    with np.load(directory / "population.npz") as saved:
        times, clean, source_index = saved["time_days"], saved["clean"], saved["source_index"]
    months = _months(times)
    selections = {}
    seasonal_name = f"spaced_{seasonal_count}"
    for name, filename in [("current", "current"), (seasonal_name, f"n{seasonal_count}_seed2228"),
                           ("spaced_maximum", "earliest_first_maximum")]:
        with np.load(directory / f"{filename}.npz") as saved:
            rows = saved["control"]
        if not len(rows) or len(np.unique(rows)) != len(rows) or not np.isin(rows, clean).all():
            raise ValueError(f"invalid clean control rows in {filename}")
        if name != "current":
            if np.diff(np.sort(times[rows])).min() < metadata["settings"]["gap_days"]:
                raise ValueError(f"spacing violated in {filename}")
            if name == seasonal_name and len(rows) != seasonal_count:
                raise ValueError(f"the requested {seasonal_count}-point selection is incomplete")
        selections[name] = rows
    # Fixed before generating any PIT. Random comparators have the exact same
    # month counts as their spaced partner, including the unbalanced maximum.
    for index, name in enumerate((seasonal_name, "spaced_maximum")):
        selections[name.replace("spaced", "random")] = matched_random_rows(
            clean, months, selections[name], np.random.SeedSequence([selection_seed, index]))
    for name, rows in selections.items():
        selections[name] = rows[np.argsort(times[rows], kind="stable")]
    return times, source_index, selections, metadata


def load_comparison(directories, selection_seed, seasonal_count=75):
    """Combine saved spacing inventories for paired trials on the same calendar.

    Keep one current-control baseline. Prefix the other arms by their minimum
    gap, so all spacings can share the same simulated PIT realization per trial.
    A single inventory keeps the original names and metadata.
    """
    if not directories:
        raise ValueError("require at least one inventory")
    first = load_selections(directories[0], selection_seed, seasonal_count)
    if len(directories) == 1:
        return first
    times, source_index, baseline, _ = first
    combined = {"current": baseline["current"]}
    sources = {}
    for i, directory in enumerate(directories):
        other_times, other_source, selections, metadata = (
            first if i == 0 else load_selections(directory, selection_seed, seasonal_count))
        if not np.array_equal(times, other_times) or not np.array_equal(source_index, other_source):
            raise ValueError("comparison inventories must use the same observation calendar and row IDs")
        if not np.array_equal(baseline["current"], selections["current"]):
            raise ValueError("comparison inventories must use the same current-control baseline")
        prefix = f"gap{metadata['settings']['gap_days']:g}"
        if prefix in sources:
            raise ValueError("comparison inventories must have distinct minimum gaps")
        sources[prefix] = dict(path=str(directory), settings=metadata["settings"])
        combined.update({f"{prefix}_{name}": rows for name, rows in selections.items() if name != "current"})
    return times, source_index, combined, {"settings": {"inventories": sources}}


def load_saved_control(directory, split):
    """Use actual screening rows and cached dates, without selecting new rows."""
    population = json.loads((directory / "results.json").read_text())
    metadata = json.loads(split.with_suffix(".json").read_text())
    if metadata["seeds"]["dataset"] != population["settings"]["dataset_seed"]:
        raise ValueError("saved split and calendar have different dataset seeds")
    with np.load(directory / "population.npz") as saved:
        times, source_index = saved["time_days"], saved["source_index"]
        clean, wildfire = saved["clean"], saved["wildfire"]
    with np.load(split) as saved:
        rows = saved["control"]
        if str(saved["evaluation_scope"]) != "arrival":
            raise ValueError("this scalar PIT audit requires an arrival split")
        if not np.array_equal(saved["wildfire"], wildfire):
            raise ValueError("saved split and calendar have different wildfire rows")
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(rows) < 2 or len(np.unique(rows)) != len(rows)
            or not np.isin(rows, clean).all()):
        raise ValueError("saved controls must be distinct clean row indices")
    if len(rows) != metadata["counts"]["control"]:
        raise ValueError("saved control count disagrees with its metadata")
    if not np.isfinite(times[rows]).all():
        raise ValueError("saved control timestamps must be finite")
    rows = rows[np.argsort(times[rows], kind="stable")]
    if np.diff(times[rows]).min() < metadata["settings"]["control_gap_days"]:
        raise ValueError("saved control spacing disagrees with its metadata")
    return times, source_index, {"saved_control": rows}, metadata


def rejection_summary(p_values, alpha):
    valid = [p for p in p_values if p is not None]
    rejected = sum(p < alpha for p in valid)
    ci = binomtest(rejected, len(valid)).proportion_ci() if valid else None
    return dict(valid=len(valid), invalid=len(p_values) - len(valid), rejected=rejected,
                rejection_rate=rejected / len(valid) if valid else None,
                ci95=[ci.low, ci.high] if ci else None)


def run_audit(times, selections, scenario, *, repetitions, bootstrap_replicates,
              null_simulations, seed, alpha=.05, progress=None):
    """Paired comparisons: each trial uses one PIT realization for all arms."""
    validate_ad_settings(alpha, bootstrap_replicates, seed)
    if repetitions < 1 or null_simulations < 1:
        raise ValueError("require positive trial and reference counts")
    # Simulating just the union has exactly the same Gaussian law as simulating
    # all clean dates and subsetting, and avoids unused values on every trial.
    union = np.unique(np.concatenate(list(selections.values())))
    union = union[np.argsort(times[union], kind="stable")]
    positions = {int(row): i for i, row in enumerate(union)}
    subsets = {name: np.array([positions[int(row)] for row in rows])
               for name, rows in selections.items()}
    references = {n: iid_ad_reference(n, null_simulations, np.random.SeedSequence([seed, 1, n]))
                  for n in sorted({len(rows) for rows in selections.values()})}
    results = {name: dict(n_selected=len(rows), n_blocks=len(np.unique(time_blocks(times[rows], 30.))),
                         block_bootstrap=[], iid_reference=[], invalid_reasons={})
               for name, rows in selections.items()}
    scenario_index = list(SCENARIOS).index(scenario)
    for trial in range(repetitions):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, scenario_index, trial]))
        pit = simulate_pit(times[union], rng, **SCENARIOS[scenario])
        # Separate deterministic bootstrap RNG, shared across arms for pairing.
        boot_seed = int(np.random.SeedSequence([seed, 2, scenario_index, trial]).generate_state(1)[0])
        for name, rows in selections.items():
            values = pit[subsets[name]]
            check = ad_block_bootstrap(values, times[rows], alpha=alpha,
                                       n_bootstrap=bootstrap_replicates, seed=boot_seed)
            result = results[name]
            result["block_bootstrap"].append(check["p_value"] if check["valid"] else None)
            if not check["valid"]:
                reason = check["reason"]
                result["invalid_reasons"][reason] = result["invalid_reasons"].get(reason, 0) + 1
            result["iid_reference"].append(iid_ad_pvalue(values, references[len(rows)]))
        if progress is not None and ((trial + 1) % 100 == 0 or trial + 1 == repetitions):
            progress(trial + 1, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, nargs="+", required=True,
                        help="One or more saved control-spacing inventories; multiple gaps are paired.")
    parser.add_argument("--control-split", type=Path,
                        help="Audit this screening_split.npz only; use one inventory for its cached calendar.")
    parser.add_argument("--seasonal-count", type=int, default=75,
                        help="Common complete seasonal count, saved with selection seed 2228 in each inventory.")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--null-simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2433)
    parser.add_argument("--selection-seed", type=int, default=2434)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_ad_settings(args.alpha, args.bootstrap_replicates, args.seed)
    if min(args.repetitions, args.null_simulations) < 1 or args.selection_seed < 0 or args.seasonal_count < 2:
        parser.error("require seasonal-count >= 2, positive simulation counts, and nonnegative seeds")
    if args.control_split is not None:
        if len(args.inventory) != 1:
            parser.error("--control-split requires exactly one calendar inventory")
        times, source_index, selections, source_metadata = load_saved_control(
            args.inventory[0], args.control_split)
    else:
        times, source_index, selections, source_metadata = load_comparison(
            args.inventory, args.selection_seed, args.seasonal_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "selections.npz", time_days=times,
                        source_index=source_index, **selections)
    months = _months(times)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["inventory"] = (str(args.inventory[0]) if len(args.inventory) == 1
                             else [str(path) for path in args.inventory])
    settings.update(scenario_parameters=SCENARIOS[args.scenario], source_settings=source_metadata["settings"],
                    scope="arrival", block_days=30., minimum_blocks=10,
                    pit_seed_rule="SeedSequence([seed, 0, scenario_index, trial])",
                    reference_seed_rule="SeedSequence([seed, 1, sample_size])",
                    bootstrap_seed_rule="SeedSequence([seed, 2, scenario_index, trial]).generate_state(1)[0]",
                    selection_seed_rule=(None if args.control_split else
                                         "SeedSequence([selection_seed, pair_index])"))
    if args.control_split:
        settings["seasonal_count"] = settings["selection_seed"] = None
        settings["source_split_seeds"] = source_metadata["seeds"]
    manifest = {name: dict(n=len(rows), min_gap_days=float(np.diff(times[rows]).min()),
                           month_counts=np.bincount(months[rows], minlength=13)[1:].tolist())
                for name, rows in selections.items()}
    print(json.dumps(dict(settings=settings, selections=manifest)), flush=True)

    def save(completed, results):
        summary = [dict(selection=name, method=method, n_selected=result["n_selected"],
                        n_blocks=result["n_blocks"], **rejection_summary(result[method], args.alpha))
                   for name, result in results.items()
                   for method in ("block_bootstrap", "iid_reference")]
        payload = dict(settings=settings, selections=manifest, repetitions_completed=completed,
                       complete=completed == args.repetitions,
                       known_uniform_null=not SCENARIOS[args.scenario]["miscalibrated"],
                       summary=summary, results=results,
                       interpretation=[
                           "Uniform scenarios measure false rejections; wide scenarios measure power against one alternative.",
                           "Block bootstrap calls the unchanged production AD test. IID reference is a separate proposed test.",
                           "Paired trials use the same process realization across selections; selections never use PIT values.",
                           ("Control rows are taken unchanged from the saved screening split; only sorted by date."
                            if args.control_split else
                            f"Seasonal selections contain {args.seasonal_count} points with saved seed 2228; maxima ignore seasonal matching."),
                           "Random comparators, when included, match their partner's count and month distribution, not its spacing.",
                           "Intervals describe trial variation for these fixed selections and IID Monte Carlo references.",
                           "Dependence strengths and time scales are stress tests, not fitted ENA correlation estimates.",
                           "Exact synthetic PIT only: no model fitting, finite-draw error, or recalibration-fit uncertainty.",
                           "These results do not establish independence or reliable p-values for real counterfactual models."])
        temporary = args.output_dir / "results.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output_dir / "results.json")
        with (args.output_dir / "summary.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"{args.scenario}: {completed}/{args.repetitions} trials saved", flush=True)
        if completed == args.repetitions:
            for row in summary:
                print(json.dumps(row), flush=True)

    run_audit(times, selections, args.scenario, repetitions=args.repetitions,
              bootstrap_replicates=args.bootstrap_replicates, null_simulations=args.null_simulations,
              seed=args.seed, alpha=args.alpha, progress=save)


if __name__ == "__main__":
    main()
