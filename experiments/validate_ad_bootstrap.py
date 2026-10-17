"""Synthetic rejection-rate audit for AD; run substantial repetitions via SLURM.

This measures finite-block performance, not a proof of validity for real weather.
The recalibrated scenario also measures finite predictive sampling and fitting
uncertainty that the conditional block bootstrap itself does not resample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.calibration import ad_block_bootstrap

SCENARIOS = ("iid_uniform", "dependent_uniform", "unequal_dependent_uniform",
             "recalibrated_uniform", "miscalibrated", "moderate_dependent_uniform",
             "across_block_uniform")


def simulate_pit(scenario, n_blocks, rng, prediction_samples=400, block_sizes=None):
    """Known uniform marginals, with controlled Gaussian-copula dependence.

    The across-block case deliberately violates independent-block resampling:
    stationary block latent effects have AR(1) correlation 0.5.
    """
    sizes = (np.asarray(block_sizes, dtype=int) if block_sizes is not None else
             rng.integers(5, 36, n_blocks) if scenario == "unequal_dependent_uniform"
             else np.full(n_blocks, 20))
    if sizes.shape != (n_blocks,) or np.any(sizes < 1):
        raise ValueError("require one positive size per block")
    block = np.repeat(np.arange(n_blocks), sizes)
    rho = 0. if scenario == "iid_uniform" else .3 if scenario == "moderate_dependent_uniform" else .8
    latent = rng.normal(size=n_blocks)
    if scenario == "across_block_uniform":
        for index in range(1, n_blocks):
            latent[index] = .5 * latent[index - 1] + np.sqrt(.75) * latent[index]
    z = np.sqrt(rho) * latent[block] + np.sqrt(1 - rho) * rng.normal(size=len(block))
    pit = norm.cdf(z)
    if scenario == "miscalibrated":
        pit = .2 + .6 * pit  # Predictive distribution too wide.
    elif scenario == "recalibrated_uniform":
        # Base prediction Uniform(0,1); truth Uniform(.25,.75). Calibration
        # and control are independent draws with the same block dependence.
        fit_z = np.sqrt(rho) * rng.normal(size=n_blocks)[block] + np.sqrt(1 - rho) * rng.normal(size=len(block))
        fit_y = .25 + .5 * norm.cdf(fit_z)
        raw = RecalibratedEngressor(None)
        fit_pit = raw.pit_from_samples(rng.uniform(size=(len(block), prediction_samples)),
                                       fit_y, seed=int(rng.integers(2**32)))
        corrected = raw.fit_from_pit(fit_pit, [.025, .05, .95, .975])
        pit = corrected.pit_from_samples(rng.uniform(size=(len(block), prediction_samples)),
                                         .25 + .5 * pit, seed=int(rng.integers(2**32)))
    return pit, block * 30. + .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--blocks", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument("--block-sizes-from", type=Path,
                        help="Use only block sizes from a saved candidate calibration JSON; overrides --blocks.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=2430)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--prediction-samples", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    block_sizes = None
    if args.block_sizes_from is not None:
        block_sizes = json.loads(args.block_sizes_from.read_text())["metrics"]["ad"]["block_sizes"]
        if not block_sizes or any(not isinstance(n, int) or n < 1 for n in block_sizes):
            parser.error("source must contain positive integer block sizes")
        args.blocks = [len(block_sizes)]
    if args.repetitions < 1 or min(args.blocks) < 10:
        parser.error("require positive repetitions and at least 10 blocks")
    records = []
    for scenario_idx, scenario in enumerate(SCENARIOS):
        if scenario not in args.scenarios:
            continue
        for n_blocks in args.blocks:
            rejected, invalid = 0, 0
            p_values = []
            for repetition in range(args.repetitions):
                # Scenario IDs keep each case stable when running a subset.
                rng = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, scenario_idx, n_blocks, repetition]))
                pit, times = simulate_pit(scenario, n_blocks, rng, args.prediction_samples, block_sizes)
                result = ad_block_bootstrap(pit, times, alpha=args.alpha,
                    n_bootstrap=args.bootstrap_replicates, seed=int(rng.integers(2**32)))
                invalid += not result["valid"]
                if result["valid"]:
                    rejected += not result["accepted"]
                    p_values.append(result["p_value"])
            valid = args.repetitions - invalid
            interval = binomtest(rejected, valid).proportion_ci() if valid else None
            record = dict(scenario=scenario, n_blocks=n_blocks, rejected=rejected, invalid=invalid,
                          rejection_rate=rejected / valid if valid else None,
                          rejection_rate_ci95=[interval.low, interval.high] if interval else None,
                          p_values=p_values)
            record["known_uniform_null"] = scenario not in ("miscalibrated", "recalibrated_uniform")
            record["threshold_checks"] = {}
            # The same simulated p-values check multiple thresholds for free.
            for threshold in sorted({.01, .05, .10, args.alpha}):
                count = sum(p < threshold for p in p_values)
                ci = binomtest(count, valid).proportion_ci() if valid else None
                record["threshold_checks"][str(threshold)] = {
                    "rejected": count, "rate": count / valid if valid else None,
                    "ci95": [ci.low, ci.high] if ci else None,
                }
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "p_values"}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            settings["block_sizes"] = block_sizes
            args.output.write_text(json.dumps({"settings": settings,
                "results": records, "interpretation": "Approximate conditional block-bootstrap check; "
                "known_uniform_null identifies cases for false-rejection assessment. "
                "across_block_uniform stress-tests dependence across resampled blocks. "
                "recalibrated_uniform is NOT a guaranteed calibrated null after fitting the mapping. "
                "Inspect rejection rates before using AD for research classifications."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
