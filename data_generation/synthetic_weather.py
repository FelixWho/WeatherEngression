"""CLI and dispatch helpers for synthetic weather-like data generators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover - supports direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_generation.common import SyntheticWeatherConfig
    from data_generation.hurdle_lognormal import generate_hurdle_lognormal_dataset
    from data_generation.narx_garch import generate_narx_garch_dataset
    from data_generation.narx_gaussian import generate_narx_gaussian_dataset
    from data_generation.narx_student_t import generate_narx_student_t_dataset
    from data_generation.preadditive import generate_preadditive_dataset, monotone_response
    from data_generation.regime_mixture import generate_regime_mixture_dataset
else:
    from .common import SyntheticWeatherConfig
    from .hurdle_lognormal import generate_hurdle_lognormal_dataset
    from .narx_garch import generate_narx_garch_dataset
    from .narx_gaussian import generate_narx_gaussian_dataset
    from .narx_student_t import generate_narx_student_t_dataset
    from .preadditive import generate_preadditive_dataset, monotone_response
    from .regime_mixture import generate_regime_mixture_dataset


MODEL_GENERATORS = {
    "preadditive": generate_preadditive_dataset,
    "narx_gaussian": generate_narx_gaussian_dataset,
    "narx_student_t": generate_narx_student_t_dataset,
    "narx_garch": generate_narx_garch_dataset,
    "regime_mixture": generate_regime_mixture_dataset,
    "hurdle_lognormal": generate_hurdle_lognormal_dataset,
}

MODEL_NAMES = tuple(MODEL_GENERATORS)
NON_SAMPLE_FIELDS = {"feature_names", "model"}


def generate_dataset(model: str, config: SyntheticWeatherConfig) -> dict[str, np.ndarray]:
    return MODEL_GENERATORS[model](config)


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_samples(dataset: dict[str, np.ndarray]) -> None:
    """Print generated observations as JSON Lines."""
    n = len(dataset["y"])
    metadata: dict[str, object] = {"type": "metadata", "n_observations": n}
    for key, value in dataset.items():
        array = np.asarray(value)
        if key in NON_SAMPLE_FIELDS or array.shape[:1] != (n,):
            metadata[key] = _to_jsonable(array)
    print(json.dumps(metadata))

    for i in range(n):
        record: dict[str, object] = {"type": "sample", "index": i}
        for key, value in dataset.items():
            if key in NON_SAMPLE_FIELDS:
                continue
            array = np.asarray(value)
            if array.shape[:1] == (n,):
                record[key] = _to_jsonable(array[i])
        print(json.dumps(record))


def reference_conditional_samples(
    dataset: dict[str, np.ndarray],
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample from the known conditional law saved in a generated dataset."""
    rng = np.random.default_rng(seed)
    model = str(np.asarray(dataset["model"]))
    n = len(dataset["y"])

    if model == "preadditive":
        phi = np.asarray(dataset["phi"], dtype=np.float64)
        noise_scale = float(np.asarray(dataset["noise_scale"]))
        eta = rng.normal(0.0, noise_scale, size=(n, n_samples))
        return monotone_response(phi[:, None] + eta)

    if model == "narx_gaussian":
        mu = np.asarray(dataset["mu"], dtype=np.float64)
        sigma = np.asarray(dataset["sigma"], dtype=np.float64)
        return rng.normal(mu[:, None], sigma[:, None], size=(n, n_samples))

    if model == "narx_student_t":
        mu = np.asarray(dataset["mu"], dtype=np.float64)
        scale = np.asarray(dataset["scale"], dtype=np.float64)
        df = float(np.asarray(dataset["df"]))
        return mu[:, None] + scale[:, None] * rng.standard_t(df, size=(n, n_samples))

    if model == "narx_garch":
        mu = np.asarray(dataset["mu"], dtype=np.float64)
        sigma = np.asarray(dataset["sigma"], dtype=np.float64)
        return rng.normal(mu[:, None], sigma[:, None], size=(n, n_samples))

    if model == "regime_mixture":
        weights = np.asarray(dataset["mixture_weights"], dtype=np.float64)
        weights = weights / weights.sum(axis=1, keepdims=True)
        means = np.asarray(dataset["mixture_means"], dtype=np.float64)
        sigmas = np.asarray(dataset["mixture_sigmas"], dtype=np.float64)
        draws = np.empty((n, n_samples), dtype=np.float64)
        for i in range(n):
            regimes = rng.choice(weights.shape[1], size=n_samples, p=weights[i])
            draws[i] = rng.normal(means[i, regimes], sigmas[i, regimes])
        return draws

    if model == "hurdle_lognormal":
        p_wet = np.asarray(dataset["p_wet"], dtype=np.float64)
        log_mu = np.asarray(dataset["log_mu"], dtype=np.float64)
        log_sigma = np.asarray(dataset["log_sigma"], dtype=np.float64)
        wet = rng.binomial(1, p_wet[:, None], size=(n, n_samples)).astype(bool)
        draws = np.zeros((n, n_samples), dtype=np.float64)
        mean_grid = np.broadcast_to(log_mu[:, None], (n, n_samples))
        sigma_grid = np.broadcast_to(log_sigma[:, None], (n, n_samples))
        draws[wet] = rng.lognormal(mean=mean_grid[wet], sigma=sigma_grid[wet])
        return draws

    raise ValueError(f"unknown model in dataset: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="narx_gaussian")
    parser.add_argument("--out", type=Path, default=None, help="Optional .npz output path. Omit to print JSON Lines.")
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    args = parser.parse_args()

    config = SyntheticWeatherConfig(
        n_steps=args.n_steps,
        window=args.window,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    dataset = generate_dataset(args.model, config)
    if args.out is None:
        print_samples(dataset)
        return

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dataset)
    print(
        f"wrote {out} with model={args.model}, "
        f"X={dataset['X'].shape}, y={dataset['y'].shape}"
    )


if __name__ == "__main__":
    main()
