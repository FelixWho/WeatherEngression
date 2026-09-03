"""Shared ENA wildfire/control partitioning.

The wildfire arm uses one requested biomass-burning criterion. The clean arm is
stricter: neither biomass-burning criterion may be active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_generation.ena_weather import (
    ENADataset,
    load_ena_supervised_dataset,
    select_real_split,
)
from experiments.pipeline import set_reproducible_seeds


@dataclass(frozen=True)
class WildfireArms:
    """Train/test row indices for the wildfire and clean arms."""

    train_wildfire: np.ndarray
    train_clean: np.ndarray
    test_wildfire: np.ndarray
    test_clean: np.ndarray


def partition_wildfire_arms(
    dataset: ENADataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    wildfire_flag: str = "BB_criterion1",
) -> WildfireArms:
    """Partition existing train/test indices into wildfire and clean arms."""

    required_flags = {wildfire_flag, "BB_criterion1", "BB_criterion2"}
    missing_flags = required_flags.difference(dataset.flags)
    if missing_flags:
        raise KeyError(
            f"missing wildfire flags {sorted(missing_flags)}; "
            f"available: {sorted(dataset.flags)}"
        )

    wildfire_index = np.flatnonzero(dataset.flags[wildfire_flag])
    clean_index = np.flatnonzero(
        ~(dataset.flags["BB_criterion1"] | dataset.flags["BB_criterion2"])
    )

    return WildfireArms(
        train_wildfire=np.intersect1d(train_index, wildfire_index),
        train_clean=np.intersect1d(train_index, clean_index),
        test_wildfire=np.intersect1d(test_index, wildfire_index),
        test_clean=np.intersect1d(test_index, clean_index),
    )


def load_wildfire_arms(
    *,
    mat_path: str,
    target: str,
    log_ccn: bool,
    split: str,
    seq_stride: int,
    max_samples: int | None,
    train_size: int | None,
    test_size: int | None,
    seed: int,
    wildfire_flag: str,
) -> tuple[ENADataset, WildfireArms]:
    """Load ENA data, build the requested split, and partition its arms."""

    set_reproducible_seeds(seed)
    dataset = load_ena_supervised_dataset(
        mat_path=mat_path,
        target=target,
        max_samples=max_samples,
        seq_stride=seq_stride,
        seed=seed,
        log_ccn=log_ccn,
    )
    train_index, test_index = select_real_split(
        dataset=dataset,
        split=split,
        train_size=train_size,
        test_size=test_size,
        seed=seed + 17,
    )
    arms = partition_wildfire_arms(
        dataset=dataset,
        train_index=train_index,
        test_index=test_index,
        wildfire_flag=wildfire_flag,
    )
    return dataset, arms
