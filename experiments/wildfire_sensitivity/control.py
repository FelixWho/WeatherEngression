"""Clean control group for the wildfire-sensitivity screening.

The screening asks whether a candidate covariate's wildfire-period values fall
outside what a clean-trained model predicts. Comparing that coverage against the
nominal interval level conflates three things: a real wildfire effect, model
under-dispersion, and the seasonal covariate shift (most wildfire trajectories
arrive in Jun-Sep). Only the first is of interest.

This module builds a reference intended to reduce the other two: clean trajectories
the model never trained on, drawn to match the wildfire group's month
distribution. Coverage on that control is what wildfire coverage gets compared
against. Month matching does not guarantee equal model error in the two groups.

Two details the ENA data forces:

- **Time-blocked split.** Trajectories are hourly and overlapping, so two rows an
  hour apart share 240 of their 241 hours. A random holdout would be the training
  data again under a different name. The split therefore cuts contiguous blocks of
  arrival time, discards the leading window of each holdout block, and purges fit
  rows immediately after a holdout block. Both boundaries need a history gap.
- **Month matching.** Without it the control is a different season than the
  wildfire group, which is the confound it exists to remove.
"""

from __future__ import annotations

import numpy as np

# MATLAB datenum is in days; the rest of this module works in hours.
HOURS_PER_DAY = 24.0


def time_blocks(time_days: np.ndarray, block_days: float) -> np.ndarray:
    """Bucket arrival times into contiguous blocks of ``block_days``.

    Block 0 starts at the earliest time given. Used both to split the data and,
    later, to measure how much a metric moves from one stretch of weather to the
    next, which is the only honest way to size its sampling error when neighbouring
    trajectories overlap.
    """

    hours = np.asarray(time_days, dtype=np.float64) * HOURS_PER_DAY
    return ((hours - hours.min()) / (block_days * HOURS_PER_DAY)).astype(np.int64)


def split_clean_holdout(
    clean_index: np.ndarray,
    time_days: np.ndarray,
    *,
    window_hours: float,
    holdout_fraction: float = 0.25,
    block_days: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split clean rows into training and holdout sets along arrival time.

    Rows are bucketed into contiguous ``block_days``-wide blocks of arrival time
    and whole blocks are assigned to one side or the other, so the two sets are
    separated in time rather than interleaved. Blocks are drawn per calendar
    month, which keeps every month represented in the holdout for the matching
    step. Each holdout block then drops its first ``window_hours`` of rows: those
    rows' back-trajectories extend into the preceding block, which may be a
    training block. Fit rows immediately following a holdout block are also
    purged, since their histories can reach backwards into the holdout.

    ``block_days`` must exceed the trajectory window, since the guard gap costs
    one window's worth of rows from every holdout block.

    Returns ``(fit_index, holdout_index)`` as indices into the full dataset.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must lie in (0, 1); got {holdout_fraction}")
    if block_days * HOURS_PER_DAY <= window_hours:
        raise ValueError(
            f"block_days={block_days:g} ({block_days * HOURS_PER_DAY:g}h) must exceed the "
            f"{window_hours:g}h trajectory window, or the guard gap empties every block"
        )

    clean_index = np.asarray(clean_index)
    hours = np.asarray(time_days, dtype=np.float64)[clean_index] * HOURS_PER_DAY
    block_of_row = time_blocks(np.asarray(time_days)[clean_index], block_days)
    block_ids = np.unique(block_of_row)
    if block_ids.size < 2:
        raise ValueError(
            f"clean rows span under two {block_days:g}-day blocks; "
            "widen the span or shrink block_days"
        )

    # The guard gap costs each block its leading ``window_hours``, and that depends
    # only on the block's own boundaries -- not on which side it is assigned to. So
    # work out what each block would actually contribute first, then choose.
    month_of_row = _months(time_days)[clean_index]
    survives = np.zeros(clean_index.size, dtype=bool)
    for block in block_ids:
        in_block = block_of_row == block
        survives |= in_block & (hours >= hours[in_block].min() + window_hours)

    # Label each block by the month it would actually contribute, which is not the
    # month it starts in: a block opening on 25 Jan has its first ten days eaten by
    # the guard, so everything it contributes is February. Labelling by start month
    # would file it under January and leave February unrepresented.
    block_month: dict[int, int] = {}
    for block in block_ids:
        contributed = month_of_row[(block_of_row == block) & survives]
        if contributed.size:
            block_month[int(block)] = int(np.bincount(contributed).argmax())

    # Draw a share of the blocks from each calendar month. Choosing at random
    # across all blocks instead would leave whole months out of the holdout, and
    # the month matching downstream would have nothing to draw on for those months.
    rng = np.random.default_rng(seed)
    holdout_blocks: set[int] = set()
    for month in range(1, 13):
        month_blocks = np.array(
            [b for b, m in block_month.items() if m == month], dtype=np.int64
        )
        if not month_blocks.size:
            continue
        n_take = max(1, int(round(holdout_fraction * month_blocks.size)))
        holdout_blocks.update(
            int(b) for b in rng.choice(month_blocks, size=n_take, replace=False)
        )

    is_holdout_block = np.isin(block_of_row, list(holdout_blocks))
    fit_mask = ~is_holdout_block
    holdout_mask = is_holdout_block & survives

    # Guard the other boundary too: a later fit trajectory must not reach back
    # into the end of a holdout block. Guarding only the holdout's start is not
    # enough when the next block is assigned to fitting.
    for block in holdout_blocks:
        end_hour = hours[block_of_row == block].max()
        fit_mask &= ~((hours > end_hour) & (hours < end_hour + window_hours))

    if not fit_mask.any() or not holdout_mask.any():
        raise ValueError("time-block split leaves an empty fit or holdout set")

    return clean_index[fit_mask], clean_index[holdout_mask]


def month_matched_subsample(
    pool_index: np.ndarray,
    reference_index: np.ndarray,
    time_days: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw rows from ``pool_index`` matching ``reference_index``'s month mix.

    Takes ``k * n_ref[m]`` pool rows from each calendar month ``m``, with ``k`` the
    largest factor every represented month can supply. This reproduces the
    reference group's monthly proportions up to integer rounding while keeping as much of the
    pool as possible. Mirrors ``covariate_wildfire_shift.month_matched_clean``,
    but matches between two arbitrary index sets rather than a boolean mask over
    the whole dataset.

    Reject a pool missing any reference month: otherwise the downstream comparison
    would retain wildfire rows from months that have no clean counterpart.
    """

    pool_index = np.asarray(pool_index)
    reference_index = np.asarray(reference_index)
    months = _months(time_days)
    pool_months = months[pool_index]
    reference_months = months[reference_index]

    usable: list[tuple[int, int, np.ndarray]] = []
    unmatched: list[int] = []
    for month in range(1, 13):
        n_reference = int((reference_months == month).sum())
        if not n_reference:
            continue
        candidates = pool_index[pool_months == month]
        if candidates.size:
            usable.append((month, n_reference, candidates))
        else:
            unmatched.append(month)

    if not usable:
        raise ValueError("no calendar month is represented in both the pool and the reference")
    if unmatched:
        raise ValueError(f"cannot month-match wildfire rows: no clean holdout rows in month(s) {unmatched}")

    k = min(candidates.size / n_reference for _, n_reference, candidates in usable)
    rng = np.random.default_rng(seed)
    picks = [
        rng.choice(candidates, size=max(1, int(round(k * n_reference))), replace=False)
        for _, n_reference, candidates in usable
    ]
    return np.sort(np.concatenate(picks))


def _months(time_days: np.ndarray) -> np.ndarray:
    """Calendar month (1-12) for each MATLAB datenum, as an array over all rows."""

    from data_generation.ena_weather import _matlab_datenum_to_datetime

    return np.array(
        [_matlab_datenum_to_datetime(float(t)).month for t in np.asarray(time_days)],
        dtype=np.int64,
    )
