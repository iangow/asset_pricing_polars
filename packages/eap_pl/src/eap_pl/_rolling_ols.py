"""Calendar-month rolling ordinary least squares from monthly cross-products."""

from collections.abc import Sequence
from itertools import combinations_with_replacement

import numpy as np
import polars as pl


def _monthly_moments(data, keys, date_col, models):
    """Aggregate each set of variables over its own complete cases."""
    aggregates = []
    for prefix, terms in models.items():
        complete = pl.all_horizontal(pl.col(terms).is_finite()).fill_null(False)
        aggregates.append(complete.cast(pl.Int64).sum().alias(f"{prefix}_n"))
        aggregates.extend(
            pl.when(complete).then(pl.col(term)).sum().alias(f"{prefix}_sum_{i}")
            for i, term in enumerate(terms)
        )
        aggregates.extend(
            pl.when(complete).then(pl.col(terms[i]) * pl.col(terms[j])).sum()
            .alias(f"{prefix}_cross_{i}_{j}")
            for i, j in combinations_with_replacement(range(len(terms)), 2)
        )
    columns = dict.fromkeys(term for terms in models.values() for term in terms)
    return (
        data.lazy()
        .select(keys[0], pl.col(date_col).cast(pl.Date).dt.month_start().alias("month"),
                pl.col(list(columns)).cast(pl.Float64))
        .filter(pl.col("month").is_not_null())
        .group_by(keys)
        .agg(aggregates)
        .collect(engine="streaming")
    )


def _month_grid(monthly, anchors):
    keys = anchors.columns
    return (
        monthly.join(anchors, on=keys, how="full", coalesce=True, validate="1:1")
        .with_columns(pl.exclude(keys).fill_null(0))
        .sort(keys)
    )


def _roll_moments(grid, anchors, months):
    keys = anchors.columns
    values = [name for name, dtype in grid.schema.items() if dtype == pl.Float64 and name not in keys]
    # Sum the monthly floats in each window independently to avoid accumulating
    # roundoff as large observations enter and leave the rolling window.
    # Process contiguous groups in batches to bound the size of the window lists.
    ends = grid.group_by(keys[0], maintain_order=True).len()["len"].cum_sum()
    stops = ends.gather_every(512, offset=511).to_list()
    if not stops or stops[-1] != len(grid):
        stops.append(len(grid))
    totals = pl.concat([
        grid.slice(start, stop - start)
        .rolling("month", period=f"{months}mo", group_by=keys[0], closed="right")
        .agg(pl.exclude([*keys, *values]).sum(), pl.col(values).implode().list.sum())
        for start, stop in zip([0, *stops], stops)
    ])
    return anchors.join(totals, on=keys, how="left", maintain_order="left", validate="1:1")


def _solve_moments(rolled, prefix, predictors, min_obs):
    """Return intercept, slopes and residual standard error in anchor order."""
    n = rolled[f"{prefix}_n"].to_numpy()
    eligible = n >= max(min_obs, predictors + 2)
    selected = rolled.filter(pl.Series(eligible))
    count = n[eligible]
    width = predictors + 1
    estimates = np.full((len(rolled), width + 1), np.nan)
    if not len(count):
        return estimates

    sums = selected.select(f"{prefix}_sum_{i}" for i in range(width)).to_numpy()
    means = sums / count[:, None]
    cross = np.empty((len(count), width, width), dtype=np.float64)
    for i, j in combinations_with_replacement(range(width), 2):
        cross[:, i, j] = cross[:, j, i] = selected[f"{prefix}_cross_{i}_{j}"].to_numpy()

    # Center to fit an intercept and scale to check rank independently of units.
    centered = cross - sums[:, :, None] * sums[:, None, :] / count[:, None, None]
    xx = centered[:, :-1, :-1]
    variance = xx.diagonal(axis1=1, axis2=2)
    uncentered = cross.diagonal(axis1=1, axis2=2)[:, :-1]
    # A constant column can acquire a small positive variance through cancellation.
    variable = (variance > 8 * np.finfo(np.float64).eps * uncentered).all(axis=1)
    norm = np.sqrt(np.maximum(variance, 0))
    scale = np.where(norm > 0, norm, 1)
    gram = xx / (scale[:, :, None] * scale[:, None, :])
    usable = variable & (np.linalg.det(gram) > 1e-10)
    projected = centered[usable, :-1, -1] / scale[usable]
    coefficients = np.linalg.solve(gram[usable], projected[:, :, None]).squeeze(axis=2)
    slopes = coefficients / scale[usable]
    intercept = means[usable, -1] - np.sum(means[usable, :-1] * slopes, axis=1)
    rss = centered[usable, -1, -1] - np.sum(projected * coefficients, axis=1)
    rmse = np.sqrt(np.maximum(rss, 0) / (count[usable] - width))
    estimates[np.flatnonzero(eligible)[usable]] = np.column_stack((intercept, slopes, rmse))
    return estimates


def rolling_ols(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    y: str,
    x: Sequence[str],
    group_by: str,
    date_col: str,
    anchors: pl.DataFrame,
    months: int,
    min_obs: int,
) -> pl.DataFrame:
    """Fit rolling OLS with an intercept on a panel of numeric returns.

    Each fit includes the full formation month and the preceding
    `months - 1` calendar months. It uses all rows with finite outcomes and
    predictors, with any forecast or holding period after the formation month.

    Args:
        data: Panel with group, date, outcome, and predictor columns.
        y: Outcome column.
        x: Predictor columns in the desired output order.
        group_by: Group column, shared by data and anchors.
        date_col: Date column in data.
        anchors: Unique group and month keys. The `month` column contains
            the first date of each requested formation month as a Polars Date.
        months: Number of calendar months in each window.
        min_obs: Minimum matched observations required per window.
            A fit also requires positive residual degrees of freedom.

    Returns:
        (polars.DataFrame): In anchor order with the keys, `nobs`, `Intercept`,
            the named slope columns, and `rmse` (the residual standard error).
            Fit statistics are null when the window is too small or rank deficient.
    """
    keys = [group_by, "month"]
    if isinstance(x, str) or not x:
        raise ValueError("x must be a nonempty sequence of predictor column names")
    terms = [*x, y]
    output = ["Intercept", *x, "rmse"]
    if months < 1 or min_obs < 1:
        raise ValueError("months and min_obs must be positive")
    if len(set(terms)) != len(terms) or set(terms) & {*keys, date_col}:
        raise ValueError("Outcome and predictors must be distinct data columns")
    if len(set([*keys, "nobs", *output])) != len(keys) + 1 + len(output):
        raise ValueError("Group and predictor names must differ from result columns")
    if anchors.schema["month"] != pl.Date:
        raise TypeError("Anchor months must use the Polars Date type")
    if anchors[group_by].null_count() or not (
        anchors["month"].eq(anchors["month"].dt.month_start()).fill_null(False).all()
    ):
        raise ValueError("Anchors must have non-null groups and month-start dates")

    requested = anchors.select(keys)
    prefix = "rolling_ols" if group_by.startswith("ols_") else "ols"
    monthly = _monthly_moments(data, keys, date_col, {prefix: terms})
    rolled = _roll_moments(_month_grid(monthly, requested), requested, months)
    estimates = _solve_moments(rolled, prefix, len(x), min_obs)
    return rolled.select(*keys, pl.col(f"{prefix}_n").alias("nobs")).with_columns(
        pl.Series(name, values, nan_to_null=True) for name, values in zip(output, estimates.T)
    )
