"""Calendar-month return windows for the BEM momentum chapter."""

import polars as pl

from era_pl import load_parquet

from ._data import KEYS


def _compound_trailing_return(returns, name, length, skip, minimum):
    gross = pl.col("gross")
    return returns.rolling(
        "month", period=f"{length}mo", offset=f"-{length + skip}mo",
        group_by="permno", closed="right",
    ).agg(
        pl.when(gross.count() >= minimum).then(100 * (gross.product() - 1)).alias(name)
    )


def build_momentum(first=pl.date(1963, 6, 1), last=pl.date(2012, 11, 1)) -> pl.DataFrame:
    """Build the seven BEM momentum signals from legacy CRSP monthly returns.

    Each signal compounds finite returns in its calendar-month window.
    A signal is missing when its window has too few observed returns.
    The output covers rows in the CRSP monthly stock file. Join it to
    the formation-month universe separately.

    Args:
        first (polars.Expr): First formation month, inclusive.
        last (polars.Expr): Last formation month, inclusive.

    Returns:
        (polars.DataFrame): Stock-month keys and seven signals in
            percentage points. `mom`, `lag12`, and `lag6` exclude the
            formation month. `r12m`, `r9m`, `r6m`, and `r3m` include it.
    """
    windows = [
        ("mom", 11, 1, 9),
        ("lag12", 12, 1, 10),
        ("lag6", 6, 1, 5),
        ("r12m", 12, 0, 10),
        ("r9m", 9, 0, 7),
        ("r6m", 6, 0, 5),
        ("r3m", 3, 0, 3),
    ]
    returns = (
        load_parquet("msf", "crsp")
        .filter(pl.col("date").is_between(first.dt.offset_by("-12mo"), last.dt.month_end()))
        .select(
            "permno", month=pl.col("date").dt.month_start(),
            gross=pl.when(pl.col("ret").is_finite()).then(1 + pl.col("ret")),
        )
        .sort(KEYS)
        .collect()
    )
    in_sample = pl.col("month").is_between(first, last)
    keys = returns.filter(in_sample).select(KEYS)
    signals = [
        _compound_trailing_return(returns, name, length, skip, minimum)
        .filter(in_sample).sort(KEYS).select(name)
        for name, length, skip, minimum in windows
    ]
    return pl.concat([keys, *signals], how="horizontal")
