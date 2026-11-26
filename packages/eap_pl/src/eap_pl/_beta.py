"""CRSP common-stock formation dates and daily beta inputs."""

import polars as pl

from era_pl import load_parquet


def load_beta_universe(first: pl.Expr, last: pl.Expr) -> pl.LazyFrame:
    """Select stock-month formation dates with common share codes.

    Uses historical CRSP names records. Eligibility is determined at the
    end of the formation month.

    Args:
        first: First formation month, inclusive, as a Polars date expression.
        last: Last formation month, inclusive, as a Polars date expression.

    Returns:
        (polars.LazyFrame): Unique `permno` and `month` pairs.
    """
    names = load_parquet("msenames", "crsp").filter(pl.col("shrcd").is_in([10, 11]))
    return (
        names
        .filter(pl.col("namedt") <= last.dt.month_end(), pl.col("nameendt") >= first)
        .with_columns(month=pl.date_ranges(
            pl.max_horizontal(pl.col("namedt").dt.month_start(), first),
            pl.min_horizontal(pl.col("nameendt").dt.month_start(), last),
            interval="1mo",
        ))
        .explode("month")
        .filter(pl.col("month").dt.month_end().is_between("namedt", "nameendt"))
        .select("permno", "month")
        .unique()
    )


def load_daily_beta_inputs(first: pl.Expr, last: pl.Expr) -> tuple[pl.DataFrame, pl.LazyFrame]:
    """Prepare the daily factors and stock observations used to estimate beta.

    Shifts the market factor on the trading-day calendar before joining it
    to stock returns. The contemporaneous market factor is `mkt_5`.

    Args:
        first: First stock-return date, inclusive, as a Polars date expression.
        last: Last stock-return date, inclusive, as a Polars date expression.

    Returns:
        (tuple[polars.DataFrame, polars.LazyFrame]): Daily factors and a query
            for matched stock excess returns with finite market predictors.
    """
    offsets = {k: f"mkt_{k + 5}" for k in range(-5, 6)}
    daily_factors = (
        load_parquet("factors_daily", "ff")
        .select("date", "mktrf", "rf")
        .sort("date")
        .with_columns(pl.col("mktrf").shift(-k).alias(name) for k, name in offsets.items())
        .with_columns(month=pl.col("date").dt.month_start())
        .with_columns(days_left=pl.len().over("month") - pl.col("date").cum_count().over("month"))
        .collect()
    )
    daily_excess = (
        load_parquet("dsf", "crsp")
        .filter(pl.col("date").is_between(first, last))
        .select("permno", "date", "ret")
        .join(daily_factors.lazy(), on="date", validate="m:1")
        .with_columns(retrf=pl.col("ret") - pl.col("rf"))
        .filter(pl.all_horizontal(pl.col("retrf", *offsets.values()).is_finite()))
        .select("permno", "month", "retrf", "days_left", *offsets.values())
    )
    return daily_factors, daily_excess


def load_annual_betas(start_year: int = 1988, end_year: int = 2012) -> pl.LazyFrame:
    """Load the December values of the 12-month daily beta.

    Estimates one slope per stock and calendar year with at least 200
    matched daily returns. The query reads the legacy CRSP daily file
    when collected.

    Args:
        start_year: First formation year, inclusive.
        end_year: Last formation year, inclusive.

    Returns:
        (polars.LazyFrame): December observations with `permno`, `year`, and `Beta`.
    """
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    anchors = (
        load_beta_universe(pl.date(start_year, 12, 1), pl.date(end_year, 12, 1))
        .filter(pl.col("month").dt.month() == 12)
        .select("permno", year=pl.col("month").dt.year())
    )
    _, daily_excess = load_daily_beta_inputs(pl.date(start_year, 1, 1), pl.date(end_year, 12, 31))
    variance = pl.col("mkt_5").var()
    estimates = (
        daily_excess.with_columns(year=pl.col("month").dt.year())
        .group_by("permno", "year")
        .agg(Beta=pl.when((pl.len() >= 200) & (variance > 0))
             .then(pl.cov("retrf", "mkt_5") / variance))
    )
    return anchors.join(estimates, on=["permno", "year"], how="left", validate="1:1")
