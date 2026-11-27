"""CRSP common-stock formation dates and daily beta inputs."""

from itertools import combinations_with_replacement

import polars as pl

from era_pl import load_parquet


_MONTH_KEYS = ["permno", "month"]


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


def load_daily_beta_inputs(
    first: pl.Expr,
    last: pl.Expr,
    *,
    preserve_order: bool = False,
) -> tuple[pl.DataFrame, pl.LazyFrame]:
    """Prepare the daily factors and stock observations used to estimate beta.

    Shifts the market factor on the trading-day calendar before joining it
    to stock returns. The contemporaneous market factor is `mkt_5`.

    Args:
        first: First stock-return date, inclusive, as a Polars date expression.
        last: Last stock-return date, inclusive, as a Polars date expression.
        preserve_order: Keep the stock-return order through the factor join.

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
        .join(daily_factors.lazy(), on="date", validate="m:1",
              maintain_order="left" if preserve_order else "none")
        .with_columns(retrf=pl.col("ret") - pl.col("rf"))
        .filter(pl.all_horizontal(pl.col("retrf", *offsets.values()).is_finite()))
        .select("permno", "month", "retrf", "days_left", *offsets.values())
    )
    return daily_factors, daily_excess


def product_name(left, right):
    return f"{left}_x_{right}"


def monthly_moments(returns, predictors, *, joint=False):
    pairs = [(x, "retrf") for x in predictors]
    pairs += list(combinations_with_replacement(predictors, 2)) if joint else [(x, x) for x in predictors]
    return returns.group_by(_MONTH_KEYS).agg(
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("retrf", *predictors).sum(),
        *((pl.col(a) * pl.col(b)).sum().alias(product_name(a, b)) for a, b in pairs),
    )


def add_anchors(monthly, anchors):
    return (
        monthly.join(anchors, on=_MONTH_KEYS, how="full", coalesce=True, validate="1:1")
        .sort(_MONTH_KEYS)
    )


def roll_moments(monthly, months):
    return (
        monthly.rolling("month", period=f"{months}mo", group_by="permno")
        .agg(pl.exclude(_MONTH_KEYS).sum())
    )


def daily_slope(market):
    n = pl.col("n")
    x = pl.col(market)
    numerator = pl.col(product_name(market, "retrf")) - x * pl.col("retrf") / n
    denominator = pl.col(product_name(market, market)) - x.pow(2) / n
    return pl.when(denominator > 0).then(numerator / denominator)


def rolling_daily_beta(monthly, months, minimum, label):
    return roll_moments(monthly, months).select(
        *_MONTH_KEYS,
        pl.when(pl.col("n") >= minimum).then(daily_slope("mkt_5")).alias(label),
    )


def load_monthly_betas(first: pl.Expr, last: pl.Expr) -> pl.DataFrame:
    """Estimate the 12-month daily beta for each common stock and formation month.

    Uses the same matched daily observations as the Beta chapter. A window
    includes the formation month and requires at least 200 observations.

    Args:
        first: First formation month, inclusive, as a month-start date expression.
        last: Last formation month, inclusive, as a month-start date expression.

    Returns:
        (polars.DataFrame): Formation-month observations with ``permno``, ``month``,
            and ``beta``. An unavailable estimate is null.
    """
    anchors = load_beta_universe(first, last).collect()
    _, daily_excess = load_daily_beta_inputs(
        first.dt.offset_by("-11mo"), last.dt.month_end(), preserve_order=True,
    )
    # Keep the order of the sums consistent for stocks at portfolio cutoffs.
    observations = daily_excess.select(*_MONTH_KEYS, "retrf", "mkt_5")
    moments = monthly_moments(observations.collect(), ["mkt_5"])
    monthly = add_anchors(moments, anchors)
    estimates = rolling_daily_beta(monthly, 12, 200, "beta")
    return anchors.join(estimates, on=_MONTH_KEYS, validate="1:1")


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
