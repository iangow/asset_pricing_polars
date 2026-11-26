"""Local Parquet tables and the legacy CRSP annual and monthly panels."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl

from era_pl import load_parquet


KEYS = ["permno", "month"]


def load_annual_betas(start_year: int = 1988, end_year: int = 2012) -> pl.LazyFrame:
    """Load the December values of the 12-month daily beta.

    The returned query reads `DATA_DIR/eap/betas.parquet` when collected.

    Args:
        start_year: First formation year, inclusive.
        end_year: Last formation year, inclusive.

    Returns:
        (polars.LazyFrame): December observations with `permno`, `year`, and `Beta`.
    """
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year")
    return (
        load_parquet("betas", "eap")
        .with_columns(year=pl.col("month").dt.year())
        .filter(pl.col("month").dt.month() == 12, pl.col("year").is_between(start_year, end_year))
        .select("permno", "year", Beta="12M")
    )


def build_annual_panel(start_year: int = 1988, end_year: int = 2012) -> pl.DataFrame:
    """Join December stock characteristics to next-year excess returns.

    The beta sample defines the stock-year rows. Compounds available monthly
    CRSP returns and subtracts the compounded risk-free return, which requires
    all 12 months. The final formation year has no next-year return because
    the sample ends in that calendar year.

    Args:
        start_year: First formation year, inclusive.
        end_year: Last formation year and last calendar year of returns, inclusive.

    Returns:
        (polars.DataFrame): Stock-year observations with `permno`, `year`,
            `Beta`, market capitalization in millions (`MktCap`), its log
            (`Size`), signed book-to-market (`BM`), and next-year excess
            return in percentage points (`excess`).
    """
    betas = load_annual_betas(start_year, end_year)
    keys = ["permno", "year"]
    capitalizations = (
        load_parquet("sizes", "eap")
        .filter(pl.col("month").dt.month() == 12)
        .select("permno", year=pl.col("month").dt.year(), MktCap="cap", Size="log_cap")
        .filter(pl.col("year").is_between(start_year, end_year))
    )
    book_to_market = (
        load_parquet("book_to_market", "eap")
        .filter(pl.col("month").dt.month() == 12)
        .select("permno", year=pl.col("month").dt.year(), BM="bm_raw")
        .filter(pl.col("year").is_between(start_year, end_year))
    )
    stock_returns = (
        load_parquet("msf", "crsp")
        .filter(
            pl.col("date").is_between(pl.date(start_year + 1, 1, 1), pl.date(end_year, 12, 31)),
            pl.col("ret").is_finite(),
        )
        .with_columns(year=pl.col("date").dt.year())
        .group_by(keys)
        .agg(gross_stock=(1 + pl.col("ret")).product())
    )
    risk_free = (
        load_parquet("factors_monthly", "ff")
        .with_columns(year=pl.col("date").dt.year())
        .filter(pl.col("year").is_between(start_year + 1, end_year))
        .group_by("year")
        .agg(gross_rf=(1 + pl.col("rf")).product(), n=pl.col("rf").count())
        .filter(pl.col("n") == 12)
    )
    next_year = (
        stock_returns.join(risk_free, on="year", validate="m:1")
        .select(
            "permno",
            year=pl.col("year") - 1,
            excess=100 * (pl.col("gross_stock") - pl.col("gross_rf")),
        )
    )
    return (
        betas.join(capitalizations, on=keys, how="left", validate="1:1")
        .join(book_to_market, on=keys, how="left", validate="1:1")
        .join(next_year, on=keys, how="left", validate="1:1")
        .collect()
    )


def monthly_panel(sizes, first, last):
    """Join formation-month characteristics to next-month excess returns.

    Reads `eap/betas`, `crsp/msf`, `crsp/msedelist`, and
    `ff/factors_monthly` from the local Parquet repository. On a delisting,
    the recorded delisting return replaces the usual stock return. Missing
    delisting returns use −30% for poor performance and −100% otherwise.

    Args:
        sizes (polars.DataFrame or polars.LazyFrame): Characteristics keyed
            by `permno` and the first day of the formation `month`.
        first (polars.Expr): First formation month, such as `pl.date(1963, 6, 1)`.
        last (polars.Expr): Last formation month, inclusive.

    Returns:
        (polars.DataFrame): One row per beta observation, including the
            12-month daily `beta`, the supplied characteristics, and
            next-month `excess` and `market` in percentage points.
    """
    betas = load_parquet("betas", "eap").select(*KEYS, beta="12M")
    stocks = (
        load_parquet("msf", "crsp")
        .select("permno", month=pl.col("date").dt.month_start(), ret="ret")
        .filter(pl.col("month").is_between(first.dt.offset_by("1mo"), last.dt.offset_by("1mo")))
    )
    delists = (
        load_parquet("msedelist", "crsp")
        .select("permno", "dlret", "dlstcd", month=pl.col("dlstdt").dt.month_start())
        .filter(pl.col("month").is_between(first.dt.offset_by("1mo"), last.dt.offset_by("1mo")))
    )
    poor_performance = (
        pl.col("dlstcd").is_in([500, 520, 580, 584]) | pl.col("dlstcd").is_between(551, 574)
    )
    returns = (
        stocks.join(delists, on=KEYS, how="full", coalesce=True, validate="1:1")
        .select(
            "permno", holding="month",
            ret=pl.when(pl.col("dlstcd").is_not_null())
            .then(pl.coalesce("dlret", pl.when(poor_performance).then(-0.30).otherwise(-1.00)))
            .otherwise(pl.col("ret")),
        )
    )
    market = (
        load_parquet("factors_monthly", "ff")
        .select(
            holding=pl.col("date").dt.month_start(),
            market=100 * pl.col("mktrf"), rf=pl.col("rf"),
        )
    )
    return (
        betas.filter(pl.col("month").is_between(first, last))
        .join(sizes.lazy(), on=KEYS, how="left", validate="1:1")
        .with_columns(holding=pl.col("month").dt.offset_by("1mo"))
        .join(returns, on=["permno", "holding"], how="left", validate="1:1")
        .join(market, on="holding", how="left", validate="m:1")
        .with_columns(excess=100 * (pl.col("ret") - pl.col("rf")))
        .drop("ret", "rf", "holding")
        .collect()
    )


def write_eap_table(frame, table):
    """Write a derived stock-month table to `DATA_DIR/eap`.

    Sorts by `permno` and `month` before writing. A successful write
    atomically replaces the named table.

    Args:
        frame (polars.DataFrame): Table with `permno` and `month` columns.
        table (str): File stem, such as `betas`.
    """
    target = Path(os.environ["DATA_DIR"]).expanduser() / "eap" / f"{table}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, suffix=".parquet", delete=False) as temp:
        staging = Path(temp.name)
    try:
        frame.sort("permno", "month").write_parquet(staging)
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
