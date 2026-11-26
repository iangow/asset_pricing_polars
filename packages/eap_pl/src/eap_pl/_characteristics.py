"""Construct the stock-month size and book-to-market inputs."""

from pathlib import Path

import polars as pl

from era_pl import load_parquet


def build_sizes(cpi_path: str | Path = "data/cpi-aucns.csv") -> pl.DataFrame:
    """Build monthly and June-based market capitalization for the BEM sample.

    Reads legacy CRSP monthly stock and names records. Retains earlier
    December capitalizations for book-to-market and marks common-stock
    eligibility separately from capitalization.

    Args:
        cpi_path: Local CPI-U CSV used to express market capitalization in
            December 2012 dollars.

    Returns:
        (polars.DataFrame): Stock-month keys, eligibility, exchange code,
            nominal and real capitalizations, and their logs.
    """
    prices = (
        load_parquet("msf", "crsp")
        .filter(pl.col("date").is_between(pl.date(1962, 12, 1), pl.date(2012, 12, 31)))
        .with_columns(
            month=pl.col("date").dt.month_start(),
            cap=pl.when((pl.col("altprc") != 0) & (pl.col("shrout") > 0))
            .then((pl.col("altprc") * pl.col("shrout")).abs() / 1000),
        )
        .select("permno", "month", "cap")
    )
    names = (
        load_parquet("msenames", "crsp")
        .select("permno", "namedt", "nameendt", "shrcd", "exchcd")
        .sort("namedt")
    )
    classified_months = (
        prices
        .with_columns(month_end=pl.col("month").dt.month_end())
        .sort("month_end")
        .join_asof(
            names, left_on="month_end", right_on="namedt",
            by="permno", check_sortedness=False,
        )
        .with_columns(
            common=(pl.col("month_end").is_between("namedt", "nameendt")
                    & pl.col("shrcd").is_in([10, 11])).fill_null(False),
            exchcd=pl.when(pl.col("month_end").is_between("namedt", "nameendt"))
            .then(pl.col("exchcd")),
        )
        .select("permno", "month", "common", "exchcd", "cap")
    )

    june_caps = (
        prices
        .filter(pl.col("month").dt.month() == 6)
        .select("permno", june="month", cap_june="cap")
    )
    june_year = pl.col("month").dt.year() - (pl.col("month").dt.month() < 6)
    stock_months = (
        classified_months
        .with_columns(june=pl.date(june_year, 6, 1))
        .join(june_caps, on=["permno", "june"], how="left", validate="m:1")
        .drop("june")
    )

    cpi = pl.read_csv(cpi_path, try_parse_dates=True).rename(
        {"observation_date": "month", "CPIAUCNS": "cpi"}
    )
    cpi_2012 = cpi.filter(pl.col("month") == pl.date(2012, 12, 1))["cpi"].item()

    all_sizes = (
        stock_months
        .join(cpi.lazy(), on="month", how="left", validate="m:1")
        .with_columns(
            cap_real=pl.col("cap") * cpi_2012 / pl.col("cpi"),
            cap_june_real=pl.col("cap_june") * cpi_2012 / pl.col("cpi"),
        )
        .drop("cpi")
        .with_columns(
            pl.col(var).log().alias(f"log_{var}")
            for var in ["cap", "cap_real", "cap_june", "cap_june_real"]
        )
        .collect()
    )
    return all_sizes


def build_book_to_market(sizes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Construct monthly book-to-market using prior-year fiscal book equity.

    Uses valid CRSP–Compustat links or unambiguous historical CUSIPs.
    Each ratio becomes available the June after the book year.
    The output includes signed ratios for the annual methods chapters and
    positive ratios for the monthly value replication.

    Args:
        sizes: The stock-month capitalization table from `build_sizes()`.

    Returns:
        (polars.DataFrame): Stock-month keys, book year, book equity,
            prior December market capitalization, and book-to-market measures.
    """
    last = pl.date(2012, 11, 1)
    keys = ["permno", "month"]

    funda = (
        load_parquet("funda", "comp")
        .filter(
            pl.col("indfmt") == "INDL", pl.col("datafmt") == "STD",
            pl.col("consol") == "C", pl.col("popsrc") == "D",
            pl.col("datadate").dt.year().is_between(1962, 2011),
        )
        .select(
            "gvkey", "datadate", year=pl.col("datadate").dt.year(),
            cusip=pl.col("cusip").str.slice(0, 8),
            be=pl.col("seq") + pl.col("txdb") + pl.col("itcb").fill_null(0)
            - pl.coalesce("pstkrv", "pstkl", "pstk", pl.lit(0)),
        )
        .filter(pl.col("datadate") == pl.col("datadate").max().over("gvkey", "year"))
        .unique()
        .filter(pl.len().over("gvkey", "year") == 1)
    )

    annual_keys = ["permno", "year"]
    links = (
        load_parquet("ccmxpf_lnkused", "crsp")
        .filter(pl.col("usedflag") == 1, pl.col("ulinktype").is_in(["LU", "LC"]), pl.col("upermno") > 0)
        .select(gvkey="ugvkey", permno=pl.col("upermno").cast(pl.Int32),
                start="ulinkdt", end="ulinkenddt")
    )
    linked_book = (
        funda.join(links, on="gvkey")
        .filter(pl.col("datadate") >= pl.col("start"),
                (pl.col("datadate") <= pl.col("end")) | pl.col("end").is_null())
        .select("gvkey", *annual_keys, "be").unique()
    )
    cusips = load_parquet("msenames", "crsp").select("permno", "namedt", "nameendt", cusip="ncusip")
    cusip_book = (
        funda.join(linked_book.select("gvkey", "year"), on=["gvkey", "year"], how="anti")
        .filter(pl.col("cusip").str.len_chars() == 8)
        .join(cusips, on="cusip")
        .filter(pl.col("datadate").is_between("namedt", "nameendt"))
        .select("gvkey", *annual_keys, "be").unique()
        .filter(pl.len().over("gvkey", "year") == 1)
    )
    book = (
        pl.concat([linked_book, cusip_book])
        .filter(pl.len().over(*annual_keys) == 1, pl.col("be").is_finite())
        .select(*annual_keys, "be")
    )

    december_cap = (
        sizes.lazy().filter(pl.col("month").dt.month() == 12)
        .select("permno", year=pl.col("month").dt.year(), me="cap")
    )
    book_to_market = (
        book.join(december_cap, on=["permno", "year"], validate="1:1")
        .with_columns(
            available=pl.date(pl.col("year") + 1, 6, 1),
            bm_raw=pl.col("be") / pl.col("me"),
        )
        .with_columns(
            month=pl.date_ranges(
                pl.col("available"), pl.col("available").dt.offset_by("11mo"),
                interval="1mo",
            ),
            bm=pl.when(pl.col("bm_raw") > 0).then(pl.col("bm_raw")),
        )
        .explode("month")
        .filter(pl.col("month") <= last.dt.offset_by("1mo"))
        .select(*keys, book_year="year", be="be", me="me", bm_raw="bm_raw",
                bm="bm", log_bm=pl.col("bm").log())
        .collect()
    )
    return book_to_market
