"""Pinned historical downloads from Kenneth French's data library."""

import re
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import polars as pl


_FRENCH = "mba.tuck.dartmouth.edu/pages/faculty/ken.french"
_FF3 = ("mktrf", "smb", "hml", "rf")
_COLUMNS = {"daily": _FF3, "monthly": _FF3, "momentum": ("mom",), "momentum_daily": ("mom",)}


def _wayback(timestamp: str, filename: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/http://{_FRENCH}/ftp/{filename}"


# Hash the member's original text so ZIP metadata do not affect verification.
_ARCHIVES = {
    ("daily", "201406"): (
        _wayback("20140827010427", "F-F_Research_Data_Factors_daily.zip"),
        "903bfba99f4f9c16e6f225368ce6c222fe91189b54975f7f4ab452af79af5c7e",
    ),
    ("monthly", "201406"): (
        _wayback("20140816081752", "F-F_Research_Data_Factors.zip"),
        "5eb23c6226bd566e6d2695d8c9d60ee830039009cbb587fc59f9255dfaf5a167",
    ),
    ("momentum", "201406"): (
        _wayback("20140908110708", "F-F_Momentum_Factor.zip"),
        "e34ede45fa759bc8b7c6c78921c6a4501158ecc8b18791073ee4b8673fb44fde",
    ),
    ("momentum_daily", "201406"): (
        _wayback("20140908112823", "F-F_Momentum_Factor_daily.zip"),
        "ca91ddd027296c5616813824b94b9d863258b455557f9a25e922afe7668fcb00",
    ),
    ("daily", "201509"): (
        _wayback("20151120080654", "F-F_Research_Data_Factors_daily_CSV.zip"),
        "bd272e7d362ad6782bc8b4d480a7e0db585ca0c462034f7596d4f96ad252d645",
    ),
    ("monthly", "201509"): (
        _wayback("20151120083153", "F-F_Research_Data_Factors_CSV.zip"),
        "5e7b04b8843ec2c1de8bd469c73d971ddd291000be4b9fce91a4aad68b4376fc",
    ),
    ("momentum", "201509"): (
        _wayback("20151117213821", "F-F_Momentum_Factor_CSV.zip"),
        "be34c1ff965adf806c241ed9e9f303ea21bffb4a10acaad75ef18907eab5a5bf",
    ),
    ("momentum_daily", "201509"): (
        _wayback("20151117165912", "F-F_Momentum_Factor_daily_CSV.zip"),
        "d2445bd8993974ccd815c036227282a4aefc148b6ef6c8783d0e45f8763c6598",
    ),
    ("monthly", "201607"): (
        f"https://{_FRENCH}/Data_Library/Historical_Archives/08%202016%20Update/"
        "ftp/F-F_Research_Data_Factors_CSV.zip",
        "d7bc4ad48ef22da8ccac45b4867262600b6e23972b61a67f27677e72f661cfcb",
    ),
    ("daily", "202412"): (
        f"https://{_FRENCH}/ftp_202412/F-F_Research_Data_Factors_daily_CSV.zip",
        "405edd66965f6a70d4956091fc37787faad85e86597c76ecad4bfb607253a05d",
    ),
    ("monthly", "202412"): (
        f"https://{_FRENCH}/ftp_202412/F-F_Research_Data_Factors_CSV.zip",
        "7d82ab71c4d15e5f2170dc6c8d464769e79c0b446271f8220ad2afcfbde5e56f",
    ),
    ("momentum", "202412"): (
        f"https://{_FRENCH}/ftp_202412/F-F_Momentum_Factor_CSV.zip",
        "2f83217efe88c4bfa496b707216c184290e85ffe00b85c7903d30fa444e507c3",
    ),
    ("momentum_daily", "202412"): (
        f"https://{_FRENCH}/ftp_202412/F-F_Momentum_Factor_daily_CSV.zip",
        "758deb481131aeebd33468504be92eefe4beba9b573cebc1d72abc2c32ce2b86",
    ),
}


def _archive_text(target: Path, url: str, expected_hash: str, timeout: float) -> bytes:
    try:
        contents = target.read_bytes()
    except FileNotFoundError:
        pass
    else:
        if sha256(contents).hexdigest() == expected_hash:
            return contents

    try:
        with urlopen(url, timeout=timeout) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            member, = [entry for entry in archive.infolist() if not entry.is_dir()]
            contents = archive.read(member)
    except (OSError, BadZipFile, ValueError) as err:
        raise RuntimeError(f"Could not read the Fama–French archive at {url}: {err}") from err

    if sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"Fama–French archive does not match its pinned checksum: {url}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        staging = Path(temp.name)
    try:
        staging.write_bytes(contents)
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return contents


def get_ff_factors(
    dataset: str,
    *,
    vintage: str | int = "201406",
    cache_dir: str | Path = "data/french-factors",
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download and cache a vintage of Kenneth French's US factors.

    Args:
        dataset: "daily" or "monthly" for the three Fama–French factors
            and the risk-free rate, "momentum" for monthly momentum, or
            "momentum_daily" for daily momentum.
        vintage: CRSP vintage in YYYYMM format. June 2014, September 2015,
            and December 2024 cover all four datasets. July 2016 covers
            monthly three-factor data.
        cache_dir: Directory for the original text from the ZIP file. A
            cached file is reused when it matches the source checksum.
        timeout: Download timeout in seconds.

    Returns:
        (polars.DataFrame): Sorted dates and decimal returns. Monthly dates
            are the first day of the month. Factor columns are
            mktrf, smb, hml, and rf, or mom.
    """
    if dataset not in _COLUMNS:
        raise ValueError(f"Unknown factor dataset {dataset!r}. Choose from: {', '.join(_COLUMNS)}.")
    vintage = str(vintage)
    if (dataset, vintage) not in _ARCHIVES:
        available = ", ".join(date for kind, date in _ARCHIVES if kind == dataset)
        raise ValueError(f"No archive for {dataset!r} vintage {vintage!r}. Available: {available}.")

    url, expected_hash = _ARCHIVES[dataset, vintage]
    target = Path(cache_dir).expanduser() / f"{dataset}-{vintage}.txt"
    contents = _archive_text(target, url, expected_hash, timeout)
    columns = _COLUMNS[dataset]
    daily = dataset in ("daily", "momentum_daily")
    date_digits = 8 if daily else 6
    date_pattern = re.compile(rf"^\s*\d{{{date_digits}}}[\s,]")
    rows = (
        re.sub(r"[\s,]+", ",", line.strip())
        for line in contents.decode("latin-1").splitlines()
        if date_pattern.match(line)
    )
    date = pl.col("date") if daily else pl.col("date") + "01"
    return (
        pl.read_csv(
            StringIO("\n".join(rows)),
            has_header=False,
            schema={"date": pl.String, **dict.fromkeys(columns, pl.Float64)},
            null_values=["-99.99", "-999"],
        )
        .with_columns(date.str.to_date("%Y%m%d").alias("date"), pl.col(columns) / 100)
        .sort("date")
    )
