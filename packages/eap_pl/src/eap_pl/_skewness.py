"""Rolling skewness measures for legacy CRSP returns."""

import numpy as np
import polars as pl

from ._data import KEYS
from ._return_windows import _demean, _regress, window_batches


def _skew(values, valid, count, minimum):
    centered = _demean(values, valid, count)
    divisor = np.maximum(count, 1)
    variance = (centered ** 2).sum(axis=1) / divisor
    scale = np.max(np.where(valid, np.abs(values), 0), axis=1)
    valid_variance = (count >= minimum) & (variance > (np.finfo(float).eps * scale) ** 2)
    return np.divide(
        (centered ** 3).sum(axis=1) / divisor,
        variance ** 1.5,
        out=np.full(len(count), np.nan),
        where=valid_variance,
    )


def rolling_skewness(anchors, returns, factors, windows):
    """Calculate total, co- and idiosyncratic skewness in calendar windows.

    Anchors have unique PERMNO and month-start dates. The stock return query
    has PERMNO, date and decimal return, and the factor frame has unique
    dates and decimal market, SMB, HML and risk-free returns. Use the same
    frequency and date convention for both inputs. Window definitions map
    labels to calendar-month lengths and minimum observation counts.
    """
    batches = window_batches(anchors, returns, factors, ["mktrf", "smb", "hml", "rf"])
    estimates = {}
    for label, (months, minimum) in windows.items():
        result = np.full((len(anchors), 3), np.nan)
        for rows, raw, within, count, (market, smb, hml, rf) in batches(months, minimum):
            excess = raw - rf

            result[rows, 0] = _skew(raw, within, count, minimum)
            valid = within & np.isfinite(excess) & np.isfinite(market)
            slopes, _ = _regress(excess, (market, market ** 2), valid, valid.sum(axis=1), minimum)
            result[rows, 1] = slopes[:, 1]
            valid &= np.isfinite(smb) & np.isfinite(hml)
            matched = valid.sum(axis=1)
            _, residuals = _regress(excess, (market, smb, hml), valid, matched, minimum)
            result[rows, 2] = _skew(residuals, valid, matched, minimum)

        for index, prefix in enumerate(("skew", "coskew", "idioskew")):
            name = f"{prefix}_{label}"
            estimates[name] = pl.Series(name, result[:, index], nan_to_null=True)
    return anchors.select(*KEYS).with_columns(estimates.values())
