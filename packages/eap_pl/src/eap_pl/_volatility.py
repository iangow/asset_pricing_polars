"""Rolling total and idiosyncratic volatility for the BEM replication."""

import numpy as np
import polars as pl

from ._data import KEYS
from ._rolling_ols import _month_grid, _monthly_moments, _roll_moments, _solve_moments


def rolling_volatility(anchors, returns, factors, windows, periods_per_year):
    """Estimate annualized total and idiosyncratic volatility.

    Return and factor dates have the same frequency and convention. Factor
    returns and raw stock returns are in decimals. Each model applies its
    minimum observation count to the dates on which its inputs are finite.
    """
    requested = anchors.select(KEYS)
    models = {
        "iv_capm": ("mktrf", "excess"),
        "iv_ff": ("mktrf", "smb", "hml", "excess"),
        "iv_ffc": ("mktrf", "smb", "hml", "mom", "excess"),
    }
    panel = (
        returns.lazy().select("permno", "date", "ret")
        .join(factors.lazy().select("date", "rf", "mktrf", "smb", "hml", "mom"),
              on="date", how="left", validate="m:1")
        .with_columns(excess=pl.col("ret") - pl.col("rf"))
    )
    monthly = _monthly_moments(panel, KEYS, "date", {"raw": ("ret",), **models})
    grid = _month_grid(monthly, requested)
    annualize = 100 * np.sqrt(periods_per_year)
    estimates = []
    for label, (months, minimum) in windows.items():
        rolled = _roll_moments(grid, requested, months)
        n = pl.col("raw_n")
        sum_ret = pl.col("raw_sum_0")
        sum_squares = pl.col("raw_cross_0_0")
        centered_squares = (sum_squares - sum_ret.pow(2) / n).clip(lower_bound=0)
        total = pl.when(n >= max(minimum, 2)).then(annualize * (centered_squares / (n - 1)).sqrt())
        squared = pl.when(n >= max(minimum, 1)).then(annualize * (sum_squares / n).sqrt())
        estimates.extend(rolled.select(total.alias(f"vol_{label}"), squared.alias(f"vol_ss_{label}")).get_columns())
        for name, terms in models.items():
            rmse = _solve_moments(rolled, name, len(terms) - 1, minimum)[:, -1]
            estimates.append(pl.Series(f"{name}_{label}", annualize * rmse, nan_to_null=True))
    return requested.with_columns(estimates)
