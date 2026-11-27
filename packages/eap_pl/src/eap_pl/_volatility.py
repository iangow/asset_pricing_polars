"""Rolling total and idiosyncratic volatility for the BEM replication."""

import numpy as np
import polars as pl

from ._data import KEYS
from ._return_windows import _demean, _regress, window_batches


def rolling_volatility(anchors, returns, factors, windows, periods_per_year):
    """Estimate annualized total and idiosyncratic volatility.

    Return and factor dates have the same frequency and convention. Factor
    returns and raw stock returns are in decimals. Each model applies its
    minimum observation count to the dates on which its inputs are finite.
    """
    batches = window_batches(anchors, returns, factors, ["mktrf", "smb", "hml", "rf", "mom"])
    annualize = 100 * np.sqrt(periods_per_year)
    estimates = {}
    measures = ("vol", "vol_ss", "iv_capm", "iv_ff", "iv_ffc")
    for label, (months, minimum) in windows.items():
        values = {name: np.full(len(anchors), np.nan) for name in measures}
        for rows, raw, within, count, (market, smb, hml, rf, mom) in batches(months, minimum):
            centered = _demean(raw, within, count)
            values["vol"][rows] = annualize * np.sqrt((centered ** 2).sum(axis=1) / (count - 1))
            values["vol_ss"][rows] = annualize * np.sqrt(
                (np.where(within, raw, 0) ** 2).sum(axis=1) / count
            )

            excess = raw - rf
            capm = within & np.isfinite(excess) & np.isfinite(market)
            ff = capm & np.isfinite(smb) & np.isfinite(hml)
            ffc = ff & np.isfinite(mom)
            models = [
                ("iv_capm", (market,), capm),
                ("iv_ff", (market, smb, hml), ff),
                ("iv_ffc", (market, smb, hml, mom), ffc),
            ]
            for name, regressors, valid in models:
                matched = valid.sum(axis=1)
                parameters = len(regressors) + 1
                _, residuals = _regress(excess, regressors, valid, matched,
                                        max(minimum, parameters + 1))
                rss = (np.where(valid, residuals, 0) ** 2).sum(axis=1)
                rse = np.sqrt(rss / np.maximum(matched - parameters, 1))
                values[name][rows] = np.where(matched >= max(minimum, parameters + 1),
                                              annualize * rse, np.nan)
        estimates.update({f"{name}_{label}": values[name] for name in measures})
    return anchors.select(*KEYS).with_columns(
        pl.Series(name, values, nan_to_null=True) for name, values in estimates.items()
    )
