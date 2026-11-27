"""Batched calendar windows and centered regressions for rolling return measures."""

import numpy as np
import polars as pl


def _demean(values, valid, count):
    mean = np.where(valid, values, 0).sum(axis=1) / np.maximum(count, 1)
    return np.where(valid, values - mean[:, None], 0)


def _regress(outcome, predictors, valid, count, minimum):
    # Centering fits an intercept. Scaling means the rank check is independent of units.
    y = _demean(outcome, valid, count)
    x = np.stack([_demean(p, valid, count) for p in predictors], axis=2)
    lengths = np.sqrt(np.einsum("ijk,ijk->ik", x, x))
    normalized = x / np.where(lengths > 0, lengths, 1)[:, None, :]
    transpose = normalized.transpose(0, 2, 1)
    gram = transpose @ normalized
    usable = (count >= minimum) & (np.linalg.det(gram) > 1e-10)
    projected = (transpose @ y[:, :, None])[usable]
    coefficients = np.linalg.solve(gram[usable], projected).squeeze(axis=2)
    slopes = np.full_like(lengths, np.nan)
    slopes[usable] = coefficients / lengths[usable]
    residual = np.full_like(y, np.nan)
    residual[usable] = y[usable] - (normalized[usable] @ coefficients[:, :, None]).squeeze(axis=2)
    return slopes, residual


def window_batches(anchors, returns, factors, factor_columns):
    """Prepare stock returns and align factors for trailing calendar windows."""
    history = (
        returns.select("permno", "date", "ret")
        .filter(pl.col("ret").is_finite())
        .sort("permno", "date")
        .collect()
    )
    # A key packs stock and date so a binary search cannot cross security boundaries.
    stride = 1_000_000
    days = history["date"].cast(pl.Int32).to_numpy()
    stock_key = history["permno"].cast(pl.Int64).to_numpy() * stride + days
    anchor_key = anchors["permno"].cast(pl.Int64).to_numpy() * stride
    stock_return = history["ret"].to_numpy()
    first_day, last_day = days.min(), days.max()
    factor_days = factors["date"].cast(pl.Int32).to_numpy()
    covered = (factor_days >= first_day) & (factor_days <= last_day)
    factor_by_day = np.full((int(last_day) - int(first_day) + 1, len(factor_columns)), np.nan)
    factor_by_day[factor_days[covered] - first_day] = factors.select(factor_columns).to_numpy()[covered]

    def bound(month_offset):
        dates = anchors.select(pl.col("month").dt.offset_by(f"{month_offset}mo").cast(pl.Int32))
        return np.searchsorted(stock_key, anchor_key + dates.to_series().to_numpy())

    end = bound(1)

    def batches(months, minimum):
        start = bound(1 - months)
        counts = end - start
        eligible = np.flatnonzero(counts >= minimum)
        for offset in range(0, len(eligible), 4096):
            rows = eligible[offset:offset + 4096]
            count = counts[rows]
            steps = np.arange(count.max())
            within = steps[None, :] < count[:, None]
            indices = np.where(within, start[rows, None] + steps, 0)
            raw = stock_return[indices]
            aligned = factor_by_day[days[indices] - first_day].transpose(2, 0, 1)
            yield rows, raw, within, count, aligned

    return batches
