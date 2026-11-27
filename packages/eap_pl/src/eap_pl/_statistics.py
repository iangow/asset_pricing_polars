"""Period-by-period descriptive statistics and correlations."""

from itertools import combinations

import polars as pl

from ._data import KEYS


def cross_section(variable):
    x = pl.col(variable)
    return [
        x.mean().alias("Mean"), x.std().alias("SD"),
        x.skew(bias=False).alias("Skew"), x.kurtosis(fisher=True, bias=False).alias("Kurt."),
        x.min().alias("Min"), x.quantile(.05, interpolation="linear").alias("5%"),
        x.quantile(.25, interpolation="linear").alias("25%"), x.median().alias("Median"),
        x.quantile(.75, interpolation="linear").alias("75%"),
        x.quantile(.95, interpolation="linear").alias("95%"), x.max().alias("Max"),
        x.count().alias("n"),
    ]


def mean_cross_sections(panel, variables):
    """Average each descriptive statistic across monthly cross-sections.

    Args:
        panel (polars.DataFrame): Stock-month data with a `month` column.
        variables (dict[str, str]): Variable columns mapped to row labels.

    Returns:
        (polars.DataFrame): A row per variable, with the average of each
            month's mean, standard deviation, skewness, excess kurtosis,
            quantiles, extrema, and non-null count (rounded).
    """
    return pl.concat([
        panel.group_by("month").agg(cross_section(column))
        .select(pl.lit(label).alias("Measure"), pl.exclude("month").mean())
        for column, label in variables.items()
    ]).with_columns(pl.col("n").round().cast(pl.Int64))


def winsorized_corr(a, b):
    """Compute a Pearson correlation after pairwise finite-value filtering.

    Each variable is winsorized at the 0.5th and 99.5th percentiles of the
    paired sample.

    Args:
        a (polars.Expr): First variable.
        b (polars.Expr): Second variable.

    Returns:
        (polars.Expr): Pearson correlation of the winsorized values.
    """
    both = pl.all_horizontal(a.is_finite(), b.is_finite())
    return pl.corr(
        a.filter(both).eap.winsorize(prob=0.005, quantile_method="linear"),
        b.filter(both).eap.winsorize(prob=0.005, quantile_method="linear"),
    )


def period_correlations(panel, variables, period):
    """Compute cross-sectional correlations separately for each period.

    Pearson correlations use winsorized values. Spearman correlations use
    ranks of the original values. Both use pairwise finite observations.

    Args:
        panel (polars.DataFrame): Cross-sections stacked across periods.
        variables (Iterable[str]): Variable columns in display order.
        period (str): Period column, such as `month` or `year`.

    Returns:
        (polars.DataFrame): One row per period, with Pearson and Spearman
            columns for each pair.
    """
    cells = []
    for a, b in combinations(variables, 2):
        both = pl.all_horizontal(pl.col(a, b).is_finite())
        x, y = pl.col(a).filter(both), pl.col(b).filter(both)
        cells.extend([
            winsorized_corr(pl.col(a), pl.col(b)).alias(f"Pearson {a}–{b}"),
            pl.corr(x, y, method="spearman").alias(f"Spearman {a}–{b}"),
        ])
    return panel.group_by(period).agg(cells).with_columns(pl.exclude(period).fill_nan(None))


def correlation_matrix(averages, variables):
    """Put Pearson correlations below the diagonal and Spearman above it.

    Args:
        averages (Mapping[str, float]): Average correlations keyed by the
            column names produced by `period_correlations()`.
        variables (dict[str, str]): Variable columns mapped to display labels,
            in the order used to compute the correlations.

    Returns:
        (polars.DataFrame): A labeled matrix with a blank diagonal.
    """
    return pl.DataFrame({
        "Measure": list(variables.values()),
        **{
            label: [
                None if i == j else averages[
                    f"Spearman {a}–{b}" if i < j else f"Pearson {b}–{a}"
                ]
                for i, a in enumerate(variables)
            ]
            for j, (b, label) in enumerate(variables.items())
        },
    })


def persistence(panel, variables, lags):
    """Average cross-sectional correlations with future observations.

    Match stocks by `permno` across each elapsed calendar-month lag, use
    winsorized Pearson correlations, and weight each formation month equally.

    Args:
        panel (polars.DataFrame): Data keyed by `permno` and `month`.
        variables (dict[str, str]): Variable columns mapped to display labels.
        lags (Iterable[int]): Positive lags in months.

    Returns:
        (polars.DataFrame): One row per lag, with correlations formatted to
            three decimal places.
    """
    sample = panel.select(*KEYS, *variables)
    rows = []
    for lag in lags:
        future = sample.with_columns(month=pl.col("month").dt.offset_by(f"-{lag}mo"))
        coefficient = (
            sample.join(future, on=KEYS, suffix="_future", validate="1:1")
            .group_by("month")
            .agg(winsorized_corr(pl.col(x), pl.col(f"{x}_future")).alias(x) for x in variables)
            .select(pl.col(list(variables)).fill_nan(None).mean()).row(0)
        )
        rows.append([lag, *[f"{v:.3f}" if v is not None else "" for v in coefficient]])
    return pl.DataFrame(rows, schema=["Lag (months)", *variables.values()], orient="row")
