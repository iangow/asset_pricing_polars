"""Cumulative factor and market return plots."""

import polars as pl

from era_pl import load_parquet


def plot_cumulative_returns(returns, y_label="Cumulative return"):
    """Plot compounded and cumulative log returns in separate panels.

    Args:
        returns (polars.DataFrame): Data with `date`, `compounded`, and
            `cumulative_log` columns. Supply returns as fractions.
        y_label (str): Vertical axis label.

    Returns:
        (plotnine_polars.ggplot): A plot with percentage axis labels.
    """
    from plotnine_polars import aes

    labels = {"compounded": "Compounded", "cumulative_log": "Cumulative log"}
    return (
        returns.select("date", *labels).rename(labels)
        .unpivot(index="date", variable_name="series", value_name="cumulative_return")
        .with_columns(pl.col("series").cast(pl.Enum(list(labels.values()))))
        .ggplot(aes(x="date", y="cumulative_return", color="series", linetype="series"))
        .geom_line()
        .facet_wrap("series", ncol=1, scales="free_y")
        .scale_x_date(date_breaks="10 years", date_labels="%Y")
        .scale_y_continuous(labels=lambda values: [f"{value:,.0%}" for value in values])
        .scale_color_manual(values=["#15627c", "#9b552a"], guide=None)
        .scale_linetype_manual(values=["solid", "dashed"], guide=None)
        .labs(x="Month", y=y_label)
        .theme_minimal()
        .add_theme(figure_size=(8.2, 5))
    )


def plot_factor(factor):
    """Plot cumulative Fama–French factor returns from July 1926 to December 2012.

    Reads `ff/factors_monthly` from the local Parquet repository and
    expects a complete monthly series over that interval.

    Args:
        factor (str): Factor column in the monthly file, such as `smb` or `hml`.

    Returns:
        (plotnine_polars.ggplot): Compounded and cumulative log return panels.
    """
    returns = (
        load_parquet("factors_monthly", "ff")
        .filter(pl.col("date").is_between(pl.date(1926, 7, 1), pl.date(2012, 12, 31)))
        .select("date", factor)
        .sort("date")
        .with_columns(
            compounded=(1 + pl.col(factor)).cum_prod() - 1,
            cumulative_log=pl.col(factor).log1p().cum_sum(),
        )
        .collect()
    )
    assert returns.height == 1038 and returns[factor].null_count() == 0
    return plot_cumulative_returns(returns)
