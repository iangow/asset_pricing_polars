"""Shared calculations for the book's legacy CRSP portfolio chapters."""

from itertools import combinations

import polars as pl
import statspai as sp

from era_pl import load_parquet


KEYS = ["permno", "month"]


def monthly_panel(sizes, first, last):
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
    return pl.concat([
        panel.group_by("month").agg(cross_section(column))
        .select(pl.lit(label).alias("Measure"), pl.exclude("month").mean())
        for column, label in variables.items()
    ]).with_columns(pl.col("n").round().cast(pl.Int64))


def clip_tails(x):
    return x.clip(
        x.quantile(.005, interpolation="linear"),
        x.quantile(.995, interpolation="linear"),
    )


def winsorized_corr(a, b):
    both = pl.all_horizontal(a.is_finite(), b.is_finite())
    return pl.corr(clip_tails(a.filter(both)), clip_tails(b.filter(both)))


def period_correlations(panel, variables, period):
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


def assign_portfolio(data, variable, bins, group, *, nyse=False, label="portfolio"):
    x = pl.col(variable)
    reference = x.filter(pl.col("exchcd") == 1) if nyse else x
    assignment = 1 + pl.sum_horizontal(
        (
            (x > reference.quantile(i / bins, interpolation="linear")).cast(pl.UInt8)
            for i in range(1, bins)
        ),
        ignore_nulls=False,
    )
    return data.with_columns(assignment.over(group).alias(label))


def portfolio_months(members, groups, metrics=(), *, weight="cap"):
    valid = pl.all_horizontal(pl.col("excess", weight).is_finite())
    w = pl.col(weight).filter(valid)
    y = pl.col("excess")
    return members.group_by("month", *groups).agg(
        *metrics, market=pl.col("market").first(),
        EW=y.mean(), VW=(w * y.filter(valid)).sum() / w.sum(),
    ).with_columns(pl.col("VW").fill_nan(None))


def hac(formula, data):
    return sp.regress(formula, data=data, robust="hac", hac_lags=6, hac_small=True)


def format_fit(fit, term):
    return f"{fit.params[term]:.2f}<br>({fit.tvalues[term]:.2f})"


def portfolio_inference(series, outcome, *, alpha=False):
    formula = f"{outcome} ~ {'market' if alpha else '1'}"
    return format_fit(hac(formula, series.sort("month")), "Intercept")


def spread(monthly, high, low, key, value):
    def leg(index):
        return monthly.filter(pl.col(key) == index).select("month", value, "market")

    return leg(high).join(leg(low), on="month", validate="1:1").select(
        "month", "market", (pl.col(value) - pl.col(f"{value}_right")).alias(value),
    )


def univariate_panel(panel, variable, nyse, characteristics, *, weight="cap"):
    members = assign_portfolio(
        panel.filter(pl.col(variable).is_finite()), variable, 10, "month", nyse=nyse,
    )
    monthly = portfolio_months(members, ["portfolio"], characteristics, weight=weight)
    by_portfolio = monthly.partition_by("portfolio", as_dict=True)
    rows = []
    for metric in monthly.select(pl.exclude("month", "portfolio", "market", "EW", "VW")).columns:
        rows.append([metric, *[f"{by_portfolio[i,][metric].mean():.2f}" for i in range(1, 11)], ""])
    for weight in ("EW", "VW"):
        high_minus_low = spread(monthly, 10, 1, "portfolio", weight)
        series = [by_portfolio[i,] for i in range(1, 11)] + [high_minus_low]
        for alpha in (False, True):
            outcome = "CAPM alpha" if alpha else "Excess return"
            rows.append([f"{weight} {outcome}", *[
                portfolio_inference(data, weight, alpha=alpha) for data in series
            ]])
    return pl.DataFrame(rows, schema=["Statistic", *map(str, range(1, 11)), "10–1"], orient="row")


def bivariate_months(panel, control, variable, *, nyse, dependent):
    members = assign_portfolio(panel, control, 5, "month", nyse=nyse, label="control")
    groups = ["month", "control"] if dependent else ["month"]
    members = assign_portfolio(members, variable, 5, groups, nyse=nyse, label="target")
    return portfolio_months(members.drop_nulls(["control", "target"]), ["control", "target"])


def bivariate_panel(monthly, value, *, independent, spread_only=False):
    # Group 0 represents an equally weighted average of the five cell returns.
    def average(over, index):
        return (
            monthly.group_by("month", over).agg(
                pl.col(value).mean(), pl.col("market").first(),
            )
            .with_columns(pl.lit(0, dtype=monthly.schema[index]).alias(index))
            .select(monthly.columns)
        )

    monthly = monthly.select("month", "control", "target", value, "market")
    all_cells = [monthly, average("target", "control")]
    if independent:
        all_cells.append(average("control", "target"))
    monthly = pl.concat(all_cells)
    cells = monthly.partition_by(["control", "target"], as_dict=True)
    rows = []
    columns = [*range(1, 6), 0]
    for target in [*range(1, 6), *([0] if independent else [])]:
        group = monthly.filter(pl.col("target") == target)
        if spread_only:
            data = cells[0, target]
            rows.append([str(target), portfolio_inference(data, value),
                         portfolio_inference(data, value, alpha=True)])
            continue
        means = [
            f"{cells[c, target][value].mean():.2f}" if (c, target) in cells else ""
            for c in columns
        ]
        extra = []
        if independent:
            control_spread = spread(group, 5, 1, "control", value)
            extra = [portfolio_inference(control_spread, value),
                     portfolio_inference(control_spread, value, alpha=True)]
        rows.append([str(target) if target else "Average", *means, *extra])
    spread_series = [
        spread(monthly.filter(pl.col("control") == control), 5, 1, "target", value)
        for control in columns
    ]
    if spread_only:
        rows.append(["5–1", portfolio_inference(spread_series[-1], value),
                     portfolio_inference(spread_series[-1], value, alpha=True)])
        return (
            pl.DataFrame(rows, schema=["Target", "Excess return", "CAPM alpha"], orient="row"),
            pl.DataFrame(
                [[str(c) if c else "Average", portfolio_inference(data, value),
                  portfolio_inference(data, value, alpha=True)]
                 for c, data in zip(columns, spread_series)],
                schema=["Control", "5–1 excess return", "5–1 CAPM alpha"], orient="row",
            ),
        )
    extra = ["", ""] if independent else []
    for label, alpha in [("5–1", False), ("5–1 CAPM alpha", True)]:
        rows.append([label, *[portfolio_inference(data, value, alpha=alpha) for data in spread_series], *extra])
    schema = ["Target", *map(str, range(1, 6)), "Control avg."]
    if independent:
        schema += ["Control 5–1", "Control 5–1 CAPM alpha"]
    return pl.DataFrame(rows, schema=schema, orient="row")


def fama_macbeth_table(panel, specifications, labels):
    clipped = panel.with_columns(
        clip_tails(pl.col(term)).over("month").alias(term) for term in labels
    )
    result = {"Term": [*labels.values(), "Intercept", "Adjusted R²", "n"]}
    for i, terms in enumerate(specifications, 1):
        fit = sp.fama_macbeth("excess ~ " + " + ".join(terms), data=clipped.select(
            "month", "excess", *terms,
        ).drop_nulls(), time="month", lags=6)
        periods = fit.model_info["period_coefs"]
        adjust = 1 - (1 - periods["r2"]) * (periods["nobs"] - 1) / (periods["nobs"] - len(terms) - 1)
        result[f"({i})"] = [
            format_fit(fit, term) if term in terms else "" for term in labels
        ] + [format_fit(fit, "Intercept"), f"{adjust.mean():.2f}", f"{periods['nobs'].mean():.0f}"]
    return pl.DataFrame(result)


def plot_cumulative_returns(returns, y_label="Cumulative return"):
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
