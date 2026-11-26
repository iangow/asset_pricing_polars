"""Portfolio sorts and inference for the monthly replications."""

import polars as pl
import statspai as sp


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
    """Fit a StatsPAI regression with six Newey–West lags.

    The regression uses the small-sample correction. Supply the rows in
    time order.

    Args:
        formula (str): StatsPAI regression formula.
        data (polars.DataFrame): Regression sample in time order.

    Returns:
        (statspai.EconometricResults): Fitted coefficients, HAC standard
            errors, and test statistics.
    """
    return sp.regress(formula, data=data, robust="hac", hac_lags=6, hac_small=True)


def format_fit(fit, term, *, digits=2):
    """Format a coefficient and its t statistic for a book table.

    Args:
        fit (statspai.EconometricResults): Fit exposing `params` and `tvalues`
            by term.
        term (str): Coefficient name in the fit.
        digits (int): Decimal places for the coefficient.

    Returns:
        (str): Coefficient and parenthesized t statistic separated by an HTML
            line break. The t statistic uses two decimal places.
    """
    return f"{fit.params[term]:.{digits}f}<br>({fit.tvalues[term]:.2f})"


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
    """Summarize monthly characteristic deciles and the high-minus-low spread.

    The sample must populate all ten deciles for the table.

    Args:
        panel (polars.DataFrame): Stock-month data containing `month`,
            next-month `excess` and `market` in percentage points, the sort
            variable and weight, and `exchcd` for NYSE breakpoints.
        variable (str): Characteristic used for the sort.
        nyse (bool): Compute cutoffs from NYSE stocks when true and all
            stocks in the sample otherwise.
        characteristics (Iterable[polars.Expr]): Additional portfolio
            aggregates to average across months.
        weight (str): Column containing the formation-month value weights.

    Returns:
        (polars.DataFrame): A formatted table of characteristics, equal- and
            value-weighted excess returns, and CAPM alphas, with HAC t
            statistics for returns and alphas.
    """
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
    """Form monthly quintile portfolios on a control and a target.

    Args:
        panel (polars.DataFrame): Stock-month data containing `month`,
            next-month `excess` and `market` in percentage points, `cap`,
            the sort variables, and `exchcd` for NYSE breakpoints.
        control (str): Characteristic used to form the control groups.
        variable (str): Target characteristic.
        nyse (bool): Compute cutoffs from NYSE stocks when true and all
            stocks in the sample otherwise.
        dependent (bool): Set target cutoffs within each control quintile
            when true and within the entire monthly sample otherwise.

    Returns:
        (polars.DataFrame): One row per month and occupied control-target
            cell, with equal-weighted (`EW`) and value-weighted (`VW`)
            returns and the market excess return (`market`).
    """
    members = assign_portfolio(panel, control, 5, "month", nyse=nyse, label="control")
    groups = ["month", "control"] if dependent else ["month"]
    members = assign_portfolio(members, variable, 5, groups, nyse=nyse, label="target")
    return portfolio_months(members.drop_nulls(["control", "target"]), ["control", "target"])


def bivariate_panel(monthly, value, *, independent, spread_only=False):
    """Report means and quintile spreads from a monthly two-way sort.

    Args:
        monthly (polars.DataFrame): Monthly cell returns from
            `bivariate_months()`.
        value (str): Return column, usually `EW` or `VW`.
        independent (bool): Include averages and spreads in both sort
            dimensions for independently sorted portfolios.
        spread_only (bool): Return two abbreviated tables covering target
            portfolios averaged across controls and target spreads by control.

    Returns:
        (polars.DataFrame or tuple[polars.DataFrame, polars.DataFrame]):
            A formatted panel of means and spreads, or two abbreviated
            panels when `spread_only` is true.
    """
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


def fama_macbeth_table(panel, specifications, labels, *, digits=2):
    """Build a table of monthly Fama–MacBeth regressions.

    Winsorizes the regressors within formation months at the 0.5th and
    99.5th percentiles, then uses six Newey–West lags for inference on
    average coefficients.

    Args:
        panel (polars.DataFrame): Data containing `month`, next-month
            `excess` in percentage points, and the regressors.
        specifications (Iterable[list[str]]): Regressor columns for each
            model.
        labels (dict[str, str]): Regressor columns mapped to display labels
            in table order.
        digits (int): Decimal places for coefficients and intercepts.

    Returns:
        (polars.DataFrame): Formatted coefficients and t statistics,
            average adjusted R², and mean monthly sample size by model.
    """
    clipped = panel.with_columns(
        pl.col(term).era.winsorize(prob=0.005, quantile_method="linear").over("month").alias(term)
        for term in labels
    )
    result = {"Term": [*labels.values(), "Intercept", "Adjusted R²", "n"]}
    for i, terms in enumerate(specifications, 1):
        fit = sp.fama_macbeth("excess ~ " + " + ".join(terms), data=clipped.select(
            "month", "excess", *terms,
        ).drop_nulls(), time="month", lags=6)
        periods = fit.model_info["period_coefs"]
        adjust = 1 - (1 - periods["r2"]) * (periods["nobs"] - 1) / (periods["nobs"] - len(terms) - 1)
        result[f"({i})"] = [
            format_fit(fit, term, digits=digits) if term in terms else "" for term in labels
        ] + [format_fit(fit, "Intercept", digits=digits), f"{adjust.mean():.2f}", f"{periods['nobs'].mean():.0f}"]
    return pl.DataFrame(result)
