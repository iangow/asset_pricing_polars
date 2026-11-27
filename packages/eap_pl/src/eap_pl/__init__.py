"""Polars helpers for Empirical Asset Pricing: A Python Reader’s Companion."""

from ._beta import load_annual_betas, load_beta_universe, load_daily_beta_inputs
from ._characteristics import build_book_to_market, build_sizes
from ._data import (
    build_annual_panel,
    load_pastor_stambaugh_2013,
    monthly_panel,
    write_eap_table,
)
from ._french import get_ff_factors
from ._momentum import build_momentum
from ._plots import plot_cumulative_returns, plot_factor
from ._portfolios import (
    bivariate_months,
    bivariate_panel,
    fama_macbeth_table,
    format_fit,
    hac,
    univariate_panel,
)
from ._rolling_ols import rolling_ols
from ._statistics import (
    correlation_matrix,
    mean_cross_sections,
    period_correlations,
    persistence,
    winsorized_corr,
)
from ._tables import EAPDataFrameNamespace

__all__ = [
    "EAPDataFrameNamespace",
    "bivariate_months",
    "bivariate_panel",
    "build_annual_panel",
    "build_book_to_market",
    "build_momentum",
    "build_sizes",
    "correlation_matrix",
    "fama_macbeth_table",
    "format_fit",
    "get_ff_factors",
    "hac",
    "load_annual_betas",
    "load_beta_universe",
    "load_daily_beta_inputs",
    "load_pastor_stambaugh_2013",
    "mean_cross_sections",
    "monthly_panel",
    "period_correlations",
    "persistence",
    "plot_cumulative_returns",
    "plot_factor",
    "rolling_ols",
    "univariate_panel",
    "winsorized_corr",
    "write_eap_table",
]
