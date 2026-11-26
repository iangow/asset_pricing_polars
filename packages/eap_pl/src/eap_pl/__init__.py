"""Polars helpers for Empirical Asset Pricing with Polars."""

from ._data import build_annual_panel, load_annual_betas, monthly_panel, write_eap_table
from ._plots import plot_cumulative_returns, plot_factor
from ._portfolios import (
    bivariate_months,
    bivariate_panel,
    fama_macbeth_table,
    format_fit,
    hac,
    univariate_panel,
)
from ._statistics import (
    correlation_matrix,
    mean_cross_sections,
    period_correlations,
    persistence,
    winsorized_corr,
)

__all__ = [
    "bivariate_months",
    "bivariate_panel",
    "build_annual_panel",
    "correlation_matrix",
    "fama_macbeth_table",
    "format_fit",
    "hac",
    "load_annual_betas",
    "mean_cross_sections",
    "monthly_panel",
    "period_correlations",
    "persistence",
    "plot_cumulative_returns",
    "plot_factor",
    "univariate_panel",
    "winsorized_corr",
    "write_eap_table",
]
