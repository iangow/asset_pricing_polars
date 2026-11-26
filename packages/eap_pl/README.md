# eap_pl

`eap_pl` provides Polars helpers for [Empirical Asset Pricing with Polars](https://iangow.github.io/asset_pricing_polars/).
It supports descriptive statistics, portfolio sorts, inference, and plots for the book's legacy CRSP replications.

From the book checkout, run `uv sync` to install the package into the book environment.
To install the package into another environment, run `python -m pip install ./packages/eap_pl` from the checkout with that environment's Python.

The package includes the December 2013 Pástor–Stambaugh liquidity series used in BEM Chapter 13. Load it with `eap_pl.load_pastor_stambaugh_2013()`. [Archived Chicago Booth file](https://web.archive.org/web/20141121162425id_/http://faculty.chicagobooth.edu:80/lubos.pastor/research/liq_data_1962_2013.txt).

Use `eap_pl.get_ff_factors("daily", vintage="201406")` to download daily Fama–French factors for a pinned vintage.
The loader also supports monthly three-factor data and monthly momentum.

The [function reference](https://iangow.github.io/asset_pricing_polars/eap_pl/reference/) describes each function's inputs and outputs.
