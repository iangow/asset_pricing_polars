# Empirical Asset Pricing with Polars

A local Quarto book, not a Python package. The examples use CRSP CIZ and Fama–French factors from local Parquet: size
quintiles and factor-adjusted returns, size–momentum sorts and diagnostics,
Fama–MacBeth regressions, and size–illiquidity sorts. No data or rendered outputs are tracked.

## Render

With Quarto and uv, prepare `.env` (below), then render:

```sh
uv sync
mkdir -p data
curl -fLsS 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS&cosd=1963-06-01&coed=2012-12-01' \
  -o data/cpi-aucns.csv
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render --to html
```

Before rendering, copy `.env.example` to `.env` in the project and set
`DATA_DIR` to your Parquet root. The book uses `era_pl.load_parquet()` to read `crsp/msf_v2.parquet`,
`crsp/dsf_v2.parquet`, and `ff/factors_monthly.parquet` under that root.
The Table 9.1 reproduction also uses legacy `crsp/msf.parquet` and
`crsp/msenames.parquet`, plus the public CPI CSV fetched above.
An exported `DATA_DIR` overrides the gitignored `.env`.

Open `_book/index.html` after rendering.

## Numerical comparison

For developers, install the optional reference dependencies with
`uv sync --group parity`. Clone the reference separately and check out
commit [ab045b8](https://github.com/FinPhD/AnomalyLab/commit/ab045b8e1aa4de8d4c2c854538013f00f420f2de).
Set `ANOMALYLAB_CHECKOUT` to its absolute path when rendering:

```sh
ANOMALYLAB_CHECKOUT="$HOME/git/AnomalyLab" \
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render --to html
```

This enables chapter-by-chapter comparisons with the upstream numeric
results on identical inputs. The examples are not replications of the
textbook's historical tables. StatsPAI is used in the Fama–MacBeth chapter.

The initial Mac mini check (January 2000 through December 2002) matched all
215,991 stock-month assignments and all 216 portfolio-months per weighting
method (five quintiles plus the spread in each of 36 months). The largest
absolute return difference was 1.67e-16 in decimal units.

The expanded check matched all 12 three-factor alphas and HAC t-statistics,
both momentum sorts (202,682 memberships per design), and both illiquidity
sorts (208,501 memberships per design), including their populated return
cells and spreads. Fama–MacBeth matched all 36 monthly coefficient vectors;
HAC standard errors agreed after aligning the finite-sample correction.
Momentum persistence is calendar-aligned rather than shifting by observed rows.


## Publish

Render locally with access to the inputs above, inspect the generated
`_book/` HTML, then run
`quarto publish gh-pages --no-render --no-prompt --no-browser`.
GitHub Pages serves the already-rendered output from `gh-pages`; it does
not run the CRSP analyses. Do not put licensed data or `.env` on either branch.
