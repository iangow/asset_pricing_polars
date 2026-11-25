# Empirical Asset Pricing with Polars

A local Quarto book, not a Python package. The first chapter builds monthly
size portfolios from CRSP CIZ Parquet. No data or rendered outputs are tracked.

## Render

With Quarto, uv, and access to CRSP monthly Parquet:

```sh
uv sync
QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render --to html
```

Before rendering, copy `.env.example` to `.env` in the project and set
`DATA_DIR` to your Parquet root. The book reads `$DATA_DIR/crsp/msf_v2.parquet` with `era_pl.load_parquet()`.
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

This compares portfolio membership and raw monthly returns on identical
complete cases, before inference or factor adjustments. The sample is not a
replication of the book's historical tables. StatsPAI is not needed for
this check; it may be useful when adding inference or factor regressions.

The initial Mac mini check (January 2000 through December 2002) matched all
215,991 stock-month assignments and all 216 portfolio-months per weighting
method (five quintiles plus the spread in each of 36 months). The largest
absolute return difference was 1.67e-16 in decimal units.
