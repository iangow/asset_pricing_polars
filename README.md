# Empirical Asset Pricing with Polars

A Quarto book following Bali, Engle, and Murray's *Empirical Asset Pricing*.
The rendered chapters use Python Polars, legacy CRSP, Compustat, and
Fama–French factors.

## Render

Copy `.env.example` to `.env`, set `DATA_DIR` to your local Parquet root,
and set `WRDS_ID` to your WRDS username if you need to download the inputs.
An exported `DATA_DIR` takes precedence.

```sh
uv sync
```

If you need to create the Parquet repository, run the download script from
this checkout with a WRDS account that has access to the source tables.
`db2pq` prompts for your WRDS password if needed. If you already have the
source files under `DATA_DIR`, skip this step. The required tables and the
vintages used in the book are listed in [Preliminaries](data.qmd).

```sh
uv run scripts/download_wrds_tables.py
```

Download the public inputs for the replication chapters.

```sh
mkdir -p data
curl -fLsS 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS&cosd=1925-12-01&coed=2012-12-01' \
  -o data/cpi-aucns.csv
curl -fLsS 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/Historical_Archives/08%202016%20Update/ftp/F-F_Research_Data_Factors_CSV.zip' \
  | funzip > data/ff3-201607.csv
curl -fLsS 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp_202412/F-F_Momentum_Factor_CSV.zip' \
  | funzip > data/momentum-202412.csv
curl -fLsS 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp_202412/F-F_ST_Reversal_Factor_CSV.zip' \
  | funzip > data/st-reversal-202412.csv
```

Render the full book and open `_book/index.html`. The annual methods chapters
build their inputs on demand. Set `EAP_WRITE_BETAS=1` so the Beta chapter
writes `DATA_DIR/eap/betas.parquet` for the monthly analyses in Size, Value, Momentum, Short-Term Reversal, and Liquidity.

```sh
EAP_WRITE_BETAS=1 QUARTO_PYTHON="$PWD/.venv/bin/python" \
  quarto render --to html
```

For a standalone render of one of these chapters, use an existing `eap/betas.parquet`
or render `beta-replication.qmd` with `EAP_WRITE_BETAS=1` first.
Set `EAP_WRITE_SIZES=1` or `EAP_WRITE_BOOK_TO_MARKET=1` to save those tables
for use outside the book.

## CIZ examples

The original CIZ examples remain in `size.qmd`, `momentum.qmd`,
`fama-macbeth.qmd`, and `illiquidity.qmd` outside the active book
configuration. They use AnomalyLab as an optional numerical reference.
The original comparisons used [commit ab045b8](https://github.com/FinPhD/AnomalyLab/commit/ab045b8e1aa4de8d4c2c854538013f00f420f2de).
The 2000–2002 size check matched all 215,991 stock-month assignments,
all 216 portfolio-months per weighting method, and all 12 three-factor
alphas and HAC t-statistics.

## Package and help site

`uv sync` installs [`eap_pl`](packages/eap_pl/README.md) from this checkout.
The package has its own build metadata. To build its wheel and source
distribution, run `uv build packages/eap_pl`.

The help site uses Quarto and quartodoc to generate the function reference
from the package docstrings. After rendering the book, build and copy the
help site into the book's GitHub Pages output.

```sh
sh scripts/build-package-docs.sh
```

Open `_book/eap_pl/index.html` to preview the help site locally.

## Publish

Render the book, build the help site, and inspect the HTML in `_book/`.
Push the source and run
`quarto publish gh-pages --no-render --no-prompt --no-browser`.
GitHub Pages serves the help site under `eap_pl/` alongside the book.
Keep licensed inputs, generated Parquet, and `.env` off both public branches.
