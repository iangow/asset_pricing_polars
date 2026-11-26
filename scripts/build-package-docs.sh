#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
(
    cd docs/eap_pl
    uv run --locked --group docs quartodoc build
    quarto render
)

mkdir -p _book/eap_pl
rsync -a --delete docs/eap_pl/_site/ _book/eap_pl/
