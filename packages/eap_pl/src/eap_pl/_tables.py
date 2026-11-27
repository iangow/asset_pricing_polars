"""Table display for Polars DataFrames."""

import re
from html import escape

import polars as pl
import polars.selectors as cs
from era_pl.namespaces import EraDataFrame


def _book_style(table):
    return table.tab_options(
        table_width="100%",
        table_font_names="inherit",
        table_font_size="0.875rem",
        column_labels_font_weight="bold",
        data_row_padding="0.25rem",
    )


@pl.api.register_dataframe_namespace("eap")
class EAPDataFrameNamespace(EraDataFrame):
    """Table display and grouped operations available on ``DataFrame.eap``."""

    def show_table(self) -> None:
        """Display the DataFrame with Great Tables in a notebook or Quarto.

        Importing ``eap_pl`` registers ``.eap`` on Polars DataFrames. The
        table shows floats to two decimal places and leaves null cells blank.
        Column names and text cells can contain Markdown, such as inline code.
        A cell containing a single ``$...$`` expression displays as inline math.

        Examples:
            >>> import eap_pl
            >>> import polars as pl
            >>> frame = pl.DataFrame({"Term": ["`mkt`"], "Estimate": [0.823]})
            >>> frame.eap.show_table()
        """
        from great_tables import html, md
        from IPython.display import display

        def format_text(value: str | None) -> str:
            if value is None:
                return ""
            math = re.fullmatch(r"\$([^$\n]+)\$", value)
            if math:
                return rf'<span class="math inline">\({escape(math[1])}\)</span>'
            return md(value).to_html()

        frame = self._df
        if not frame.width:
            display(frame)
            return

        table = (
            frame.style
            .cols_label(cases={name: html(format_text(name)) for name in frame.columns})
            .cols_align("left", columns=frame.columns[0])
            .cols_align("right", columns=frame.columns[1:])
            .fmt(format_text, columns=cs.string(include_categorical=True))
            .fmt(lambda value: "" if value is None else f"{value:.2f}", columns=cs.float())
            .sub_missing(columns=~cs.float(), missing_text="")
            .pipe(_book_style)
            .tab_options(
                column_labels_padding_horizontal="0.45rem",
                data_row_padding_horizontal="0.45rem",
            )
        )
        display(table)

    def show_grid(self, control: str, target: str, *, holding: bool = False) -> None:
        """Display a pivoted portfolio grid with optional year headings.

        The frame has a ``target`` column of quartile numbers, an optional
        ``year`` column, and one value column per control portfolio. Target
        codes 5 and 6 denote the high-minus-low spread and the average.
        Optional ``Diff`` and ``Avg`` columns contain the spread and average
        across the control portfolios.

        Args:
            control: Display label for the control sort, such as ``"Beta"``.
            target: Display label for the target sort, such as ``"Market cap"``.
            holding: Label ``year`` as the formation year and show the next
                calendar year as the holding year. Requires a ``year`` column.
        """
        from IPython.display import display

        frame = self._df.with_columns(
            pl.col("target").replace_strict({
                group: f"{target} {suffix}"
                for group, suffix in enumerate(("1", "2", "3", "4", "4–1", "average"), 1)
            })
        )
        stub = ["target"]
        labels = ["Portfolio"]
        if holding:
            frame = frame.with_columns(holding_year=pl.col("year") + 1)
            stub = ["year", "holding_year", *stub]
            labels = ["Formation year", "Holding year", *labels]
        elif "year" in frame.columns:
            stub = ["year", *stub]
            labels = ["Year", *labels]

        columns = [name for name in frame.columns if name not in stub]
        table = (
            frame.style
            .tab_stub(rowname_col=stub)
            .tab_stubhead(labels)
            .tab_spanner(f"{control} portfolios", columns=columns)
            .cols_label(cases={
                name: label
                for name, label in {"Diff": "3–1", "Avg": "Average"}.items()
                if name in columns
            })
            .cols_align("right", columns=columns)
            .fmt_integer(columns=cs.by_name(columns) & cs.integer(), use_seps=True)
            .fmt_number(columns=cs.by_name(columns) & cs.float(), decimals=2, use_seps=False)
            .sub_missing(missing_text="")
            .pipe(_book_style)
            .tab_options(stub_font_weight="normal")
        )
        display(table)
