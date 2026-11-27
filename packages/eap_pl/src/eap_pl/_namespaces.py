"""Expression helpers available through the book's Polars namespace."""

import polars as pl
from era_pl.namespaces import EraExpr


@pl.api.register_expr_namespace("eap")
class EAPExprNamespace(EraExpr):
    """Portfolio, tail-treatment, and ratio helpers on ``Expr.eap``."""
