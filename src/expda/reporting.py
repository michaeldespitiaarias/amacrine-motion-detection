"""Output writing for every stage's results.

Every diagnostic and results table in this project is a plain CSV — raw
numbers, no styling, no narrative document (``results.csv`` /
``posthoc.csv``). This module used to render styled, self-contained HTML
fragments instead; that was removed (2026-08) to keep this repository's
output to the statistical analysis itself, reproducible straight from the
CSVs without a browser.

Presentation only — nothing in this module affects a statistic.
"""

from __future__ import annotations

import os

import pandas as pd


def save_csv(df: pd.DataFrame, path: str, precision: int | None = None) -> None:
    """Write *df* to *path* as a plain CSV.

    Parameters
    ----------
    df : DataFrame
        Table to write.
    path : str
        Destination file. Parent directories are created if needed.
    precision : int, optional
        When given, every float column is rounded to this many decimals
        first — on a copy, the caller's DataFrame is never mutated.
    """
    if precision is not None:
        df = df.copy()
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(precision)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)
