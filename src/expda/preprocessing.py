"""Stage 1 - preprocessing, quality control and assumption reporting.

Pipeline position
-----------------
``<input_dir>/<Dataset>.csv``
    -> handle_nulls -> average_trials
    -> detect_and_replace_outliers
    -> ``<results_dir>/<Dataset>/<report_folders["intermediate"]>/<Dataset>_no_outliers.csv``
    plus diagnostic reports elsewhere under ``<results_dir>/<Dataset>/``.

Contents
--------
1.  Reporting helpers
2.  Data loading
3.  Missing-data handling
4.  Trial aggregation
5.  Outlier handling

Normality and homoscedasticity are no longer reported here as standalone
per-dataset CSVs: they are recomputed per test in ``inference.py`` /
``rm_anova.py`` and written as columns alongside the test they gated,
so a reader never has to cross-reference a separate assumption-check
file against the result it explains.

Every diagnostic report is a plain CSV (``results.csv`` / ``posthoc.csv``
— raw numbers, no styling, no narrative document). This module does not
generate plots: no distribution figures, no correlation heatmap. Nothing
about that is a technical limitation — matplotlib/seaborn were removed
on purpose, 2026-08, keeping this repository's output to the statistical
analysis itself, reproducible straight from the CSVs.
"""


import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .reporting import save_csv

# ANSI escape codes used by the progress output.
bold_start = "\033[1m"
bold_end = "\033[0m"



def _is_identifier(column: str) -> bool:
    """True when a column name looks like a subject or record identifier."""
    name = str(column).strip().lower()
    return name == "id" or name.endswith(" id") or name.endswith("_id")


# =============================================================================
# SECTION 1. REPORTING HELPERS
# =============================================================================
#
# save_csv() lives in reporting.py — the single write path every diagnostic
# report in this module goes through — and is imported at the top of this
# file (see the `from .reporting import save_csv` import).


# Maps a p-value onto the ***/**/*/ns ladder used in every report table.
def significance_stars(p_value: float) -> str:
    """Return significance symbols based on p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


# =============================================================================
# SECTION 2. DATA LOADING
# =============================================================================


# Column-name suffixes that are numeric (or boolean, which pandas'
# read_csv/to_numeric both happily coerce) but are diagnostic metadata,
# never a measurement to analyse — e.g. the `{col}_outlier` flag columns
# `detect_and_replace_outliers` adds for cells its skip gate withheld.
NON_MEASURE_SUFFIXES = ("_outlier",)


# Reads one split dataset and infers numeric vs categorical columns.
# A column is treated as categorical when its dtype is object or category,
# when its name looks like a subject/record identifier (`_is_identifier` —
# "id", "... id", "..._id"), or when it IS this dataset's configured
# `subject_column` — which catches identifiers that don't follow that
# naming pattern at all (e.g. a registry using "Subject" or "Sample" as the
# subject column name). Either way, this is how the subject identifier is
# kept out of the numeric analysis even when its values are literal
# integers (e.g. "Mouse ID" holding 24228, 24229, ... parses as numeric
# otherwise, and without this exclusion is fed straight into missing-value
# imputation, outlier detection, and the assumption-check reports as if it
# were a real measurement). Columns ending in NON_MEASURE_SUFFIXES are
# excluded from numeric_cols even when they parse as numbers (they always
# do — see above).
def load_dataset(name, cfg):
    csv_path = cfg["location"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV {csv_path}: {e}")

    subject_col = cfg.get("subject_column")

    numeric_cols = []
    for col in df.columns:
        if str(col).lower().endswith(NON_MEASURE_SUFFIXES):
            continue
        if _is_identifier(col) or col == subject_col:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
            numeric_cols.append(col)
        except:
            pass

    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    print(f"Loaded '{name}' with {len(df)} rows, {len(df.columns)} columns")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    return df, numeric_cols, categorical_cols


# =============================================================================
# SECTION 3. MISSING-DATA HANDLING
# =============================================================================


# Group-wise imputation or row dropping. Numeric columns are typically filled with
# method='median' and categorical columns with method='mode', grouped by the
# dataset's group column.
def handle_nulls(
    df: pd.DataFrame,
    variables: list,
    method: str,
    group_column=None,
    grouped: bool = True,
    mv_drop_pct: float = 30.0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame columns using the chosen method.
    Simplified version: group_column is no longer a per-dataset dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    variables : list
        Columns to process.
    method : str
        Fill/drop method. Options:
        {'unknown', 'mode', 'ffill', 'bfill', 'drop', 'mean', 'median', 'interpolate'}.
    group_column : str or list, optional
        Column(s) to group by when grouped=True.
    grouped : bool, default=False
        Whether to apply the missing-data method within groups.
    mv_drop_pct : float, default=30.0
        Per-column skip gate, applied to every method except ``'drop'``
        (which removes rows, not columns, and is left untouched): a
        column whose missing share exceeds this percentage keeps its
        NaNs rather than having more than that share of its values
        invented by a single-imputation method (mean/median/interpolate/
        mode). Nothing is dropped from the dataset — only the fill is
        skipped for that column; downstream tests already handle NaN
        honestly by reporting the real n. The default is picked for
        small-n basic-research groups (typically 3-15 subjects) where
        the commonly-cited caution zone for simple single imputation is
        well below 50 %.
    verbose : bool, default=False
        Print processing information.

    Returns
    -------
    pd.DataFrame
        DataFrame after handling missing values.
    """

    valid_methods = {"unknown", "mode", "ffill", "bfill", "drop", "mean", "median", "interpolate"}
    if method not in valid_methods:
        raise ValueError(f"❌ Invalid method '{method}'. Choose from {valid_methods}.")

    df = df.copy()

    # Validate variables exist in the DataFrame
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        raise ValueError(f"❌ Variables not found in DataFrame: {missing_vars}")

    # Determine grouping columns
    if grouped:
        if group_column is None:
            raise ValueError("❌ 'group_column' must be provided when grouped=True.")

        # Normalize to list
        if isinstance(group_column, str):
            group_cols = [group_column]
        elif isinstance(group_column, list):
            group_cols = group_column
        else:
            raise TypeError("'group_column' must be a string or a list of strings.")

        # Validate group columns
        for col in group_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Group column '{col}' not found in DataFrame.")
    else:
        group_cols = None

    # Store null counts before
    nulls_before = df[variables].isna().sum()

    # Process each variable
    for col in variables:

        n_null = df[col].isna().sum()
        if n_null == 0:
            if verbose:
                print(f"✅ No nulls detected in '{col}'.")
            continue

        # Drop rows if requested
        if method == "drop":
            before_n = len(df)
            df = df.dropna(subset=[col])
            after_n = len(df)
            dropped = before_n - after_n

            if verbose:
                print(f"🗑️ Dropped {dropped} rows with nulls in '{col}'.")
            continue

        # Skip-imputation gate: leave the column's NaNs untouched above
        # mv_drop_pct % missing rather than fabricating most of it from a
        # handful of real values. Nothing is dropped — only the fill.
        missing_frac = n_null / len(df) if len(df) else 0.0
        if missing_frac > (mv_drop_pct / 100.0):
            if verbose:
                print(f"⏭️  Skipped imputation for '{col}': "
                      f"{missing_frac:.1%} missing exceeds mv_drop_pct="
                      f"{mv_drop_pct:.0f}% (left as NaN).")
            continue

        # Fill values
        if grouped and group_cols:
            # Apply fill method within each group
            df[col] = df.groupby(group_cols)[col].transform(
                lambda s: _apply_fill_method(s, method)
            )
        else:
            # Apply fill method on the whole column
            df[col] = _apply_fill_method(df[col], method)

        if verbose:
            print(f"🔧 Filled nulls in '{col}' using method '{method}'"
                  f"{' (grouped)' if grouped else ''}.")

    # Report changes
    if verbose:
        nulls_after = df[variables].isna().sum()
        print("\n📊 Nulls handled per column:")
        for c in variables:
            diff = nulls_before[c] - nulls_after[c]
            print(f" - {c}: {diff} filled or removed")

    return df


# Single-Series backend for handle_nulls().
def _apply_fill_method(series: pd.Series, method: str) -> pd.Series:
    """
    Apply the selected null-handling method to a single pandas Series.

    Parameters
    ----------
    series : pd.Series
        Column to process.
    method : str
        Method used to fill missing values.

    Returns
    -------
    pd.Series
        Series with missing values handled.
    """

    # Fill with the literal string "Unknown"
    if method == "unknown":
        return series.fillna("Unknown")

    # Forward-fill or backward-fill
    if method in {"ffill", "bfill"}:
        return series.fillna(method=method)

    # Replace with the mode (most frequent value)
    if method == "mode":
        modes = series.mode()
        if not modes.empty:
            return series.fillna(modes.iloc[0])
        else:
            return series  # nothing to fill with

    # Numeric-only methods
    if pd.api.types.is_numeric_dtype(series):

        if method == "mean":
            return series.fillna(series.mean())

        if method == "median":
            return series.fillna(series.median())

        if method == "interpolate":
            return series.interpolate(method="linear")

    # If method is not applicable, return unchanged
    return series


# =============================================================================
# SECTION 4. TRIAL AGGREGATION
# =============================================================================


# Collapses repeated trials to one row per subject and condition, making the
# subject the unit of analysis.
def average_trials(df, trial_col, group_cols, numeric_cols, subject_col):
    """
    Averages numeric columns per subject based on the group columns and the subject column,
    handles categorical columns using the first occurrence (mode), drops the trial column
    and updates numeric and categorical column lists.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    trial_col : str
        Column name that represents the trial (to be averaged and dropped).
    group_cols : list
        List of columns to group by for averaging.
    numeric_cols : list
        List of numeric columns in the dataframe.
    subject_col : str
        Column identifying the subject, kept as a grouping key.

    Returns:
    --------
    df : pd.DataFrame
        Dataframe with averaged numeric values, categorical columns preserved, trial column dropped.
    numeric_cols : list
        Updated list of numeric columns in the dataframe.
    categorical_cols : list
        Updated list of categorical columns in the dataframe.
    """
    
    # Store original column order
    original_order = df.columns.tolist()
    
    # If trial column doesn't exist or is empty, return original dataframe
    if not trial_col or trial_col not in df.columns:
        updated_numeric_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if not _is_identifier(col) and col != subject_col
        ]
        updated_categorical_cols = [col for col in df.columns if col not in updated_numeric_cols]
        return df, updated_numeric_cols, updated_categorical_cols
    
    # Columns used for grouping (group_cols + the subject column)
    avg_group_cols = group_cols + [subject_col]
    
    # Numeric columns to average (exclude grouping columns)
    numeric_cols_to_avg = [col for col in numeric_cols if col not in avg_group_cols]
    
    # Average numeric columns
    df_avg = df.groupby(avg_group_cols, as_index=False)[numeric_cols_to_avg].mean()
    
    # Handle categorical columns: keep first occurrence for each group
    cat_cols = [col for col in df.columns if col not in numeric_cols_to_avg + avg_group_cols]
    if cat_cols:
        df_cat = df.groupby(avg_group_cols, as_index=False)[cat_cols].first()
        df = pd.merge(df_avg, df_cat, on=avg_group_cols, how="left")
    else:
        df = df_avg
    
    # Drop the trial column
    df = df.drop(columns=[trial_col])
    
    # Reorder columns to match original dataframe
    df = df[[col for col in original_order if col in df.columns]]
    
    # Update numeric and categorical columns
    updated_numeric_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if not _is_identifier(col) and col != subject_col
    ]
    updated_categorical_cols = [col for col in df.columns if col not in updated_numeric_cols]

    return df, updated_numeric_cols, updated_categorical_cols


# =============================================================================
# SECTION 5. OUTLIER HANDLING
# =============================================================================


# Tukey IQR rule applied within each group, default factor 1.5. Flagged
# values are replaced by the median of their group. Returns the modified
# frames together with a boolean table marking which cells were replaced;
# qc_report summarises that table.
def detect_and_replace_outliers(
    dataframes: dict,
    variables: list,
    group_column: dict,
    iqr_factors: dict = None,
    outlier_pct_skip: float = 15.0,
    outlier_replace_max_n=2,
    outlier_replace_max_n_group_ceiling=20,
    verbose: bool = True
) -> tuple[dict, dict]:
    """
    Detect and replace outliers in each DataFrame using the IQR rule.

    Outliers are replaced with the median of their group, subject to a
    replacement-skip gate (see below). Detection always runs on every
    flagged cell; only the replacement step can be withheld.

    Supports per-dataset configuration:
    - different grouping columns per dataset
    - customizable IQR multipliers per dataset (default 1.5)
    - large datasets (memory-conscious)

    Parameters
    ----------
    dataframes : dict
        Dictionary {dataset_name: DataFrame}.
    variables : list
        Variables to evaluate for outliers (numeric only).
    group_column : dict
        Mapping {dataset_name: grouping columns (str or list)}.
    iqr_factors : dict, optional
        Mapping {dataset_name: float} defining IQR multiplier.
        If not provided, ALL datasets use default 1.5.
    outlier_pct_skip : float, default=15.0
        Replacement-skip gate: when more than this share of a (group,
        column) is flagged, that is evidence the detector is being
        applied to a genuinely skewed/multimodal distribution rather
        than routine contamination — replacing that many points would
        erase real variance rather than clean noise. Those cells are
        excluded from replacement and instead recorded in a
        non-destructive ``{col}_outlier`` diagnostic column.
    outlier_replace_max_n : int or float or None, default=2
        Absolute cap on the number of points replaced in a single
        (group, column), independent of group size but only evaluated
        while the group is at or under
        ``outlier_replace_max_n_group_ceiling`` (see below). A pure
        percentage gate can still let a narrow boundary case slip
        through at small n (e.g. exactly 15 % on the nose); this patches
        that case for small groups only. ``None`` (or ``0``) disables
        the cap.
    outlier_replace_max_n_group_ceiling : int or float or None, default=20
        Group-size ceiling above which ``outlier_replace_max_n`` stops
        applying entirely and only ``outlier_pct_skip`` governs — a
        fixed count is a sensible patch for a 3-15-subject group and an
        absurd one for a 1000-subject group, where only the percentage
        should ever bind. ``None`` disables the ceiling (the absolute
        cap then always applies, regardless of group size).
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    modified_dfs : dict
        DataFrames with outliers replaced (plus any ``{col}_outlier``
        diagnostic columns for cells the skip gate withheld).
    outlier_flags : dict
        Boolean DataFrames marking which values were actually replaced
        (cells withheld by the skip gate are NOT marked here — they are
        marked instead by the ``{col}_outlier`` column added to the
        returned DataFrame itself).

    Notes
    -----
    The dual skip gate (percentage skip + absolute cap) keeps automatic
    cleaning from over-reaching on small research samples — see the
    ``outlier_pct_skip`` / ``outlier_replace_max_n`` parameters below.
    """

    # Default IQR factor = 1.5
    DEFAULT_IQR = 1.5
    iqr_factors = iqr_factors or {}

    skip_frac = float(outlier_pct_skip) / 100.0
    max_n_raw = (float(outlier_replace_max_n)
                 if outlier_replace_max_n not in (None, "", 0)
                 else float("inf"))
    max_n_ceiling = (float(outlier_replace_max_n_group_ceiling)
                      if outlier_replace_max_n_group_ceiling not in (None, "")
                      else float("inf"))

    modified_dfs = {}
    outlier_flags = {}

    for name, df in dataframes.items():

        if name not in group_column:
            raise KeyError(f"Dataset '{name}' missing entry in group_column dict.")

        # If dataset missing in iqr_factors → default = 1.5
        iqr_factor = iqr_factors.get(name, DEFAULT_IQR)

        grouping = group_column[name]

        # Convert grouping column to list if needed
        if isinstance(grouping, str):
            grouping = [grouping]
        elif not isinstance(grouping, list):
            raise TypeError("Entries in group_column must be str or list of str.")

        # Ensure grouping columns exist
        for g in grouping:
            if g not in df.columns:
                raise ValueError(f"Grouping column '{g}' not found in dataset '{name}'.")

        valid_vars = [
            v for v in variables
            if v in df.columns and pd.api.types.is_numeric_dtype(df[v])
        ]

        if len(valid_vars) == 0:
            if verbose:
                print(f"⚠️ No valid numeric variables found in '{name}', skipping.")
            modified_dfs[name] = df.copy()
            outlier_flags[name] = pd.DataFrame(False, index=df.index, columns=variables)
            continue

        if verbose:
            print(f"🔎 Processing '{name}' ({len(df)} rows)")
            print(f"   • Grouped by: {grouping}")
            print(f"   • IQR factor: {iqr_factor}")
            print(f"   • Variables:  {valid_vars}")

        df_copy = df.copy()
        flags = pd.DataFrame(False, index=df_copy.index, columns=valid_vars)
        skipped = pd.DataFrame(False, index=df_copy.index, columns=valid_vars)

        grouped = df_copy.groupby(grouping)

        for group_key, group_df in grouped:

            Q1 = group_df[valid_vars].quantile(0.25)
            Q3 = group_df[valid_vars].quantile(0.75)
            IQR = Q3 - Q1

            # Skip variables with no variability
            non_zero_IQR = IQR[IQR > 0].index.tolist()
            if len(non_zero_IQR) == 0:
                continue

            lower = Q1[non_zero_IQR] - iqr_factor * IQR[non_zero_IQR]
            upper = Q3[non_zero_IQR] + iqr_factor * IQR[non_zero_IQR]

            mask = (group_df[non_zero_IQR] < lower) | (group_df[non_zero_IQR] > upper)
            n_group = len(group_df)
            # Absolute cap only active while the group is at or under the
            # ceiling; above it, only the percentage gate binds.
            max_n = max_n_raw if n_group <= max_n_ceiling else float("inf")

            if not mask.values.any():
                continue

            medians = group_df[non_zero_IQR].median()
            for col in non_zero_IQR:
                col_outliers = mask[col]
                n_flagged = int(col_outliers.sum())
                if n_flagged == 0:
                    continue
                flagged_idx = col_outliers[col_outliers].index

                if (n_flagged / n_group > skip_frac) or (n_flagged > max_n):
                    # Replacement-skip gate: leave these cells untouched,
                    # mark them non-destructively instead.
                    skipped.loc[flagged_idx, col] = True
                    continue

                df_copy.loc[flagged_idx, col] = medians[col]
                flags.loc[flagged_idx, col] = True

        for col in valid_vars:
            if skipped[col].any():
                df_copy[f"{col}_outlier"] = skipped[col]

        total_replaced = flags.sum().sum()
        total_skipped = skipped.sum().sum()
        if verbose:
            msg = f"✅ '{name}': replaced {total_replaced} outlier values."
            if total_skipped:
                msg += (f" {total_skipped} flagged but left unreplaced "
                        f"(skip gate; see *_outlier columns).")
            print(msg + "\n")

        modified_dfs[name] = df_copy
        outlier_flags[name] = flags

    if verbose:
        print("🏁 Outlier detection completed for all datasets.\n")

    return modified_dfs, outlier_flags

