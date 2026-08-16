"""Stage 1 - preprocessing, quality control and assumption reporting.

Pipeline position
-----------------
``<input_dir>/<Dataset>.csv``
    -> handle_nulls -> average_trials
    -> transform_variables (shape-stabilizing methods only: log, sqrt,
       boxcox, reciprocal, yeojohnson)
    -> detect_and_replace_outliers
    -> transform_variables (scale-standardizing methods only: zscore,
       minmax)
    -> ``<working_dir>/<Dataset>_no_outliers.csv``
    plus diagnostic reports under ``<results_dir>/<Dataset>/``.

The transform is split across two points around outlier handling rather
than run once, driven by which family the configured method falls
into (``PRE_OUTLIER_TRANSFORMS`` / ``POST_OUTLIER_TRANSFORMS`` below):
shape-stabilizing transforms need to run before outlier detection so a
skewed raw tail isn't mistaken for contamination; scale-standardizing
transforms need to run after outlier replacement, since they are
computed from the column's own mean/std or min/max and a genuine
outlier would otherwise distort those statistics for every other value
in the column before it gets a chance to be replaced.

Contents
--------
1.  Reporting helpers
2.  Data loading
3.  Missing-data handling
4.  Trial aggregation
5.  Variable transformation and ratio normalisation
6.  Outlier handling
7.  Assumption checks
8.  Correlation structure
9.  Multicollinearity

Every diagnostic report is a plain CSV (``results.csv`` / ``posthoc.csv``
— raw numbers, no styling, no narrative document). This module does not
generate plots: no distribution figures, no correlation heatmap. Nothing
about that is a technical limitation — matplotlib/seaborn were removed
on purpose, 2026-08, keeping this repository's output to the statistical
analysis itself, reproducible straight from the CSVs.
"""


import gc
import os
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
# SECTION 5. VARIABLE TRANSFORMATION AND RATIO NORMALISATION
# =============================================================================


# Which `transform` methods run before vs. after outlier replacement in
# the stage-1 script (see module docstring). Shape-stabilizing methods
# change the distribution's shape and need to run BEFORE the outlier
# detector sees the data; scale-standardizing methods rescale off the
# column's own mean/std or min/max and need to run AFTER outliers are
# replaced, or a genuine outlier distorts those statistics for every
# other value in the column.
PRE_OUTLIER_TRANSFORMS = {"log", "sqrt", "boxcox", "reciprocal", "yeojohnson"}
POST_OUTLIER_TRANSFORMS = {"zscore", "minmax"}


# Dispatcher for log / sqrt / boxcox / yeojohnson / reciprocal / zscore /
# minmax / ratio. Pass method='none' to leave the variables on their
# original scale.
def transform_variables(
    data: pd.DataFrame | dict,
    columns: list,
    method: str,
    *,
    reference_group: str = None,
    group_column: str = None,
    subject_column: str = None,
    comparison_factor: str = None,
    paired: bool = False,
    verbose: bool = False
) -> pd.DataFrame | dict:
    """
    Apply transformations (log, sqrt, zscore, minmax, ratio, etc.) to selected variables.

    This function supports both:
    - Single DataFrame input
    - Multiple datasets provided as a dict of DataFrames

    Parameters
    ----------
    data : pd.DataFrame or dict[str, pd.DataFrame]
        Input dataset(s). A dict allows applying transformations to several
        datasets independently.
    columns : list
        Columns to transform.
    method : str
        Type of transformation. Options: 'log', 'sqrt', 'boxcox', 'reciprocal',
        'zscore', 'minmax', 'ratio', 'none'.
    reference_group : str, optional
        Control/reference group (used only for ratio normalization).
    group_column : str, optional
        Column name identifying groups for ratio normalization.
    subject_column : str, optional
        Subject identifier column used for paired ratio transformations.
    comparison_factor : str, optional
        Factor differentiating datasets in ratio mode.
    paired : bool, default=False
        Whether ratio normalization is paired (subject-by-subject).
    verbose : bool, default=False
        Print progress messages.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
        The transformed dataset(s).
    """

    # Handle multiple datasets
    if isinstance(data, dict):
        # Apply transformation independently to each dataset
        return {
            name: _transform_single_df(
                df=df,
                columns=columns,
                method=method,
                reference_group=reference_group,
                group_column=group_column,
                subject_column=subject_column,
                comparison_factor=comparison_factor,
                paired=paired,
                verbose=verbose,
                dataset_name=name
            )
            for name, df in data.items()
        }

    # Handle single dataset
    return _transform_single_df(
        df=data,
        columns=columns,
        method=method,
        reference_group=reference_group,
        group_column=group_column,
        subject_column=subject_column,
        comparison_factor=comparison_factor,
        paired=paired,
        verbose=verbose
    )


# Single-DataFrame backend for transform_variables().
def _transform_single_df(
    df: pd.DataFrame,
    columns: list,
    method: str,
    *,
    reference_group: str = None,
    group_column: str | list | dict = None,
    subject_column: str = None,
    comparison_factor: str = None,
    paired: bool = False,
    verbose: bool = True,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Apply a specific transformation to a single DataFrame.

    This function is called internally by `transform_variables` and handles
    both scalar transformations (log, sqrt, zscore...) and ratio normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to transform.
    columns : list
        Columns to apply transformations to.
    method : str
        Transformation method.
    reference_group : str, optional
        Used only for ratio normalization.
    group_column : str, list or dict, optional
        Column(s) defining groups. Dict allows different group columns per dataset.
    subject_column : str, optional
        Subject identifier column used for paired ratios.
    comparison_factor : str, optional
        Factor differentiating datasets for ratio mode.
    paired : bool, default=False
        Whether ratio mode should be paired (subject-matched).
    verbose : bool, default=True
        Print informational messages.
    dataset_name : str, optional
        Name of current dataset when processing multiple datasets.

    Returns
    -------
    pd.DataFrame
        Transformed dataset.
    """

    df = df.copy()
    valid_methods = {
        "log", "sqrt", "boxcox", "yeojohnson", "reciprocal",
        "zscore", "minmax", "ratio", "none"
    }

    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Valid options: {valid_methods}")

    if group_column is not None:
        
        # Dictionary mode: different grouping per dataset
        if isinstance(group_column, dict):
            if dataset_name is None or dataset_name not in group_column:
                raise ValueError("dataset_name must be provided and exist in group_column dict")
            group_cols = group_column[dataset_name]
        else:
            group_cols = group_column

        # Normalize to list format
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        elif not isinstance(group_cols, list):
            raise TypeError("'group_column' must be a string or list of strings")
    else:
        group_cols = None

    if method == "ratio":
        if reference_group is None or group_column is None:
            raise ValueError("'reference_group' and 'group_column' are required for ratio normalization.")

        return _compute_ratios(
            df=df,
            variables=columns,
            reference_group=reference_group,
            group_columns=group_cols,
            subject_column=subject_column,
            paired=paired,
            comparison_factor=comparison_factor,
            verbose=verbose,
            dataset_name=dataset_name
        )

    scaler_z = StandardScaler()
    scaler_minmax = MinMaxScaler()

    for col in columns:
        if col not in df.columns:
            if verbose:
                print(f"⚠️ Column '{col}' not found. Skipping.")
            continue

        series = df[col]

        if method == "none":
            if verbose:
                print(f"➡️ No transformation applied to '{col}'.")
            continue

        try:
            # ---- Mathematical transforms ----
            if method == "log":
                offset = 1e-6
                min_val = series.min()
                adjusted = series - min_val + offset if min_val <= 0 else series
                df[col] = np.log(adjusted)

            elif method == "sqrt":
                df[col] = np.sqrt(series.clip(lower=0))

            elif method == "boxcox":
                if (series <= 0).any():
                    raise ValueError("Box-Cox requires strictly positive values.")
                transformed, _ = stats.boxcox(series)
                df[col] = transformed

            elif method == "yeojohnson":
                # Handles negative and zero values, unlike Box-Cox — the
                # transform to reach for whenever a column can't
                # guarantee strictly-positive values (Box-Cox raises
                # above instead of silently doing the wrong thing).
                transformed, _ = stats.yeojohnson(series.astype(float))
                df[col] = transformed

            elif method == "reciprocal":
                df[col] = np.where(series != 0, 1 / series, np.nan)

            # ---- Scaling transforms ----
            elif method == "zscore":
                df[col] = scaler_z.fit_transform(series.to_numpy().reshape(-1, 1)).flatten()

            elif method == "minmax":
                df[col] = scaler_minmax.fit_transform(series.to_numpy().reshape(-1, 1)).flatten()

            if verbose:
                print(f"✅ Applied '{method}' to '{col}'.")

        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped '{col}' due to error: {e}")

    if verbose:
        suffix = f" for dataset '{dataset_name}'" if dataset_name else ""
        print(f"\n🎯 Transformation '{method}' completed{suffix}.\n")

    return df


# Builds the *_Ratio datasets as post/pre x 100. With paired=True the
# reference is the animal's own baseline; otherwise it is the
# reference-group mean.
def _compute_ratios(
    df: pd.DataFrame,
    variables: list,
    reference_group: str,
    group_columns: list,
    subject_column: str = None,
    paired: bool = False,
    comparison_factor: str = None,
    verbose: bool = True,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Compute ratio-normalized variables relative to a reference group.

    Supports:
    - Unpaired ratio normalization (simple normalization relative to reference group mean)
    - Paired normalization (per-subject normalization using subject_column)
    - Multiple grouping columns
    - Use inside transform_variables with multiple datasets

    IMPORTANT:
    This implementation avoids unnecessary memory expansion and keeps computation
    vectorized whenever possible.
    """

    df = df.copy()

    numeric_vars = [
        v for v in variables
        if v in df.columns and pd.api.types.is_numeric_dtype(df[v])
    ]

    if len(numeric_vars) == 0:
        raise ValueError("No numeric variables available for ratio computation.")

    if verbose:
        print(f"\n⚙️ Computing ratios for: {numeric_vars} in dataset '{dataset_name or 'data'}'.")

    if group_columns is None:
        raise ValueError("group_columns must be provided for ratio normalization.")

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    for col in group_columns:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found in dataset.")

    group_col = group_columns[0]  # ratio always uses one main grouping column

    groups = df[group_col].dropna().unique()

    if reference_group not in groups:
        raise ValueError(f"Reference group '{reference_group}' not found in dataset.")

    if paired:
        if subject_column is None:
            raise ValueError("subject_column must be provided when paired=True.")

        if subject_column not in df.columns:
            raise ValueError(f"Subject column '{subject_column}' not found in DataFrame.")

        # Split reference and non-reference
        ref_df = df[df[group_col] == reference_group]

        # Merge df with its reference values per subject
        merged = df.merge(
            ref_df[[subject_column] + numeric_vars],
            on=subject_column,
            how="left",
            suffixes=("", "_ref")
        )

        # Compute ratios (vectorized)
        for var in numeric_vars:
            merged[var] = np.where(
                merged[f"{var}_ref"] == 0,
                np.nan,
                merged[var] / merged[f"{var}_ref"] * 100
            )

        # Keep only relevant columns
        result = merged[[subject_column, group_col] + numeric_vars]

        if verbose:
            print(f"✅ Paired ratio computation completed for dataset '{dataset_name or 'data'}'.")

        return result

    # Compute global reference means
    ref_means = df[df[group_col] == reference_group][numeric_vars].mean()

    # Avoid division by zero
    ref_means = ref_means.replace({0: np.nan})

    # Vectorized normalization. subject_column is not needed for the
    # unpaired computation itself (there is no per-subject pairing), but
    # is carried through when available so a reader can still trace a
    # ratio row back to the animal it came from.
    id_cols = ([subject_column] if subject_column
               and subject_column in df.columns else [])
    result = df[id_cols + [group_col] + numeric_vars].copy()

    for var in numeric_vars:
        result[var] = df[var] / ref_means[var] * 100

    if verbose:
        print(f"✅ Unpaired ratio computation completed for dataset '{dataset_name or 'data'}'.")

    return result


# =============================================================================
# SECTION 6. OUTLIER HANDLING
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

# =============================================================================
# SECTION 7. ASSUMPTION CHECKS
# =============================================================================


# Per-group normality report, written to '2. Normality report'. The test
# selection in inference.py recomputes normality independently.
def normality_test(
    dataframes: Dict[str, pd.DataFrame],
    variables: List[str],
    group_column: Union[str, List[str], Dict[str, Union[str, List[str]]], None],
    folder: str,
    *,
    min_n: int = 3,
    max_n_shapiro: int = 5000,
    verbose: bool = False
) -> None:
    """
    Perform normality tests (Shapiro–Wilk or D’Agostino K^2) for each dataset, variable and group (optional).
    Results are exported as a plain CSV — no HTML, no plots.

    Test selection:
    - Shapiro–Wilk for n <= max_n_shapiro
    - D’Agostino K^2 for n > max_n_shapiro

    Parameters
    ----------
    dataframes : dict
        Mapping of dataset_name → DataFrame.
    variables : list
        Variables to test for normality.
    group_column : str | list | dict | None
        Grouping configuration. Can be:
            - str: column name
            - list: multiple columns
            - dict: per-dataset grouping columns
            - None: no grouping
    folder : str
        Directory where the CSV results will be saved.
    min_n : int, default=3
        Minimum number of observations required to run a test.
    max_n_shapiro : int, default=5000
        Maximum sample size for Shapiro–Wilk; larger samples use D’Agostino.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    None
    """

    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    # Iterate over all datasets
    for name, df in dataframes.items():

        # --- Determine grouping columns ---
        if isinstance(group_column, dict):
            group_cols = group_column.get(name, None)
        else:
            group_cols = group_column

        # Validate group columns exist in DataFrame
        if isinstance(group_cols, str):
            if group_cols not in df.columns:
                if verbose:
                    print(f"⚠️ Group column '{group_cols}' not found in dataset '{name}'. Using no grouping.")
                group_cols = None

        elif isinstance(group_cols, list):
            missing = [c for c in group_cols if c not in df.columns]
            if missing:
                if verbose:
                    print(f"⚠️ Missing group columns {missing} in '{name}'. Using no grouping.")
                group_cols = None

        # --- Generate groups ---
        if group_cols is None:
            # Single group for the whole dataset
            groups = [(None, df)]
        else:
            if isinstance(group_cols, str):
                # Single grouping column
                groups = [(val, df[df[group_cols] == val]) for val in df[group_cols].dropna().unique()]
            else:
                # Multiple columns → use groupby
                groups = list(df.groupby(group_cols))

        rows = []

        # --- Loop over groups and variables ---
        for group_key, group_df in groups:

            for var in variables:

                # Skip missing or non-numeric variables
                if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]):
                    continue

                x = group_df[var].dropna()
                n_total = len(x)

                # Skip if too few data points
                if n_total < min_n:
                    rows.append([group_key, var, n_total, 0, np.nan, np.nan, "skipped"])
                    continue

                # --- Select normality test ---
                if n_total <= max_n_shapiro:
                    # Use Shapiro–Wilk for small/medium samples
                    try:
                        stat, p = stats.shapiro(x)
                    except Exception:
                        stat, p = np.nan, np.nan
                else:
                    # Use D’Agostino K^2 for larger samples
                    try:
                        stat, p = stats.normaltest(x)
                    except Exception:
                        stat, p = np.nan, np.nan

                # Determine significance symbols
                sig = significance_stars(p)

                # Record row
                rows.append([group_key, var, n_total, len(x), stat, p, sig])

        # --- Build result DataFrame ---
        result_df = pd.DataFrame(
            rows,
            columns=["Group", "Variable", "N total", "N used", "Statistic", "p-value", "Significance"]
        )

        csv_path = os.path.join(folder, f"normality - {name}.csv")
        save_csv(result_df, csv_path, precision=4)

        if verbose:
            print(f"📄 CSV saved: {csv_path}")


# Per-variable variance-homogeneity report, written to
# '3. Homoscedasticity report'.
def homoscedasticity_test(
    dataframes: Dict[str, pd.DataFrame],
    variables: List[str],
    folder: str,
    normality_results: Dict[str, Dict[str, Dict[str, bool]]],
    group_column: Union[str, List[str], Dict[str, Union[str, List[str]]], None] = None,
    *,
    min_n: int = 2,
    max_n: int = 5000,
    subsample_if_large: bool = True,
    random_state: int = 42,
    verbose: bool = True
) -> None:
    """
    Test homogeneity of variances (homoscedasticity) using an automatically chosen test:
    Bartlett, Levene, Brown-Forsythe or Fligner-Killeen depending on group normality
    and dataset size, leveraging previously computed normality results.

    Saves results as a plain CSV.

    Parameters
    ----------
    dataframes : dict
        Dictionary {dataset_name: DataFrame}.
    variables : list
        Numeric variables to test.
    folder : str
        Output folder for the CSV.
    normality_results : dict
        {dataset_name: {variable: {group: True/False}}} from normality_test.
    group_column : str | list | dict | None
        Column(s) used for grouping.
    min_n : int
        Minimum entries per group to perform test.
    max_n : int
        Maximum entries per group; subsampling applied if exceeded.
    subsample_if_large : bool
        Whether to subsample large groups.
    random_state : int
        RNG seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    None
    """
    os.makedirs(folder, exist_ok=True)

    for name, df in dataframes.items():
        # --- Determine grouping columns ---
        grp_col = None
        if isinstance(group_column, dict):
            grp_col = group_column.get(name, None)
        else:
            grp_col = group_column

        # Validate grouping columns
        if isinstance(grp_col, str) and grp_col not in df.columns:
            grp_col = None
        elif isinstance(grp_col, list):
            missing_cols = [c for c in grp_col if c not in df.columns]
            if missing_cols:
                grp_col = None

        # Generate groups
        if grp_col is None:
            groups = [(None, df)]
        else:
            if isinstance(grp_col, str):
                groups = [(val, df[df[grp_col] == val]) for val in df[grp_col].dropna().unique()]
            else:
                groups = list(df.groupby(grp_col))

        rows = []

        # --- Loop over variables ---
        for var in variables:
            if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]):
                continue

            # Collect samples per group
            group_samples = []
            n_total, n_used = 0, 0
            valid = True

            for group_key, group_df in groups:
                x = group_df[var].dropna()
                n_total += len(group_df[var])
                if len(x) < min_n:
                    valid = False
                    break
                if subsample_if_large and len(x) > max_n:
                    x = x.sample(max_n, random_state=random_state)
                group_samples.append(x)
                n_used += len(x)

            if not valid:
                rows.append([var, n_total, n_used, np.nan, np.nan, "skipped", "skipped"])
                continue

            # --- Determine test based on normality ---
            use_test = "Levene"
            if normality_results and name in normality_results and var in normality_results[name]:
                group_flags = normality_results[name][var]
                if all(group_flags.get(k, False) for k in group_flags):
                    use_test = "Bartlett"
                else:
                    if n_total > 1000:
                        use_test = "Fligner"
                    else:
                        use_test = "Brown-Forsythe"
            else:
                # fallback
                if n_total <= 5000:
                    use_test = "Bartlett"
                elif n_total > 1000:
                    use_test = "Fligner"
                else:
                    use_test = "Brown-Forsythe"

            # --- Perform test ---
            try:
                if use_test == "Bartlett":
                    stat, p_value = stats.bartlett(*group_samples)
                elif use_test == "Levene":
                    stat, p_value = stats.levene(*group_samples, center="mean")
                elif use_test == "Brown-Forsythe":
                    stat, p_value = stats.levene(*group_samples, center="median")
                elif use_test == "Fligner":
                    stat, p_value = stats.fligner(*group_samples)
                else:
                    stat, p_value = np.nan, np.nan
            except Exception:
                stat, p_value = np.nan, np.nan

            sig = significance_stars(p_value)
            rows.append([var, n_total, n_used, use_test, stat, p_value, sig])

        # --- Build result DataFrame ---
        result_df = pd.DataFrame(rows, columns=[
            "Variable", "N total", "N used", "Test Used", "Statistic", "p-value", "Significance"
        ])

        csv_path = os.path.join(folder, f"homoscedasticity - {name}.csv")
        save_csv(result_df, csv_path)

        if verbose:
            tested = result_df['Significance'].isin(['*','**','***','ns']).sum()
            skipped = (result_df['Significance'] == "skipped").sum()
            print(f"✅ CSV saved: {csv_path} (tested: {tested}, skipped: {skipped})")


# =============================================================================
# SECTION 8. CORRELATION STRUCTURE
# =============================================================================


# Label-encodes categorical variables and writes the correlation report.
def correlation_and_encoding_auto(
    dataframes: dict,
    variables: list,
    folder: str,
    *,
    datasets_config: dict,
    normality_results: dict = None,
    verbose: bool = True,
    min_n: int = 3
) -> tuple[dict, dict]:
    """
    Perform correlation analysis across all variables using a fully predefined
    categorical encoding scheme.

    This function computes a full correlation matrix (all variables vs. all variables)
    for each dataset, strictly respecting user-defined categorical encodings provided
    in `datasets_config["categorical_mappings"]`.

    Identifier variables and constant variables are automatically
    excluded from the correlation analysis and plots.

    The function:
    • Applies user-defined categorical encodings only.
    • Removes identifier and constant variables.
    • Drops any non-numeric or non-encoded variables.
    • Automatically selects Pearson or Spearman correlation based on provided
      Shapiro normality test results.
    • Saves the correlation matrix as a CSV per dataset, plus a small
      label-encodings CSV alongside it when categorical mappings were used.
    • Returns both the label mappings used and the encoded datasets.

    Returns
    -------
    all_label_mappings : dict
        Reverse mappings {encoded_value → original_label} used per dataset.

    encoded_dataframes : dict
        Encoded datasets used for correlation analysis.
    """
    os.makedirs(folder, exist_ok=True)

    all_label_mappings = {}
    encoded_dataframes = {}

    for name, df in dataframes.items():

        cfg = datasets_config.get(name, {})
        cat_maps = cfg.get("categorical_mappings", {})

        sub_df = df[variables].copy()
        label_mappings = {}

        for col, mapping in cat_maps.items():
            if col in sub_df.columns:
                sub_df[col] = sub_df[col].map(mapping)
                label_mappings[col] = {v: k for k, v in mapping.items()}

        sub_df = sub_df.dropna(how="all")
        numeric_df = sub_df.select_dtypes(include=[np.number])

        if numeric_df.shape[0] < min_n:
            if verbose:
                print(f"⚠️ Skipped {name}: insufficient data")
            continue

        # Remove identifier variables
        id_cols = [
            c for c in numeric_df.columns
            if _is_identifier(c)
        ]
        numeric_df = numeric_df.drop(columns=id_cols, errors="ignore")

        # Remove constant variables
        constant_cols = [
            c for c in numeric_df.columns
            if numeric_df[c].nunique(dropna=True) <= 1
        ]
        numeric_df = numeric_df.drop(columns=constant_cols)

        if numeric_df.shape[1] < 2:
            if verbose:
                print(
                    f"⚠️ Skipped {name}: "
                    f"not enough non-constant numeric variables"
                )
            continue

        method_used = "spearman"

        if normality_results and name in normality_results:
            pvals = normality_results[name]
            p_list = []

            for var in numeric_df.columns:
                if var in pvals:
                    valid_p = [
                        p for p in pvals[var].values()
                        if not np.isnan(p)
                    ]
                    if valid_p:
                        p_list.append(np.mean(valid_p))

            if p_list and all(p > 0.05 for p in p_list):
                method_used = "pearson"

        corr_matrix = numeric_df.corr(method=method_used)

        _save_correlation_csv(name, corr_matrix, method_used, folder,
                                label_mappings, verbose)

        all_label_mappings[name] = label_mappings
        encoded_dataframes[name] = sub_df.copy()

    return all_label_mappings, encoded_dataframes


# CSV writer backend for correlation_and_encoding_auto(). The
# correlation matrix keeps its variable names as both the index and
# the header row (written via to_csv's default index=True, the one
# exception to this module's usual index=False — a correlation matrix
# needs its row labels to be readable at all); label encodings, when
# any categorical variable was mapped, go to a small sibling CSV
# rather than being folded into the same table.
def _save_correlation_csv(name, corr, method, folder, label_mappings, verbose=True):
    os.makedirs(folder, exist_ok=True)

    corr_path = os.path.join(folder, f"correlation ({method}) - {name}.csv")
    corr.round(4).to_csv(corr_path)

    if label_mappings:
        # label_mappings[col] is {code: label} (reversed from the
        # user-supplied {label: code} categorical_mappings config).
        rows = [{"column": col, "code": code, "label": label}
                 for col, mapping in label_mappings.items()
                 for code, label in mapping.items()]
        enc_path = os.path.join(folder, f"correlation label encodings - {name}.csv")
        save_csv(pd.DataFrame(rows), enc_path)

    if verbose:
        print(f"📄 CSV saved: {corr_path}")


# =============================================================================
# SECTION 9. MULTICOLLINEARITY
# =============================================================================


# VIF-based multicollinearity report, written to
# '5. Multicollinearity report'.
def multicollinearity_analysis_auto(
    dataframes,
    variables,
    vif_threshold=5.0,
    ridge_alpha=1.0,
    lasso_alpha=0.01,
    folder=".",
    verbose=True
):
    """
    Automatic multicollinearity analysis for multiple datasets.

    Computes VIF iteratively, uses Ridge/Lasso coefficients to identify worst variables,
    generates recommendations ("Keep"/"Remove"), saves a plain CSV report and returns
    summary and cleaned numeric datasets.

    Args:
        dataframes (dict): Dictionary of pandas DataFrames to process.
        variables (list): Numeric columns to check for multicollinearity.
        vif_threshold (float): VIF threshold for high collinearity.
        ridge_alpha (float): Ridge regression alpha.
        lasso_alpha (float): Lasso regression alpha.
        folder (str): Path to save the CSV report.
        verbose (bool): Print progress messages.

    Returns:
        results (dict): Dictionary with 'summary' DataFrame per dataset.
        transformed_dataframes (dict): Dictionary with cleaned numeric DataFrames per dataset.
    """

    os.makedirs(folder, exist_ok=True)
    results = {}
    transformed_dataframes = {}

    for name, df in dataframes.items():
        if verbose:
            print(f"\n🔄 Processing dataset '{name}'")

        # Select numeric columns
        sub_df_numeric = df.select_dtypes(include=[np.number])
        variables_iter = [v for v in variables if v in sub_df_numeric.columns]

        if not variables_iter:
            if verbose:
                print(f"⚠️ Dataset '{name}' has no numeric columns to process. Skipping.")
            continue

        constant_cols = [col for col in variables_iter if sub_df_numeric[col].nunique() == 1]
        if constant_cols and verbose:
            print(f"⚠️ Columns with constant values excluded from VIF/correlation: {constant_cols}")
        variables_iter = [col for col in variables_iter if col not in constant_cols]

        # All-NaN columns (e.g. a ratio computed against an all-zero
        # reference, or a sparse assay sub-panel with no valid rows left
        # after the missing-value skip gate) can't contribute to VIF at
        # all and would otherwise wipe out every row once combined with
        # the row-wise dropna below.
        all_nan_cols = [col for col in variables_iter if sub_df_numeric[col].isna().all()]
        if all_nan_cols and verbose:
            print(f"⚠️ All-NaN columns excluded from VIF/correlation: {all_nan_cols}")
        variables_iter = [col for col in variables_iter if col not in all_nan_cols]

        if not variables_iter:
            if verbose:
                print(f"⚠️ All numeric columns are constant or all-NaN. Skipping dataset '{name}'.")
            continue

        keep_vars = []
        remove_vars = []
        vifs_history = {}
        ridge_coefs_history = {}
        lasso_coefs_history = {}

        while len(variables_iter) > 1:
            sub_df_iter = sub_df_numeric[variables_iter].select_dtypes(include=[np.number])
            # variance_inflation_factor (like Ridge/Lasso below) needs a
            # complete-case matrix — a row with a NaN in even one of the
            # remaining columns would otherwise reach statsmodels' OLS
            # and raise MissingDataError rather than being screened out
            # the way handle_nulls/detect_and_replace_outliers already
            # screen every other step in this pipeline.
            sub_df_iter = sub_df_iter.dropna(axis=0, how="any")
            if sub_df_iter.empty or len(sub_df_iter) < len(variables_iter) + 2:
                if verbose:
                    print(f"⚠️ Too few complete-case rows for VIF on {name} "
                          f"with {variables_iter}. Skipping remaining "
                          f"multicollinearity check.")
                keep_vars.extend(variables_iter)
                break

            # Calculate VIF
            vifs = pd.Series([variance_inflation_factor(sub_df_iter.values, i)
                              for i in range(len(variables_iter))],
                             index=variables_iter).replace([np.inf, -np.inf], np.nan).fillna(np.inf).astype(float)

            # Dummy target for Ridge/Lasso
            y_dummy = sub_df_iter[variables_iter[0]].values
            X_scaled = StandardScaler().fit_transform(sub_df_iter)

            ridge = Ridge(alpha=ridge_alpha).fit(X_scaled, y_dummy)
            lasso = Lasso(alpha=lasso_alpha, max_iter=10000).fit(X_scaled, y_dummy)

            ridge_coefs = pd.Series(np.abs(ridge.coef_), index=variables_iter)
            lasso_coefs = pd.Series(np.abs(lasso.coef_), index=variables_iter)

            # Remove variable with max VIF if exceeds threshold
            max_vif_var = vifs.idxmax()
            max_vif_val = vifs[max_vif_var]

            if max_vif_val > vif_threshold:
                vifs_history[max_vif_var] = max_vif_val
                ridge_coefs_history[max_vif_var] = ridge_coefs[max_vif_var]
                lasso_coefs_history[max_vif_var] = lasso_coefs[max_vif_var]

                worst_var = max_vif_var
                remove_vars.append(worst_var)
                variables_iter.remove(worst_var)
                if verbose:
                    print(f"⚠️ Removing {worst_var} with VIF={max_vif_val:.2f}")
            else:
                keep_vars.extend(variables_iter)
                for var in variables_iter:
                    vifs_history[var] = vifs[var]
                    ridge_coefs_history[var] = ridge_coefs[var]
                    lasso_coefs_history[var] = lasso_coefs[var]
                break

        # Build summary DataFrame
        summary_records = []
        for var in keep_vars:
            summary_records.append({
                "Variable": var,
                "VIF": vifs_history.get(var, np.nan),
                "Ridge_coef": ridge_coefs_history.get(var, np.nan),
                "Lasso_coef": lasso_coefs_history.get(var, np.nan),
                "Recommendation": "Keep"
            })
        for var in remove_vars:
            summary_records.append({
                "Variable": var,
                "VIF": vifs_history.get(var, np.nan),
                "Ridge_coef": ridge_coefs_history.get(var, np.nan),
                "Lasso_coef": lasso_coefs_history.get(var, np.nan),
                "Recommendation": "Remove"
            })

        final_summary = pd.DataFrame(summary_records)

        csv_path = os.path.join(folder, f"multicollinearity - {name}.csv")
        save_csv(final_summary, csv_path, precision=4)
        if verbose:
            print(f"✅ CSV saved: {csv_path}")

        # Save cleaned dataset
        transformed_dataframes[name] = sub_df_numeric[keep_vars].copy()
        results[name] = {"summary": final_summary}

        # Clean memory. ridge/lasso are only bound once the VIF loop runs
        # at least one full iteration — a dataset skipped early (e.g. too
        # few complete-case rows once NaNs are dropped) never binds them,
        # so they're freed defensively rather than assumed to exist.
        del sub_df_numeric
        gc.collect()

    return results, transformed_dataframes
