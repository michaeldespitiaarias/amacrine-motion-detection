"""Stage 2 - two-group hypothesis contrasts.

Compares two levels of a grouping factor across every numeric variable of a
dataset, choosing the test from the data and writing one plain CSV table per
dataset (raw numbers, no styling, no narrative document).

Test selection
--------------
    normality per group : Shapiro-Wilk, alpha = 0.05
                          (D'Agostino K^2 above ``max_n_shapiro``)
    equal variances     : Bartlett when both groups pass normality

    paired   + normal      -> paired t-test            effect: Cohen's dz
    paired   + non-normal  -> Wilcoxon signed-rank      effect: r
    unpaired + normal + equal var    -> Student's t     effect: Cohen's d
    unpaired + normal + unequal var  -> Welch's t       effect: Cohen's d
    unpaired + non-normal            -> Mann-Whitney U  effect: r

Benjamini-Hochberg FDR is applied across every variable tested in one
dataset's table (one dataset = one family; see ``multicomparison.bh_fdr``),
reported alongside the raw, uncorrected p-value rather than replacing it.

Designs measured at more than two levels declare a ``levels`` pair in the
registry, which selects the two conditions to contrast and their order.

Output
------
One CSV table per dataset. Alongside the test, its p-value, significance class
and effect size, each row carries n, mean, SD and median per group, the
direction of the effect, the pretest p-values, Hedges' g, a bootstrap
confidence interval, a signed rank-biserial, and the BH-FDR-adjusted q-value
with its own significance flag.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from .effect_sizes import (
    _cohen_d,
    _effect_r_from_u,
    _stars,
    _wilcoxon_r,
    bootstrap_ci,
    describe_group,
    hedges_g,
    rank_biserial,
)
from .multicomparison import bh_fdr
from .reporting import save_csv

# Alpha for the normality and variance pretests and for the significance ladder,
# stated once rather than repeated as a literal at each decision point.
ALPHA = 0.05


# ============================================================================= #
# SECTION 1. DATA LOADING
# ============================================================================= #


# Column-name suffixes that are numeric/boolean but are diagnostic
# metadata, never a variable to test — e.g. the `{col}_outlier` flag
# columns preprocessing.detect_and_replace_outliers adds for cells its
# skip gate withheld. Kept in sync with preprocessing.NON_MEASURE_SUFFIXES.
NON_MEASURE_SUFFIXES = ("_outlier",)


def load_dataset(name: str, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load one preprocessed dataset and split its columns by role.

    A column is treated as categorical when its dtype is object or category, or
    when its name contains the substring ``id`` (case-insensitive), which keeps
    subject identifiers out of the numeric analysis. Columns ending in
    NON_MEASURE_SUFFIXES are also routed to categorical_cols even though
    pandas parses their True/False values as boolean — they are diagnostic
    flags, not a variable to contrast (booleans would also break the
    parametric/non-parametric tests below, which assume a continuous DV).

    Parameters
    ----------
    name : str
        Dataset key, used for the progress message.
    cfg : dict
        Configuration entry; must provide ``location``.

    Returns
    -------
    tuple
        ``(dataframe, numeric_columns, categorical_columns)``.
    """
    df = pd.read_csv(cfg["location"])

    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object"
        or df[col].dtype.name == "category"
        or df[col].dtype == "bool"
        or "id" in col.lower()
        or str(col).lower().endswith(NON_MEASURE_SUFFIXES)
    ]
    numeric_cols = [col for col in df.columns if col not in categorical_cols]

    print(f"Loaded '{name}' with {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    return df, numeric_cols, categorical_cols


# ============================================================================= #
# SECTION 2. SAMPLE EXTRACTION AND PAIRING
# ============================================================================= #


def _extract_samples(
    df: pd.DataFrame,
    group_col: str,
    var: str,
    group1,
    group2,
    paired: bool,
    subject_col: str | None,
    dataset: str = "",
) -> tuple[pd.Series, pd.Series, str | None]:
    """Return the two samples to compare, aligned by subject when paired.

    Paired samples are matched through ``subject_col`` so that each subject's
    two values are compared with each other regardless of row order.
    Where the identifier cannot resolve the pairing, the samples are taken in
    row order instead, which assumes both groups list their subjects in the
    same sequence.

    Returns
    -------
    tuple
        ``(sample1, sample2, problem)``, where ``problem`` is None on success or
        a short description of why the contrast cannot be run.
    """
    if not paired:
        return (
            df.loc[df[group_col] == group1, var].dropna(),
            df.loc[df[group_col] == group2, var].dropna(),
            None,
        )

    if subject_col is None or subject_col not in df.columns:
        s1 = df.loc[df[group_col] == group1, var].dropna()
        s2 = df.loc[df[group_col] == group2, var].dropna()
        return s1, s2, None if len(s1) == len(s2) else "Paired size mismatch"

    subset = df[[subject_col, group_col, var]].dropna(subset=[var])

    if subset.duplicated(subset=[subject_col, group_col]).any():
        # An identifier appears more than once within a group, so the pairing
        # cannot be resolved from it; fall back to row order.
        s1 = df.loc[df[group_col] == group1, var].dropna()
        s2 = df.loc[df[group_col] == group2, var].dropna()
        return s1, s2, None if len(s1) == len(s2) else "Paired size mismatch"

    wide = subset.pivot(index=subject_col, columns=group_col, values=var)
    if group1 not in wide.columns or group2 not in wide.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float), "Paired size mismatch"

    complete = wide[[group1, group2]].dropna()
    dropped = len(wide) - len(complete)
    if dropped:
        warnings.warn(
            f"[{dataset}:{var}] {dropped} subject(s) dropped for lacking one "
            f"member of the pair.",
            RuntimeWarning,
            stacklevel=2,
        )

    return complete[group1], complete[group2], None


# ============================================================================= #
# SECTION 3. ASSUMPTION PRETESTS
# ============================================================================= #


def _normality(data1, data2, max_n_shapiro: int) -> tuple[float, float]:
    """Per-group normality p-values, Shapiro-Wilk or D'Agostino K^2 by size.

    On failure both p-values are set to 0, routing the contrast to the
    non-parametric branch.
    """
    n_total = len(data1) + len(data2)
    try:
        if n_total <= max_n_shapiro:
            return stats.shapiro(data1)[1], stats.shapiro(data2)[1]
        return stats.normaltest(data1)[1], stats.normaltest(data2)[1]
    except Exception:
        return 0.0, 0.0


def _equal_variance(data1, data2, both_normal: bool) -> tuple[str, float]:
    """Variance-homogeneity pretest and the name of the test used.

    Bartlett is applied when both groups pass normality, which is the case in
    which the result is consulted to choose between Student's and Welch's t.
    Brown-Forsythe and Fligner cover the non-normal case, where the contrast
    proceeds to Mann-Whitney.
    """
    n_total = len(data1) + len(data2)
    if both_normal:
        name = "Bartlett"
    elif n_total > 1000:
        name = "Fligner"
    else:
        name = "Brown-Forsythe"

    try:
        if name == "Bartlett":
            return name, stats.bartlett(data1, data2)[1]
        if name == "Fligner":
            return name, stats.fligner(data1, data2)[1]
        return name, stats.levene(data1, data2, center="median")[1]
    except Exception:
        return name, np.nan


# ============================================================================= #
# SECTION 4. THE TWO-GROUP CONTRAST
# ============================================================================= #


def _run_test(data1, data2, paired: bool, both_normal: bool, equal_var: bool):
    """Apply the selected test and its matching effect size.

    Returns ``(test_name, statistic, p_value, effect_value, effect_metric)``.
    """
    if paired:
        if both_normal:
            stat, p_value = stats.ttest_rel(data1, data2)
            return ("Paired t-test", stat, p_value,
                    _cohen_d(data1.values, data2.values, paired=True), "Cohen's dz")

        stat, p_value = stats.wilcoxon(data1, data2)
        return ("Wilcoxon signed-rank test", stat, p_value,
                _wilcoxon_r(p_value, len(data1)), "r")

    if both_normal:
        if equal_var:
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            test_name = "Student's t-test"
        else:
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test"
        return (test_name, stat, p_value,
                _cohen_d(data1.values, data2.values, paired=False), "Cohen's d")

    stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")
    return ("Mann-Whitney U test", stat, p_value,
            _effect_r_from_u(stat, len(data1), len(data2)), "r")


def compare_two_groups(
    dataframes: dict,
    comparison_by: dict,
    variables: list,
    folder: str,
    paired: dict,
    subject: dict | None = None,
    table_prefix: str = "Contrast table",
    min_n: int = 3,
    max_n_shapiro: int = 5000,
    with_intervals: bool = True,
    verbose: bool = True,
) -> dict:
    """Compare two groups across numeric variables and write one CSV table.

    For each dataset and each numeric variable this runs the normality and
    variance pretests, selects the test as documented in the module docstring
    and records the result together with the descriptive statistics needed to
    read the direction of the effect.

    Parameters
    ----------
    dataframes : dict
        ``{dataset name: DataFrame}``.
    comparison_by : dict
        ``{dataset name: grouping column}``.
    variables : list
        Numeric variables to test.
    folder : str
        Output directory for the CSV table.
    paired : dict
        ``{dataset name: bool}``.
    subject : dict, optional
        ``{dataset name: subject column}``, used to align paired samples.
    table_prefix : str
        Leading text of the output CSV's file name.
    min_n : int
        Minimum observations per group; below this the contrast is skipped.
    max_n_shapiro : int
        Above this total n, normality uses D'Agostino K^2 instead of Shapiro-Wilk.
    with_intervals : bool
        Compute bootstrap confidence intervals. Set False for a fast pass.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{dataset name: {variable: result dict}}``.
    """
    os.makedirs(folder, exist_ok=True)
    subject = subject or {}
    results_all: dict[str, dict] = {}

    for name, df in dataframes.items():
        grp_col = comparison_by.get(name)
        is_paired = paired.get(name, False)
        subject_col = subject.get(name)

        if grp_col is None:
            if verbose:
                print(f"No comparison column defined for '{name}'. Skipping.")
            continue

        # Order of appearance, not alphabetical: this fixes which level is
        # group 1 and therefore the sign of every signed effect size.
        groups = df[grp_col].dropna().unique()
        if len(groups) != 2:
            if verbose:
                print(f"Dataset '{name}' has {len(groups)} levels in "
                      f"'{grp_col}', expected 2: {list(groups)}")
            continue

        group1, group2 = groups
        res_dict: dict[str, dict] = {}

        if verbose:
            print(f"\nProcessing '{name}': {group1} vs {group2} "
                  f"({'paired' if is_paired else 'independent'})")

        for var in variables:
            if var not in df.columns:
                continue

            data1, data2, problem = _extract_samples(
                df, grp_col, var, group1, group2, is_paired, subject_col, name)

            if problem is None and (len(data1) < min_n or len(data2) < min_n):
                problem = "Insufficient data"

            if problem is not None:
                res_dict[var] = _blank_row(group1, group2, problem, data1, data2)
                continue

            p_norm1, p_norm2 = _normality(data1, data2, max_n_shapiro)
            both_normal = p_norm1 > ALPHA and p_norm2 > ALPHA

            variance_test, p_variance = _equal_variance(data1, data2, both_normal)
            equal_var = bool(p_variance > ALPHA) if np.isfinite(p_variance) else False

            test_name, _stat, p_value, effect_value, effect_metric = _run_test(
                data1, data2, is_paired, both_normal, equal_var)

            row = _blank_row(group1, group2, test_name, data1, data2)
            row.update({
                "p_value": round(float(p_value), 6),
                "significance": _stars(p_value if p_value is not None else 1),
                "effect size metric": effect_metric,
                "effect size value": (
                    round(float(effect_value), 3)
                    if effect_value is not None and not np.isnan(effect_value)
                    else np.nan
                ),
                "normality p (group 1)": round(float(p_norm1), 4),
                "normality p (group 2)": round(float(p_norm2), 4),
                "variance test": variance_test,
                "variance p": (round(float(p_variance), 4)
                               if np.isfinite(p_variance) else np.nan),
                "Hedges g": round(hedges_g(data1.values, data2.values, is_paired), 3),
                "rank-biserial (signed)": round(
                    rank_biserial(data1.values, data2.values, is_paired), 3),
            })

            if with_intervals:
                low, high = bootstrap_ci(data1.values, data2.values, paired=is_paired)
                row["effect 95% CI"] = (
                    f"[{low:.2f}, {high:.2f}]" if np.isfinite(low) else "n/a")

            res_dict[var] = row

            if verbose:
                print(f"  {var}: {test_name}, p = {p_value:.4g} "
                      f"{row['significance']}, {effect_metric} = "
                      f"{row['effect size value']}")

        # Benjamini-Hochberg FDR across every variable tested in THIS
        # table (one dataset = one family). Adds two columns without
        # touching the raw p-value, its significance stars, or any
        # effect size — a reader who wants the uncorrected result still
        # has it. Variables skipped above (blank rows, p_value = NaN)
        # pass through untouched, per bh_fdr's NaN handling.
        variables_in_table = list(res_dict.keys())
        q_values = bh_fdr([res_dict[v]["p_value"] for v in variables_in_table])
        for var, q in zip(variables_in_table, q_values):
            res_dict[var]["p_value_fdr_bh"] = (
                round(float(q), 6) if np.isfinite(q) else np.nan)
            res_dict[var]["significant_fdr"] = bool(
                np.isfinite(q) and q < ALPHA)

        results_all[name] = res_dict

        result_df = pd.DataFrame(
            [{"Variable": var, **res} for var, res in res_dict.items()])
        csv_path = os.path.join(folder, f"{table_prefix} - {name}.csv")
        save_csv(result_df, csv_path)

        if verbose:
            print(f"Results saved for '{name}' -> {csv_path}")

    return results_all


def _blank_row(group1, group2, test_label: str, data1=None, data2=None) -> dict:
    """Build a result row carrying the descriptives, with no test outcome yet.

    Descriptives are attached even when the contrast is skipped, so that the
    table still shows how much data each variable had. ``direction`` states in
    words which group scores higher.
    """
    d1 = describe_group(data1 if data1 is not None else [])
    d2 = describe_group(data2 if data2 is not None else [])

    if np.isfinite(d1["mean"]) and np.isfinite(d2["mean"]):
        if d1["mean"] > d2["mean"]:
            direction = f"{group1} > {group2}"
        elif d1["mean"] < d2["mean"]:
            direction = f"{group1} < {group2}"
        else:
            direction = "equal"
    else:
        direction = "n/a"

    return {
        "group 1": group1,
        "group 2": group2,
        "n (group 1)": d1["n"],
        "n (group 2)": d2["n"],
        "mean (group 1)": round(d1["mean"], 3) if np.isfinite(d1["mean"]) else np.nan,
        "SD (group 1)": round(d1["sd"], 3) if np.isfinite(d1["sd"]) else np.nan,
        "mean (group 2)": round(d2["mean"], 3) if np.isfinite(d2["mean"]) else np.nan,
        "SD (group 2)": round(d2["sd"], 3) if np.isfinite(d2["sd"]) else np.nan,
        "direction": direction,
        "test": test_label,
        "p_value": np.nan,
        "significance": "ns",
        "effect size metric": "n/a",
        "effect size value": np.nan,
    }


# ============================================================================= #
# SECTION 5. PIPELINE DRIVER
# ============================================================================= #


def run_two_group_pipeline(datasets: dict, folder: str,
                           table_prefix: str = "Contrast table",
                           with_intervals: bool = True,
                           verbose: bool = True) -> dict:
    """Run the two-group contrast for every configured dataset.

    Datasets whose comparison column does not hold exactly two levels are
    skipped with a message.

    Parameters
    ----------
    datasets : dict
        ``{name: {location, comparison_by, paired, subject_column, ...}}``.
    folder : str
        Directory receiving one CSV table per dataset.
    table_prefix : str
        Leading text of the output file name.
    with_intervals : bool
        Compute bootstrap confidence intervals.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{dataset name: {variable: result dict}}`` for the datasets that ran.
    """
    os.makedirs(folder, exist_ok=True)
    all_results: dict[str, dict] = {}

    for name, cfg in datasets.items():
        df, numeric_cols, _ = load_dataset(name, cfg)
        comparison_col = cfg["comparison_by"]

        # `levels`, where declared, selects the two conditions to contrast and
        # the order in which they are taken, so that a design measured at more
        # than two timepoints can be analysed one contrast at a time.
        levels = cfg.get("levels")
        if levels:
            df = df[df[comparison_col].isin(levels)].copy()
            df[comparison_col] = pd.Categorical(
                df[comparison_col], categories=levels, ordered=True)
            df = df.sort_values(comparison_col, kind="stable")
            df[comparison_col] = df[comparison_col].astype(str)
            if verbose:
                print(f"Restricted '{name}' to {levels[0]!r} vs {levels[1]!r}")

        n_levels = df[comparison_col].nunique()
        if n_levels != 2:
            if verbose:
                print(f"Skipping '{name}': {n_levels} levels in "
                      f"'{comparison_col}'; not a two-group design.\n")
            continue

        # Identifier columns are never a variable to test. `load_dataset`
        # only routes a column to `numeric_cols` when it parses as numbers
        # AND its name doesn't contain "id" — a numeric subject/sample
        # column named e.g. "Subject" or "Mouse" slips through that filter,
        # and comparing the subject column against itself corrupts the
        # paired-pivot in `_extract_samples` (two same-named columns
        # selected at once). Excluded explicitly here by the configured
        # `subject_column`, on top of the existing "id" substring filter.
        subject_col = cfg.get("subject_column")
        test_vars = [v for v in numeric_cols if v != subject_col]

        all_results.update(compare_two_groups(
            dataframes={name: df},
            comparison_by={name: comparison_col},
            variables=test_vars,
            folder=folder,
            paired={name: cfg.get("paired", False)},
            subject={name: cfg.get("subject_column")},
            table_prefix=table_prefix,
            with_intervals=with_intervals,
            verbose=verbose,
        ))

    if verbose:
        print(f"\nFinished. {len(all_results)} dataset(s) analysed.")
    return all_results
