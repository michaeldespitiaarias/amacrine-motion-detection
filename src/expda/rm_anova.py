"""Stage 2b - repeated-measures contrasts for within-subject level families.

Some variables in a dataset are not independent measurements: they are the
same construct sampled at several ordered levels within one test session for
the same subject (e.g. contrast sensitivity at six spatial frequencies, ERG
amplitude at nine light intensities). Testing each level separately with
``inference.compare_two_groups`` and then correcting with BH-FDR treats them
as an unordered bag of variables, which (a) is statistically weaker than it
needs to be — it throws away the correlation between adjacent levels instead
of using it — and (b) still tests every level whether or not the deficit
actually depends on the level at all.

This module runs a single 2-way repeated-measures ANOVA per family instead
(Condition x Level, both within-subject), and only drills down level-by-level
when there is a reason to: **the per-level post-hoc runs if and only if the
Condition x Level interaction is itself significant** (GG-corrected). That
is the gate — evidence the effect's *shape* differs by level — before any
levels are compared individually. A significant Condition main effect with a
non-significant interaction is reported as a uniform, level-independent
shift and is not decomposed further; doing so would reintroduce exactly the
unguarded multiple-comparison problem this module exists to avoid.

Engine selection (per family, all three omnibus rows together — never
per-term, so the table never mixes p-values computed two different ways):

    all (Condition, Level) cells pass Shapiro-Wilk (alpha=0.05) and n >= 5,
    and Levene across cells (center='median') passes
        -> parametric ``pingouin.rm_anova``, Greenhouse-Geisser corrected
    otherwise
        -> Aligned Rank Transform (ART, Wobbrock 2011), adapted for the
           within-subject case

A between-subjects ART fallback (no subject term at all) would be the wrong
model here. This module's version is adapted for the repeated-measures case
instead: subject enters the closed-form decomposition as an
additive blocking term (matching the additivity ``pingouin.rm_anova`` itself
assumes — no subject x factor interactions), so the paired structure
survives into the aligned response, then the same ``pingouin.rm_anova`` is
refit on the aligned-and-ranked values to read off that one term's F/p — one
alignment per term, per Wobbrock's procedure, never the same ranked column
reused across terms. See ``_align_and_rank``'s docstring for the exact
decomposition and a double-counting bug it replaced.

Post-hoc (only when gated open): the same paired-contrast dispatch as
``inference.compare_two_groups`` (paired t-test or Wilcoxon signed-rank, by
per-level normality — this per-level pretest is unaffected by which engine
the omnibus used), one comparison per level, corrected across levels with
Holm-Bonferroni (``multicomparison.holm_adjust`` — exact family-wise
control, no independence assumption, appropriate for adjacent, correlated
levels).

Both outputs carry the assumption checks that gated the engine/test choice
as leading columns, before the F/p or test/p columns they gated — the
omnibus table's normality/homoscedasticity describe the whole family (one
value per family, repeated on all three rows, since the engine was chosen
once for the family, not per term); the post-hoc table's normality is
per-level, per the same pretest inference.compare_two_groups uses. A
starred assumption significance means that assumption was rejected, not
that a real effect was found — same convention as inference.py.

A family is a named set of columns in one dataset sharing a single ordered
axis (frequency, intensity, ...). Not every variable belongs to one:
qualitatively distinct named measurements (e.g. the ERG named waves Scotopic
b / Mixed a / Mixed b / Photopic b / OPs / Flicker) are not levels of a
common factor and are deliberately left to the existing per-variable
``compare_two_groups`` + BH-FDR path.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats

from .effect_sizes import _stars, describe_group, hedges_g
from .inference import ALPHA, _equal_variance, _extract_samples, _normality, _run_test
from .multicomparison import holm_adjust
from .reporting import save_csv

warnings.filterwarnings("ignore", category=FutureWarning, module="pingouin")

MIN_CELL_N = 5  # below this, or on a failed Shapiro-Wilk/Levene, ART takes over


# ============================================================================= #
# SECTION 1. ASSUMPTION GATE
# ============================================================================= #


def _cell_diagnostics(long: pd.DataFrame, cond_col: str, alpha: float = ALPHA) -> dict:
    """Per-(Condition, Level) cell Shapiro-Wilk, plus Levene across all cells.

    Returns ``{"parametric_ok": bool, "cells": [...]}``. A cell with fewer
    than 3 observations cannot be tested for normality and counts as a
    failure — too small to trust either way. ``normality_p`` is the
    worst-case (minimum) Shapiro-Wilk p-value across every cell — the
    single number that actually decided ``all_normal`` (a family passes
    only if every cell does, so its weakest cell is the binding one),
    reported alongside as a queryable column rather than only the bool.
    """
    cells = []
    groups_for_levene = []
    for (cond, lvl), sub in long.groupby([cond_col, "Level"]):
        vals = sub["Value"].dropna().values
        n = len(vals)
        normal = False
        p_norm = np.nan
        if n >= 3:
            try:
                p_norm = float(stats.shapiro(vals)[1])
                normal = bool(p_norm >= alpha)
            except Exception:
                normal = False
        cells.append({"Condition": cond, "Level": lvl, "n": n, "normal": normal, "p": p_norm})
        if n >= 2:
            groups_for_levene.append(vals)

    all_normal = all(c["normal"] for c in cells)
    min_n = min((c["n"] for c in cells), default=0)
    finite_p = [c["p"] for c in cells if np.isfinite(c["p"])]
    normality_p = min(finite_p) if finite_p else np.nan

    homoscedastic = True
    levene_p = np.nan
    if len(groups_for_levene) >= 2:
        try:
            levene_p = float(stats.levene(*groups_for_levene, center="median")[1])
            homoscedastic = bool(levene_p >= alpha)
        except Exception:
            homoscedastic = False

    parametric_ok = all_normal and homoscedastic and min_n >= MIN_CELL_N
    return {"parametric_ok": parametric_ok, "cells": cells,
            "all_normal": all_normal, "homoscedastic": homoscedastic, "min_n": min_n,
            "normality_p": normality_p, "levene_p": levene_p}


# ============================================================================= #
# SECTION 2. ALIGNED RANK TRANSFORM, ADAPTED FOR WITHIN-SUBJECT DESIGNS
# ============================================================================= #


def _align_and_rank(long: pd.DataFrame, cond_col: str, subject_col: str,
                     target: str) -> pd.Series:
    """ART alignment for one term, closed-form, subject as a blocking factor.

    ``target`` is one of ``"cond"``, ``"level"``, ``"interaction"``. Uses the
    classic additive repeated-measures decomposition (matching what
    ``pingouin.rm_anova`` itself assumes — no subject x factor interaction
    terms):

        Y_sij = grand_mean + subject_s + cond_i + level_j + (cond:level)_ij + error_sij

    Every term on the right is a centred marginal/cell mean, computable in
    closed form because the design is fully balanced (every subject
    contributes exactly one observation per (Condition, Level) cell — this
    function is not called until that has been guaranteed upstream). The
    full-model residual (pure error) is shared across all three alignments;
    only the target's own effect is added back, exactly once:

        aligned_for_target = (Y - subject_s - cell_mean_ij) + target_effect

    An earlier version of this function instead fit a *reduced* nuisance
    model (missing the target term entirely) and added the target's group
    means back on top — but a reduced model's residual already contains the
    full target effect by construction (it was never subtracted), so adding
    it again double-counted the signal and manufactured spurious
    significance. Caught by cross-checking against a case with a known
    non-significant parametric result (ERG Photopic) that this bug flipped
    to significant across the board — reproducing that check is the
    regression test to keep in mind if this function is touched again.
    """
    w = long[["Value", cond_col, "Level", subject_col]].dropna().copy()
    w.columns = ["_y", "_cond", "_level", "_subject"]

    grand_mean = w["_y"].mean()
    cell_mean = w.groupby(["_cond", "_level"])["_y"].transform("mean")
    cond_mean = w.groupby("_cond")["_y"].transform("mean")
    level_mean = w.groupby("_level")["_y"].transform("mean")
    subject_mean = w.groupby("_subject")["_y"].transform("mean")

    cond_effect = cond_mean - grand_mean
    level_effect = level_mean - grand_mean
    interaction_effect = cell_mean - cond_mean - level_mean + grand_mean
    subject_offset = subject_mean - grand_mean

    full_fitted = grand_mean + subject_offset + cond_effect + level_effect + interaction_effect
    full_residual = w["_y"].values - full_fitted.values

    target_effect = {"cond": cond_effect, "level": level_effect,
                      "interaction": interaction_effect}[target].values

    aligned = full_residual + target_effect
    ranked = stats.rankdata(aligned)
    return pd.Series(ranked, index=w.index)


def _art_omnibus_row(long: pd.DataFrame, cond_col: str, subject_col: str,
                      target: str, source_label: str) -> dict:
    """Aligned-rank F/p for one term, via a fresh alignment + rm_anova refit."""
    work = long[["Value", cond_col, "Level", subject_col]].dropna().copy()
    work["_ranked"] = _align_and_rank(long, cond_col, subject_col, target).values

    aov = pg.rm_anova(data=work, dv="_ranked", within=[cond_col, "Level"],
                       subject=subject_col, detailed=True)
    row = aov[aov["Source"] == source_label]
    if row.empty:
        return {"F": np.nan, "ddof1": np.nan, "ddof2": np.nan,
                "p_unc": np.nan, "p_GG_corr": np.nan, "eps": np.nan, "ng2": np.nan}
    r = row.iloc[0]
    return {"F": float(r["F"]), "ddof1": r["ddof1"], "ddof2": r["ddof2"],
            "p_unc": float(r["p-unc"]), "p_GG_corr": float(r["p-GG-corr"]),
            "eps": float(r["eps"]), "ng2": float(r["ng2"])}


# ============================================================================= #
# SECTION 3. THE 2-WAY REPEATED-MEASURES CONTRAST
# ============================================================================= #


def compare_repeated_levels(
    df: pd.DataFrame,
    id_col: str,
    cond_col: str,
    level_cols: list,
    family_name: str,
    folder: str,
    level_labels: dict | None = None,
    min_n: int = 3,
    max_n_shapiro: int = 5000,
    verbose: bool = True,
) -> dict | None:
    """Run the 2-way Condition x Level contrast for one family.

    Dispatches to a parametric ``pingouin.rm_anova`` (Greenhouse-Geisser
    corrected) or, when any (Condition, Level) cell fails normality or
    homoscedasticity, or is smaller than ``MIN_CELL_N``, to the
    within-subject Aligned Rank Transform described in the module
    docstring. The whole family switches engine together — never one row
    parametric and another ART in the same table.

    Parameters
    ----------
    df : DataFrame
        Wide-format data: one row per (subject, condition), one column per
        level (e.g. one column per spatial frequency).
    id_col : str
        Subject identifier column.
    cond_col : str
        The two-level within-subject Condition column (e.g. Pre/Post).
    level_cols : list
        The wide columns making up this family's Level factor.
    family_name : str
        Used in printed messages and output filenames.
    folder : str
        Output directory (one dataset's ``(2) Statistical inference`` folder).
    level_labels : dict, optional
        ``{column name: display label}``. Defaults to the column names
        themselves.
    min_n : int
        Minimum complete subjects to run the design at all.
    max_n_shapiro : int
        Passed through to the per-level post-hoc normality pretest.
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        ``{"omnibus": DataFrame, "posthoc": DataFrame or None}``, or None if
        the design could not be run.
    """
    os.makedirs(folder, exist_ok=True)
    level_labels = level_labels or {c: c for c in level_cols}

    groups = df[cond_col].dropna().unique()
    if len(groups) != 2:
        if verbose:
            print(f"'{family_name}': {len(groups)} levels in {cond_col!r}, "
                  f"expected 2. Skipping.")
        return None
    group1, group2 = groups

    long = df.melt(id_vars=[id_col, cond_col], value_vars=level_cols,
                    var_name="_col", value_name="Value")
    long["Level"] = long["_col"].map(level_labels)
    long = long.dropna(subset=["Value"])

    # pingouin's rm_anova needs a fully balanced design: every subject
    # present at every (Condition, Level) cell. Subjects missing any cell
    # are dropped rather than imputed here — this omnibus test is meant to
    # summarise the complete-case pattern, not paper over missingness.
    cell_counts = long.groupby(id_col).size()
    n_cells = cell_counts.max()
    complete_ids = cell_counts[cell_counts == n_cells].index
    n_dropped = long[id_col].nunique() - len(complete_ids)
    if n_dropped and verbose:
        print(f"'{family_name}': dropping {n_dropped} subject(s) with "
              f"incomplete (Condition x Level) cells.")
    long = long[long[id_col].isin(complete_ids)]

    if long[id_col].nunique() < min_n:
        if verbose:
            print(f"'{family_name}': fewer than {min_n} complete subjects. "
                  f"Skipping.")
        return None

    diag = _cell_diagnostics(long, cond_col)
    use_art = not diag["parametric_ok"]

    # Shared assumption-check columns, identical on every row of this
    # family's omnibus table (they gated the engine choice for the whole
    # family, not one term at a time) — positioned before F/p, same
    # before-the-test-it-gated convention as inference.py's per-variable
    # table. "normality p" is the worst (minimum) cell in the family; a
    # starred significance here means that assumption was rejected, not
    # that a real effect was found.
    norm_p = diag["normality_p"]
    lev_p = diag["levene_p"]
    assumption_cols = {
        "normality test": "Shapiro-Wilk",
        "normality p (worst cell)": round(norm_p, 4) if np.isfinite(norm_p) else np.nan,
        "normality significance": _stars(norm_p) if np.isfinite(norm_p) else "n/a",
        "homoscedasticity test": "Levene",
        "homoscedasticity p": round(lev_p, 4) if np.isfinite(lev_p) else np.nan,
        "homoscedasticity significance": _stars(lev_p) if np.isfinite(lev_p) else "n/a",
    }

    if use_art:
        if verbose:
            print(f"'{family_name}': assumptions not met (all-cells-normal="
                  f"{diag['all_normal']}, homoscedastic={diag['homoscedastic']}, "
                  f"min cell n={diag['min_n']}) -> Aligned Rank Transform.")
        term_specs = [("cond", cond_col), ("level", "Level"),
                      ("interaction", f"{cond_col} * Level")]
        omnibus_rows = []
        for target, source_label in term_specs:
            stats_row = _art_omnibus_row(long, cond_col, id_col, target, source_label)
            p_gg = stats_row["p_GG_corr"]
            omnibus_rows.append({
                "Family": family_name, "Source": source_label, "engine": "ART",
                **assumption_cols,
                "F": round(stats_row["F"], 4) if np.isfinite(stats_row["F"]) else np.nan,
                "ddof1": stats_row["ddof1"], "ddof2": stats_row["ddof2"],
                "p_unc": round(stats_row["p_unc"], 6) if np.isfinite(stats_row["p_unc"]) else np.nan,
                "p_GG_corr": round(p_gg, 6) if np.isfinite(p_gg) else np.nan,
                "significance": _stars(p_gg) if np.isfinite(p_gg) else "ns",
                "eps": round(stats_row["eps"], 4) if np.isfinite(stats_row["eps"]) else np.nan,
                "generalized_eta_sq": round(stats_row["ng2"], 4) if np.isfinite(stats_row["ng2"]) else np.nan,
                "significant": bool(np.isfinite(p_gg) and p_gg < ALPHA),
            })
        omnibus_df = pd.DataFrame(omnibus_rows)
        interaction_row = omnibus_df[omnibus_df["Source"] == f"{cond_col} * Level"].iloc[0]
        interaction_sig = bool(interaction_row["significant"])
    else:
        aov = pg.rm_anova(data=long, dv="Value", within=[cond_col, "Level"],
                           subject=id_col, detailed=True)
        omnibus_rows = []
        for _, r in aov.iterrows():
            p_gg = float(r["p-GG-corr"])
            omnibus_rows.append({
                "Family": family_name, "Source": r["Source"], "engine": "parametric",
                **assumption_cols,
                "F": round(float(r["F"]), 4),
                "ddof1": r["ddof1"], "ddof2": r["ddof2"],
                "p_unc": round(float(r["p-unc"]), 6),
                "p_GG_corr": round(p_gg, 6),
                "significance": _stars(p_gg),
                "eps": round(float(r["eps"]), 4),
                "generalized_eta_sq": round(float(r["ng2"]), 4),
                "significant": bool(p_gg < ALPHA),
            })
        omnibus_df = pd.DataFrame(omnibus_rows)
        interaction_source = f"{cond_col} * Level"
        interaction_rows = omnibus_df[omnibus_df["Source"] == interaction_source]
        interaction_sig = (not interaction_rows.empty
                            and bool(interaction_rows.iloc[0]["significant"]))

    # ------------------------------------------------------------------- #
    # Post-hoc: only when the interaction cleared its own gate. Uses the
    # same paired dispatch as compare_two_groups regardless of which
    # omnibus engine ran — the per-level pretest decides parametric vs
    # non-parametric independently at each level, same as everywhere else
    # in this repository.
    # ------------------------------------------------------------------- #
    posthoc_df = None
    if interaction_sig:
        if verbose:
            print(f"'{family_name}': interaction significant -> running "
                  f"per-level post-hoc.")
        level_names = list(dict.fromkeys(level_labels.values()))
        col_by_label = {lab: col for col, lab in level_labels.items()}

        raw_p, rows = [], []
        for lvl in level_names:
            col = col_by_label[lvl]
            data1, data2, problem = _extract_samples(
                df, cond_col, col, group1, group2, True, id_col, family_name)

            if problem is None and (len(data1) < min_n or len(data2) < min_n):
                problem = "Insufficient data"

            d1 = describe_group(data1)
            d2 = describe_group(data2)
            row = {
                "Level": lvl,
                "group 1": group1, "group 2": group2,
                "n (group 1)": d1["n"], "n (group 2)": d2["n"],
                "mean (group 1)": round(d1["mean"], 3) if np.isfinite(d1["mean"]) else np.nan,
                "SD (group 1)": round(d1["sd"], 3) if np.isfinite(d1["sd"]) else np.nan,
                "mean (group 2)": round(d2["mean"], 3) if np.isfinite(d2["mean"]) else np.nan,
                "SD (group 2)": round(d2["sd"], 3) if np.isfinite(d2["sd"]) else np.nan,
            }

            if problem is not None:
                row.update({"normality test": "n/a",
                            "normality p (group 1)": np.nan, "normality significance (group 1)": "n/a",
                            "normality p (group 2)": np.nan, "normality significance (group 2)": "n/a",
                            "test": problem, "p_value": np.nan,
                            "effect size metric": "n/a", "effect size value": np.nan,
                            "Hedges g": np.nan})
                raw_p.append(np.nan)
                rows.append(row)
                continue

            normality_test, p_norm1, p_norm2 = _normality(data1, data2, max_n_shapiro)
            both_normal = p_norm1 > ALPHA and p_norm2 > ALPHA
            test_name, _stat, p_value, effect_value, effect_metric = _run_test(
                data1, data2, True, both_normal, False)

            # Assumption columns before the test they gated, same
            # before-the-test-it-gated convention as inference.py.
            row.update({
                "normality test": normality_test,
                "normality p (group 1)": round(float(p_norm1), 4) if np.isfinite(p_norm1) else np.nan,
                "normality significance (group 1)": _stars(p_norm1) if np.isfinite(p_norm1) else "n/a",
                "normality p (group 2)": round(float(p_norm2), 4) if np.isfinite(p_norm2) else np.nan,
                "normality significance (group 2)": _stars(p_norm2) if np.isfinite(p_norm2) else "n/a",
                "test": test_name,
                "p_value": round(float(p_value), 6),
                "effect size metric": effect_metric,
                "effect size value": (round(float(effect_value), 3)
                                       if effect_value is not None
                                       and not np.isnan(effect_value) else np.nan),
                "Hedges g": round(hedges_g(data1.values, data2.values, True), 3),
            })
            raw_p.append(p_value)
            rows.append(row)

        p_for_holm = [p if (p is not None and np.isfinite(p)) else 1.0 for p in raw_p]
        p_holm = holm_adjust(p_for_holm)
        for row, p_adj, p_raw in zip(rows, p_holm, raw_p):
            has_p = p_raw is not None and np.isfinite(p_raw)
            row["p_value_holm"] = round(float(p_adj), 6) if has_p else np.nan
            row["significant_holm"] = bool(has_p and p_adj < ALPHA)
            row["significance"] = _stars(p_adj) if has_p else "ns"

        posthoc_df = pd.DataFrame(rows)
    elif verbose:
        print(f"'{family_name}': interaction not significant -> reporting "
              f"the omnibus only (no per-level post-hoc; the effect, if "
              f"any, is treated as level-independent).")

    omnibus_path = os.path.join(folder, f"Repeated-measures omnibus - {family_name}.csv")
    save_csv(omnibus_df, omnibus_path)
    if verbose:
        print(f"✅ CSV saved: {omnibus_path}")

    if posthoc_df is not None:
        posthoc_path = os.path.join(folder, f"Repeated-measures posthoc - {family_name}.csv")
        save_csv(posthoc_df, posthoc_path)
        if verbose:
            print(f"✅ CSV saved: {posthoc_path}")

    return {"omnibus": omnibus_df, "posthoc": posthoc_df}


# ============================================================================= #
# SECTION 4. DATASET-LEVEL DRIVER
# ============================================================================= #


def run_repeated_families(
    df: pd.DataFrame,
    cfg: dict,
    folder: str,
    verbose: bool = True,
) -> dict:
    """Run every family declared in a dataset's ``repeated_families`` config.

    Parameters
    ----------
    df : DataFrame
        The dataset's preprocessed (working) DataFrame.
    cfg : dict
        The dataset's registry entry. Reads ``repeated_families``,
        ``group_column`` (whose first entry is the Condition factor) and
        ``subject_column``.
    folder : str
        The dataset's ``(2) Statistical inference`` output folder.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{family_name: {"omnibus": DataFrame, "posthoc": DataFrame or None}}``
        for the families that ran.
    """
    families = cfg.get("repeated_families") or {}
    if not families:
        return {}

    cond_col = cfg["group_column"][0]
    id_col = cfg["subject_column"]
    results = {}

    for family_name, spec in families.items():
        prefix = spec.get("column_prefix")
        explicit_cols = spec.get("columns")
        if explicit_cols:
            level_cols = [c for c in explicit_cols if c in df.columns]
        elif prefix:
            level_cols = [c for c in df.columns
                          if c.startswith(prefix) and not c.endswith("_outlier")]
        else:
            if verbose:
                print(f"'{family_name}': no 'columns' or 'column_prefix' "
                      f"in repeated_families config. Skipping.")
            continue

        if len(level_cols) < 2:
            if verbose:
                print(f"'{family_name}': fewer than 2 level columns found. "
                      f"Skipping.")
            continue

        strips = spec.get("label_strip")
        if strips:
            strips = [strips] if isinstance(strips, str) else strips
            level_labels = {}
            for c in level_cols:
                label = c
                for s in strips:
                    label = label.replace(s, "")
                level_labels[c] = label.strip()
        else:
            level_labels = None

        res = compare_repeated_levels(
            df, id_col=id_col, cond_col=cond_col, level_cols=level_cols,
            family_name=family_name, folder=folder,
            level_labels=level_labels, verbose=verbose)
        if res is not None:
            results[family_name] = res

    return results


def repeated_family_columns(cfg: dict, df_columns) -> set:
    """Every wide column claimed by any of a dataset's repeated families.

    Used by the stage-2 driver to exclude these columns from the plain
    per-variable ``compare_two_groups`` + BH-FDR path, so a level is never
    tested by both engines at once.
    """
    families = cfg.get("repeated_families") or {}
    claimed = set()
    for spec in families.values():
        explicit_cols = spec.get("columns")
        prefix = spec.get("column_prefix")
        if explicit_cols:
            claimed.update(c for c in explicit_cols if c in df_columns)
        elif prefix:
            claimed.update(c for c in df_columns
                           if c.startswith(prefix) and not c.endswith("_outlier"))
    return claimed
