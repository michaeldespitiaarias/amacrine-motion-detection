#!/usr/bin/env python3
"""Stage 1 - preprocess every dataset and write the diagnostic reports.

Runs the pipeline in order:

    load -> handle missing values -> average trials -> transform
         -> detect and replace outliers
         -> normality -> homoscedasticity -> correlation -> multicollinearity

Every diagnostic report is a plain CSV (no plots, no HTML — see
``expda.preprocessing``'s module docstring).

Datasets, grouping factors and directory names come from the JSON registry;
see ``registry.example.json``.

Usage
-----
    python scripts/01_preprocess.py                      # every dataset
    python scripts/01_preprocess.py --datasets DatasetA  # a subset
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from expda import config, preprocessing  # noqa: E402


def preprocess_dataset(name: str, cfg: dict, registry: dict,
                       clean: bool = False, verbose: bool = True) -> None:
    """Run the full stage-1 pipeline for one dataset."""
    print(f"\n{'=' * 70}\nProcessing dataset: {name}\n{'=' * 70}")

    layout = registry["layout"]
    datasets_config = registry["datasets"]

    dataset_root = config.DATA_ROOT / layout["results_dir"] / name
    if clean and dataset_root.exists():
        shutil.rmtree(dataset_root)
    for key in layout["report_folders"]:
        config.results_path(name, key, layout).mkdir(parents=True, exist_ok=True)

    group_column = cfg["group_column"]
    subject_column = cfg["subject_column"]
    cfg = {**cfg, "location": str(config.input_path(name, layout))}

    # 1. Load -------------------------------------------------------------- #
    df, numeric_cols, categorical_cols = preprocessing.load_dataset(name, cfg)

    # 2. Missing values ---------------------------------------------------- #
    # mv_drop_pct: a column above this % missing is left unimputed rather
    # than filled from too few real values (see preprocessing.handle_nulls).
    mv_drop_pct = cfg.get("mv_drop_pct", 30.0)
    df = preprocessing.handle_nulls(df, variables=numeric_cols, method="median",
                                    group_column=group_column,
                                    mv_drop_pct=mv_drop_pct, verbose=verbose)
    df = preprocessing.handle_nulls(df, variables=categorical_cols, method="mode",
                                    group_column=group_column,
                                    mv_drop_pct=mv_drop_pct, verbose=verbose)

    # 3. Collapse trials so that the unit of analysis is the subject ------- #
    trial_col = cfg.get("trial_column", "")
    df, numeric_cols, categorical_cols = preprocessing.average_trials(
        df, trial_col, group_column, numeric_cols, subject_column)

    # 4a. Shape-stabilizing transform — BEFORE outliers, so the outlier
    # detector sees the transformed scale rather than confusing a skewed
    # raw tail for genuine contamination. See preprocessing.py's
    # PRE_OUTLIER_TRANSFORMS / POST_OUTLIER_TRANSFORMS split.
    transform_method = cfg.get("transform", "none")
    if transform_method in preprocessing.PRE_OUTLIER_TRANSFORMS:
        df = preprocessing.transform_variables(
            df, columns=numeric_cols, method=transform_method)

    # 4b. Outliers: flagged by the IQR rule, replaced by the group median,
    # subject to the replacement-skip gate (outlier_pct_skip /
    # outlier_replace_max_n). Identifier columns are excluded, so that
    # subject ids pass through the pipeline unchanged.
    outlier_vars = [c for c in numeric_cols if "id" not in c.lower()]

    modified, _flags = preprocessing.detect_and_replace_outliers(
        {name: df}, variables=outlier_vars,
        group_column={name: group_column},
        outlier_pct_skip=cfg.get("outlier_pct_skip", 15.0),
        outlier_replace_max_n=cfg.get("outlier_replace_max_n", 2),
        outlier_replace_max_n_group_ceiling=cfg.get(
            "outlier_replace_max_n_group_ceiling", 20),
        verbose=verbose)
    df_no_outliers = modified[name]

    # 4c. Scale-standardizing transform — AFTER outliers: zscore/minmax are
    # computed from the column's own mean/std or min/max, and a genuine
    # outlier would otherwise distort those for every other value before
    # it gets a chance to be replaced.
    if transform_method in preprocessing.POST_OUTLIER_TRANSFORMS:
        df_no_outliers = preprocessing.transform_variables(
            df_no_outliers, columns=numeric_cols, method=transform_method)

    # Written to two places: the per-dataset report folder (alongside this
    # dataset's other diagnostic CSV output, for a human browsing
    # results/<Dataset>/), and config.working_path() — the canonical
    # stage-1 -> stage-2 handoff location that 02_two_group_contrasts.py
    # actually reads from by default. These used to diverge (results-
    # folder copy only), which silently broke the two-stage pipeline
    # end-to-end unless --output was passed explicitly to stage 2.
    out_csv = (config.results_path(name, "intermediate", layout)
               / f"{name}_no_outliers.csv")
    df_no_outliers.to_csv(out_csv, index=False)
    print(f"Preprocessed data -> {out_csv}")

    working_csv = config.working_path(name, layout)
    working_csv.parent.mkdir(parents=True, exist_ok=True)
    df_no_outliers.to_csv(working_csv, index=False)
    print(f"Preprocessed data -> {working_csv} (stage-2 input)")

    # 6-7. Assumption reports ----------------------------------------------- #
    normality_results = preprocessing.normality_test(
        {name: df_no_outliers}, numeric_cols,
        group_column={name: group_column},
        folder=str(config.results_path(name, "normality", layout)), verbose=verbose)

    preprocessing.homoscedasticity_test(
        {name: df_no_outliers}, numeric_cols,
        folder=str(config.results_path(name, "homoscedasticity", layout)),
        normality_results=normality_results,
        group_column={name: group_column}, verbose=verbose)

    # 8. Correlation and label encoding -------------------------------------- #
    mappings = cfg.get("categorical_mappings", {})
    _label_mappings, encoded = preprocessing.correlation_and_encoding_auto(
        dataframes={name: df_no_outliers},
        variables=numeric_cols + list(mappings.keys()),
        folder=str(config.results_path(name, "correlation", layout)),
        datasets_config=datasets_config,
        normality_results=normality_results, verbose=verbose)

    # 9. Multicollinearity ---------------------------------------------------- #
    # `correlation_and_encoding_auto` skips (and omits from `encoded`) any
    # dataset left with fewer than 2 non-constant numeric variables after
    # its own identifier/constant-column filtering — VIF needs at least 2
    # variables to say anything. Common now that the subject identifier is
    # correctly excluded from numeric_cols (a dataset whose only measured
    # variable is, say, a single Activity_au column has nothing left to
    # test collinearity against).
    if name not in encoded:
        if verbose:
            print(f"⚠️ Skipped multicollinearity for '{name}': "
                  f"not enough non-constant numeric variables.")
    else:
        numeric_vars = encoded[name].select_dtypes(include=[np.number]).columns.tolist()
        _multicol, transformed = preprocessing.multicollinearity_analysis_auto(
            dataframes=encoded, variables=numeric_vars,
            folder=str(config.results_path(name, "multicollinearity", layout)),
            verbose=verbose)
        transformed[name].to_csv(
            config.results_path(name, "intermediate", layout)
            / f"{name}_multicollinearity.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset to process (default: all in the registry)")
    parser.add_argument("--clean", action="store_true",
                        help="delete each dataset's results folder first")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        registry = config.load_registry()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(config.describe_layout(registry["layout"]))
    datasets = config.select(registry, args.datasets)

    for name, cfg in datasets.items():
        preprocess_dataset(name, cfg, registry,
                           clean=args.clean, verbose=not args.quiet)

    print(f"\nPreprocessing complete for {len(datasets)} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
