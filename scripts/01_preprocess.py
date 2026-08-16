#!/usr/bin/env python3
"""Stage 1 - preprocess every dataset and write the diagnostic reports.

Runs the pipeline in order:

    load -> handle missing values -> average trials -> transform
         -> detect and replace outliers

Normality and homoscedasticity are no longer computed here as standalone
reports: stage 2 (``inference.py`` / ``rm_anova.py``) recomputes them per
test and writes them as columns in the results table itself.

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

from expda import config, preprocessing  # noqa: E402


def preprocess_dataset(name: str, cfg: dict, registry: dict,
                       clean: bool = False, verbose: bool = True) -> None:
    """Run the full stage-1 pipeline for one dataset."""
    print(f"\n{'=' * 70}\nProcessing dataset: {name}\n{'=' * 70}")

    layout = registry["layout"]

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

    # Stage 1's output and stage 2's input in one place — see
    # config.preprocessed_csv_path's docstring for why this used to be two
    # separate copies and isn't anymore.
    out_csv = config.preprocessed_csv_path(name, layout)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_no_outliers.to_csv(out_csv, index=False)
    print(f"Preprocessed data -> {out_csv}")


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
