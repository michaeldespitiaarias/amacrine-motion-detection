"""Preprocessing and two-group statistical analysis for tabular experiments.

A small pipeline for repeated-measures tabular data: it cleans and screens a
dataset, reports the assumptions behind the tests that follow and runs
two-group contrasts with effect sizes and confidence intervals.

Nothing about a particular study is hard-coded. Datasets, grouping factors and
directory names are declared in an external JSON registry; see
``registry.example.json``.

Modules
-------
``config``        paths and the dataset registry
``preprocessing`` stage 1: cleaning, aggregation, screening, assumption reports
``effect_sizes``  effect-size estimators and their confidence intervals
``inference``     stage 2: two-group contrasts
``reporting``     plain-CSV writer shared by every stage
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "effect_sizes",
    "inference",
    "preprocessing",
    "reporting",
]
