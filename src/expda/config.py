"""Filesystem layout and dataset registry.

Nothing about a particular study is hard-coded. Both the location of the data
and the description of each analysis unit come from outside the package:

* ``EXPDA_DATA`` — root directory holding the data and the output tree.
* ``EXPDA_REGISTRY`` — path to a JSON file describing the datasets.
  Defaults to ``registry.json`` beside the data root.

See ``registry.example.json`` for the expected shape.

Expected tree under the data root::

    <DATA_ROOT>/
        <input_dir>/         <Dataset>.csv               input to stage 1
        <working_dir>/       <Dataset>_no_outliers.csv   input to stage 2
        <results_dir>/<Dataset>/<report subfolders>

The dataset name is the join key across every stage: the same string names the
input CSV, the working CSV, the results folder and the table inside it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("EXPDA_DATA", PACKAGE_ROOT / "data")).expanduser()

# Directory names, overridable through the registry file.
DEFAULT_LAYOUT = {
    "input_dir": "input",
    "working_dir": "working",
    "results_dir": "results",
    "report_folders": {
        "intermediate": "00_working",
        "normality": "01_normality",
        "homoscedasticity": "02_homoscedasticity",
        "correlation": "03_correlation",
        "multicollinearity": "04_multicollinearity",
        "hypothesis_contrast": "05_contrasts",
    },
    "table_prefix": "Contrast table",
}


def registry_path() -> Path:
    """Location of the JSON registry describing the datasets."""
    return Path(
        os.environ.get("EXPDA_REGISTRY", DATA_ROOT / "registry.json")
    ).expanduser()


def load_registry() -> dict:
    """Read the registry, merged over the default layout.

    Returns
    -------
    dict
        ``{"layout": {...}, "datasets": {name: {...}}}``.

    Raises
    ------
    FileNotFoundError
        If no registry file is present. There is no built-in dataset list.
    """
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"No dataset registry at {path}. Set EXPDA_REGISTRY, or copy "
            f"registry.example.json and adapt it."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    layout = {**DEFAULT_LAYOUT, **raw.get("layout", {})}
    layout["report_folders"] = {
        **DEFAULT_LAYOUT["report_folders"],
        **raw.get("layout", {}).get("report_folders", {}),
    }
    return {"layout": layout, "datasets": raw.get("datasets", {})}


def input_path(name: str, layout: dict) -> Path:
    """Path to the raw CSV that stage 1 consumes."""
    return DATA_ROOT / layout["input_dir"] / f"{name}.csv"


def working_path(name: str, layout: dict) -> Path:
    """Path to the preprocessed CSV that stage 2 consumes."""
    return DATA_ROOT / layout["working_dir"] / f"{name}_no_outliers.csv"


def results_path(name: str, folder_key: str, layout: dict) -> Path:
    """Path to one report subfolder of one dataset."""
    return DATA_ROOT / layout["results_dir"] / name / layout["report_folders"][folder_key]


def select(registry: dict, names: list[str] | None) -> dict:
    """Return the requested datasets, or all of them when ``names`` is None."""
    datasets = registry["datasets"]
    if not names:
        return dict(datasets)
    missing = [n for n in names if n not in datasets]
    if missing:
        raise KeyError(f"Not in the registry: {missing}")
    return {n: datasets[n] for n in names}


def describe_layout(layout: dict) -> str:
    """Summary of where the data is being read from, for logs."""
    return (
        f"DATA_ROOT = {DATA_ROOT}  (exists: {DATA_ROOT.is_dir()})\n"
        f"registry  = {registry_path()}"
    )
