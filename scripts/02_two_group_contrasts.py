#!/usr/bin/env python3
"""Stage 2 - run the two-group contrasts.

Reads the preprocessed CSV of each dataset and writes one plain CSV table per
dataset into its contrasts folder.

Datasets, grouping factors and directory names come from the JSON registry;
see ``registry.example.json``.

Usage
-----
    python scripts/02_two_group_contrasts.py
    python scripts/02_two_group_contrasts.py --datasets DatasetA DatasetB
    python scripts/02_two_group_contrasts.py --output /tmp/tables --no-intervals
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expda import config, inference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset to analyse (default: all in the registry)")
    parser.add_argument("--output", type=Path, default=None,
                        help="write every table to this directory instead of "
                             "each dataset's contrasts folder")
    parser.add_argument("--no-intervals", action="store_true",
                        help="skip the bootstrap confidence intervals")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        registry = config.load_registry()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    layout = registry["layout"]
    print(config.describe_layout(layout))

    # Surface every warning raised while pairing or testing.
    warnings.simplefilter("always", RuntimeWarning)

    datasets = config.select(registry, args.datasets)
    results: dict = {}

    for name, cfg in datasets.items():
        cfg = {**cfg, "location": str(config.preprocessed_csv_path(name, layout))}
        folder = args.output or config.results_path(name, "hypothesis_contrast", layout)
        Path(folder).mkdir(parents=True, exist_ok=True)
        results.update(inference.run_two_group_pipeline(
            {name: cfg}, folder=str(folder),
            table_prefix=layout["table_prefix"],
            with_intervals=not args.no_intervals, verbose=not args.quiet))

    if not results:
        print("\nNo datasets were analysed.", file=sys.stderr)
        return 1

    total = sum(len(v) for v in results.values())
    print(f"\nAnalysed {len(results)} dataset(s), {total} variable contrasts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
