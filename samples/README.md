# Samples

Synthetic demo data and its real, pre-baked pipeline output — proof the
code runs end-to-end without needing the actual (unpublished)
experimental dataset behind the paper.

- `input/` — two small synthetic datasets with a planted effect
  (`DSGC_PrePost_SCH23390`: paired pre/post drug design; `DSGC_WT_vs_Drd1KO`:
  independent-groups genotype comparison), plus planted missing values and
  outliers to exercise the preprocessing safety gates.
- `registry.json` — the dataset registry describing both.
- `working/`, `results/` — real output of running the pipeline against
  `input/` (not hand-edited).

## Reproduce

```bash
export EXPDA_DATA="$(pwd)/samples"
export EXPDA_REGISTRY="$(pwd)/samples/registry.json"
python scripts/01_preprocess.py --clean
python scripts/02_two_group_contrasts.py
```

Every output file is a plain CSV — no plots, no HTML — matching the
project's `results.csv` / `posthoc.csv` convention (see the root
[README](../README.md)).
