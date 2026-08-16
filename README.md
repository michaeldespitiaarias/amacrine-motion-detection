# Paper 1 — analysis code

Analysis code accompanying **"Dopaminergic amacrine cells modulate retinal
movement detection."** It cleans and screens a dataset, reports the
assumptions behind the tests that follow, and runs two-group contrasts with
effect sizes, confidence intervals, and Benjamini-Hochberg FDR correction
across the variables tested.

Every output is a plain CSV — raw numbers, no styling, no narrative
document, no figures. This is a statistical-analysis repository, not a
reporting tool.

This repository exists to reproduce the exact analysis behind that specific
paper — it is not a general-purpose statistics tool. When the manuscript is
final, this code is frozen at a tagged release (`v1.0.0`) and archived on
Zenodo, so the DOI cited in the paper always resolves to the exact code that
produced its results, independent of any later work.

## Layout

```
src/expda/
    config.py                 paths and the dataset registry
    preprocessing.py          stage 1: cleaning, aggregation, screening
    effect_sizes.py           effect-size estimators and their intervals
    multicomparison.py        Benjamini-Hochberg FDR correction
    inference.py              stage 2: two-group contrasts
    reporting.py              plain-CSV writer shared by every stage
scripts/
    01_preprocess.py          run stage 1 over every dataset
    02_two_group_contrasts.py run stage 2 and write the results tables
registry.example.json         template for the dataset registry
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Configuration

Two environment variables:

```bash
export EXPDA_DATA=/path/to/data-root      # required
export EXPDA_REGISTRY=/path/to/registry.json   # optional
```

`EXPDA_REGISTRY` defaults to `registry.json` inside the data root. Copy
`registry.example.json` and adapt it:

```json
{
  "layout": {
    "input_dir": "input",
    "working_dir": "working",
    "results_dir": "results",
    "table_prefix": "Contrast table"
  },
  "datasets": {
    "DatasetA": {
      "group_column": ["Condition"],
      "trial_column": "Trial",
      "subject_column": "Subject",
      "comparison_by": "Condition",
      "paired": true,
      "categorical_mappings": {
        "Condition": {"Baseline": 0, "Follow-up": 1}
      }
    }
  }
}
```

| Key | Meaning |
|---|---|
| `group_column` | strata for imputation and outlier screening |
| `trial_column` | column collapsed so that the subject is the unit of analysis |
| `subject_column` | subject identifier, used to align paired samples |
| `comparison_by` | the factor whose two levels are contrasted |
| `paired` | whether the contrast uses a paired test |
| `levels` | optional pair selecting which two levels to contrast and their order |
| `categorical_mappings` | label encoding used by the correlation report |
| `transform` | optional variable transformation, default `none` |
| `mv_drop_pct` | missing-value skip gate, default `30` (%) — see Pipeline below |
| `outlier_pct_skip` | outlier replacement-skip gate, default `15` (%) |
| `outlier_replace_max_n` | absolute cap on points replaced per (group, column), default `2` |
| `outlier_replace_max_n_group_ceiling` | group size above which the cap stops applying, default `20` |

Expected tree under the data root:

```
<EXPDA_DATA>/
    <input_dir>/    <Dataset>.csv                input to stage 1
    <working_dir>/  <Dataset>_no_outliers.csv    input to stage 2
    <results_dir>/<Dataset>/<report subfolders>
```

The dataset name is the join key across every stage: the same string names the
input CSV, the working CSV, the results folder and the table inside it.

## Usage

```bash
# Stage 1 — clean, screen and write the diagnostic reports
python scripts/01_preprocess.py
python scripts/01_preprocess.py --datasets DatasetA

# Stage 2 — two-group contrasts, one CSV table per dataset
python scripts/02_two_group_contrasts.py
python scripts/02_two_group_contrasts.py --output /tmp/tables --no-intervals
```

## Pipeline

```
<input_dir>/<Dataset>.csv
    ↓  handle_nulls                 missing values → group median or mode,
    │                                skipped above mv_drop_pct % missing
    ↓  average_trials               collapse trials → one row per subject
    ↓  transform_variables          shape-stabilizing methods only
    │                                (log/sqrt/boxcox/yeojohnson)
    ↓  detect_and_replace_outliers  1.5 × IQR within group → group median,
    │                                skipped above outlier_pct_skip % or
    │                                outlier_replace_max_n flagged (small
    │                                groups only) — see Safety gates below
    ↓  transform_variables          scale-standardizing methods only
                                     (zscore/minmax)
<working_dir>/<Dataset>_no_outliers.csv
    ↓  normality · homoscedasticity · correlation · multicollinearity reports
    ↓  compare_two_groups
<results_dir>/<Dataset>/<contrasts>/…csv
```

`average_trials` makes the subject, rather than the trial, the unit of analysis.
Designs measured at more than two levels declare a `levels` pair in the registry
to select which two conditions are contrasted.

The two `transform_variables` passes are not the same call twice: which one
runs (if either) depends on which family the configured `transform` method
falls into. Shape-stabilizing transforms (log, sqrt, Box-Cox, Yeo-Johnson)
run *before* outlier detection, so the detector sees the transformed scale
rather than mistaking a skewed raw tail for contamination. Scale-
standardizing transforms (z-score, min-max) run *after* outlier replacement,
since they are computed from the column's own mean/std or min/max and a
genuine outlier would otherwise distort those statistics for every other
value in the column before it gets replaced.

### Safety gates

Two independent gates keep automatic cleaning from over-reaching on small
research samples (typically 3–15 subjects per group):

- **Missing values.** A column whose missing share exceeds `mv_drop_pct`
  (default 30 %) keeps its `NaN`s rather than having more than that share
  of its values invented by a single-imputation method. Nothing is
  dropped from the dataset — only the fill is skipped for that column; the
  downstream tests already handle `NaN` honestly by reporting the real n.
- **Outliers.** A (group, column) cell where more than `outlier_pct_skip`
  (default 15 %) of values are flagged, or where the absolute count exceeds
  `outlier_replace_max_n` (default 2, active only while the group has at
  most `outlier_replace_max_n_group_ceiling` = 20 members), is left
  unreplaced rather than treated as contamination — that density of
  flagged points is more likely a genuinely skewed or multimodal subgroup.
  Those cells get a non-destructive `{column}_outlier` diagnostic column
  instead of being overwritten.

## Test selection

Chosen from the data rather than fixed in advance:

| Design | Normality | Variance | Test | Effect size |
|---|---|---|---|---|
| paired | pass | — | paired t-test | Cohen's dz |
| paired | fail | — | Wilcoxon signed-rank | r |
| independent | pass | equal | Student's t-test | Cohen's d |
| independent | pass | unequal | Welch's t-test | Cohen's d |
| independent | fail | — | Mann-Whitney U | r |

Normality is Shapiro-Wilk per group at α = 0.05, with D'Agostino K² above
`max_n_shapiro`. Variance homogeneity is Bartlett when both groups pass
normality. Benjamini-Hochberg FDR is applied across every variable tested in
one dataset's table (one dataset = one family), reported as `p_value_fdr_bh`
alongside — never in place of — the raw `p_value`.

## Output

One CSV table per dataset, carrying per variable: the test, its p-value and
significance class, the effect size, n, mean, SD and median per group, the
direction of the effect, the pretest p-values, Hedges' g, a bootstrap 95 %
confidence interval, a signed rank-biserial, and the BH-FDR-adjusted q-value
with its own significance flag.

The bootstrap uses a fixed seed (`effect_sizes.BOOTSTRAP_SEED`), so the
intervals are reproducible across runs and machines.

## Citation

See [`CITATION.cff`](CITATION.cff). Cite the Zenodo DOI of the specific
release used for the paper's results, not this repository's latest state.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) —
see [`LICENSE`](LICENSE). Free for any noncommercial purpose (academic
research, teaching, peer review, personal study). Commercial use requires
a separate license from the author — contact esspitia@gmail.com.
