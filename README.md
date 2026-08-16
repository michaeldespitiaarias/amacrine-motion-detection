# Dopaminergic amacrine cells modulate retinal movement detection — analysis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21969128.svg)](https://doi.org/10.5281/zenodo.21969128)

Analysis code accompanying this paper. It cleans and screens a dataset,
reports the assumptions behind the tests that follow, and runs two-group
contrasts with effect sizes, confidence intervals, and Benjamini-Hochberg
FDR correction across the variables tested. Variables that are levels of one within-subject
factor (e.g. contrast sensitivity at six spatial frequencies) are instead
routed to a 2-way repeated-measures ANOVA, so the correlation between
adjacent levels is used rather than discarded, and per-level comparisons only
run once there's evidence the effect actually depends on the level — see
Repeated-measures level families below.

Every output is a plain CSV — raw numbers, no styling, no narrative
document, no figures. This is a statistical-analysis repository, not a
reporting tool.

## Layout

```
src/expda/
    config.py                 paths and the dataset registry
    preprocessing.py          stage 1: cleaning, aggregation, screening
    effect_sizes.py           effect-size estimators and their intervals
    multicomparison.py        Benjamini-Hochberg FDR + Holm-Bonferroni
    inference.py              stage 2: two-group contrasts
    rm_anova.py               stage 2b: repeated-measures level families
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
    "results_dir": "results",
    "table_prefix": "Two-group comparisons"
  },
  "datasets": {
    "DatasetA": {
      "group_column": ["Condition"],
      "trial_column": "Trial",
      "subject_column": "Subject",
      "paired": true,
      "repeated_families": {
        "Contrast sensitivity": {
          "column_prefix": "Optomotor",
          "label_strip": ["Optomotor (", ")"]
        }
      }
    }
  }
}
```

| Key | Meaning |
|---|---|
| `group_column` | strata for imputation and outlier screening; its first entry also names the factor whose two levels are contrasted |
| `trial_column` | column collapsed so that the subject is the unit of analysis |
| `subject_column` | subject identifier, used to align paired samples |
| `paired` | whether the contrast uses a paired test |
| `levels` | optional pair selecting which two levels to contrast and their order |
| `mv_drop_pct` | missing-value skip gate, default `30` (%) — see Pipeline below |
| `outlier_pct_skip` | outlier replacement-skip gate, default `15` (%) |
| `outlier_replace_max_n` | absolute cap on points replaced per (group, column), default `2` |
| `outlier_replace_max_n_group_ceiling` | group size above which the cap stops applying, default `20` |
| `repeated_families` | named groups of columns that are levels of one within-subject factor — see below |

Expected tree under the data root:

```
<EXPDA_DATA>/
    <input_dir>/    <Dataset>.csv                          input to stage 1
    <results_dir>/<Dataset>/(1) Preprocessed/<Dataset>_no_outliers.csv
                                                 stage 1's output, stage 2's input
    <results_dir>/<Dataset>/(2) Statistical inference/…csv
```

The dataset name is the join key across every stage: the same string names the
input CSV, the results folder and the table inside it. Stage 2 reads the
preprocessed CSV straight from `(1) Preprocessed/`.

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
    ↓  detect_and_replace_outliers  1.5 × IQR within group → group median,
    │                                skipped above outlier_pct_skip % or
    │                                outlier_replace_max_n flagged (small
    │                                groups only) — see Safety gates below
<results_dir>/<Dataset>/(1) Preprocessed/<Dataset>_no_outliers.csv
    ↓  compare_two_groups         variables outside any repeated_families group
    ↓  rm_anova.run_repeated_families   one 2-way RM-ANOVA per declared family
<results_dir>/<Dataset>/(2) Statistical inference/…csv
```

`average_trials` makes the subject, rather than the trial, the unit of analysis.
Designs measured at more than two levels declare a `levels` pair in the registry
to select which two conditions are contrasted.

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

## Repeated-measures level families

A variable is a **level of one within-subject factor**, not an independent
measurement, when it is the same construct sampled several times in one test
session for the same subject — contrast sensitivity at six spatial
frequencies, ERG amplitude at nine light intensities. Testing each level
separately with `compare_two_groups` and correcting with BH-FDR treats them
as an unordered bag of variables: it throws away the correlation between
adjacent levels, and it still tests every level whether or not the deficit
actually depends on the level at all.

A dataset's `repeated_families` config (a `column_prefix` or explicit
`columns` list per named family) routes those columns to `rm_anova.py`
instead: a single 2-way repeated-measures ANOVA per family (Condition x
Level, both within-subject). Columns claimed by a family are excluded from
the plain per-variable path, so a level is never tested by both engines at
once.

**Engine, chosen once per family (never per term):**

```
all (Condition, Level) cells pass Shapiro-Wilk (alpha=0.05) and n >= 5,
and Levene across cells (center='median') passes
    -> parametric pingouin.rm_anova, Greenhouse-Geisser corrected
otherwise
    -> Aligned Rank Transform (ART, Wobbrock 2011), adapted for the
       within-subject case (see rm_anova.py's module docstring for the
       closed-form decomposition)
```

**Post-hoc is gated**: per-level pairwise comparisons (paired t-test or
Wilcoxon, by per-level normality, Holm-Bonferroni-corrected across levels)
run *if and only if* the Condition x Level interaction is itself significant
— evidence the effect's shape differs by level, before any level is tested
individually. A significant Condition main effect with a non-significant
interaction is reported as a uniform, level-independent shift and is not
decomposed further, which is exactly the unguarded multiple-comparison
problem this module exists to avoid.

Output: `Repeated-measures omnibus - {family}.csv` (always) and
`Repeated-measures posthoc - {family}.csv` (only when the interaction gate opens),
alongside the dataset's plain `Two-group comparisons - {dataset}.csv` for whatever
variables aren't part of a family.

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

One CSV table per dataset, carrying per variable: n, mean and SD per group,
the direction of the effect, the assumption checks that gated the test
choice (normality test name + p-value + significance per group,
homoscedasticity test name + p-value + significance — positioned *before*
the test they gated, so the reason a test was chosen reads before the
result it produced), then the test itself: its name, p-value, significance
class, effect size, Hedges' g, a bootstrap 95 % confidence interval, a
signed rank-biserial, and the BH-FDR-adjusted q-value with its own
significance stars. Variables belonging to a `repeated_families`
group instead land in that family's `Repeated-measures omnibus` / `Repeated-measures posthoc`
CSVs — see Repeated-measures level families above.

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
