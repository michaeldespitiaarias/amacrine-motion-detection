"""Effect-size estimators for two-group contrasts.

Section 1 holds the primary estimators, one per test in the selection tree.
Section 2 adds companion statistics reported as extra columns: a small-sample
bias correction, bootstrap confidence intervals, a signed rank-based estimate
and per-group descriptives.

Sign conventions
----------------
``_cohen_d`` is signed: positive means group 1 scores above group 2, where
group 1 is the first level encountered in the data, in order of appearance
rather than alphabetically. The ``r`` estimators are unsigned;
:func:`rank_biserial` supplies a signed rank-based alternative.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

# Seed for the bootstrap, fixed so that confidence intervals are reproducible
# across runs and machines.
BOOTSTRAP_SEED = 20260101
BOOTSTRAP_RESAMPLES = 10_000


# ============================================================================= #
# SECTION 1. PRIMARY ESTIMATORS
# ============================================================================= #


def _stars(p):
    """
    Returns significance stars based on the p-value.

    Parameters:
        p (float): p-value from a statistical test.

    Returns:
        str: Significance stars ('***', '**', '*', 'ns').
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# paired=False -> pooled-SD Cohen's d, for independent samples.
# paired=True  -> mean(diff) / SD(diff), i.e. Cohen's dz, for paired samples.
# The two are different quantities: dz is standardised by the SD of the
# difference scores and therefore grows with the pre-post correlation.
def _cohen_d(x, y, paired=False):
    """
    Calculate Cohen's d effect size between two samples.

    Parameters:
        x, y (array-like): Samples.
        paired (bool): Whether the samples are paired (dependent).

    Returns:
        float: Cohen's d effect size.
    """
    x, y = np.array(x), np.array(y)

    if paired:
        diff = x - y
        return np.mean(diff) / np.std(diff, ddof=1)
    else:
        n1, n2 = len(x), len(y)
        s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
        s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
        return (np.mean(x) - np.mean(y)) / s_pooled


# r derived from the normal approximation of the Mann-Whitney U statistic,
# without tie or continuity correction. Returned unsigned.
def _effect_r_from_u(U, n1, n2):
    """
    Compute effect size r from Mann-Whitney U statistic.

    Parameters:
        U (float): Mann-Whitney U value.
        n1, n2 (int): Sample sizes of the two groups.

    Returns:
        float: Effect size r.
    """
    if n1 <= 0 or n2 <= 0 or np.isnan(U):
        return np.nan
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U - mean_U) / std_U
    return abs(z) / np.sqrt(n1 + n2)


# r for the Wilcoxon signed-rank test, obtained from the p-value by inverting
# the two-tailed normal quantile: z = |Phi^-1(p/2)|, r = z / sqrt(n_pairs).
# Returned unsigned. `rank_biserial()` gives the signed, statistic-based
# alternative and is reported alongside.
def _wilcoxon_r(p_value, n):
    """
    Compute the effect size r for Wilcoxon signed-rank test.

    Parameters:
        p_value (float): p-value from Wilcoxon test.
        n (int): Number of non-null pairs (sample size).

    Returns:
        float: Effect size r, rounded to 3 decimals.
    """
    if pd.isna(p_value) or p_value <= 0 or n <= 0:
        return np.nan
    try:
        z = abs(stats.norm.ppf(p_value / 2))  # two-tailed
        r = z / np.sqrt(n)
        return round(r, 3)
    except Exception:
        return np.nan


# ============================================================================= #
# SECTION 2. COMPANION STATISTICS
#
# Reported as additional columns. They do not feed into test selection,
# p-values or the estimators above.
# ============================================================================= #


def hedges_g(x, y, paired: bool = False) -> float:
    """Small-sample bias-corrected standardised mean difference.

    Multiplies Cohen's d by J = 1 - 3 / (4 * df - 1). The correction matters most at small n:
    with eight observations per group it is about 5 per cent.

    Parameters
    ----------
    x, y : array-like
        The two samples, in the same order convention as :func:`_cohen_d`.
    paired : bool
        Correct dz rather than the pooled-SD d.

    Returns
    -------
    float
        Bias-corrected effect size, or NaN if it cannot be computed.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = _cohen_d(x, y, paired=paired)
    if not np.isfinite(d):
        return np.nan

    df = (len(x) - 1) if paired else (len(x) + len(y) - 2)
    if df <= 1:
        return np.nan
    return float(d * (1.0 - 3.0 / (4.0 * df - 1.0)))


def bootstrap_ci(
    x,
    y,
    paired: bool = False,
    confidence: float = 0.95,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the standardised effect.

    Paired samples are resampled by subject, so the pairing is preserved in every
    replicate; independent samples are resampled within each group.

    Parameters
    ----------
    x, y : array-like
        The two samples.
    paired : bool
        Resample pairs (True) or each group independently (False).
    confidence : float
        Coverage of the interval, default 0.95.
    n_resamples : int
        Bootstrap replicates.
    seed : int
        Fixed for reproducibility; see :data:`BOOTSTRAP_SEED`.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds, or (NaN, NaN) when the sample is too small.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)

    with warnings.catch_warnings():
        # A replicate in which every value is identical has zero SD and yields a
        # non-finite estimate; those replicates are discarded below.
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n_resamples):
            if paired:
                idx = rng.integers(0, len(x), len(x))
                estimates[i] = _cohen_d(x[idx], y[idx], paired=True)
            else:
                xi = rng.integers(0, len(x), len(x))
                yi = rng.integers(0, len(y), len(y))
                estimates[i] = _cohen_d(x[xi], y[yi], paired=False)

    estimates = estimates[np.isfinite(estimates)]
    if estimates.size < n_resamples // 2:
        return (np.nan, np.nan)

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return (float(lower), float(upper))


def rank_biserial(x, y, paired: bool = False) -> float:
    """Signed rank-based effect size.

    For paired samples this is the matched-pairs rank-biserial correlation
    (Kerby 2014): the share of signed-rank mass favouring an increase minus the
    share favouring a decrease. For independent samples it is the Mann-Whitney
    rank-biserial, ``2U / (n1 * n2) - 1``.

    Both range from -1 to +1 and are signed, so positive means group 1 scores
    above group 2.

    Returns
    -------
    float
        Effect size in [-1, 1], or NaN when undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if paired:
        diff = x - y
        diff = diff[diff != 0]  # zero differences carry no rank information
        if diff.size == 0:
            return np.nan
        ranks = stats.rankdata(np.abs(diff))
        total = ranks.sum()
        if total == 0:
            return np.nan
        return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)

    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    u_stat = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2.0 * u_stat / (n1 * n2) - 1.0)


def describe_group(values) -> dict[str, float]:
    """Per-group summary statistics reported next to every contrast."""
    values = pd.Series(values, dtype="float64").dropna()
    if values.empty:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        "median": float(values.median()),
    }
