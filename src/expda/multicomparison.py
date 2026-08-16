"""Multiple-comparison corrections shared across the inference modules.

Kept in its own module (rather than inline in ``inference.py``) so a
second correction can be added alongside this one without touching the
inference logic. This repository only needs BH-FDR: with exactly one
test per variable (a single two-group contrast), there is no
within-variable family of pairwise comparisons to additionally correct.
"""

from __future__ import annotations

import numpy as np


def bh_fdr(p_values) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values.

    Corrects across the p-values passed in as one "family" — the caller
    decides what that family is (here: every variable tested in one
    dataset's contrast table). NaN entries pass through as NaN and do
    not participate in the correction (neither counted toward *m* nor
    assigned a rank). Monotonicity is enforced (q-values are
    non-decreasing when read in ascending p-value order), the standard
    BH step-up guarantee.

    Parameters
    ----------
    p_values : array-like
        Raw p-values, one per test in the family.

    Returns
    -------
    list[float]
        q-values in the same order and length as *p_values*; NaN where
        the input was NaN.
    """
    p_arr = np.array([
        float(p) if p is not None and not np.isnan(float(p)) else np.nan
        for p in p_values
    ])
    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]
    m = len(valid_p)
    if m == 0:
        return list(p_arr)

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    q_sorted = ranked * m / (np.arange(m) + 1)
    # Enforce monotonicity: right-to-left cumulative minimum.
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty(m)
    q_valid[order] = q_sorted
    q_full = np.full_like(p_arr, np.nan)
    q_full[valid_mask] = q_valid
    return list(q_full)
