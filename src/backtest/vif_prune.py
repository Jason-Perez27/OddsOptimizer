"""
src/backtest/vif_prune.py

Variance Inflation Factor (VIF) computation and iterative pruning for the
compression-fix v2 de-collinearization step.

VIF measures how much the variance of a coefficient estimate is inflated
by collinearity with other regressors. Rule of thumb:
    VIF <  5  — acceptable
    VIF <  10 — borderline (some literature says 5)
    VIF >= 10 — problematic; coefficient is unreliable and can be sign-flipped

The iterative-prune procedure mirrors the standard "backward-elimination by
VIF" approach:
    1. Compute VIF for all columns in the candidate set.
    2. If max VIF >= threshold, drop the column with the highest VIF.
    3. Repeat until all VIFs < threshold or only one column remains.

This is strictly for feature-selection diagnostics, not model selection —
the survivors still need a walk-forward OOS comparison before promotion.

Public API
----------
compute_vif(df, cols)
    -> DataFrame: col | vif  (one row per feature)

iterative_vif_prune(df, cols, max_vif=10.0, min_features=1)
    -> (surviving_cols: list, prune_log: list[dict])

format_vif_table_md(vif_df)
    -> str: markdown table

format_prune_log_md(prune_log)
    -> str: multiline markdown describing each drop step
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _vif_one(X_const: np.ndarray, col_idx: int) -> float:
    """
    VIF for column col_idx given all other columns (including the appended
    constant at position -1).

    Uses OLS R² via the closed-form projection: R² = 1 - RSS/TSS.
    VIF = 1 / (1 - R²).  Returns inf if R² is (near) 1.0 (perfect collinearity).
    """
    n_cols = X_const.shape[1]
    other_idxs = [i for i in range(n_cols) if i != col_idx]
    X_others = X_const[:, other_idxs]
    y = X_const[:, col_idx]

    # OLS: β = (X'X)^{-1} X'y;  y_hat = X β;  R² = 1 - RSS/TSS
    try:
        beta, _, _, _ = np.linalg.lstsq(X_others, y, rcond=None)
        y_hat = X_others @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-12:
            return float("inf")  # zero-variance column → perfect "collinearity"
        r2 = 1.0 - ss_res / ss_tot
        if r2 >= 1.0 - 1e-10:
            return float("inf")
        return 1.0 / (1.0 - r2)
    except np.linalg.LinAlgError:
        return float("inf")


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Compute VIF for each column in `cols` given all the others.

    Parameters
    ----------
    df : DataFrame
        Data frame containing the feature columns. NaN rows are dropped
        silently before computation (same rows the GLM would drop).
    cols : list[str]
        Columns to include in the VIF computation. Must be numeric.
        A constant column (all-same value) causes singularity; such columns
        are detected and returned with VIF = inf.

    Returns
    -------
    DataFrame with columns:
        col  -- feature name
        vif  -- VIF value (float; inf when perfect collinearity with others)

    Notes
    -----
    VIF is computed by regressing each feature on all the others (via OLS)
    and computing 1 / (1 - R²). A constant column is appended internally
    so the regression has an intercept; callers must NOT include a constant
    in `cols` — it would make the OLS singular.

    Implementation uses pure numpy (no statsmodels dependency) for portability.
    The results are numerically identical to statsmodels.variance_inflation_factor
    for well-conditioned data.
    """
    if len(cols) < 2:
        # VIF is undefined for a single column (no "others" to regress on)
        return pd.DataFrame({"col": cols, "vif": [float("nan")] * len(cols)})

    X = df[cols].dropna().copy()
    if X.empty or len(X) < len(cols) + 1:
        return pd.DataFrame({"col": cols, "vif": [float("nan")] * len(cols)})

    # Add constant so VIF regresses each feature on others + intercept
    X_const = np.column_stack([X.values.astype(float), np.ones(len(X))])

    rows = []
    for i, col in enumerate(cols):
        vif_val = _vif_one(X_const, i)
        rows.append({"col": col, "vif": round(float(vif_val), 2)})

    return pd.DataFrame(rows)


def iterative_vif_prune(
    df: pd.DataFrame,
    cols: list[str],
    max_vif: float = 10.0,
    min_features: int = 1,
) -> tuple[list[str], list[dict]]:
    """
    Iteratively drop the highest-VIF feature until all remaining features
    have VIF < `max_vif` or `min_features` survivors remain.

    Parameters
    ----------
    df : DataFrame
        Feature data. NaN rows are dropped inside compute_vif.
    cols : list[str]
        Initial candidate set (e.g. CORE_PITCHER_FORM_COLUMNS).
    max_vif : float
        Target ceiling — prune until every VIF < max_vif (default 10.0).
    min_features : int
        Stop pruning when only this many features remain, even if VIF
        still exceeds the ceiling. Prevents pruning to an empty set.

    Returns
    -------
    surviving_cols : list[str]
        Columns remaining after pruning (in original order).
    prune_log : list[dict]
        One dict per pruning step:
            step          -- 1-based iteration number
            dropped       -- column dropped
            dropped_vif   -- VIF of that column before the drop
            remaining     -- list of columns still in the set
            max_vif_after -- max VIF among remaining after drop (nan if <2 remain)
    """
    remaining = list(cols)
    prune_log = []
    step = 0

    while True:
        if len(remaining) <= min_features:
            break

        vif_df = compute_vif(df, remaining)
        vif_df = vif_df.sort_values("vif", ascending=False).reset_index(drop=True)

        worst_vif = vif_df.iloc[0]["vif"]
        if np.isnan(worst_vif) or worst_vif < max_vif:
            break  # all OK

        step += 1
        worst_col = vif_df.iloc[0]["col"]
        remaining.remove(worst_col)

        # Re-compute max VIF after drop (for the log entry)
        if len(remaining) >= 2:
            vif_after = compute_vif(df, remaining)
            max_vif_after = float(vif_after["vif"].max())
        else:
            max_vif_after = float("nan")

        prune_log.append({
            "step": step,
            "dropped": worst_col,
            "dropped_vif": round(float(worst_vif), 2),
            "remaining": list(remaining),
            "max_vif_after": round(max_vif_after, 2) if not np.isnan(max_vif_after) else float("nan"),
        })

    return remaining, prune_log


def format_vif_table_md(vif_df: pd.DataFrame) -> str:
    """Render a compute_vif output as a markdown table."""
    if vif_df.empty:
        return "*(no data)*"
    header = "| feature | VIF |\n|---------|-----|\n"
    lines = [header]
    for _, row in vif_df.sort_values("vif", ascending=False).iterrows():
        vif_str = f"{row['vif']:.2f}" if not np.isnan(row["vif"]) else "n/a"
        flag = " ⚠" if (not np.isnan(row["vif"]) and row["vif"] >= 10) else ""
        lines.append(f"| {row['col']} | {vif_str}{flag} |\n")
    return "".join(lines)


def format_prune_log_md(prune_log: list[dict]) -> str:
    """Render the iterative_vif_prune log as a human-readable markdown block."""
    if not prune_log:
        return "*(no features pruned — initial VIF already below threshold)*"
    lines = []
    for entry in prune_log:
        after_str = (
            f"{entry['max_vif_after']:.2f}"
            if not np.isnan(entry["max_vif_after"])
            else "n/a"
        )
        lines.append(
            f"**Step {entry['step']}**: drop `{entry['dropped']}` "
            f"(VIF={entry['dropped_vif']:.2f}) → "
            f"remaining: {entry['remaining']}; max VIF after = {after_str}"
        )
    return "\n\n".join(lines)
