"""
src/backtest/tail_calibration.py

Tail calibration diagnostics for strikeout projection models.

Aggregate metrics (ECE, Brier, log-loss) can look fine while high-μ starts
are systematically under-predicted — this is the "projection compression" bug
described in docs/design/specs/2026-06-30-fix-projection-compression-
and-matchup-design.md. This module surfaces that bias by bucketing OOS
predictions by μ decile and reporting predicted-vs-realized per bucket.

Key insight: when a model was trained on a data-starved window (e.g. half-
season), fitted coefficients shrink toward zero and μ barely departs from
the baseline intercept. The error is worst at the tails (aces and soft arms),
but aggregate calibration hides it because the error is *symmetric* in sign
across the distribution while being large in absolute value.

Public API
----------
compute_mu_decile_table(oos_df, n_deciles=10)
    -> DataFrame: decile | mu_lo | mu_hi | n | mean_pred | mean_actual | bias | bias_pct

flag_high_mu_underprediction(decile_table, top_n_deciles=2, min_bias_pct=-10.0)
    -> bool: True if top deciles show systematic under-prediction (actual > predicted)

high_mu_reliability_curve(oos_df, top_fraction=0.2, n_bins=10)
    -> DataFrame: mu_bin_lo | mu_bin_hi | n | mean_pred | mean_actual

format_decile_table_md(decile_table)
    -> str: markdown table ready to paste into a report

Usage
-----
    from src.backtest.tail_calibration import (
        compute_mu_decile_table,
        flag_high_mu_underprediction,
        format_decile_table_md,
    )
    oos_df = pd.read_csv("data/processed/backtest/walk_forward_oos.csv")
    tbl = compute_mu_decile_table(oos_df)
    flagged = flag_high_mu_underprediction(tbl)
    print(format_decile_table_md(tbl))
"""

import numpy as np
import pandas as pd


def _resolve_mu_col(df: pd.DataFrame) -> str:
    """Return the μ column name (pred_mean or mu), raising if neither present."""
    for candidate in ("pred_mean", "mu"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "oos_df must contain a 'pred_mean' or 'mu' column (the model's predicted mean K count)"
    )


def compute_mu_decile_table(
    oos_df: pd.DataFrame,
    n_deciles: int = 10,
    subset_mask: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Split OOS predictions into μ deciles and report predicted-vs-realized K
    per decile.

    Parameters
    ----------
    oos_df : DataFrame
        Walk-forward OOS frame with columns 'pred_mean' (or 'mu') and
        'strikeouts' (realized count). Additional columns are ignored.
    n_deciles : int
        Number of equal-frequency buckets (default 10 = deciles).
    subset_mask : boolean Series aligned with oos_df.index, optional
        When provided, only the rows where mask is True are included in the
        decile computation. Use this to score a year-specific OOS subset
        (e.g. mask = oos_df['game_date'].dt.year == 2026) without requiring
        the caller to pre-filter the frame. If None (default), all rows are
        used.

    Returns
    -------
    DataFrame with columns:
        decile      -- 1-based integer bucket (1 = lowest μ, n = highest)
        mu_lo       -- minimum μ in this bucket
        mu_hi       -- maximum μ in this bucket
        n           -- number of starts in this bucket
        mean_pred   -- mean predicted μ
        mean_actual -- mean realized strikeout count
        bias        -- mean_pred − mean_actual  (negative = under-prediction)
        bias_pct    -- bias / mean_actual × 100  (negative = under-prediction)
    """
    if oos_df.empty:
        return pd.DataFrame(
            columns=["decile", "mu_lo", "mu_hi", "n", "mean_pred", "mean_actual", "bias", "bias_pct"]
        )

    # Apply subset mask before doing anything else
    working_df = oos_df if subset_mask is None else oos_df.loc[subset_mask]
    if working_df.empty:
        return pd.DataFrame(
            columns=["decile", "mu_lo", "mu_hi", "n", "mean_pred", "mean_actual", "bias", "bias_pct"]
        )

    mu_col = _resolve_mu_col(working_df)
    if "strikeouts" not in working_df.columns:
        raise ValueError("oos_df must have a 'strikeouts' column (realized K count)")

    df = working_df[[mu_col, "strikeouts"]].copy()
    df.columns = ["_mu", "_k"]
    df["_mu"] = df["_mu"].astype(float)
    df["_k"] = df["_k"].astype(float)

    # Drop rows where μ or realized K is NaN (can't bucket them)
    df = df.dropna(subset=["_mu", "_k"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=["decile", "mu_lo", "mu_hi", "n", "mean_pred", "mean_actual", "bias", "bias_pct"]
        )

    # Assign decile labels (0-based, allowing fewer bins when data is sparse)
    df["_decile"] = pd.qcut(df["_mu"], q=n_deciles, labels=False, duplicates="drop")

    rows = []
    for decile_idx, grp in df.groupby("_decile", sort=True):
        mean_pred = float(grp["_mu"].mean())
        mean_actual = float(grp["_k"].mean())
        bias = mean_pred - mean_actual
        # Avoid divide-by-zero; 0-actual starts are edge cases (no K game)
        bias_pct = (bias / mean_actual * 100.0) if mean_actual != 0.0 else float("nan")
        rows.append({
            "decile": int(decile_idx) + 1,
            "mu_lo": float(grp["_mu"].min()),
            "mu_hi": float(grp["_mu"].max()),
            "n": len(grp),
            "mean_pred": round(mean_pred, 3),
            "mean_actual": round(mean_actual, 3),
            "bias": round(bias, 3),
            "bias_pct": round(bias_pct, 1) if not np.isnan(bias_pct) else float("nan"),
        })

    return pd.DataFrame(rows)


def flag_high_mu_underprediction(
    decile_table: pd.DataFrame,
    top_n_deciles: int = 2,
    min_bias_pct: float = -10.0,
    *,
    oos_df: "pd.DataFrame | None" = None,
    subset_mask: "pd.Series | None" = None,
    n_deciles: int = 10,
) -> bool:
    """
    Return True if the top `top_n_deciles` deciles ALL show under-prediction
    beyond the `min_bias_pct` threshold.

    Parameters
    ----------
    decile_table : DataFrame
        Output of compute_mu_decile_table.
    top_n_deciles : int
        How many top deciles to inspect (default 2 = top 20% when n_deciles=10).
    min_bias_pct : float
        Bias-percent threshold that triggers the flag (default −10.0).
        bias_pct = (pred − actual) / actual × 100; negative means under-prediction.
        A value of −10 triggers when the model is 10%+ below actual in the top deciles.

    Returns
    -------
    bool: True = systematic high-μ under-prediction detected, adopt the fix.
          False = top deciles look OK, may not need the compression fix.
    """
    # If an oos_df + mask were passed instead of a pre-built table, build it now.
    if oos_df is not None:
        decile_table = compute_mu_decile_table(oos_df, n_deciles=n_deciles, subset_mask=subset_mask)

    if decile_table.empty:
        return False
    if "bias_pct" not in decile_table.columns:
        raise ValueError("decile_table must contain a 'bias_pct' column (from compute_mu_decile_table)")

    top = decile_table.nlargest(top_n_deciles, "decile")
    if top.empty:
        return False

    # All top deciles must have bias_pct < min_bias_pct (i.e. sufficiently negative)
    return bool((top["bias_pct"] < min_bias_pct).all())


def high_mu_reliability_curve(
    oos_df: pd.DataFrame,
    top_fraction: float = 0.2,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Reliability curve restricted to the top `top_fraction` of starts by μ.

    Within those high-μ starts, bucket by predicted μ (equal-frequency bins)
    and report mean predicted vs mean actual — a finer-grained view of where
    the tail compression is worst.

    Parameters
    ----------
    oos_df : DataFrame
        Walk-forward OOS frame with 'pred_mean' (or 'mu') and 'strikeouts'.
    top_fraction : float
        Fraction of starts to keep (default 0.2 = top 20% by μ).
    n_bins : int
        Number of bins within the high-μ subset.

    Returns
    -------
    DataFrame: mu_bin_lo | mu_bin_hi | n | mean_pred | mean_actual
    """
    if oos_df.empty:
        return pd.DataFrame(columns=["mu_bin_lo", "mu_bin_hi", "n", "mean_pred", "mean_actual"])

    mu_col = _resolve_mu_col(oos_df)
    df = oos_df[[mu_col, "strikeouts"]].copy()
    df.columns = ["_mu", "_k"]
    df = df.dropna(subset=["_mu", "_k"])
    if df.empty:
        return pd.DataFrame(columns=["mu_bin_lo", "mu_bin_hi", "n", "mean_pred", "mean_actual"])

    cutoff = df["_mu"].quantile(1.0 - top_fraction)
    top_df = df[df["_mu"] >= cutoff].copy().reset_index(drop=True)
    if top_df.empty:
        return pd.DataFrame(columns=["mu_bin_lo", "mu_bin_hi", "n", "mean_pred", "mean_actual"])

    top_df["_bin"] = pd.qcut(top_df["_mu"], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for _, grp in top_df.groupby("_bin", sort=True):
        rows.append({
            "mu_bin_lo": float(grp["_mu"].min()),
            "mu_bin_hi": float(grp["_mu"].max()),
            "n": len(grp),
            "mean_pred": round(float(grp["_mu"].mean()), 3),
            "mean_actual": round(float(grp["_k"].mean()), 3),
        })
    return pd.DataFrame(rows)


def format_decile_table_md(decile_table: pd.DataFrame) -> str:
    """
    Render a compute_mu_decile_table output as a markdown table.

    Returns a string like:
        | decile | μ range | n | mean_pred | mean_actual | bias | bias_pct |
        |--------|---------|---|-----------|-------------|------|----------|
        | 1 | 2.3–3.8 | 427 | 3.1 | 3.2 | -0.1 | -3.1% |
        ...
    """
    if decile_table.empty:
        return "*(no data)*"

    header = "| decile | μ range | n | mean_pred | mean_actual | bias | bias_pct |\n"
    separator = "|--------|---------|---|-----------|-------------|------|----------|\n"
    lines = [header, separator]
    for _, row in decile_table.iterrows():
        mu_range = f"{row['mu_lo']:.2f}–{row['mu_hi']:.2f}"
        bias_pct_str = f"{row['bias_pct']:.1f}%" if not np.isnan(row["bias_pct"]) else "n/a"
        lines.append(
            f"| {int(row['decile'])} | {mu_range} | {int(row['n'])} | "
            f"{row['mean_pred']:.3f} | {row['mean_actual']:.3f} | "
            f"{row['bias']:+.3f} | {bias_pct_str} |\n"
        )
    return "".join(lines)
