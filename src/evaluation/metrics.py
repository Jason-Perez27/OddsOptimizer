"""
Model-honesty evaluation metrics for task #10 (spec section 4(a)).

Design: docs/design/specs/2026-06-27-outcome-tracking-design.md, section
4 "(a) Model honesty -- src/evaluation/metrics.py". Pure functions: graded
frames (the output of src/evaluation/grading.py) in, numbers/tables out -- no
file IO, no network -- matching the task #7 evaluation-helper convention
(src/models/baseline_model.py's own mean_absolute_error / root_mean_squared_
error / brier_score / log_loss, reused directly here, never re-implemented).

Two groups of helpers:
- Calibration / reliability + ECE (new) -- bucket p_over into N bins, compare
  predicted vs. empirical hit frequency.
- Brier score / log loss / MAE / RMSE (reuse) -- thin wrappers around
  baseline_model's pure helpers, applied to the over/under binary at the
  posted line and at the "representative" sweep thresholds.
"""

import numpy as np
import pandas as pd

from src.models.baseline_model import (
    brier_score,
    log_loss,
    mean_absolute_error,
    root_mean_squared_error,
)

try:
    from scipy import stats as scipy_stats
except ImportError:  # scipy is optional; pit_histogram raises loudly if used without it
    scipy_stats = None

# The spec names "representative sweep thresholds task #7 named (~5.5, 6.5)"
# -- those are PrizePicks-style *line* values; the 1-10 sweep keys on the
# *threshold* (floor(line)+1, the same convention as
# src/predictions/tiering.line_to_threshold), so 5.5 -> 6 and 6.5 -> 7.
REPRESENTATIVE_THRESHOLDS = (6, 7)

DEFAULT_N_BUCKETS = 10


# ---------------------------------------------------------------------------
# Calibration / reliability + ECE
# ---------------------------------------------------------------------------

RELIABILITY_TABLE_COLUMNS = ["bucket_lo", "bucket_hi", "n", "mean_predicted", "empirical_rate"]


def reliability_table(graded_sweep_df: pd.DataFrame, n_buckets: int = DEFAULT_N_BUCKETS) -> pd.DataFrame:
    """
    Bucket `p_over` into `n_buckets` equal-width bins over [0, 1] and compare
    the mean predicted probability in each bucket to the empirical hit
    frequency (mean `over_hit`). Only settled rows (`over_hit` not NaN)
    contribute -- pending/void rows carry no realized outcome to calibrate
    against.

    Empty buckets are included with n=0 and NaN rates (never silently
    dropped, so the table's bucket count always equals `n_buckets`).
    """
    settled = graded_sweep_df[graded_sweep_df["over_hit"].notna()].copy()

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows = []
    if settled.empty:
        for i in range(n_buckets):
            rows.append({
                "bucket_lo": edges[i], "bucket_hi": edges[i + 1],
                "n": 0, "mean_predicted": np.nan, "empirical_rate": np.nan,
            })
        return pd.DataFrame(rows, columns=RELIABILITY_TABLE_COLUMNS)

    settled["over_hit"] = settled["over_hit"].astype(bool)
    # Right-inclusive on the last bucket so p_over == 1.0 lands in the top
    # bucket rather than overflowing past it.
    bucket_idx = np.clip(
        np.digitize(settled["p_over"].to_numpy(dtype=float), edges[1:-1], right=False),
        0, n_buckets - 1,
    )
    settled["_bucket"] = bucket_idx

    for i in range(n_buckets):
        in_bucket = settled[settled["_bucket"] == i]
        n = len(in_bucket)
        rows.append({
            "bucket_lo": edges[i],
            "bucket_hi": edges[i + 1],
            "n": n,
            "mean_predicted": float(in_bucket["p_over"].mean()) if n else np.nan,
            "empirical_rate": float(in_bucket["over_hit"].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows, columns=RELIABILITY_TABLE_COLUMNS)


def expected_calibration_error(reliability_df: pd.DataFrame) -> float:
    """
    ECE = sum over non-empty buckets of (n_bucket / n_total) * |mean_predicted
    - empirical_rate|. Decoupled from `reliability_table` so it's directly
    testable against a hand-computed reference table.
    """
    non_empty = reliability_df[reliability_df["n"] > 0]
    total = non_empty["n"].sum()
    if total == 0:
        return float("nan")
    weighted_gap = (
        non_empty["n"] * (non_empty["mean_predicted"] - non_empty["empirical_rate"]).abs()
    ).sum()
    return float(weighted_gap / total)


# ---------------------------------------------------------------------------
# Brier score / log loss (reuse baseline_model)
# ---------------------------------------------------------------------------

def brier_and_log_loss_at_line(graded_line_picks_df: pd.DataFrame) -> dict:
    """
    Brier score and log loss of `p_over` vs. the realized over/under binary
    (`over_hit`), at the posted line. Settled rows only.
    """
    settled = graded_line_picks_df[graded_line_picks_df["over_hit"].notna()]
    if settled.empty:
        return {"brier_score": float("nan"), "log_loss": float("nan"), "n": 0}
    y = settled["over_hit"].astype(bool).astype(float)
    p = settled["p_over"].astype(float)
    return {
        "brier_score": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "n": int(len(settled)),
    }


def brier_and_log_loss_at_thresholds(
    graded_sweep_df: pd.DataFrame, thresholds=REPRESENTATIVE_THRESHOLDS,
) -> dict:
    """
    Brier score and log loss of `p_over` vs. `over_hit`, computed separately
    at each of `thresholds` (default: the spec's representative ~5.5/6.5
    line thresholds, 6 and 7). Settled rows only, per threshold.
    """
    out = {}
    for t in thresholds:
        subset = graded_sweep_df[
            (graded_sweep_df["threshold"] == t) & (graded_sweep_df["over_hit"].notna())
        ]
        if subset.empty:
            out[t] = {"brier_score": float("nan"), "log_loss": float("nan"), "n": 0}
            continue
        y = subset["over_hit"].astype(bool).astype(float)
        p = subset["p_over"].astype(float)
        out[t] = {
            "brier_score": brier_score(y, p),
            "log_loss": log_loss(y, p),
            "n": int(len(subset)),
        }
    return out


# ---------------------------------------------------------------------------
# PIT (probability integral transform) for counts -- task #11
# ---------------------------------------------------------------------------

# Mid-PIT (a.k.a. randomized-PIT's expectation): for a discrete count model,
# PIT_i = F(y_i - 1) + 0.5 * f(y_i), where F is the fitted family's CDF and f
# its PMF at the realized count y_i. This is the standard correction for
# discrete outcomes (Czado, Gneiting & Held 2009) -- a fully randomized PIT
# (drawing U ~ Uniform(0,1) and using F(y-1) + U*f(y)) is also unbiased but
# non-deterministic, which would make "matches a hand-computed reference"
# (testing item #5) impossible to pin down across runs. Using the
# distribution's mean instead of a random draw keeps this pure and
# reproducible, at the cost of mild underdispersion in the resulting PIT
# histogram relative to a true randomized PIT -- acceptable for this
# diagnostic use. A well-calibrated model's PIT values are ~Uniform(0, 1);
# pit_histogram's bucket counts are the read on that.
def pit_values(graded_df: pd.DataFrame) -> pd.Series:
    """
    Mid-PIT value per row of `graded_df`, which must carry `family` ("poisson"
    or "negative_binomial"), `mu`, `alpha` (NaN/ignored for poisson rows), and
    `realized_strikeouts`. Only settled rows (`realized_strikeouts` not NaN)
    are scored; the returned Series is indexed like the settled subset (NaN
    rows are dropped, not included as NaN).
    """
    if scipy_stats is None:
        raise ImportError("scipy is required for pit_values (pip install scipy)")

    settled = graded_df[graded_df["realized_strikeouts"].notna()]
    if settled.empty:
        return pd.Series([], dtype=float)

    y = settled["realized_strikeouts"].astype(float).to_numpy()
    mu = settled["mu"].astype(float).to_numpy()
    family = settled["family"].to_numpy()
    alpha = settled["alpha"].astype(float).to_numpy() if "alpha" in settled.columns else np.full(len(settled), np.nan)

    is_poisson = family == "poisson"

    cdf_below = np.empty(len(settled), dtype=float)
    pmf_at = np.empty(len(settled), dtype=float)

    if is_poisson.any():
        cdf_below[is_poisson] = scipy_stats.poisson.cdf(y[is_poisson] - 1, mu[is_poisson])
        pmf_at[is_poisson] = scipy_stats.poisson.pmf(y[is_poisson], mu[is_poisson])

    is_nb = ~is_poisson
    if is_nb.any():
        n = 1.0 / alpha[is_nb]
        p = 1.0 / (1.0 + alpha[is_nb] * mu[is_nb])
        cdf_below[is_nb] = scipy_stats.nbinom.cdf(y[is_nb] - 1, n, p)
        pmf_at[is_nb] = scipy_stats.nbinom.pmf(y[is_nb], n, p)

    pit = cdf_below + 0.5 * pmf_at
    return pd.Series(pit, index=settled.index, name="pit")


PIT_HISTOGRAM_COLUMNS = ["bucket_lo", "bucket_hi", "n"]


def pit_histogram(graded_df: pd.DataFrame, n_buckets: int = DEFAULT_N_BUCKETS) -> pd.DataFrame:
    """
    Bucket the mid-PIT values from `pit_values(graded_df)` into `n_buckets`
    equal-width bins over [0, 1] and count how many fall in each. A
    well-calibrated model's PIT histogram is approximately flat (uniform);
    systematic skew or U/inverted-U shape signals mis-calibration. Empty
    buckets are included with n=0, matching `reliability_table`'s convention
    of never silently dropping a bucket.
    """
    pit = pit_values(graded_df)

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows = []
    if pit.empty:
        for i in range(n_buckets):
            rows.append({"bucket_lo": edges[i], "bucket_hi": edges[i + 1], "n": 0})
        return pd.DataFrame(rows, columns=PIT_HISTOGRAM_COLUMNS)

    bucket_idx = np.clip(
        np.digitize(pit.to_numpy(dtype=float), edges[1:-1], right=False),
        0, n_buckets - 1,
    )
    for i in range(n_buckets):
        rows.append({
            "bucket_lo": edges[i],
            "bucket_hi": edges[i + 1],
            "n": int((bucket_idx == i).sum()),
        })
    return pd.DataFrame(rows, columns=PIT_HISTOGRAM_COLUMNS)


# ---------------------------------------------------------------------------
# Slices -- by sweep-tier and over time (by walk-forward step) -- task #11
# spec section 3 / testing item #6: "Slices: by-threshold, by-sweep-tier, and
# over-time aggregations are correct on a fixture." By-threshold slicing
# already exists (brier_and_log_loss_at_thresholds); these two are new.
#
# Both take the same long, one-row-per-(pitcher-game, threshold) shape as
# brier_and_log_loss_at_thresholds (e.g. walk_forward.melt_oos_sweep's output,
# or the live GRADED_THRESHOLD_SWEEP_COLUMNS frame, which also carries a
# `tier` column) -- never reimplemented per-caller, just grouped on a
# different column. Settled rows only, per group; an empty/all-unsettled
# input returns an empty (correctly-columned) frame rather than crashing.
# ---------------------------------------------------------------------------

SLICE_COLUMNS = ["n", "mean_predicted", "empirical_rate", "brier_score", "log_loss"]


def _grouped_calibration_slice(graded_sweep_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    settled = graded_sweep_df[graded_sweep_df["over_hit"].notna()].copy()
    columns = [group_col] + SLICE_COLUMNS
    if settled.empty:
        return pd.DataFrame(columns=columns)

    settled["over_hit"] = settled["over_hit"].astype(bool)
    rows = []
    for group_value, group in settled.groupby(group_col, sort=True):
        y = group["over_hit"].astype(float)
        p = group["p_over"].astype(float)
        rows.append({
            group_col: group_value,
            "n": int(len(group)),
            "mean_predicted": float(p.mean()),
            "empirical_rate": float(y.mean()),
            "brier_score": brier_score(y, p),
            "log_loss": log_loss(y, p),
        })
    return pd.DataFrame(rows, columns=columns)


def by_sweep_tier(graded_sweep_df: pd.DataFrame, tier_col: str = "tier") -> pd.DataFrame:
    """
    Calibration (mean predicted vs. empirical hit rate) plus Brier/log loss,
    grouped by sweep tier. A first read on whether the threshold-sweep tiers
    (low/medium/high) are themselves calibrated -- distinct from the live
    line-pick tier hit-rate, which still needs Track B accumulation (spec
    section 2's "tiers bonus").
    """
    return _grouped_calibration_slice(graded_sweep_df, tier_col)


def over_time(graded_sweep_df: pd.DataFrame, time_col: str = "wf_step") -> pd.DataFrame:
    """
    The same calibration/Brier/log-loss slice, grouped by walk-forward step
    (`wf_step`, the step's cutoff date) instead of tier -- shows whether
    calibration holds or drifts as the season progresses (spec section 3).
    """
    return _grouped_calibration_slice(graded_sweep_df, time_col)


def point_accuracy(attached_predictions_df: pd.DataFrame) -> dict:
    """
    MAE / RMSE of the model's mean prediction `mu` vs. realized
    `realized_strikeouts` -- settled rows only. Expects the output of
    `grading.attach_outcomes(predictions_df, realized_df, ...)` (the only
    frame in this pipeline that carries both `mu` and `realized_strikeouts`).
    """
    settled = attached_predictions_df[attached_predictions_df["realized_strikeouts"].notna()]
    if settled.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    y_true = settled["realized_strikeouts"].astype(float)
    y_pred = settled["mu"].astype(float)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "n": int(len(settled)),
    }
