"""
No-action band calibration helper (spec ①, 2026-06-30).

Accepts a graded line-picks frame (output of src/pipeline/settle.py after
src/evaluation/grading.grade_line_picks) that includes `conviction` and
`pick_correct` columns, and returns the ROI-optimal conviction cutoff per
bucket (tier × threshold bracket).

Until ≥100 settled picks/bucket exist the function returns the documented
provisional default and a `validated=False` flag.  Do NOT remove that label
or treat the provisional threshold as proven edge.

Usage (once Track-B data accrues):
    from src.backtest.conviction import calibrate_no_action_band
    result = calibrate_no_action_band(graded_df)
    # result["buckets"] → list of per-bucket dicts
    # result["validated"] → bool (True only if ALL buckets have ≥100 settled)
"""

import numpy as np
import pandas as pd

from src.predictions.tiering import NO_ACTION_CONVICTION_THRESHOLD, NO_ACTION_EDGE_THRESHOLD

MIN_SAMPLES_PER_BUCKET = 100  # gate: below this, return provisional defaults

# Conviction grid to scan when searching for the ROI-optimal cutoff.
_CONVICTION_GRID = np.arange(0.0, 5.05, 0.25)


def _roi_at_cutoff(settled: pd.DataFrame, cutoff: float) -> dict:
    """Even-money ROI for picks with conviction >= cutoff."""
    above = settled[settled["conviction"] >= cutoff]
    n = len(above)
    if n == 0:
        return {"cutoff": cutoff, "n": 0, "hit_rate": float("nan"), "roi": float("nan")}
    wins = above["pick_correct"].sum()
    losses = n - wins
    roi = (wins - losses) / n  # ∈ [−1, +1]; break-even = 0
    return {
        "cutoff": cutoff,
        "n": int(n),
        "hit_rate": float(wins / n),
        "roi": float(roi),
    }


def _best_cutoff(settled: pd.DataFrame) -> dict:
    """Return the conviction cutoff on _CONVICTION_GRID with the highest ROI.
    Tie-breaks toward higher cutoffs (stricter = fewer but better picks)."""
    results = [_roi_at_cutoff(settled, c) for c in _CONVICTION_GRID]
    # Filter to cutoffs with at least 5 picks to avoid noise at the extreme tail.
    eligible = [r for r in results if r["n"] >= 5]
    if not eligible:
        return {"cutoff": NO_ACTION_CONVICTION_THRESHOLD, "n": 0,
                "hit_rate": float("nan"), "roi": float("nan")}
    best = max(eligible, key=lambda r: (r["roi"], r["cutoff"]))
    return best


def calibrate_no_action_band(graded: pd.DataFrame, bucket_col: str = "tier") -> dict:
    """
    Given a graded line-picks frame with columns ``conviction``, ``pick_correct``,
    and ``bucket_col`` (default ``tier``), return the ROI-optimal conviction
    cutoff per bucket.

    Parameters
    ----------
    graded : pd.DataFrame
        Graded picks with at least:
          - ``conviction``   (float)  — computed by tiering.build_line_picks
          - ``pick_correct`` (float)  — 1.0 win / 0.0 loss / NaN push|unsettled
          - ``tier``         (str)    — or the column named by ``bucket_col``
    bucket_col : str
        Column to group by.  Default ``tier`` (high / medium / low).

    Returns
    -------
    dict with keys:
      - ``validated`` (bool): True only if EVERY bucket has ≥ MIN_SAMPLES_PER_BUCKET
        settled (non-push) picks.  If False, every bucket uses the provisional
        defaults and ``reason`` is set.
      - ``provisional_conviction_threshold``: the global provisional default
      - ``provisional_edge_threshold``: the global provisional edge default
      - ``buckets``: list of per-bucket result dicts, each with:
          - ``bucket``
          - ``n_settled``
          - ``validated``
          - ``cutoff`` (float) — use this in production if validated
          - ``hit_rate``, ``roi`` — at the returned cutoff
          - ``reason`` (str) — human-readable note
    """
    required = {"conviction", "pick_correct"}
    if not required.issubset(graded.columns):
        missing = required - set(graded.columns)
        raise ValueError(f"graded frame missing required columns: {sorted(missing)}")
    if bucket_col not in graded.columns:
        raise ValueError(f"bucket column {bucket_col!r} not in graded frame")

    # Only settled non-push rows count toward the ROI denominator.
    settled = graded[graded["pick_correct"].notna()].copy()
    settled["pick_correct"] = settled["pick_correct"].astype(float)

    buckets = []
    global_validated = True

    for bucket, grp in settled.groupby(bucket_col):
        n = len(grp)
        if n < MIN_SAMPLES_PER_BUCKET:
            global_validated = False
            buckets.append({
                "bucket": bucket,
                "n_settled": n,
                "validated": False,
                "cutoff": NO_ACTION_CONVICTION_THRESHOLD,
                "hit_rate": float("nan"),
                "roi": float("nan"),
                "reason": (
                    f"Only {n} settled picks in bucket '{bucket}' "
                    f"(need ≥{MIN_SAMPLES_PER_BUCKET}); using provisional default "
                    f"conviction ≥ {NO_ACTION_CONVICTION_THRESHOLD} — NOT ROI-validated."
                ),
            })
        else:
            best = _best_cutoff(grp)
            buckets.append({
                "bucket": bucket,
                "n_settled": n,
                "validated": True,
                "cutoff": best["cutoff"],
                "hit_rate": best["hit_rate"],
                "roi": best["roi"],
                "reason": (
                    f"ROI-optimal cutoff {best['cutoff']:.2f} found on {n} settled picks "
                    f"(hit-rate {best['hit_rate']:.3f}, roi {best['roi']:+.3f})."
                ),
            })

    if not buckets:
        # No settled picks at all — fully provisional.
        global_validated = False

    return {
        "validated": global_validated,
        "provisional_conviction_threshold": NO_ACTION_CONVICTION_THRESHOLD,
        "provisional_edge_threshold": NO_ACTION_EDGE_THRESHOLD,
        "min_samples_per_bucket": MIN_SAMPLES_PER_BUCKET,
        "buckets": buckets,
    }
