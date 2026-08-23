"""
Pick-profitability backtest for task #10 (spec section 4(b)).

Design: docs/design/specs/2026-06-27-outcome-tracking-design.md, section
4 "(b) Pick profitability -- src/backtest/roi.py". Pure functions: the graded
line-picks frame (src/evaluation/grading.grade_line_picks' output, after
settle.py has merged in the authoritative `settlement_status`) in,
numbers/tables out -- no file IO, no network.

Payout convention (2026-08 migration): real EV. A win pays the CHOSEN side's
`over_payout_multiplier` / `under_payout_multiplier` (Underdog's profit-per-
winning-unit-stake convention -- see src/data/underdog_lines.payout_to_decimal),
a loss costs the full unit stake (-1) regardless of payout convention, and a
push refunds (0). Settled historical rows from before the Underdog migration
(or any matched-but-unpriced line) carry no multiplier -- those fall back to
the old v1 flat/even-money proxy (win +1, loss -1) instead of silently being
treated as real EV. Every row is tagged `pnl_source` ("real_ev" /
"flat_fallback") precisely so a report CAN separate pre- and post-migration
ROI rather than averaging two different payout conventions together as if
they were one number.

The denominator convention used throughout -- "settled, non-push" -- is
exactly the set of rows where `pick_correct` is not NaN: grading.py already
NaNs `pick_correct` for both unsettled rows (no realized outcome yet) and
pushes (refunded, not scored), so filtering on it gives the honest hit-rate
and ROI denominator in one step without re-deriving settlement_status here.
"""

import numpy as np
import pandas as pd


def _settled_non_push(graded_line_picks_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real win/loss verdict -- excludes both unsettled and push rows."""
    if graded_line_picks_df.empty or "pick_correct" not in graded_line_picks_df.columns:
        return graded_line_picks_df.iloc[0:0]
    return graded_line_picks_df[graded_line_picks_df["pick_correct"].notna()]


def hit_rate(graded_line_picks_df: pd.DataFrame) -> dict:
    """
    wins / (wins + losses); pushes and unsettled (pending/void_scratched)
    excluded from the denominator -- the honest version (spec, testing item 4).
    """
    settled = _settled_non_push(graded_line_picks_df)
    n = len(settled)
    if n == 0:
        return {"wins": 0, "losses": 0, "n": 0, "hit_rate": float("nan")}
    wins = int(settled["pick_correct"].astype(bool).sum())
    return {"wins": wins, "losses": n - wins, "n": n, "hit_rate": wins / n}


def _resolve_row_pnl(row) -> tuple:
    """
    Real-EV pnl for one row, and where it came from.

    - No realized `pnl_units` at all (unsettled) -> (NaN, "flat_fallback").
    - `pnl_units == 0` (push) -> (0.0, "flat_fallback") -- a push refunds
      under either payout convention, so the source label is moot; kept
      "flat_fallback" simply because no multiplier was consulted.
    - Win: the chosen side's (`lean`) payout multiplier if present and
      parseable -> (multiplier, "real_ev"); otherwise the v1 flat proxy
      -> (1.0, "flat_fallback").
    - Loss: always -1 (the full stake is lost regardless of payout
      convention) -- tagged "real_ev" only when a multiplier was actually
      available for this row (i.e. it's a post-migration row), purely so
      pre/post slicing groups it with its era, not because the -1 itself
      differs.
    """
    pnl = row.get("pnl_units")
    if pnl is None or (isinstance(pnl, float) and np.isnan(pnl)):
        return np.nan, "flat_fallback"
    if pnl == 0:
        return 0.0, "flat_fallback"

    lean = row.get("lean")
    over_mult = row.get("over_payout_multiplier")
    under_mult = row.get("under_payout_multiplier")
    multiplier = over_mult if lean == "over" else under_mult
    has_multiplier = multiplier is not None and not (
        isinstance(multiplier, float) and np.isnan(multiplier)
    )

    pick_correct = row.get("pick_correct")
    if bool(pick_correct):
        if has_multiplier:
            return float(multiplier), "real_ev"
        return 1.0, "flat_fallback"
    return -1.0, ("real_ev" if has_multiplier else "flat_fallback")


def _with_real_pnl(graded_line_picks_df: pd.DataFrame) -> pd.DataFrame:
    """Add `real_pnl_units` / `pnl_source` columns -- see _resolve_row_pnl."""
    out = graded_line_picks_df.copy()
    if out.empty:
        out["real_pnl_units"] = pd.Series(dtype="float64")
        out["pnl_source"] = pd.Series(dtype="object")
        return out
    resolved = out.apply(_resolve_row_pnl, axis=1)
    out["real_pnl_units"] = [r[0] for r in resolved]
    out["pnl_source"] = [r[1] for r in resolved]
    return out


def _roi_over(settled_non_push_df: pd.DataFrame, all_rows_df: pd.DataFrame) -> dict:
    """total_pnl/n/roi for a slice, summing `real_pnl_units` over `all_rows_df`
    (pushes included in the numerator, contributing exactly 0) and dividing by
    settled-non-push count."""
    settled_any = all_rows_df[all_rows_df["real_pnl_units"].notna()]
    total_pnl = float(settled_any["real_pnl_units"].sum()) if not settled_any.empty else 0.0
    denom = len(settled_non_push_df)
    roi = total_pnl / denom if denom else float("nan")
    return {"total_pnl": total_pnl, "n_settled_non_push": denom, "roi": roi}


def flat_bet_roi(graded_line_picks_df: pd.DataFrame) -> dict:
    """
    Real-EV ROI (spec, testing item 8, updated for the 2026-08 payout-aware
    migration): ROI = sum(real_pnl_units) / number of settled non-push picks.
    The numerator sums over all settled rows (push rows contribute exactly 0);
    the denominator is settled-non-push only.

    Also returns `roi_real_ev` / `roi_flat_fallback` breakdowns (same shape:
    {total_pnl, n_settled_non_push, roi}) computed ONLY over rows of that
    payout source, so a mixed pre/post-migration dataset is never silently
    averaged into one blended number -- `roi`/`total_pnl` at the top level
    remain the honest blended total across every row actually settled, but a
    caller that wants "real ROI since the migration" reads `roi_real_ev`.
    """
    df = _with_real_pnl(graded_line_picks_df)
    settled_non_push_df = _settled_non_push(df)

    result = _roi_over(settled_non_push_df, df)

    for source in ("real_ev", "flat_fallback"):
        sub_settled = settled_non_push_df[settled_non_push_df["pnl_source"] == source] \
            if "pnl_source" in settled_non_push_df.columns else settled_non_push_df.iloc[0:0]
        sub_all = df[df["pnl_source"] == source] if "pnl_source" in df.columns else df.iloc[0:0]
        result[f"roi_{source}"] = _roi_over(sub_settled, sub_all)

    return result


def by_group(graded_line_picks_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Hit rate + ROI sliced by an arbitrary column -- `tier` (High/Medium/Low)
    or `line` (the posted line value), per the spec's "slices that actually
    answer 'do the tiers mean anything'". Every group, even one with zero
    settled rows, gets a row (n=0, NaN rates) rather than disappearing.
    """
    if graded_line_picks_df.empty:
        return pd.DataFrame(columns=[group_col, "n_settled", "wins", "losses", "hit_rate", "total_pnl", "roi"])

    rows = []
    for key, group in graded_line_picks_df.groupby(group_col, dropna=False):
        hr = hit_rate(group)
        roi_info = flat_bet_roi(group)
        rows.append({
            group_col: key,
            "n_settled": hr["n"],
            "wins": hr["wins"],
            "losses": hr["losses"],
            "hit_rate": hr["hit_rate"],
            "total_pnl": roi_info["total_pnl"],
            "roi": roi_info["roi"],
        })
    return pd.DataFrame(rows, columns=[group_col, "n_settled", "wins", "losses", "hit_rate", "total_pnl", "roi"])


def by_tier(graded_line_picks_df: pd.DataFrame) -> pd.DataFrame:
    return by_group(graded_line_picks_df, "tier")


def by_line(graded_line_picks_df: pd.DataFrame) -> pd.DataFrame:
    return by_group(graded_line_picks_df, "line")


TIME_SERIES_COLUMNS = [
    "game_date", "win", "pnl_units", "pnl_source", "cum_n", "cum_wins", "cum_pnl",
    "cum_hit_rate", "cum_roi",
]


def time_series(
    graded_line_picks_df: pd.DataFrame,
    date_col: str = "game_date",
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """
    Cumulative (and, if `rolling_window` is given, rolling) hit-rate/ROI
    series over settled-non-push picks, ordered by `date_col` (spec, testing
    item 9: "cumulative and rolling ROI/hit-rate series have the right
    length and cumulative values on an ordered fixture").

    `pnl_units` here is the real-EV pnl (see module docstring) with a
    per-row `pnl_source` carried alongside so a caller can slice the series
    at the pre/post-migration boundary instead of treating the whole
    cumulative curve as one payout convention.

    Unsettled and push rows are excluded -- a cumulative curve that included
    NaN/0-but-not-really-a-result rows would misrepresent both the sample
    size and the running rate.
    """
    resolved = _with_real_pnl(graded_line_picks_df)
    settled = _settled_non_push(resolved).copy()
    if settled.empty:
        cols = list(TIME_SERIES_COLUMNS)
        if rolling_window:
            cols += ["rolling_hit_rate", "rolling_roi"]
        return pd.DataFrame(columns=cols)

    settled = settled.sort_values(date_col, kind="stable").reset_index(drop=True)
    settled["win"] = settled["pick_correct"].astype(bool).astype(int)
    settled["pnl_units"] = settled["real_pnl_units"].astype(float)

    settled["cum_n"] = np.arange(1, len(settled) + 1)
    settled["cum_wins"] = settled["win"].cumsum()
    settled["cum_pnl"] = settled["pnl_units"].cumsum()
    settled["cum_hit_rate"] = settled["cum_wins"] / settled["cum_n"]
    settled["cum_roi"] = settled["cum_pnl"] / settled["cum_n"]

    out_cols = list(TIME_SERIES_COLUMNS)

    if rolling_window:
        settled["rolling_hit_rate"] = settled["win"].rolling(rolling_window, min_periods=1).mean()
        settled["rolling_roi"] = settled["pnl_units"].rolling(rolling_window, min_periods=1).mean()
        out_cols += ["rolling_hit_rate", "rolling_roi"]

    return settled[out_cols].reset_index(drop=True)
