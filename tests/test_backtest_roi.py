"""
Unit tests for src/backtest/roi.py (task #10, spec section 4(b) + the
"Testing approach" list).

Covers spec testing-approach items:
  4. Hit rate excludes pushes and pending/unsettled rows from the denominator.
  5. By-tier / by-threshold(line) aggregations are correct on a
     hand-checkable fixture.
  8. Flat-bet ROI proxy: wins/losses/pushes map to the right units, pushes
     refund as 0 and are excluded from the denominator, unsettled excluded
     from both numerator and denominator.
  9. Over-time series: cumulative and rolling ROI/hit-rate series have the
     right length and cumulative values on an ordered fixture.

Run with: pytest tests/test_backtest_roi.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import roi


def _row(pick_correct, pnl_units, tier="low", line=6.5, game_date="2026-06-20"):
    return {
        "pick_correct": pick_correct,
        "pnl_units": pnl_units,
        "tier": tier,
        "line": line,
        "game_date": game_date,
    }


# ---------------------------------------------------------------------------
# 4. Hit rate excludes pushes and unsettled
# ---------------------------------------------------------------------------

def test_hit_rate_excludes_pushes_and_unsettled():
    df = pd.DataFrame([
        _row(True, 1),     # win
        _row(True, 1),     # win
        _row(False, -1),   # loss
        _row(np.nan, 0),   # push -- pick_correct NaN, must be excluded
        _row(np.nan, np.nan),  # unsettled -- must be excluded
    ])

    result = roi.hit_rate(df)

    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["n"] == 3
    assert result["hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_all_unsettled_is_nan():
    df = pd.DataFrame([_row(np.nan, np.nan), _row(np.nan, 0)])
    result = roi.hit_rate(df)
    assert result["n"] == 0
    assert result["wins"] == 0
    assert np.isnan(result["hit_rate"])


def test_hit_rate_empty_df():
    df = pd.DataFrame(columns=["pick_correct", "pnl_units", "tier", "line", "game_date"])
    result = roi.hit_rate(df)
    assert result["n"] == 0
    assert np.isnan(result["hit_rate"])


# ---------------------------------------------------------------------------
# 8. Flat-bet ROI proxy
# ---------------------------------------------------------------------------

def test_flat_bet_roi_hand_computed():
    # 2 wins (+1 each), 1 loss (-1), 1 push (0, excluded from denom),
    # 1 unsettled (NaN, excluded from both).
    df = pd.DataFrame([
        _row(True, 1),
        _row(True, 1),
        _row(False, -1),
        _row(np.nan, 0),       # push
        _row(np.nan, np.nan),  # unsettled
    ])

    result = roi.flat_bet_roi(df)

    # total_pnl sums all settled rows (push contributes 0): 1+1-1+0 = 1
    assert result["total_pnl"] == pytest.approx(1.0)
    # denominator is settled non-push only: 3 (2 wins + 1 loss)
    assert result["n_settled_non_push"] == 3
    assert result["roi"] == pytest.approx(1.0 / 3)


def test_flat_bet_roi_all_losses_is_negative_one():
    df = pd.DataFrame([_row(False, -1), _row(False, -1)])
    result = roi.flat_bet_roi(df)
    assert result["roi"] == pytest.approx(-1.0)


def test_flat_bet_roi_no_settled_non_push_is_nan():
    df = pd.DataFrame([_row(np.nan, 0), _row(np.nan, np.nan)])  # push + unsettled only
    result = roi.flat_bet_roi(df)
    assert result["n_settled_non_push"] == 0
    assert np.isnan(result["roi"])


# ---------------------------------------------------------------------------
# 5. By-tier / by-line aggregations
# ---------------------------------------------------------------------------

def test_by_tier_hand_computed():
    df = pd.DataFrame([
        _row(True, 1, tier="high"),
        _row(False, -1, tier="high"),
        _row(True, 1, tier="low"),
        _row(True, 1, tier="low"),
        _row(True, 1, tier="low"),
    ])

    table = roi.by_tier(df).set_index("tier")

    assert table.loc["high", "n_settled"] == 2
    assert table.loc["high", "hit_rate"] == pytest.approx(0.5)
    assert table.loc["high", "roi"] == pytest.approx(0.0)  # (1-1)/2

    assert table.loc["low", "n_settled"] == 3
    assert table.loc["low", "hit_rate"] == pytest.approx(1.0)
    assert table.loc["low", "roi"] == pytest.approx(1.0)


def test_by_line_includes_group_with_zero_settled():
    df = pd.DataFrame([
        _row(True, 1, line=6.5),
        _row(np.nan, np.nan, line=7.5),  # this line has zero settled picks
    ])

    table = roi.by_line(df).set_index("line")

    assert table.loc[6.5, "n_settled"] == 1
    assert table.loc[7.5, "n_settled"] == 0
    assert np.isnan(table.loc[7.5, "hit_rate"])


def test_by_tier_empty_df():
    df = pd.DataFrame(columns=["pick_correct", "pnl_units", "tier", "line", "game_date"])
    table = roi.by_tier(df)
    assert table.empty
    assert "tier" in table.columns


# ---------------------------------------------------------------------------
# 9. Over-time series: cumulative + rolling
# ---------------------------------------------------------------------------

def test_time_series_cumulative_values_on_ordered_fixture():
    # Deliberately out of order in the input -- time_series must sort by date.
    df = pd.DataFrame([
        _row(True, 1, game_date="2026-06-22"),   # day 3: win
        _row(True, 1, game_date="2026-06-20"),   # day 1: win
        _row(False, -1, game_date="2026-06-21"), # day 2: loss
        _row(np.nan, 0, game_date="2026-06-23"), # day 4: push -- excluded
        _row(True, 1, game_date="2026-06-24"),   # day 5: win
    ])

    series = roi.time_series(df, date_col="game_date")

    # push row excluded -> only 4 rows (the settled non-push ones)
    assert len(series) == 4
    assert list(series["game_date"]) == ["2026-06-20", "2026-06-21", "2026-06-22", "2026-06-24"]

    assert list(series["cum_n"]) == [1, 2, 3, 4]
    assert list(series["cum_wins"]) == [1, 1, 2, 3]
    assert list(series["cum_pnl"]) == [1, 0, 1, 2]
    assert series["cum_hit_rate"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 3 / 4])
    assert series["cum_roi"].tolist() == pytest.approx([1.0, 0.0, 1 / 3, 2 / 4])


def test_time_series_rolling_window_length_and_values():
    df = pd.DataFrame([
        _row(True, 1, game_date="2026-06-20"),
        _row(False, -1, game_date="2026-06-21"),
        _row(True, 1, game_date="2026-06-22"),
        _row(False, -1, game_date="2026-06-23"),
    ])

    series = roi.time_series(df, date_col="game_date", rolling_window=2)

    assert len(series) == 4
    assert "rolling_hit_rate" in series.columns
    assert "rolling_roi" in series.columns
    # rolling window=2, min_periods=1:
    # row0: window=[win] -> hit_rate=1.0, roi=1.0
    # row1: window=[win,loss] -> hit_rate=0.5, roi=0.0
    # row2: window=[loss,win] -> hit_rate=0.5, roi=0.0
    # row3: window=[win,loss] -> hit_rate=0.5, roi=0.0
    assert series["rolling_hit_rate"].tolist() == pytest.approx([1.0, 0.5, 0.5, 0.5])
    assert series["rolling_roi"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_time_series_excludes_unsettled_and_pushes():
    df = pd.DataFrame([
        _row(True, 1, game_date="2026-06-20"),
        _row(np.nan, np.nan, game_date="2026-06-21"),  # unsettled
        _row(np.nan, 0, game_date="2026-06-22"),       # push
    ])

    series = roi.time_series(df, date_col="game_date")

    assert len(series) == 1
    assert series.iloc[0]["game_date"] == "2026-06-20"


def test_time_series_empty_input():
    df = pd.DataFrame(columns=["pick_correct", "pnl_units", "tier", "line", "game_date"])
    series = roi.time_series(df, date_col="game_date")
    assert series.empty
    assert "cum_n" in series.columns


# ---------------------------------------------------------------------------
# Payout-aware real EV (2026-08 Underdog migration)
# ---------------------------------------------------------------------------

def _payout_row(pick_correct, pnl_units, lean="over", over_mult=None, under_mult=None,
                 tier="low", line=6.5, game_date="2026-08-20"):
    return {
        "pick_correct": pick_correct, "pnl_units": pnl_units, "lean": lean,
        "over_payout_multiplier": over_mult, "under_payout_multiplier": under_mult,
        "tier": tier, "line": line, "game_date": game_date,
    }


def test_flat_bet_roi_uses_payout_multiplier_on_the_chosen_side_for_a_win():
    # Win, leaning over, with a real payout multiplier of 0.71 (Underdog
    # convention: profit on a winning unit stake) -- real EV pnl is +0.71,
    # not the old flat +1.
    df = pd.DataFrame([_payout_row(True, 1, lean="over", over_mult=0.71, under_mult=1.35)])
    result = roi.flat_bet_roi(df)
    assert result["total_pnl"] == pytest.approx(0.71)
    assert result["roi"] == pytest.approx(0.71)
    assert result["roi_real_ev"]["n_settled_non_push"] == 1
    assert result["roi_real_ev"]["roi"] == pytest.approx(0.71)
    assert result["roi_flat_fallback"]["n_settled_non_push"] == 0


def test_flat_bet_roi_uses_the_under_side_multiplier_when_the_lean_is_under():
    df = pd.DataFrame([_payout_row(True, 1, lean="under", over_mult=0.71, under_mult=1.35)])
    result = roi.flat_bet_roi(df)
    assert result["total_pnl"] == pytest.approx(1.35)


def test_flat_bet_roi_loss_always_costs_the_full_stake_regardless_of_multiplier():
    df = pd.DataFrame([_payout_row(False, -1, lean="over", over_mult=0.71, under_mult=1.35)])
    result = roi.flat_bet_roi(df)
    assert result["total_pnl"] == pytest.approx(-1.0)
    assert result["roi_real_ev"]["n_settled_non_push"] == 1  # tagged real_ev: a multiplier was available


def test_flat_bet_roi_falls_back_to_flat_for_rows_with_no_multiplier():
    # Pre-migration historical row: no payout multiplier at all.
    df = pd.DataFrame([_payout_row(True, 1, lean="over", over_mult=None, under_mult=None)])
    result = roi.flat_bet_roi(df)
    assert result["total_pnl"] == pytest.approx(1.0)  # flat fallback, not a crash/NaN
    assert result["roi_flat_fallback"]["n_settled_non_push"] == 1
    assert result["roi_real_ev"]["n_settled_non_push"] == 0


def test_flat_bet_roi_does_not_silently_mix_pre_and_post_migration_rows():
    """A mix of a pre-migration (flat) row and a post-migration (real EV) row:
    the blended top-level total combines both, but the per-source breakdowns
    isolate each so a report can tell them apart."""
    df = pd.DataFrame([
        _payout_row(True, 1, lean="over", over_mult=None, under_mult=None,  # pre-migration
                    game_date="2026-06-20"),
        _payout_row(True, 1, lean="over", over_mult=0.71, under_mult=1.35,  # post-migration
                    game_date="2026-08-20"),
    ])
    result = roi.flat_bet_roi(df)

    assert result["roi_flat_fallback"]["n_settled_non_push"] == 1
    assert result["roi_flat_fallback"]["total_pnl"] == pytest.approx(1.0)
    assert result["roi_real_ev"]["n_settled_non_push"] == 1
    assert result["roi_real_ev"]["total_pnl"] == pytest.approx(0.71)

    # Blended top-level total is the honest sum across both eras.
    assert result["total_pnl"] == pytest.approx(1.71)
    assert result["n_settled_non_push"] == 2


def test_flat_bet_roi_push_is_source_agnostic_and_contributes_zero():
    df = pd.DataFrame([_payout_row(np.nan, 0, lean="over", over_mult=0.71, under_mult=1.35)])
    result = roi.flat_bet_roi(df)
    assert result["total_pnl"] == pytest.approx(0.0)
    assert result["n_settled_non_push"] == 0  # push excluded from the denominator


def test_time_series_carries_pnl_source_and_uses_real_ev_pnl():
    df = pd.DataFrame([
        _payout_row(True, 1, lean="over", over_mult=0.71, under_mult=1.35, game_date="2026-08-20"),
        _payout_row(False, -1, lean="over", over_mult=0.71, under_mult=1.35, game_date="2026-08-21"),
    ])
    series = roi.time_series(df, date_col="game_date")
    assert "pnl_source" in series.columns
    assert list(series["pnl_source"]) == ["real_ev", "real_ev"]
    assert list(series["pnl_units"]) == pytest.approx([0.71, -1.0])
    assert list(series["cum_pnl"]) == pytest.approx([0.71, -0.29])
