"""
Unit tests for src/evaluation/grading.py (task #10, spec section 2 + the
"Testing approach" list in
docs/design/specs/2026-06-27-outcome-tracking-design.md).

All fixtures are hand-built, no network, no real Statcast pulls -- matching
the rest of this repo's test style (tiering.py / refresh.py conventions).

Covers spec testing-approach items:
  1. grade_line_picks correctness (under-win, over-loss, integer-line push).
  2. Settlement status: pending within wait window, void_scratched past
     scratch_void_max_wait_days, settled when a realized row exists --
     driven entirely by the injected `now`.
  3. Join is on (pitcher, game_pk): a doubleheader grades independently;
     a game_date-only join would be caught failing.
  14. Strikeout-count grading is unconditional on game length: a short,
      low-K start grades as a normal `under`, not a void.

Run with: pytest tests/test_grading.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import grading


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _line_pick(pitcher=1, game_pk=100, line=6.5, lean="over"):
    return {
        "pitcher": pitcher,
        "game_pk": game_pk,
        "pitcher_name": "Test Pitcher",
        "team": "NYY",
        "start_time": "2026-06-26T18:00:00Z",
        "line": line,
        "line_threshold": int(np.floor(line)) + 1,
        "p_over": 0.55,
        "p_under": 0.45,
        "tier": "low",
        "lean": lean,
        "edge": 0.05,
        "push_mass": 0.0,
        "projection_id": "proj-1",
        "pulled_at": "2026-06-26T12:00:00Z",
    }


def _threshold_row(pitcher=1, game_pk=100, game_date="2026-06-26", threshold=6, tier="low"):
    return {
        "pitcher": pitcher,
        "game_pk": game_pk,
        "pitcher_name": "Test Pitcher",
        "team": "NYY",
        "opponent_team": "BOS",
        "game_date": game_date,
        "threshold": threshold,
        "p_over": 0.55,
        "tier": tier,
    }


def _realized(pitcher=1, game_pk=100, strikeouts=7):
    return {"pitcher": pitcher, "game_pk": game_pk, "strikeouts": strikeouts}


# ---------------------------------------------------------------------------
# 1. grade_line_picks correctness
# ---------------------------------------------------------------------------

def test_grade_line_picks_under_win():
    # lean=under, line=6.5 (threshold 7), realized=4 -> under is correct, win.
    picks = pd.DataFrame([_line_pick(line=6.5, lean="under")])
    realized = pd.DataFrame([_realized(strikeouts=4)])

    graded = grading.grade_line_picks(picks, realized)

    row = graded.iloc[0]
    assert row["over_hit"] == False
    assert row["push"] == False
    assert row["pick_correct"] == True
    assert row["pnl_units"] == 1


def test_grade_line_picks_over_loss():
    # lean=over, line=6.5 (threshold 7), realized=4 -> over_hit False, loss.
    picks = pd.DataFrame([_line_pick(line=6.5, lean="over")])
    realized = pd.DataFrame([_realized(strikeouts=4)])

    graded = grading.grade_line_picks(picks, realized)

    row = graded.iloc[0]
    assert row["over_hit"] == False
    assert row["push"] == False
    assert row["pick_correct"] == False
    assert row["pnl_units"] == -1


def test_grade_line_picks_integer_line_push():
    # Integer line=6, realized==6 -> push, pnl 0, pick_correct NaN.
    picks = pd.DataFrame([_line_pick(line=6.0, lean="over")])
    realized = pd.DataFrame([_realized(strikeouts=6)])

    graded = grading.grade_line_picks(picks, realized)

    row = graded.iloc[0]
    assert row["push"] == True
    assert row["pnl_units"] == 0
    assert pd.isna(row["pick_correct"])


def test_grade_line_picks_unsettled_is_all_nan_except_push():
    # No realized row at all -> over_hit/pick_correct/pnl_units all NaN,
    # push stays a real False (never NaN -- per output schema).
    picks = pd.DataFrame([_line_pick(line=6.5, lean="over")])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    graded = grading.grade_line_picks(picks, realized)

    row = graded.iloc[0]
    assert pd.isna(row["realized_strikeouts"])
    assert pd.isna(row["over_hit"])
    assert row["push"] == False
    assert pd.isna(row["pick_correct"])
    assert pd.isna(row["pnl_units"])


# ---------------------------------------------------------------------------
# 2. Settlement status, driven by injected now
# ---------------------------------------------------------------------------

def test_attach_outcomes_settled_when_realized_present():
    preds = pd.DataFrame([_threshold_row(game_date="2026-06-20")])
    realized = pd.DataFrame([_realized(strikeouts=7)])

    out = grading.attach_outcomes(
        preds, realized, now="2026-06-21", scratch_void_max_wait_days=3,
    )

    assert out.iloc[0]["settlement_status"] == grading.SETTLED
    assert out.iloc[0]["realized_strikeouts"] == 7


def test_attach_outcomes_pending_within_wait_window():
    preds = pd.DataFrame([_threshold_row(game_date="2026-06-26")])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    # Only 1 day elapsed, wait window is 3 days -> still pending.
    out = grading.attach_outcomes(
        preds, realized, now="2026-06-27", scratch_void_max_wait_days=3,
    )

    assert out.iloc[0]["settlement_status"] == grading.PENDING
    assert pd.isna(out.iloc[0]["realized_strikeouts"])


def test_attach_outcomes_void_scratched_past_max_wait():
    preds = pd.DataFrame([_threshold_row(game_date="2026-06-20")])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    # 7 days elapsed, wait window is 3 days -> void_scratched.
    out = grading.attach_outcomes(
        preds, realized, now="2026-06-27", scratch_void_max_wait_days=3,
    )

    assert out.iloc[0]["settlement_status"] == grading.VOID_SCRATCHED


def test_attach_outcomes_status_flips_as_now_advances():
    # Same fixture, only `now` changes -- pending today, void a week later.
    preds = pd.DataFrame([_threshold_row(game_date="2026-06-26")])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    soon = grading.attach_outcomes(
        preds, realized, now="2026-06-27", scratch_void_max_wait_days=3,
    )
    later = grading.attach_outcomes(
        preds, realized, now="2026-07-04", scratch_void_max_wait_days=3,
    )

    assert soon.iloc[0]["settlement_status"] == grading.PENDING
    assert later.iloc[0]["settlement_status"] == grading.VOID_SCRATCHED


# ---------------------------------------------------------------------------
# 3. Join is on (pitcher, game_pk) -- doubleheader safety
# ---------------------------------------------------------------------------

def test_grade_threshold_sweep_doubleheader_grades_independently():
    # Same pitcher, same game_date, two different game_pk values (a
    # doubleheader) -- each game's realized strikeouts must land on the
    # correct row, never cross-joined just because game_date matches.
    sweep = pd.DataFrame([
        _threshold_row(pitcher=1, game_pk=100, game_date="2026-06-26", threshold=6),
        _threshold_row(pitcher=1, game_pk=200, game_date="2026-06-26", threshold=6),
    ])
    realized = pd.DataFrame([
        _realized(pitcher=1, game_pk=100, strikeouts=8),  # game 1: over 6
        _realized(pitcher=1, game_pk=200, strikeouts=3),  # game 2: under 6
    ])

    graded = grading.grade_threshold_sweep(sweep, realized)

    game1 = graded[graded["game_pk"] == 100].iloc[0]
    game2 = graded[graded["game_pk"] == 200].iloc[0]
    assert game1["realized_strikeouts"] == 8
    assert game1["over_hit"] == True
    assert game2["realized_strikeouts"] == 3
    assert game2["over_hit"] == False


def test_grade_line_picks_doubleheader_grades_independently():
    picks = pd.DataFrame([
        _line_pick(pitcher=1, game_pk=100, line=6.5, lean="over"),
        _line_pick(pitcher=1, game_pk=200, line=6.5, lean="over"),
    ])
    realized = pd.DataFrame([
        _realized(pitcher=1, game_pk=100, strikeouts=8),
        _realized(pitcher=1, game_pk=200, strikeouts=2),
    ])

    graded = grading.grade_line_picks(picks, realized)

    game1 = graded[graded["game_pk"] == 100].iloc[0]
    game2 = graded[graded["game_pk"] == 200].iloc[0]
    assert game1["pick_correct"] == True
    assert game2["pick_correct"] == False


def test_attach_outcomes_requires_game_pk():
    preds = pd.DataFrame([{"pitcher": 1, "game_date": "2026-06-26"}])  # no game_pk
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    with pytest.raises(ValueError, match="game_pk"):
        grading.attach_outcomes(preds, realized, now="2026-06-27", scratch_void_max_wait_days=3)


def test_grade_line_picks_requires_game_pk():
    picks = pd.DataFrame([{"pitcher": 1, "line": 6.5, "lean": "over", "line_threshold": 7}])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    with pytest.raises(ValueError, match="game_pk"):
        grading.grade_line_picks(picks, realized)


# ---------------------------------------------------------------------------
# 14. Strikeout-count grading is unconditional on game length
# ---------------------------------------------------------------------------

def test_short_low_k_start_grades_as_normal_under_not_void():
    # A short start (e.g. pulled after 2 IP, 1 K) is a legitimate "under" --
    # realized=1 is a real Statcast row, so settlement_status is settled,
    # never void_scratched, regardless of how few strikeouts were recorded.
    preds = pd.DataFrame([_threshold_row(game_date="2026-06-20", threshold=6)])
    realized = pd.DataFrame([_realized(strikeouts=1)])

    out = grading.attach_outcomes(
        preds, realized, now="2026-06-27", scratch_void_max_wait_days=3,
    )

    assert out.iloc[0]["settlement_status"] == grading.SETTLED
    assert out.iloc[0]["realized_strikeouts"] == 1

    sweep = pd.DataFrame([_threshold_row(game_date="2026-06-20", threshold=6)])
    graded = grading.grade_threshold_sweep(sweep, realized)
    assert graded.iloc[0]["over_hit"] == False  # legitimate under, not NaN/void


# ---------------------------------------------------------------------------
# Empty-input handling
# ---------------------------------------------------------------------------

def test_attach_outcomes_empty_pred_df():
    preds = pd.DataFrame(columns=["pitcher", "game_pk", "game_date"])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    out = grading.attach_outcomes(preds, realized, now="2026-06-27", scratch_void_max_wait_days=3)
    assert out.empty
    assert "settlement_status" in out.columns


def test_grade_line_picks_empty_input():
    picks = pd.DataFrame(columns=[
        "pitcher", "game_pk", "line", "line_threshold", "lean",
    ])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    out = grading.grade_line_picks(picks, realized)
    assert out.empty
    assert "over_hit" in out.columns
    assert "push" in out.columns


def test_grade_threshold_sweep_empty_input():
    sweep = pd.DataFrame(columns=["pitcher", "game_pk", "threshold"])
    realized = pd.DataFrame(columns=["pitcher", "game_pk", "strikeouts"])

    out = grading.grade_threshold_sweep(sweep, realized)
    assert out.empty
    assert "over_hit" in out.columns
