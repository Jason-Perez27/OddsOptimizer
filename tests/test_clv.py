"""
Unit tests for src/evaluation/clv.py.

Strategy (matches this project's fixture-based testing convention): pure
DataFrame-in/DataFrame-in-or-out functions, hand-built fixtures, no network,
no file IO, no wall-clock reads.

Covers: resolve_closing_lines' "last genuinely pre-game tick" selection and
stale-close flagging; compute_clv's line-move-vs-price-move branching (the
non-negotiable "never blend across a line move" rule) and its market_agreed
derivation in both branches; clv_summary's per-bucket stats-with-n and the
empty-input shape.

Run with: pytest tests/test_clv.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.underdog_lines import american_to_prob, no_vig_two_way
from src.evaluation import clv


def _p_market(over_american, under_american):
    p_o = american_to_prob(over_american)
    p_u = american_to_prob(under_american)
    return no_vig_two_way(p_o, p_u)


def _tick(over_under_id, poll_at, start_time, line, over_american, under_american,
          live_event=False, game_status="scheduled", pitcher="Gerrit Cole",
          stat_type="strikeouts", game_id="m-1", game_title="NYY @ BOS"):
    return {
        "poll_at": poll_at,
        "over_under_id": over_under_id,
        "projection_id": f"{over_under_id}-proj",
        "player_id": "p-1",
        "pitcher": pitcher,
        "stat_type": stat_type,
        "line": line,
        "over_american": over_american,
        "under_american": under_american,
        "over_payout_multiplier": 0.68,
        "under_payout_multiplier": 1.24,
        "p_over_implied": american_to_prob(over_american),
        "p_under_implied": american_to_prob(under_american),
        "p_market": _p_market(over_american, under_american),
        "over_updated_at": poll_at,
        "under_updated_at": poll_at,
        "game_id": game_id,
        "game_title": game_title,
        "start_time": start_time,
        "game_status": game_status,
        "live_event": live_event,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# resolve_closing_lines
# ---------------------------------------------------------------------------

def test_resolve_closing_lines_picks_last_pre_game_tick():
    ticks = pd.DataFrame([
        _tick("ou-1", "2026-08-23T14:00:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -140, +118),
        _tick("ou-1", "2026-08-23T22:30:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -148, +124),
        # A LIVE tick after first pitch must never be selected as "the close".
        _tick("ou-1", "2026-08-23T23:10:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -200, +160,
              live_event=True, game_status="in_progress"),
    ])

    closing = clv.resolve_closing_lines(ticks)

    assert len(closing) == 1
    row = closing.iloc[0]
    assert row["over_under_id"] == "ou-1"
    assert row["over_american"] == -148  # the 22:30 tick, not the earlier 14:00 or the live 23:10 one
    assert row["close_poll_at"] == "2026-08-23T22:30:00+00:00"
    assert row["minutes_before_first_pitch"] == pytest.approx(35.0)
    assert row["close_quality"] == "good"


def test_resolve_closing_lines_flags_stale_close():
    # Only a very early tick (>60 min before first pitch) is pre-game --
    # nothing else was ever captured for this market before it went live.
    ticks = pd.DataFrame([
        _tick("ou-2", "2026-08-23T14:00:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -140, +118),
        _tick("ou-2", "2026-08-23T23:10:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -200, +160,
              live_event=True, game_status="in_progress"),
    ])

    closing = clv.resolve_closing_lines(ticks)

    assert len(closing) == 1
    row = closing.iloc[0]
    assert row["minutes_before_first_pitch"] > clv.STALE_CLOSE_MINUTES
    assert row["close_quality"] == "stale"


def test_resolve_closing_lines_one_row_per_over_under_id():
    ticks = pd.DataFrame([
        _tick("ou-1", "2026-08-23T14:00:00+00:00", "2026-08-23T23:05:00+00:00", 6.5, -148, +124),
        _tick("ou-2", "2026-08-23T14:00:00+00:00", "2026-08-23T22:10:00+00:00", 1.5, +105, -135,
              pitcher="Framber Valdez", stat_type="walks_allowed", game_id="m-2",
              game_title="HOU @ SEA"),
    ])
    closing = clv.resolve_closing_lines(ticks)
    assert sorted(closing["over_under_id"]) == ["ou-1", "ou-2"]


def test_resolve_closing_lines_empty_input():
    assert clv.resolve_closing_lines(pd.DataFrame()).empty
    assert clv.resolve_closing_lines(None).empty
    assert list(clv.resolve_closing_lines(None).columns) == clv.CLOSING_LINES_COLUMNS


# ---------------------------------------------------------------------------
# compute_clv -- line unchanged (price-move branch)
# ---------------------------------------------------------------------------

def _pick(pitcher_name="Gerrit Cole", line=6.5, lean="over", edge=0.03,
          p_market=None, over_american=-148, under_american=124,
          tier="medium", actionability="lean_over", pitcher=543037, game_pk=1001):
    p_mkt = p_market if p_market is not None else _p_market(over_american, under_american)
    return {
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": pitcher_name,
        "team": "NYY", "start_time": "2026-08-23T23:05:00+00:00", "line": line,
        "line_threshold": 7, "p_over": 0.55, "p_under": 0.45, "tier": tier,
        "lean": lean, "edge": edge, "edge_vs_coinflip": 0.05, "push_mass": 0.0,
        "projection_id": "line-1", "pulled_at": "2026-08-23T14:00:00+00:00",
        "over_american": over_american, "under_american": under_american,
        "over_payout_multiplier": 0.68, "under_payout_multiplier": 1.24,
        "p_over_implied": american_to_prob(over_american),
        "p_under_implied": american_to_prob(under_american),
        "vig": 0.075, "p_market": p_mkt,
        "p_over_lo": 0.50, "p_over_hi": 0.60, "conviction": 1.5,
        "actionability": actionability,
    }


def _close(pitcher="Gerrit Cole", line=6.5, over_american=-160, under_american=138,
           close_quality="good", minutes_before=45.0):
    return {
        "over_under_id": "ou-1", "pitcher": pitcher, "stat_type": "strikeouts",
        "line": line, "over_american": over_american, "under_american": under_american,
        "over_payout_multiplier": 0.62, "under_payout_multiplier": 1.38,
        "p_over_implied": american_to_prob(over_american),
        "p_under_implied": american_to_prob(under_american),
        "p_market": _p_market(over_american, under_american),
        "game_id": "m-1", "game_title": "NYY @ BOS", "start_time": "2026-08-23T23:05:00+00:00",
        "close_poll_at": "2026-08-23T22:00:00+00:00",
        "minutes_before_first_pitch": minutes_before, "close_quality": close_quality,
    }


def test_compute_clv_line_unchanged_price_moved_toward_lean():
    # Open: -148/+124. Close: -160/+138 -- price shortened on the "over" side
    # (lean == "over"), i.e. the market moved TOWARD this pick, line unchanged.
    picks = pd.DataFrame([_pick(lean="over", over_american=-148, under_american=124)])
    closes = pd.DataFrame([_close(line=6.5, over_american=-160, under_american=138)])

    result = clv.compute_clv(picks, closes)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["line_move"] == pytest.approx(0.0)
    assert row["market_agreed"] == "toward"
    assert row["price_move_toward_lean"] > 0
    assert not pd.isna(row["price_move_toward_lean"])


def test_compute_clv_line_unchanged_price_moved_against_lean():
    # Same open, but the close price DRIFTED AWAY from an "over" lean
    # (over got longer, not shorter).
    picks = pd.DataFrame([_pick(lean="over", over_american=-148, under_american=124)])
    closes = pd.DataFrame([_close(line=6.5, over_american=-120, under_american=100)])

    result = clv.compute_clv(picks, closes)
    row = result.iloc[0]
    assert row["market_agreed"] == "against"
    assert row["price_move_toward_lean"] < 0


# ---------------------------------------------------------------------------
# compute_clv -- line moved (direction branch; price_move_toward_lean is NaN)
# ---------------------------------------------------------------------------

def test_compute_clv_line_moved_up_toward_over_lean():
    picks = pd.DataFrame([_pick(lean="over", line=5.5)])
    closes = pd.DataFrame([_close(line=6.5)])  # line moved UP -- toward an "over" lean

    result = clv.compute_clv(picks, closes)
    row = result.iloc[0]
    assert row["line_move"] == pytest.approx(1.0)
    assert row["market_agreed"] == "toward"
    assert pd.isna(row["price_move_toward_lean"])  # never blended across a line move


def test_compute_clv_line_moved_down_against_over_lean():
    picks = pd.DataFrame([_pick(lean="over", line=6.5)])
    closes = pd.DataFrame([_close(line=5.5)])  # line moved DOWN -- against an "over" lean

    result = clv.compute_clv(picks, closes)
    row = result.iloc[0]
    assert row["line_move"] == pytest.approx(-1.0)
    assert row["market_agreed"] == "against"
    assert pd.isna(row["price_move_toward_lean"])


def test_compute_clv_line_moved_mirrors_for_under_lean():
    picks = pd.DataFrame([_pick(lean="under", line=6.5)])
    closes = pd.DataFrame([_close(line=5.5)])  # line moved DOWN -- toward an "under" lean

    result = clv.compute_clv(picks, closes)
    row = result.iloc[0]
    assert row["market_agreed"] == "toward"
    assert pd.isna(row["price_move_toward_lean"])


# ---------------------------------------------------------------------------
# compute_clv -- join fallback, missing market, empty inputs
# ---------------------------------------------------------------------------

def test_compute_clv_joins_by_normalized_pitcher_name_when_no_over_under_id():
    assert "over_under_id" not in pd.DataFrame([_pick()]).columns
    picks = pd.DataFrame([_pick(pitcher_name="Gerrit  Cole")])  # extra whitespace
    closes = pd.DataFrame([_close(pitcher="gerrit cole")])       # different case

    result = clv.compute_clv(picks, closes)
    assert len(result) == 1


def test_compute_clv_no_market_at_open_line_unchanged_is_nan_and_unchanged():
    picks = pd.DataFrame([_pick(p_market=np.nan, over_american=np.nan, under_american=np.nan)])
    closes = pd.DataFrame([_close(line=6.5)])

    result = clv.compute_clv(picks, closes)
    row = result.iloc[0]
    assert row["line_move"] == pytest.approx(0.0)
    assert pd.isna(row["price_move_toward_lean"])
    assert row["market_agreed"] == "unchanged"


def test_compute_clv_empty_inputs():
    assert clv.compute_clv(pd.DataFrame(), pd.DataFrame()).empty
    assert clv.compute_clv(None, None).empty
    assert list(clv.compute_clv(None, None).columns) == clv.CLV_COLUMNS


# ---------------------------------------------------------------------------
# clv_summary
# ---------------------------------------------------------------------------

def _clv_row(tier="medium", actionability="lean_over", edge_open=0.05,
             market_agreed="toward", line_move=0.0, price_move_toward_lean=0.02,
             close_quality="good"):
    return {
        "pitcher": 1, "game_pk": 1, "pitcher_name": "X", "team": "NYY",
        "start_time": "2026-08-23T23:05:00+00:00", "tier": tier, "lean": "over",
        "actionability": actionability, "edge_open": edge_open,
        "line_open": 6.5, "line_close": 6.5 + line_move, "line_move": line_move,
        "p_market_open": 0.5, "p_market_close": 0.52,
        "price_move_toward_lean": price_move_toward_lean if line_move == 0 else np.nan,
        "market_agreed": market_agreed,
        "close_quality": close_quality, "minutes_before_first_pitch": 45.0,
    }


def test_clv_summary_overall_stats_and_n():
    df = pd.DataFrame([
        _clv_row(market_agreed="toward", line_move=0.0, price_move_toward_lean=0.02),
        _clv_row(market_agreed="against", line_move=0.0, price_move_toward_lean=-0.01),
        _clv_row(market_agreed="toward", line_move=1.0),  # line moved -- no price_move contribution
    ])

    summary = clv.clv_summary(df)

    assert summary["n_total"] == 3
    assert summary["n_stale_excluded"] == 0
    overall = summary["overall"]
    assert overall["n"] == 3
    assert overall["pct_market_agreed"]["n"] == 3
    assert overall["pct_market_agreed"]["value"] == pytest.approx(2 / 3)
    # mean_price_move_toward_lean restricted to line_move == 0 rows only (n=2).
    assert overall["mean_price_move_toward_lean"]["n"] == 2
    assert overall["mean_price_move_toward_lean"]["value"] == pytest.approx((0.02 + -0.01) / 2)
    assert overall["mean_abs_line_move"]["n"] == 3
    assert overall["mean_abs_line_move"]["value"] == pytest.approx((0 + 0 + 1) / 3)


def test_clv_summary_excludes_stale_from_every_stat():
    df = pd.DataFrame([
        _clv_row(market_agreed="toward", close_quality="good"),
        _clv_row(market_agreed="against", close_quality="stale"),
    ])
    summary = clv.clv_summary(df)
    assert summary["n_total"] == 2
    assert summary["n_stale_excluded"] == 1
    assert summary["overall"]["n"] == 1
    assert summary["overall"]["pct_market_agreed"]["value"] == pytest.approx(1.0)


def test_clv_summary_breaks_out_by_tier_and_actionability():
    df = pd.DataFrame([
        _clv_row(tier="high", actionability="lean_over", market_agreed="toward"),
        _clv_row(tier="high", actionability="lean_over", market_agreed="toward"),
        _clv_row(tier="low", actionability="no_action", market_agreed="against"),
    ])
    summary = clv.clv_summary(df)

    assert summary["by_tier"]["high"]["n"] == 2
    assert summary["by_tier"]["low"]["n"] == 1
    assert summary["by_actionability"]["lean_over"]["pct_market_agreed"]["value"] == pytest.approx(1.0)
    assert summary["by_actionability"]["no_action"]["pct_market_agreed"]["value"] == pytest.approx(0.0)


def test_clv_summary_edge_quartile_excludes_unknown_edge():
    df = pd.DataFrame([
        _clv_row(edge_open=0.02), _clv_row(edge_open=0.04),
        _clv_row(edge_open=0.06), _clv_row(edge_open=0.08),
        _clv_row(edge_open=np.nan),
    ])
    summary = clv.clv_summary(df)
    assert summary["n_edge_unknown_excluded_from_quartiles"] == 1
    total_bucketed = sum(b["n"] for b in summary["by_edge_quartile"].values())
    assert total_bucketed == 4


def test_clv_summary_empty_input_shape():
    summary = clv.clv_summary(pd.DataFrame())
    assert summary["n_total"] == 0
    assert summary["n_stale_excluded"] == 0
    assert summary["overall"]["pct_market_agreed"] == {"value": None, "n": 0}
    assert summary["overall"]["mean_price_move_toward_lean"] == {"value": None, "n": 0}
    assert summary["overall"]["mean_abs_line_move"] == {"value": None, "n": 0}
    assert summary["by_tier"] == {}
    assert summary["by_actionability"] == {}
    assert summary["by_edge_quartile"] == {}
