"""
Unit tests for src/data/underdog_lines.py.

Strategy (matches tests/test_prizepicks_lines.py's conventions): no real
network calls -- requests.get is mocked throughout. Underdog's endpoint is
public but unofficial, so these tests pin down our own join/parsing logic
against a hand-built fixture payload matching the verified contract (see the
module docstring), not against the live API.

Covers: the appearance_id -> appearances -> players -> games join path, both
options (higher/lower) parsed, the runs_allowed/"Earned Runs Allowed" naming
trap, game_title "AWAY @ HOME" splitting, and empty/missing-stat handling.

Run with: pytest tests/test_underdog_lines.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import underdog_lines


# ---------------------------------------------------------------------------
# fetch_over_under_lines
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_fetch_over_under_lines_sends_correct_params(monkeypatch):
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(200, json_data={"over_under_lines": []})

    monkeypatch.setattr(underdog_lines.requests, "get", fake_get)

    payload = underdog_lines.fetch_over_under_lines(sport_id="MLB")

    assert captured["url"] == "https://api.underdogfantasy.com/beta/v6/over_under_lines"
    assert captured["params"]["sport_id"] == "MLB"
    assert "User-Agent" in captured["headers"]
    assert payload == {"over_under_lines": []}


def test_fetch_over_under_lines_raises_clear_error_on_403(monkeypatch):
    monkeypatch.setattr(underdog_lines.requests, "get", lambda *a, **k: _FakeResponse(403))
    with pytest.raises(RuntimeError, match="rejected the request"):
        underdog_lines.fetch_over_under_lines()


# ---------------------------------------------------------------------------
# flatten_lines -- fixture payload matching the verified contract
# ---------------------------------------------------------------------------

def _sample_payload():
    """
    Two lines: a strikeouts line for Gerrit Cole (NYY @ BOS) and a
    runs_allowed line for Framber Valdez (HOU @ SEA) -- deliberately covers
    the "key != display name" trap (runs_allowed's display_stat is "Earned
    Runs Allowed").
    """
    return {
        "over_under_lines": [
            {
                "id": "line-1",
                "over_under_id": "ou-1",
                "stat_value": "6.5",
                "line_type": "balanced",
                "live_event": False,
                "status": "active",
                "over_under": {
                    "appearance_stat": {
                        "appearance_id": "app-1",
                        "stat": "strikeouts",
                        "display_stat": "Strikeouts",
                    },
                    "has_alternates": True,
                },
                "options": [
                    {"choice": "higher", "american_price": "-148", "decimal_price": 1.68,
                     "payout_multiplier": "0.68", "status": "active",
                     "selection_header": "Gerrit Cole", "selection_subheader": "Higher"},
                    {"choice": "lower", "american_price": "+124", "decimal_price": 2.24,
                     "payout_multiplier": "1.24", "status": "active",
                     "selection_header": "Gerrit Cole", "selection_subheader": "Lower"},
                ],
            },
            {
                "id": "line-2",
                "over_under_id": "ou-2",
                "stat_value": "0.5",
                "line_type": "balanced",
                "live_event": False,
                "status": "active",
                "over_under": {
                    "appearance_stat": {
                        "appearance_id": "app-2",
                        "stat": "runs_allowed",
                        "display_stat": "Earned Runs Allowed",
                    },
                    "has_alternates": False,
                },
                "options": [
                    {"choice": "higher", "american_price": "+105", "decimal_price": 2.05,
                     "payout_multiplier": "1.05", "status": "active",
                     "selection_header": "Framber Valdez", "selection_subheader": "Higher"},
                    {"choice": "lower", "american_price": "-135", "decimal_price": 1.74,
                     "payout_multiplier": "0.74", "status": "active",
                     "selection_header": "Framber Valdez", "selection_subheader": "Lower"},
                ],
            },
        ],
        "appearances": [
            {"id": "app-1", "player_id": "p-1", "match_id": "m-1", "team_id": "t-away-uuid"},
            {"id": "app-2", "player_id": "p-2", "match_id": "m-2", "team_id": "t-home-uuid"},
        ],
        "players": [
            {"id": "p-1", "first_name": "Gerrit", "last_name": "Cole"},
            {"id": "p-2", "first_name": "Framber", "last_name": "Valdez"},
        ],
        "games": [
            {"id": "m-1", "sport_id": "MLB", "title": "NYY @ BOS",
             "scheduled_at": "2026-08-23T23:05:00Z", "status": "scheduled"},
            {"id": "m-2", "sport_id": "MLB", "title": "HOU @ SEA",
             "scheduled_at": "2026-08-23T22:10:00Z", "status": "scheduled"},
        ],
        "solo_games": [],
    }


def test_flatten_lines_joins_and_parses_both_options():
    df = underdog_lines.flatten_lines(_sample_payload(), stat="strikeouts")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["projection_id"] == "line-1"
    assert row["over_under_id"] == "ou-1"
    assert row["player_id"] == "p-1"
    assert row["pitcher"] == "Gerrit Cole"
    assert pd.isna(row["team"])  # never in this feed -- caller resolves it
    assert row["stat_type"] == "strikeouts"
    assert row["line"] == 6.5
    assert row["line_type"] == "balanced"

    # Both options parsed and coerced numeric.
    assert row["over_american"] == -148.0
    assert row["under_american"] == 124.0
    assert row["over_payout_multiplier"] == pytest.approx(0.68)
    assert row["under_payout_multiplier"] == pytest.approx(1.24)
    assert row["over_status"] == "active"
    assert row["under_status"] == "active"


def test_flatten_lines_game_title_splitting():
    df = underdog_lines.flatten_lines(_sample_payload(), stat="strikeouts")
    row = df.iloc[0]
    assert row["game_title"] == "NYY @ BOS"
    assert row["away_team"] == "NYY"
    assert row["home_team"] == "BOS"
    assert row["start_time"] == "2026-08-23T23:05:00Z"
    assert row["game_status"] == "scheduled"
    assert row["live_event"] == False  # noqa: E712 -- explicit bool check, not truthiness
    assert row["status"] == "active"


def test_flatten_lines_runs_allowed_key_vs_display_name_trap():
    """
    The Underdog stat KEY is "runs_allowed" but its display_stat is
    "Earned Runs Allowed" -- filtering must match the key, never the
    display name, and the wrong ("earned_runs_allowed") key must find nothing.
    """
    df = underdog_lines.flatten_lines(_sample_payload(), stat="runs_allowed")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pitcher"] == "Framber Valdez"
    assert row["line"] == 0.5

    wrong_key_df = underdog_lines.flatten_lines(_sample_payload(), stat="earned_runs_allowed")
    assert wrong_key_df.empty


def test_flatten_lines_missing_stat_returns_empty_frame():
    df = underdog_lines.flatten_lines(_sample_payload(), stat="hits_allowed")
    assert df.empty


def test_flatten_lines_empty_payload_returns_empty_frame():
    empty_payload = {"over_under_lines": [], "appearances": [], "players": [], "games": [], "solo_games": []}
    df = underdog_lines.flatten_lines(empty_payload, stat="strikeouts")
    assert df.empty


def test_flatten_lines_handles_missing_top_level_keys_gracefully():
    """A malformed/partial payload (e.g. missing 'players') must not crash."""
    df = underdog_lines.flatten_lines({}, stat="strikeouts")
    assert df.empty


def test_flatten_lines_filters_by_sport_id():
    payload = _sample_payload()
    payload["games"][0]["sport_id"] = "NBA"  # Cole's game is not MLB any more
    df = underdog_lines.flatten_lines(payload, stat="strikeouts", sport_id="MLB")
    assert df.empty


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def test_american_to_prob_negative_odds():
    # -148 -> 148 / 248
    assert underdog_lines.american_to_prob(-148) == pytest.approx(148 / 248)


def test_american_to_prob_positive_odds():
    # +124 -> 100 / 224
    assert underdog_lines.american_to_prob(124) == pytest.approx(100 / 224)


def test_no_vig_two_way_normalizes_to_sum_one_pair():
    p_over_implied = underdog_lines.american_to_prob(-148)
    p_under_implied = underdog_lines.american_to_prob(124)
    # Raw implied probabilities sum to > 1.0 (the vig).
    assert p_over_implied + p_under_implied > 1.0

    p_market = underdog_lines.no_vig_two_way(p_over_implied, p_under_implied)
    p_market_under = underdog_lines.no_vig_two_way(p_under_implied, p_over_implied)
    assert p_market + p_market_under == pytest.approx(1.0)
    assert 0.0 < p_market < 1.0


def test_payout_to_decimal_is_one_plus_multiplier():
    assert underdog_lines.payout_to_decimal("0.89") == pytest.approx(1.89)
    assert underdog_lines.payout_to_decimal(1.24) == pytest.approx(2.24)
