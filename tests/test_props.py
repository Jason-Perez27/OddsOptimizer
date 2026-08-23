"""
Tests for src/props.py (Phase 1 of prop expansion).

Verifies:
- PROP_REGISTRY contains the three expected props with correct fields.
- get_prop() resolves known keys and raises KeyError for unknown ones.
- DEFAULT_PROP is "strikeouts" (backward-compat guarantee).
- Prop fields are internally consistent (label_source, statcast_event, etc.).
- The registry refactor is behaviour-preserving: the strikeout pipeline
  reads the same model path, stat_type, and thresholds as the hardcoded
  values in baseline_model.py and underdog_lines.py (contract guard).

Run with: pytest tests/test_props.py -v
"""

import pytest

from src.props import PROP_REGISTRY, DEFAULT_PROP, get_prop


def test_default_prop_is_strikeouts():
    assert DEFAULT_PROP == "strikeouts"


def test_registry_contains_all_three_props():
    assert set(PROP_REGISTRY.keys()) == {"strikeouts", "walks", "earned_runs"}


def test_get_prop_resolves_known_key():
    prop = get_prop("strikeouts")
    assert prop.key == "strikeouts"


def test_get_prop_raises_key_error_for_unknown_prop():
    with pytest.raises(KeyError, match="unknown_prop"):
        get_prop("unknown_prop")


# ---------------------------------------------------------------------------
# Strikeouts prop -- contract guard: must match the hardcoded values the
# existing pipeline uses so the registry refactor is behaviour-preserving.
# ---------------------------------------------------------------------------

def test_strikeouts_underdog_stat():
    prop = get_prop("strikeouts")
    # Must match the Underdog stat KEY exactly (underdog_lines.DEFAULT_STAT).
    assert prop.underdog_stat == "strikeouts"


def test_strikeouts_label_column():
    assert get_prop("strikeouts").label_column == "strikeouts"


def test_strikeouts_label_source_is_statcast():
    assert get_prop("strikeouts").label_source == "statcast"


def test_strikeouts_statcast_event():
    assert get_prop("strikeouts").statcast_event == "strikeout"


def test_strikeouts_thresholds_match_baseline_model():
    # baseline_model.THRESHOLDS = range(1, 11)
    prop = get_prop("strikeouts")
    assert list(prop.thresholds) == list(range(1, 11))


def test_strikeouts_rate_prefix():
    assert get_prop("strikeouts").rate_feature_prefix == "k_rate"


# ---------------------------------------------------------------------------
# Walks prop
# ---------------------------------------------------------------------------

def test_walks_underdog_stat():
    assert get_prop("walks").underdog_stat == "walks_allowed"


def test_walks_label_column():
    assert get_prop("walks").label_column == "walks"


def test_walks_label_source_is_statcast():
    assert get_prop("walks").label_source == "statcast"


def test_walks_statcast_event():
    assert get_prop("walks").statcast_event == "walk"


def test_walks_thresholds_are_low_count_range():
    assert list(get_prop("walks").thresholds) == list(range(0, 6))


def test_walks_rate_prefix():
    assert get_prop("walks").rate_feature_prefix == "bb_rate"


# ---------------------------------------------------------------------------
# Earned runs prop
# ---------------------------------------------------------------------------

def test_earned_runs_underdog_stat():
    # NOTE: the key is "runs_allowed", not "earned_runs_allowed" -- Underdog's
    # display name ("Earned Runs Allowed") disagrees with its own stat key.
    assert get_prop("earned_runs").underdog_stat == "runs_allowed"


def test_earned_runs_label_column():
    assert get_prop("earned_runs").label_column == "earned_runs"


def test_earned_runs_label_source_is_statsapi_boxscore():
    # ER cannot be derived from Statcast (earned/unearned is an official
    # scoring decision) -- must use the boxscore source.
    assert get_prop("earned_runs").label_source == "statsapi_boxscore"


def test_earned_runs_statcast_event_is_none():
    # No Statcast event for earned runs -- the boxscore path is used instead.
    assert get_prop("earned_runs").statcast_event is None


def test_earned_runs_thresholds_are_low_count_range():
    assert list(get_prop("earned_runs").thresholds) == list(range(0, 6))


def test_earned_runs_rate_prefix():
    assert get_prop("earned_runs").rate_feature_prefix == "er"


# ---------------------------------------------------------------------------
# game_logs emits walks from fixture with known walk events
# ---------------------------------------------------------------------------

import pandas as pd
from src.features.game_logs import aggregate_pitcher_games, OUTPUT_COLUMNS


def _pitch_row(**overrides):
    row = {
        "pitcher": 543037,
        "game_pk": 1001,
        "game_date": "2026-04-01",
        "home_team": "NYY",
        "away_team": "BOS",
        "inning_topbot": "Top",
        "events": None,
        "description": "ball",
        "pitch_type": "FF",
        "release_speed": 96.0,
        "stand": "R",
    }
    row.update(overrides)
    return row


def test_game_logs_emits_walks_column():
    """game_logs.OUTPUT_COLUMNS must include 'walks'."""
    assert "walks" in OUTPUT_COLUMNS


def test_game_logs_counts_walks_correctly():
    """Verify walk events are counted correctly — fixture has 2 walks and 1 strikeout."""
    rows = [
        _pitch_row(events="walk", stand="R"),
        _pitch_row(events="walk", stand="L"),
        _pitch_row(events="strikeout", stand="R"),
        _pitch_row(events="field_out", stand="R"),
        _pitch_row(events=None, description="ball"),  # mid-PA pitch, not PA end
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))

    assert len(df) == 1
    game = df.iloc[0]
    assert game["walks"] == 2
    assert game["strikeouts"] == 1
    assert game["batters_faced"] == 4  # 3 outs + 1 walk = 4 PAs


def test_game_logs_walks_zero_when_no_walk_events():
    rows = [
        _pitch_row(events="strikeout"),
        _pitch_row(events="field_out"),
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))
    assert df.iloc[0]["walks"] == 0
