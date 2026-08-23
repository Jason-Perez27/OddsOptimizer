"""Tests for src/features/game_logs.py -- pitch-level -> per-game aggregation."""

import pandas as pd
import pytest

from src.features.game_logs import aggregate_pitcher_games, OUTPUT_COLUMNS


def _pitch_row(**overrides):
    row = {
        "pitcher": 543037,
        "game_pk": 1001,
        "game_date": "2026-04-01",
        "home_team": "NYY",
        "away_team": "BOS",
        "inning_topbot": "Top",  # pitcher's team (home, NYY) is pitching
        "events": None,
        "description": "ball",
        "pitch_type": "FF",
        "release_speed": 96.0,
        "stand": "R",
    }
    row.update(overrides)
    return row


def test_empty_input_returns_empty_with_expected_columns():
    out = aggregate_pitcher_games(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_single_game_aggregates_strikeouts_and_batters_faced():
    rows = [
        _pitch_row(events="strikeout", stand="R"),
        _pitch_row(events="field_out", stand="L"),
        _pitch_row(events="strikeout", stand="L"),
        _pitch_row(events=None, description="ball"),  # mid-PA pitch, not a PA end
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))

    assert len(df) == 1
    game = df.iloc[0]
    assert game["strikeouts"] == 2
    assert game["batters_faced"] == 3
    assert game["pitch_count"] == 4


def test_home_away_derived_from_inning_topbot():
    home_row = _pitch_row(inning_topbot="Top")  # away team batting -> home team pitching
    away_row = _pitch_row(
        game_pk=1002, inning_topbot="Bot", events="strikeout"
    )  # home team batting -> away team pitching

    df = aggregate_pitcher_games(pd.DataFrame([home_row, away_row]))
    df = df.set_index("game_pk")

    assert df.loc[1001, "home_away"] == "home"
    assert df.loc[1001, "pitcher_team"] == "NYY"
    assert df.loc[1001, "opponent_team"] == "BOS"

    assert df.loc[1002, "home_away"] == "away"
    assert df.loc[1002, "pitcher_team"] == "BOS"
    assert df.loc[1002, "opponent_team"] == "NYY"


def test_whiff_rate_and_fastball_velo_average_correctly():
    rows = [
        _pitch_row(description="swinging_strike", pitch_type="FF", release_speed=96.0),
        _pitch_row(description="swinging_strike_blocked", pitch_type="SL", release_speed=85.0),
        _pitch_row(description="ball", pitch_type="FF", release_speed=98.0),
        _pitch_row(description="called_strike", pitch_type="CU", release_speed=80.0),
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))
    game = df.iloc[0]

    # 2 whiffs out of 4 pitches
    assert game["whiff_rate"] == pytest.approx(0.5)
    # fastball velo avg only over FF pitches: (96.0 + 98.0) / 2
    assert game["fastball_velo_avg"] == pytest.approx(97.0)


def test_handedness_split_strikeout_counts():
    rows = [
        _pitch_row(events="strikeout", stand="L"),
        _pitch_row(events="field_out", stand="L"),
        _pitch_row(events="strikeout", stand="R"),
        _pitch_row(events="strikeout", stand="R"),
        _pitch_row(events="walk", stand="R"),
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))
    game = df.iloc[0]

    assert game["strikeouts_vs_LHB"] == 1
    assert game["batters_faced_vs_LHB"] == 2
    assert game["strikeouts_vs_RHB"] == 2
    assert game["batters_faced_vs_RHB"] == 3


def test_rest_days_is_null_for_first_game_and_correct_gap_for_second():
    game1 = [_pitch_row(game_pk=1001, game_date="2026-04-01", events="strikeout")]
    game2 = [_pitch_row(game_pk=1002, game_date="2026-04-06", events="strikeout")]

    df = aggregate_pitcher_games(pd.DataFrame(game1 + game2))
    df = df.sort_values("game_date").reset_index(drop=True)

    assert pd.isna(df.loc[0, "rest_days"])
    assert df.loc[1, "rest_days"] == 5


def test_rest_days_is_per_pitcher_not_global():
    pitcher_a_game = _pitch_row(pitcher=1, game_pk=1001, game_date="2026-04-01", events="strikeout")
    pitcher_b_game = _pitch_row(pitcher=2, game_pk=1002, game_date="2026-04-01", events="strikeout")

    df = aggregate_pitcher_games(pd.DataFrame([pitcher_a_game, pitcher_b_game]))

    # Both pitchers' first game in the input -- neither should get a rest_days
    # value borrowed from the other pitcher's game.
    assert df["rest_days"].isna().all()


def test_day_night_is_always_null_since_statcast_has_no_time_field():
    df = aggregate_pitcher_games(pd.DataFrame([_pitch_row(events="strikeout")]))
    assert df.iloc[0]["day_night"] is None


def test_innings_pitched_estimated_from_outs_recorded():
    rows = [
        _pitch_row(events="strikeout"),       # 1 out
        _pitch_row(events="field_out"),        # 1 out
        _pitch_row(events="grounded_into_double_play"),  # 2 outs
        _pitch_row(events="walk"),             # 0 outs (not an out event)
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))
    game = df.iloc[0]

    # 4 outs recorded -> 4/3 innings
    assert game["innings_pitched"] == pytest.approx(4 / 3)


def test_pitcher_throws_is_none_when_p_throws_column_absent():
    # Existing pitch rows (via _pitch_row) don't include p_throws -- the
    # column should be gracefully absent/optional, not a KeyError.
    df = aggregate_pitcher_games(pd.DataFrame([_pitch_row(events="strikeout")]))
    assert df.iloc[0]["pitcher_throws"] is None


def test_pitcher_throws_is_carried_through_when_p_throws_column_present():
    rows = [
        _pitch_row(events="strikeout", p_throws="L"),
        _pitch_row(events="field_out", p_throws="L"),
    ]
    df = aggregate_pitcher_games(pd.DataFrame(rows))
    assert df.iloc[0]["pitcher_throws"] == "L"
