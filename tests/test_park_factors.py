"""Tests for src/features/park_factors.py -- ballpark K-index feature."""

import pandas as pd
import pytest

from src.features.park_factors import (
    add_park_factors,
    STATIC_PARK_FACTORS,
    DEFAULT_STATIC_PARK_FACTOR,
    MIN_PARK_GAMES_FOR_COMPUTED,
)


def _pgame(**overrides):
    row = {
        "pitcher": 1, "game_pk": 1, "game_date": "2026-04-01", "pitcher_team": "NYY",
        "opponent_team": "BOS", "home_away": "home", "strikeouts": 5, "batters_faced": 20,
        "pitch_count": 90, "whiff_rate": 0.25, "fastball_velo_avg": 96.0, "innings_pitched": 6.0,
        "pitcher_throws": "R", "strikeouts_vs_LHB": 2, "batters_faced_vs_LHB": 8,
        "strikeouts_vs_RHB": 3, "batters_faced_vs_RHB": 12, "rest_days": 5.0, "day_night": None,
    }
    row.update(overrides)
    return row


def test_empty_input_returns_empty_with_feature_column():
    out = add_park_factors(pd.DataFrame())
    assert out.empty
    assert "park_k_factor" in out.columns


def test_cold_start_falls_back_to_static_table():
    # Only one prior game ever played at NYY's park -- nowhere near
    # MIN_PARK_GAMES_FOR_COMPUTED, so every row should use the static value.
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", pitcher_team="NYY",
               opponent_team="BOS", home_away="home", strikeouts=5, batters_faced=20),
        _pgame(pitcher=2, game_pk=1, game_date="2026-04-01", pitcher_team="BOS",
               opponent_team="NYY", home_away="away", strikeouts=3, batters_faced=18),
    ]
    df = add_park_factors(pd.DataFrame(rows))
    assert (df["park_k_factor"] == STATIC_PARK_FACTORS["NYY"]).all()


def test_unknown_team_defaults_to_league_average_static_value():
    rows = [_pgame(pitcher=1, game_pk=1, pitcher_team="XXX", opponent_team="BOS", home_away="home")]
    df = add_park_factors(pd.DataFrame(rows))
    assert df.iloc[0]["park_k_factor"] == DEFAULT_STATIC_PARK_FACTOR


def test_computed_value_used_once_enough_prior_games_at_park():
    rows = []
    gp = 1
    # NYY's park: heavy strikeout park, 50% K-rate every game.
    for day in range(1, MIN_PARK_GAMES_FOR_COMPUTED + 2):
        rows.append(_pgame(pitcher=day, game_pk=gp, game_date=f"2026-04-{day:02d}",
                            pitcher_team="NYY", opponent_team="BOS", home_away="home",
                            strikeouts=10, batters_faced=20))
        gp += 1
    # A second park with a lower K-rate, to pull the league average down
    # below NYY's park-specific rate.
    for day in range(1, MIN_PARK_GAMES_FOR_COMPUTED + 2):
        rows.append(_pgame(pitcher=1000 + day, game_pk=gp, game_date=f"2026-04-{day:02d}",
                            pitcher_team="BOS", opponent_team="NYY", home_away="home",
                            strikeouts=4, batters_faced=20))
        gp += 1

    df = add_park_factors(pd.DataFrame(rows))
    nyy_rows = df[df["pitcher_team"] == "NYY"].sort_values("game_date")
    last_row = nyy_rows.iloc[-1]

    # By the last game, enough prior NYY-park games have accumulated that the
    # computed value should be used instead of the static fallback, and
    # since NYY's park K-rate is well above the blended league rate, the
    # index should be meaningfully above 100.
    assert last_row["park_k_factor"] != STATIC_PARK_FACTORS["NYY"]
    assert last_row["park_k_factor"] > 100


def test_leakage_guardrail_earlier_rows_unchanged_by_later_game():
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", pitcher_team="NYY",
               opponent_team="BOS", home_away="home", strikeouts=5, batters_faced=20),
        _pgame(pitcher=2, game_pk=2, game_date="2026-04-03", pitcher_team="NYY",
               opponent_team="TBR", home_away="home", strikeouts=7, batters_faced=22),
        _pgame(pitcher=3, game_pk=3, game_date="2026-04-06", pitcher_team="NYY",
               opponent_team="BAL", home_away="home", strikeouts=4, batters_faced=19),
    ]
    later_game = _pgame(pitcher=4, game_pk=4, game_date="2026-04-09", pitcher_team="NYY",
                         opponent_team="BOS", home_away="home", strikeouts=20, batters_faced=20)

    df_without = add_park_factors(pd.DataFrame(rows)).set_index("game_pk")
    df_with = add_park_factors(pd.DataFrame(rows + [later_game])).set_index("game_pk")

    for gp in [1, 2, 3]:
        a = df_without.loc[gp, "park_k_factor"]
        b = df_with.loc[gp, "park_k_factor"]
        assert a == pytest.approx(b)
