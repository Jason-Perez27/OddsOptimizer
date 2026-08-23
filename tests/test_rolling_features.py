"""Tests for src/features/rolling_features.py -- pitcher-side rolling/split features."""

import pandas as pd
import pytest

from src.features.rolling_features import add_rolling_features, OPPONENT_CAREER_MIN_STARTS, FEATURE_COLUMNS


def _game(**overrides):
    row = {
        "pitcher": 1,
        "game_pk": 1,
        "game_date": "2026-04-01",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "strikeouts": 5,
        "batters_faced": 20,
        "pitch_count": 90,
        "whiff_rate": 0.25,
        "fastball_velo_avg": 96.0,
        "innings_pitched": 6.0,
        "strikeouts_vs_LHB": 2,
        "batters_faced_vs_LHB": 8,
        "strikeouts_vs_RHB": 3,
        "batters_faced_vs_RHB": 12,
        "rest_days": 5.0,
        "day_night": None,
    }
    row.update(overrides)
    return row


def test_empty_input_returns_empty_with_feature_columns():
    out = add_rolling_features(pd.DataFrame())
    assert out.empty
    assert "k_rate_last5" in out.columns
    assert "k_rate_vs_opponent_career" in out.columns


def test_first_game_has_no_prior_history_so_all_features_are_null():
    df = add_rolling_features(pd.DataFrame([_game()]))
    row = df.iloc[0]
    for col in [
        "k_rate_last5", "k_rate_season", "k_rate_vs_LHB", "k_rate_vs_RHB",
        "k_rate_home", "k_rate_away", "k_rate_vs_opponent_career",
        "ip_avg_last5", "pitch_count_avg_last5", "whiff_rate_last5", "velo_avg_last5",
    ]:
        assert pd.isna(row[col]), f"{col} should be null with no prior games"


def test_k_rate_season_uses_only_strictly_prior_games():
    rows = [
        _game(game_pk=1, game_date="2026-04-01", strikeouts=5, batters_faced=20),
        _game(game_pk=2, game_date="2026-04-06", strikeouts=10, batters_faced=20),
        _game(game_pk=3, game_date="2026-04-11", strikeouts=6, batters_faced=20),
    ]
    df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")

    assert pd.isna(df.loc[1, "k_rate_season"])
    assert df.loc[2, "k_rate_season"] == pytest.approx(5 / 20)  # only game 1's stats
    assert df.loc[3, "k_rate_season"] == pytest.approx(15 / 40)  # games 1+2


def test_k_rate_last5_is_pooled_rate_not_mean_of_rates():
    # Game A: 1/10 (low-volume relief-style outing), Game B: 10/20.
    # Pooled rate over the two prior starts should be 11/30, not the mean
    # of the two per-game rates (which would be (0.1 + 0.5)/2 = 0.3).
    rows = [
        _game(game_pk=1, game_date="2026-04-01", strikeouts=1, batters_faced=10),
        _game(game_pk=2, game_date="2026-04-06", strikeouts=10, batters_faced=20),
        _game(game_pk=3, game_date="2026-04-11", strikeouts=4, batters_faced=20),
    ]
    df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
    assert df.loc[3, "k_rate_last5"] == pytest.approx(11 / 30)


def test_home_away_splits_are_season_to_date_and_isolated_by_site():
    rows = [
        _game(game_pk=1, game_date="2026-04-01", home_away="home", strikeouts=5, batters_faced=20),
        _game(game_pk=2, game_date="2026-04-06", home_away="away", strikeouts=8, batters_faced=20),
        _game(game_pk=3, game_date="2026-04-11", home_away="home", strikeouts=6, batters_faced=20),
    ]
    df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")

    # Game 3 is home; prior home games = game 1 only.
    assert df.loc[3, "k_rate_home"] == pytest.approx(5 / 20)
    # Game 3's prior away games = game 2 only (k_rate_away reflects that,
    # even though game 3 itself isn't away).
    assert df.loc[3, "k_rate_away"] == pytest.approx(8 / 20)


def test_opponent_career_fallback_below_threshold_and_actual_rate_at_threshold():
    # Pitcher faces BOS in games 1, 3, 5, 6; other opponents in between.
    # All games use batters_faced=20 for simplicity.
    rows = [
        _game(game_pk=1, game_date="2026-04-01", opponent_team="BOS", strikeouts=5, batters_faced=20),
        _game(game_pk=2, game_date="2026-04-06", opponent_team="NYY", strikeouts=10, batters_faced=20),
        _game(game_pk=3, game_date="2026-04-11", opponent_team="BOS", strikeouts=6, batters_faced=20),
        _game(game_pk=4, game_date="2026-04-16", opponent_team="TBR", strikeouts=8, batters_faced=20),
        _game(game_pk=5, game_date="2026-04-21", opponent_team="BOS", strikeouts=4, batters_faced=20),
        _game(game_pk=6, game_date="2026-04-26", opponent_team="BOS", strikeouts=10, batters_faced=20),
    ]
    df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")

    assert OPPONENT_CAREER_MIN_STARTS == 3

    # Game 3: only 1 prior start vs BOS (game 1) -> below threshold ->
    # falls back to k_rate_season for game 3.
    assert df.loc[3, "k_rate_vs_opponent_career"] == pytest.approx(df.loc[3, "k_rate_season"])

    # Game 5: 2 prior starts vs BOS (games 1, 3) -> still below threshold ->
    # falls back to k_rate_season for game 5.
    assert df.loc[5, "k_rate_vs_opponent_career"] == pytest.approx(df.loc[5, "k_rate_season"])

    # Game 6: 3 prior starts vs BOS (games 1, 3, 5) -> meets threshold ->
    # uses the actual pooled career rate vs BOS: (5+6+4)/(20+20+20) = 15/60.
    expected_career_rate = (5 + 6 + 4) / (20 + 20 + 20)
    assert df.loc[6, "k_rate_vs_opponent_career"] == pytest.approx(expected_career_rate)
    assert df.loc[6, "k_rate_vs_opponent_career"] != pytest.approx(df.loc[6, "k_rate_season"])


def test_leakage_guardrail_earlier_rows_unchanged_by_later_game():
    rows = [
        _game(game_pk=1, game_date="2026-04-01", strikeouts=5, batters_faced=20, innings_pitched=6.0),
        _game(game_pk=2, game_date="2026-04-06", strikeouts=8, batters_faced=22, innings_pitched=7.0),
        _game(game_pk=3, game_date="2026-04-11", strikeouts=6, batters_faced=19, innings_pitched=5.5),
    ]
    later_game = _game(game_pk=4, game_date="2026-04-16", strikeouts=12, batters_faced=24, innings_pitched=7.0)

    df_without_later = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
    df_with_later = add_rolling_features(pd.DataFrame(rows + [later_game])).set_index("game_pk")

    feature_cols = [
        "k_rate_last5", "k_rate_season", "k_rate_vs_LHB", "k_rate_vs_RHB",
        "k_rate_home", "k_rate_away", "k_rate_vs_opponent_career",
        "ip_avg_last5", "pitch_count_avg_last5", "whiff_rate_last5", "velo_avg_last5",
    ]
    for game_pk in [1, 2, 3]:
        for col in feature_cols:
            a = df_without_later.loc[game_pk, col]
            b = df_with_later.loc[game_pk, col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"game {game_pk} col {col} changed when a later game was added (leakage)"
