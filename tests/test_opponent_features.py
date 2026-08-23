"""Tests for src/features/opponent_features.py -- opponent-side team features."""

import pandas as pd
import pytest

from src.features.opponent_features import (
    add_opponent_features,
    build_team_game_logs,
)


def _pgame(**overrides):
    """One per-pitcher-game row, matching game_logs.OUTPUT_COLUMNS shape."""
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
        "pitcher_throws": "R",
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
    out = add_opponent_features(pd.DataFrame())
    assert out.empty
    assert "opponent_k_rate_last10" in out.columns
    assert "opponent_k_rate_vs_hand_season" in out.columns


def test_build_team_game_logs_sums_across_multiple_pitchers_same_team_game():
    # Two pitchers (a starter and a reliever) on NYY both face BOS in the
    # same game -- BOS's team-game totals should be the sum of both rows.
    rows = [
        _pgame(pitcher=1, game_pk=100, pitcher_team="NYY", opponent_team="BOS",
               home_away="home", strikeouts=5, batters_faced=18, pitcher_throws="R"),
        _pgame(pitcher=2, game_pk=100, pitcher_team="NYY", opponent_team="BOS",
               home_away="home", strikeouts=2, batters_faced=6, pitcher_throws="L"),
    ]
    tg = build_team_game_logs(pd.DataFrame(rows))

    assert len(tg) == 1
    row = tg.iloc[0]
    assert row["team"] == "BOS"
    assert row["strikeouts"] == 7
    assert row["batters_faced"] == 24
    # BOS is batting, NYY's pitchers are home -> BOS is away.
    assert row["team_home_away"] == "away"
    # Starter proxy = whichever pitcher faced the most batters (pitcher 1, 18 > 6).
    assert row["opponent_pitcher_hand"] == "R"


def test_opponent_k_rate_last10_uses_only_prior_games_pooled_not_averaged():
    # BOS bats against 3 different opposing pitchers on 3 different days.
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", opponent_team="BOS", strikeouts=2, batters_faced=20),
        _pgame(pitcher=2, game_pk=2, game_date="2026-04-02", opponent_team="BOS", strikeouts=8, batters_faced=20),
        _pgame(pitcher=3, game_pk=3, game_date="2026-04-03", opponent_team="BOS", strikeouts=5, batters_faced=20),
    ]
    df = add_opponent_features(pd.DataFrame(rows)).set_index("game_pk")

    assert pd.isna(df.loc[1, "opponent_k_rate_last10"])
    assert df.loc[2, "opponent_k_rate_last10"] == pytest.approx(2 / 20)
    # Pooled over games 1+2, not the mean of the two per-game rates.
    assert df.loc[3, "opponent_k_rate_last10"] == pytest.approx(10 / 40)


def test_opponent_k_rate_last10_window_caps_at_ten_prior_games():
    # 12 prior BOS games with a clearly different rate in the oldest 2 vs.
    # the most recent 10 -- the 13th game's feature should only reflect the
    # most recent 10, not all 12.
    rows = []
    for day in range(1, 13):
        # First 2 games: 0 K / 20 BF. Next 10 games: 10 K / 20 BF.
        k = 0 if day <= 2 else 10
        rows.append(_pgame(
            pitcher=day, game_pk=day, game_date=f"2026-04-{day:02d}",
            opponent_team="BOS", strikeouts=k, batters_faced=20,
        ))
    df = add_opponent_features(pd.DataFrame(rows)).set_index("game_pk")

    # Game 13 doesn't exist; check game 12's feature instead, which should
    # reflect games 2-11 (the 10 games prior to game 12): one 0-K game (day 2)
    # plus nine 10-K games (days 3-11).
    prior_k = 0 + 10 * 9
    prior_bf = 20 * 10
    assert df.loc[12, "opponent_k_rate_last10"] == pytest.approx(prior_k / prior_bf)


def test_opponent_home_away_splits_are_season_to_date_and_isolated_by_site():
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", opponent_team="BOS",
               home_away="home", strikeouts=5, batters_faced=20),  # BOS away
        _pgame(pitcher=2, game_pk=2, game_date="2026-04-06", opponent_team="BOS",
               home_away="away", strikeouts=8, batters_faced=20),  # BOS home
        _pgame(pitcher=3, game_pk=3, game_date="2026-04-11", opponent_team="BOS",
               home_away="home", strikeouts=6, batters_faced=20),  # BOS away
    ]
    df = add_opponent_features(pd.DataFrame(rows)).set_index("game_pk")

    # Game 3: BOS away again; prior away games for BOS = game 1 only.
    assert df.loc[3, "opponent_k_rate_away"] == pytest.approx(5 / 20)
    # BOS's prior home games = game 2 only.
    assert df.loc[3, "opponent_k_rate_home"] == pytest.approx(8 / 20)


def test_opponent_k_rate_vs_hand_season_filters_by_this_rows_pitcher_hand():
    # BOS faces an RHP, then an LHP, then another RHP.
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", opponent_team="BOS",
               pitcher_throws="R", strikeouts=4, batters_faced=20),
        _pgame(pitcher=2, game_pk=2, game_date="2026-04-06", opponent_team="BOS",
               pitcher_throws="L", strikeouts=9, batters_faced=20),
        _pgame(pitcher=3, game_pk=3, game_date="2026-04-11", opponent_team="BOS",
               pitcher_throws="R", strikeouts=6, batters_faced=20),
    ]
    df = add_opponent_features(pd.DataFrame(rows)).set_index("game_pk")

    # Game 3 faces an RHP again; BOS's only prior game vs RHP is game 1
    # (game 2, vs an LHP, must be excluded).
    assert df.loc[3, "opponent_k_rate_vs_hand_season"] == pytest.approx(4 / 20)
    # Game 2 (first game vs an LHP) has no prior vs-LHP history -> null.
    assert pd.isna(df.loc[2, "opponent_k_rate_vs_hand_season"])


def test_leakage_guardrail_earlier_rows_unchanged_by_later_game():
    rows = [
        _pgame(pitcher=1, game_pk=1, game_date="2026-04-01", opponent_team="BOS",
               pitcher_throws="R", home_away="home", strikeouts=5, batters_faced=20),
        _pgame(pitcher=2, game_pk=2, game_date="2026-04-06", opponent_team="BOS",
               pitcher_throws="L", home_away="away", strikeouts=8, batters_faced=22),
        _pgame(pitcher=3, game_pk=3, game_date="2026-04-11", opponent_team="BOS",
               pitcher_throws="R", home_away="home", strikeouts=6, batters_faced=19),
    ]
    later_game = _pgame(pitcher=4, game_pk=4, game_date="2026-04-16", opponent_team="BOS",
                         pitcher_throws="R", home_away="away", strikeouts=12, batters_faced=24)

    df_without = add_opponent_features(pd.DataFrame(rows)).set_index("game_pk")
    df_with = add_opponent_features(pd.DataFrame(rows + [later_game])).set_index("game_pk")

    feature_cols = [
        "opponent_k_rate_last10", "opponent_k_rate_vs_hand_season",
        "opponent_k_rate_home", "opponent_k_rate_away",
    ]
    for game_pk in [1, 2, 3]:
        for col in feature_cols:
            a = df_without.loc[game_pk, col]
            b = df_with.loc[game_pk, col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"game {game_pk} col {col} changed when a later game was added"
