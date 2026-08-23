"""Tests for src/features/build_features.py -- the final joined training table."""

import pandas as pd
import pytest

from src.features.build_features import (
    build_training_table,
    REQUIRED_COLUMNS,
    _validate_required_columns,
)
from src.features.game_logs import OUTPUT_COLUMNS
from src.features.rolling_features import FEATURE_COLUMNS as ROLLING_FEATURE_COLUMNS
from src.features.opponent_features import FEATURE_COLUMNS as OPPONENT_FEATURE_COLUMNS


def _pitch_row(**overrides):
    row = {
        "pitcher": 1,
        "game_pk": 1,
        "game_date": "2026-04-01",
        "home_team": "NYY",
        "away_team": "BOS",
        "inning_topbot": "Top",
        "events": None,
        "description": "ball",
        "pitch_type": "FF",
        "release_speed": 96.0,
        "stand": "R",
        "p_throws": "R",
    }
    row.update(overrides)
    return row


def test_empty_input_returns_empty_with_all_expected_columns():
    out = build_training_table(pd.DataFrame())
    assert out.empty
    for col in OUTPUT_COLUMNS + ROLLING_FEATURE_COLUMNS + OPPONENT_FEATURE_COLUMNS + ["park_k_factor"]:
        assert col in out.columns


def test_one_row_per_pitcher_game_from_pitch_level_input():
    rows = [
        _pitch_row(pitcher=1, game_pk=1, game_date="2026-04-01", events="strikeout", stand="R"),
        _pitch_row(pitcher=1, game_pk=1, game_date="2026-04-01", events="field_out", stand="L"),
        _pitch_row(pitcher=1, game_pk=1, game_date="2026-04-01", events=None, description="ball"),
        _pitch_row(pitcher=2, game_pk=1, game_date="2026-04-01", events="strikeout", stand="R",
                   home_team="NYY", away_team="BOS", inning_topbot="Bot", p_throws="L"),
        _pitch_row(pitcher=1, game_pk=2, game_date="2026-04-06", events="strikeout", stand="L"),
    ]
    out = build_training_table(pd.DataFrame(rows))

    # 3 distinct (pitcher, game_pk) pairs: (1,1) home starter, (2,1) away
    # starter same game, (1,2) pitcher 1's next game.
    assert len(out) == 3
    assert set(zip(out["pitcher"], out["game_pk"])) == {(1, 1), (2, 1), (1, 2)}


def test_no_unexpected_nulls_in_required_columns():
    rows = [
        _pitch_row(pitcher=1, game_pk=1, game_date="2026-04-01", events="strikeout", stand="R"),
        _pitch_row(pitcher=2, game_pk=1, game_date="2026-04-01", events="strikeout", stand="R",
                   home_team="NYY", away_team="BOS", inning_topbot="Bot", p_throws="L"),
        _pitch_row(pitcher=1, game_pk=2, game_date="2026-04-06", events="strikeout", stand="L"),
    ]
    out = build_training_table(pd.DataFrame(rows))

    assert not out[REQUIRED_COLUMNS].isna().any().any()


def test_feature_columns_can_be_null_for_first_games_but_required_columns_cannot():
    # A pitcher's very first game has no prior history -- rolling/opponent
    # features are expected to be null there. That's not a bug.
    rows = [_pitch_row(pitcher=1, game_pk=1, game_date="2026-04-01", events="strikeout", stand="R")]
    out = build_training_table(pd.DataFrame(rows))

    assert pd.isna(out.iloc[0]["k_rate_season"])
    assert not out.iloc[0][REQUIRED_COLUMNS].isna().any()


def test_validate_required_columns_raises_on_null_required_value():
    df = pd.DataFrame([{col: 1 for col in REQUIRED_COLUMNS}])
    df.loc[0, "strikeouts"] = None

    with pytest.raises(ValueError):
        _validate_required_columns(df)


def test_validate_required_columns_passes_on_empty_dataframe():
    # Nothing to validate -- shouldn't raise just because the input is empty.
    _validate_required_columns(pd.DataFrame(columns=REQUIRED_COLUMNS))
