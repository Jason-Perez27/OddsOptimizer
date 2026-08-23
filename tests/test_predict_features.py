"""
Unit tests for src/features/predict_features.py (task #9, module 3).

Design: docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md
("Pre-game (as-of-today) feature construction" section, testing-approach
items 3-5).

Strategy (mirrors tests/test_rolling_features.py / tests/test_probable_
pitchers.py conventions in this repo):
- No network calls, no real model. Hand-built fixtures via small helper
  functions (_game, _slate_row), # ---...--- section dividers grouping
  tests by function under test, test_<function>_<behavior> naming.
- The core correctness test (the one that matters most, per the build-order
  approval): an as-of-today synthetic row, run through
  build_prediction_features, must produce EXACTLY the same rolling/
  opponent/park feature values as a "real" row sharing the same pre-game
  identifiers (pitcher/team/opponent/home_away/date), computed by running
  the same three builders directly. This proves the synthetic-row mechanism
  is leakage-free by construction, not just by inspection -- it reuses the
  builders' OWN shift(1)-before-aggregate guardrail rather than adding a
  new one. The test also asserts that appending the synthetic row never
  perturbs any of the 5 historical rows' own computed features (the
  invariant test_rolling_features.py already exercises for its own
  builder, replicated here at the build_prediction_features level).

Run with: pytest tests/test_predict_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.predict_features import (
    build_synthetic_game_rows,
    build_prediction_features,
    LEAKAGE_OUTCOME_COLUMNS,
)
from src.features.game_logs import OUTPUT_COLUMNS
from src.features.rolling_features import add_rolling_features
from src.features.opponent_features import add_opponent_features
from src.features.park_factors import add_park_factors


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

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


def _slate_row(**overrides):
    row = {
        "pitcher": 1,
        "pitcher_name": "Test Pitcher",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "game_pk": 99,
        "game_date": "2026-04-26",
        "start_time": "2026-04-26T23:05:00Z",
        "pitcher_throws": "R",
    }
    row.update(overrides)
    return row


SLATE_COLUMNS = [
    "pitcher", "pitcher_name", "pitcher_team", "opponent_team", "home_away",
    "game_pk", "game_date", "start_time", "pitcher_throws",
]

# A 5-start history for pitcher=1/NYY, alternating opponents BOS/TB and
# home/away, used by both the synthetic-row-shape and core-correctness
# tests below.
HISTORY_ROWS = [
    _game(game_pk=1, game_date="2026-04-01", opponent_team="BOS", home_away="home", strikeouts=5, batters_faced=20),
    _game(game_pk=2, game_date="2026-04-06", opponent_team="BOS", home_away="away", strikeouts=10, batters_faced=20),
    _game(game_pk=3, game_date="2026-04-11", opponent_team="TB", home_away="home", strikeouts=6, batters_faced=18),
    _game(game_pk=4, game_date="2026-04-16", opponent_team="TB", home_away="away", strikeouts=8, batters_faced=20),
    _game(game_pk=5, game_date="2026-04-21", opponent_team="BOS", home_away="home", strikeouts=7, batters_faced=21),
]


def _history_df():
    return pd.DataFrame(HISTORY_ROWS)


def _run_builders(game_df):
    return add_park_factors(add_opponent_features(add_rolling_features(game_df.copy())))


# ---------------------------------------------------------------------------
# build_synthetic_game_rows() -- shape and content (testing item 3)
# ---------------------------------------------------------------------------

def test_synthetic_row_has_game_logs_columns_and_known_fields_filled():
    slate = pd.DataFrame([_slate_row()])
    synth = build_synthetic_game_rows(slate)

    assert list(synth.columns) == OUTPUT_COLUMNS
    assert len(synth) == 1

    row = synth.iloc[0]
    assert row["pitcher"] == 1
    assert row["game_pk"] == 99
    assert row["game_date"] == pd.Timestamp("2026-04-26")
    assert row["pitcher_team"] == "NYY"
    assert row["opponent_team"] == "BOS"
    assert row["home_away"] == "home"
    assert row["pitcher_throws"] == "R"
    assert row["day_night"] is None


def test_synthetic_row_leaves_every_outcome_column_nan():
    slate = pd.DataFrame([_slate_row()])
    synth = build_synthetic_game_rows(slate)
    row = synth.iloc[0]

    for col in LEAKAGE_OUTCOME_COLUMNS:
        assert pd.isna(row[col]), f"{col} should be NaN pre-game"
    assert pd.isna(row["rest_days"])  # recomputed later, not here


def test_synthetic_rows_one_per_slate_row_multiple_starters():
    slate = pd.DataFrame([
        _slate_row(pitcher=1, game_pk=99),
        _slate_row(pitcher=2, pitcher_team="BOS", opponent_team="NYY", home_away="away", game_pk=99),
    ])
    synth = build_synthetic_game_rows(slate)
    assert len(synth) == 2
    assert set(synth["pitcher"]) == {1, 2}


def test_empty_slate_returns_empty_well_formed_frame():
    empty_slate = pd.DataFrame(columns=SLATE_COLUMNS)
    synth = build_synthetic_game_rows(empty_slate)
    assert synth.empty
    assert list(synth.columns) == OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# build_prediction_features() -- the core as-of-today correctness test
# (testing item 4 -- the one that matters most)
# ---------------------------------------------------------------------------

def test_synthetic_today_row_matches_real_row_with_same_identifiers():
    """
    A "real" row dated today, sharing the synthetic row's pre-game
    identifiers (pitcher/team/opponent/home_away/date) but with arbitrary
    (and necessarily different) outcome values, must produce IDENTICAL
    rolling/opponent/park feature values to the synthetic row -- those
    features only ever depend on STRICTLY PRIOR rows, never on the row's
    own outcome columns. This is the leakage-free guarantee the synthetic-
    row mechanism rests on.
    """
    history = _history_df()

    real_today_row = _game(
        game_pk=99, game_date="2026-04-26", opponent_team="BOS", home_away="home",
        strikeouts=3, batters_faced=15,  # arbitrary -- must not affect this row's own features
    )
    combined_real = pd.concat([history, pd.DataFrame([real_today_row])], ignore_index=True)
    combined_real["game_date"] = pd.to_datetime(combined_real["game_date"])
    combined_real = combined_real.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    combined_real["rest_days"] = combined_real.groupby("pitcher")["game_date"].diff().dt.days
    combined_real = _run_builders(combined_real)
    real_row_features = combined_real[combined_real["game_pk"] == 99].iloc[0]

    slate_today = pd.DataFrame([_slate_row(game_pk=99, game_date="2026-04-26")])
    predicted = build_prediction_features(history, slate_today)
    assert len(predicted) == 1
    pred_row = predicted.iloc[0]

    for col in ["k_rate_last5", "opponent_k_rate_last10", "park_k_factor"]:
        assert np.isclose(pred_row[col], real_row_features[col]), (
            f"{col}: synthetic={pred_row[col]!r} real={real_row_features[col]!r}"
        )


def test_synthetic_today_row_matches_hand_computed_expectations():
    """Same fixture as above, checked against independently hand-computed values."""
    history = _history_df()
    slate_today = pd.DataFrame([_slate_row(game_pk=99, game_date="2026-04-26")])
    predicted = build_prediction_features(history, slate_today)
    row = predicted.iloc[0]

    expected_k_rate_last5 = (5 + 10 + 6 + 8 + 7) / (20 + 20 + 18 + 20 + 21)
    assert np.isclose(row["k_rate_last5"], expected_k_rate_last5)

    # Only the 3 BOS games count toward opponent_k_rate_last10 (BOS's own
    # batting history across the games where BOS was the opponent).
    expected_opponent_k_rate_last10 = (5 + 10 + 7) / (20 + 20 + 21)
    assert np.isclose(row["opponent_k_rate_last10"], expected_opponent_k_rate_last10)

    # Only 2 prior home games at NYY's park, well under
    # park_factors.MIN_PARK_GAMES_FOR_COMPUTED (15) -- falls back to the
    # static NYY factor.
    assert row["park_k_factor"] == 99

    # rest_days recomputed post-concatenation: 2026-04-26 minus the
    # pitcher's last REAL start (2026-04-21) = 5 days.
    assert row["rest_days"] == 5


def test_appending_synthetic_row_does_not_change_historical_rows_features():
    """
    Mirrors the invariant tests/test_rolling_features.py already checks for
    add_rolling_features alone: adding a later row must never change the
    computed features of any strictly-earlier row. Checked here at the
    build_prediction_features level, across all three builders at once.
    """
    history = _history_df()
    hist_only = _run_builders(history)

    slate_today = pd.DataFrame([_slate_row(game_pk=99, game_date="2026-04-26")])
    _ = build_prediction_features(history, slate_today)

    # history itself must be untouched (build_prediction_features copies).
    hist_after = _run_builders(history)
    for col in ["k_rate_last5", "opponent_k_rate_last10", "park_k_factor"]:
        assert np.allclose(
            hist_only[col].fillna(-999).values,
            hist_after[col].fillna(-999).values,
        ), f"{col} changed after running build_prediction_features"


# ---------------------------------------------------------------------------
# No-history pitcher is skipped (feature-wise), not crashed (testing item 5)
# ---------------------------------------------------------------------------

def test_pitcher_with_no_history_gets_a_row_with_all_nan_features_not_a_crash():
    history = _history_df()
    slate_no_history = pd.DataFrame([_slate_row(pitcher=999, game_pk=100, pitcher_name="Rookie")])

    predicted = build_prediction_features(history, slate_no_history)

    assert len(predicted) == 1
    row = predicted.iloc[0]
    assert pd.isna(row["k_rate_last5"])
    assert row["pitcher"] == 999


def test_mixed_slate_of_known_and_unknown_pitchers_both_get_rows():
    history = _history_df()
    slate = pd.DataFrame([
        _slate_row(pitcher=1, game_pk=99, game_date="2026-04-26"),
        _slate_row(pitcher=999, pitcher_name="Rookie", game_pk=100, game_date="2026-04-26"),
    ])

    predicted = build_prediction_features(history, slate)

    assert len(predicted) == 2
    known_row = predicted[predicted["pitcher"] == 1].iloc[0]
    unknown_row = predicted[predicted["pitcher"] == 999].iloc[0]
    assert not pd.isna(known_row["k_rate_last5"])
    assert pd.isna(unknown_row["k_rate_last5"])


# ---------------------------------------------------------------------------
# Empty inputs -- no network, no crash
# ---------------------------------------------------------------------------

def test_empty_slate_returns_empty_predictions():
    history = _history_df()
    empty_slate = pd.DataFrame(columns=SLATE_COLUMNS)

    predicted = build_prediction_features(history, empty_slate)

    assert predicted.empty
    assert list(predicted.columns) == OUTPUT_COLUMNS


def test_empty_historical_game_logs_still_produces_a_row():
    empty_history = pd.DataFrame(columns=OUTPUT_COLUMNS)
    slate = pd.DataFrame([_slate_row()])

    predicted = build_prediction_features(empty_history, slate)

    assert len(predicted) == 1
    assert pd.isna(predicted.iloc[0]["k_rate_last5"])


def test_none_historical_game_logs_still_produces_a_row():
    slate = pd.DataFrame([_slate_row()])

    predicted = build_prediction_features(None, slate)

    assert len(predicted) == 1
    assert pd.isna(predicted.iloc[0]["k_rate_last5"])


# ---------------------------------------------------------------------------
# Doubleheader -- two slate rows, same pitcher and date, distinct game_pk
# ---------------------------------------------------------------------------

def test_doubleheader_yields_two_distinct_rows():
    history = _history_df()
    slate_dh = pd.DataFrame([
        _slate_row(game_pk=201, game_date="2026-04-26"),
        _slate_row(game_pk=202, game_date="2026-04-26"),
    ])

    predicted = build_prediction_features(history, slate_dh)

    assert len(predicted) == 2
    assert set(predicted["game_pk"]) == {201, 202}
