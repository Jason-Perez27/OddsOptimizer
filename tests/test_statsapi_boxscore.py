"""
Tests for src/data/statsapi_boxscore.py -- earned-runs label from MLB StatsAPI.

No network calls: every test injects a fixture fetcher lambda that returns a
pre-built boxscore dict, mirroring the real API's shape at
liveData.boxscore.teams.{home|away}.{pitchers, players}.

Run with: pytest tests/test_statsapi_boxscore.py -v
"""

import pandas as pd
import pytest

from src.data.statsapi_boxscore import (
    get_pitcher_earned_runs_by_game,
    OUTPUT_COLUMNS,
)


def _make_boxscore(home_pitchers=None, away_pitchers=None):
    """Build a minimal boxscore dict shaped like the MLB StatsAPI response."""
    home_pitchers = home_pitchers or []
    away_pitchers = away_pitchers or []

    def _players(pitcher_list):
        players = {}
        for pid, er in pitcher_list:
            players[f"ID{pid}"] = {
                "stats": {"pitching": {"earnedRuns": er, "inningsPitched": "6.0"}}
            }
        return players

    return {
        "teams": {
            "home": {
                "pitchers": [pid for pid, _ in home_pitchers],
                "players": _players(home_pitchers),
            },
            "away": {
                "pitchers": [pid for pid, _ in away_pitchers],
                "players": _players(away_pitchers),
            },
        }
    }


# ---------------------------------------------------------------------------
# OUTPUT_COLUMNS contract
# ---------------------------------------------------------------------------

def test_output_columns_are_correct():
    assert OUTPUT_COLUMNS == ["pitcher", "game_pk", "earned_runs"]


# ---------------------------------------------------------------------------
# Happy-path ingestion
# ---------------------------------------------------------------------------

def test_single_home_pitcher_returns_one_row():
    data = _make_boxscore(home_pitchers=[(543037, 3)])
    df = get_pitcher_earned_runs_by_game(100001, fetcher=lambda _: data)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["pitcher"] == 543037
    assert row["game_pk"] == 100001
    assert row["earned_runs"] == 3


def test_both_teams_pitchers_returned():
    data = _make_boxscore(
        home_pitchers=[(543037, 2)],
        away_pitchers=[(600001, 4)],
    )
    df = get_pitcher_earned_runs_by_game(100002, fetcher=lambda _: data)

    assert len(df) == 2
    assert set(df["pitcher"]) == {543037, 600001}
    assert set(df["game_pk"]) == {100002}


def test_zero_earned_runs_still_included():
    """A starter who allowed 0 ER should appear in the output (not filtered out)."""
    data = _make_boxscore(home_pitchers=[(543037, 0)])
    df = get_pitcher_earned_runs_by_game(100003, fetcher=lambda _: data)
    assert len(df) == 1
    assert df.iloc[0]["earned_runs"] == 0


def test_multiple_relievers_all_returned():
    data = _make_boxscore(
        home_pitchers=[(543037, 2), (612345, 1), (678901, 0)],
    )
    df = get_pitcher_earned_runs_by_game(100004, fetcher=lambda _: data)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Missing / absent data
# ---------------------------------------------------------------------------

def test_pitcher_without_earned_runs_key_is_excluded():
    """A player entry with no earnedRuns field must not produce a row."""
    data = {
        "teams": {
            "home": {
                "pitchers": [543037],
                "players": {
                    "ID543037": {"stats": {"pitching": {"inningsPitched": "6.0"}}}
                    # no earnedRuns key
                },
            },
            "away": {"pitchers": [], "players": {}},
        }
    }
    df = get_pitcher_earned_runs_by_game(100005, fetcher=lambda _: data)
    assert df.empty
    assert list(df.columns) == OUTPUT_COLUMNS


def test_empty_boxscore_returns_empty_dataframe():
    data = {"teams": {"home": {"pitchers": [], "players": {}},
                      "away": {"pitchers": [], "players": {}}}}
    df = get_pitcher_earned_runs_by_game(100006, fetcher=lambda _: data)
    assert df.empty
    assert list(df.columns) == OUTPUT_COLUMNS


def test_fetcher_exception_returns_empty_dataframe():
    """A network error or bad game_pk must return empty, not raise."""
    def bad_fetcher(_):
        raise ConnectionError("network unavailable")

    df = get_pitcher_earned_runs_by_game(999999, fetcher=bad_fetcher)
    assert df.empty
    assert list(df.columns) == OUTPUT_COLUMNS


def test_malformed_response_returns_empty_dataframe():
    """A response missing the teams key must return empty, not raise."""
    df = get_pitcher_earned_runs_by_game(100007, fetcher=lambda _: {"status": "error"})
    assert df.empty


# ---------------------------------------------------------------------------
# Column types and contract
# ---------------------------------------------------------------------------

def test_output_dtypes_are_ints():
    data = _make_boxscore(home_pitchers=[(543037, 2)])
    df = get_pitcher_earned_runs_by_game(100008, fetcher=lambda _: data)
    assert df["pitcher"].dtype in (int, "int64")
    assert df["game_pk"].dtype in (int, "int64")
    assert df["earned_runs"].dtype in (int, "int64")
