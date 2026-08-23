"""
Unit tests for src/data/probable_pitchers.py (task #9, module 1).

Strategy (mirrors tests/test_tiering.py conventions in this repo):
- No network calls. fetch_schedule() is a thin live wrapper and is
  deliberately NOT exercised here -- every test instead hand-builds a raw
  StatsAPI schedule payload and feeds it directly to parse_probable_starters().
- Hand-built fixtures via small helper functions, # ---...--- section
  dividers grouping tests by function under test, test_<function>_<behavior>
  naming.
- Covers spec testing-approach items 1-2: slate parsing edge cases (one row
  per probable starter, one-side-posted, none-posted) and the StatsAPI ->
  Statcast team crosswalk (known mismatches, unknown pass-through).

Run with: pytest tests/test_probable_pitchers.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import probable_pitchers


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _probable(pid, name, hand=None):
    pitcher = {"id": pid, "fullName": name}
    if hand is not None:
        pitcher["pitchHand"] = {"code": hand}
    return pitcher


def _side(abbrev, probable=None):
    side = {"team": {"abbreviation": abbrev}}
    if probable is not None:
        side["probablePitcher"] = probable
    return side


def _game(game_pk, home_side, away_side, start_time="2026-06-27T23:05:00Z"):
    return {
        "gamePk": game_pk,
        "gameDate": start_time,
        "teams": {"home": home_side, "away": away_side},
    }


def _schedule(*games):
    return {"dates": [{"games": list(games)}]}


GAME_DATE = "2026-06-27"


# ---------------------------------------------------------------------------
# parse_probable_starters() -- slate parsing
# ---------------------------------------------------------------------------

def test_both_sides_posted_yields_two_rows_with_correct_fields():
    home = _side("BOS", _probable(999999, "Some Guy", hand="L"))
    away = _side("NYY", _probable(543037, "Gerrit Cole", hand="R"))
    schedule = _schedule(_game(12345, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    assert list(slate.columns) == probable_pitchers.SLATE_COLUMNS
    assert len(slate) == 2

    home_row = slate[slate["pitcher"] == 999999].iloc[0]
    assert home_row["pitcher_name"] == "Some Guy"
    assert home_row["pitcher_team"] == "BOS"
    assert home_row["opponent_team"] == "NYY"
    assert home_row["home_away"] == "home"
    assert home_row["game_pk"] == 12345
    assert home_row["game_date"] == GAME_DATE
    assert home_row["pitcher_throws"] == "L"

    away_row = slate[slate["pitcher"] == 543037].iloc[0]
    assert away_row["pitcher_name"] == "Gerrit Cole"
    assert away_row["pitcher_team"] == "NYY"
    assert away_row["opponent_team"] == "BOS"
    assert away_row["home_away"] == "away"
    assert away_row["pitcher_throws"] == "R"


def test_only_one_side_posted_yields_one_row():
    home = _side("BOS", _probable(999999, "Some Guy"))
    away = _side("NYY", probable=None)  # not posted yet
    schedule = _schedule(_game(12345, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    assert len(slate) == 1
    assert slate.iloc[0]["pitcher"] == 999999
    assert slate.iloc[0]["home_away"] == "home"


def test_no_probable_pitchers_posted_yields_empty_well_formed_frame():
    home = _side("BOS", probable=None)
    away = _side("NYY", probable=None)
    schedule = _schedule(_game(12345, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    assert slate.empty
    assert list(slate.columns) == probable_pitchers.SLATE_COLUMNS


def test_no_games_at_all_yields_empty_well_formed_frame():
    slate = probable_pitchers.parse_probable_starters(_schedule(), GAME_DATE)
    assert slate.empty
    assert list(slate.columns) == probable_pitchers.SLATE_COLUMNS

    slate_empty_payload = probable_pitchers.parse_probable_starters({}, GAME_DATE)
    assert slate_empty_payload.empty
    assert list(slate_empty_payload.columns) == probable_pitchers.SLATE_COLUMNS


def test_missing_pitch_hand_defaults_to_none_not_guessed():
    home = _side("BOS", _probable(999999, "Some Guy"))  # no hand field
    away = _side("NYY", _probable(543037, "Gerrit Cole"))
    schedule = _schedule(_game(12345, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    home_row = slate[slate["pitcher"] == 999999].iloc[0]
    assert home_row["pitcher_throws"] is None


def test_multiple_games_each_contribute_rows_independently():
    g1 = _game(1, _side("BOS", _probable(1, "A")), _side("NYY", _probable(2, "B")))
    g2 = _game(2, _side("LAD", _probable(3, "C")), _side("SF", _probable(4, "D")))
    schedule = _schedule(g1, g2)

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    assert len(slate) == 4
    assert set(slate["game_pk"]) == {1, 2}
    assert set(slate["pitcher"]) == {1, 2, 3, 4}


# ---------------------------------------------------------------------------
# to_statcast_team() -- StatsAPI -> Statcast team crosswalk
# ---------------------------------------------------------------------------

def test_to_statcast_team_applies_known_crosswalk_mappings():
    assert probable_pitchers.to_statcast_team("WSH") == "WAS"
    assert probable_pitchers.to_statcast_team("CWS") == "CHW"


def test_to_statcast_team_passes_through_unmapped_codes_unchanged():
    assert probable_pitchers.to_statcast_team("NYY") == "NYY"
    assert probable_pitchers.to_statcast_team("BOS") == "BOS"


def test_to_statcast_team_handles_none_and_lowercase_without_raising():
    assert probable_pitchers.to_statcast_team(None) == ""
    assert probable_pitchers.to_statcast_team("wsh") == "WAS"


def test_crosswalk_applied_within_full_slate_parse():
    home = _side("WSH", _probable(1, "Home Guy"))
    away = _side("CWS", _probable(2, "Away Guy"))
    schedule = _schedule(_game(1, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)

    home_row = slate[slate["pitcher"] == 1].iloc[0]
    away_row = slate[slate["pitcher"] == 2].iloc[0]
    assert home_row["pitcher_team"] == "WAS"
    assert home_row["opponent_team"] == "CHW"
    assert away_row["pitcher_team"] == "CHW"
    assert away_row["opponent_team"] == "WAS"


# ---------------------------------------------------------------------------
# fetch_schedule() -- thin live wrapper, only checked for the no-network guard
# ---------------------------------------------------------------------------

def test_fetch_schedule_raises_clear_error_when_statsapi_unavailable():
    if probable_pitchers.statsapi is not None:
        pytest.skip("MLB-StatsAPI is installed in this environment; guard path not exercised")
    with pytest.raises(ImportError):
        probable_pitchers.fetch_schedule(GAME_DATE)

# ---------------------------------------------------------------------------
# MLBAM team-id fallback (2026-06-29: live schedule omits abbreviation)
# ---------------------------------------------------------------------------

def _side_by_id(team_id, team_name, probable=None):
    """Build a team side using the live payload shape: {id, name, link}, no abbreviation."""
    side = {"team": {"id": team_id, "name": team_name, "link": f"/api/v1/teams/{team_id}"}}
    if probable is not None:
        side["probablePitcher"] = probable
    return side


def test_parse_probable_starters_falls_back_to_mlbam_id_when_abbreviation_absent():
    """Live StatsAPI schedule omits team.abbreviation; MLBAM_TEAM_ID_TO_ABBREV covers it."""
    home = _side_by_id(110, "Baltimore Orioles", _probable(669358, "Shane Baz"))
    away = _side_by_id(145, "Chicago White Sox", _probable(680732, "Sean Burke"))
    schedule = _schedule(_game(12345, home, away))

    slate = probable_pitchers.parse_probable_starters(schedule, GAME_DATE)
    assert len(slate) == 2

    baz = slate[slate["pitcher"] == 669358].iloc[0]
    burke = slate[slate["pitcher"] == 680732].iloc[0]

    # id 110 -> "BAL" -> to_statcast_team -> "BAL" (no crosswalk needed)
    assert baz["pitcher_team"] == "BAL"
    # id 145 -> "CWS" -> to_statcast_team -> "CHW"
    assert burke["pitcher_team"] == "CHW"
    # Opponent crosswalk works too
    assert baz["opponent_team"] == "CHW"
    assert burke["opponent_team"] == "BAL"


def test_mlbam_team_id_to_abbrev_covers_all_thirty_teams():
    """Spot-checks that the MLBAM dict has 30 entries and known ids are right."""
    d = probable_pitchers.MLBAM_TEAM_ID_TO_ABBREV
    assert len(d) == 30
    assert d[110] == "BAL"
    assert d[145] == "CWS"
    assert d[147] == "NYY"
    assert d[120] == "WSH"
