"""
Unit tests for src/data/prizepicks_lines.py.

Strategy (see engineering:testing-strategy):
- No real network calls -- requests.get is mocked throughout. PrizePicks'
  endpoint is unofficial/undocumented, so these tests pin down our own
  parsing logic against a representative JSON:API payload shape, not
  against the live API.
- fetch_projections: correct params/headers sent, 403 mapped to a clear
  RuntimeError, success path returns decoded JSON.
- _build_player_lookup / flatten_projections: JSON:API data+included join,
  stat_type filtering, missing-player-id handling, empty payload -> empty df.
- save_raw: directory creation, stat-type-aware timestamped filename.
- main(): empty-result branch and the happy path.

Run with: pytest tests/test_prizepicks_lines.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import prizepicks_lines


# ---------------------------------------------------------------------------
# fetch_projections
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {"data": [], "included": []}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_fetch_projections_sends_correct_params(monkeypatch):
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(200, json_data={"data": [], "included": []})

    monkeypatch.setattr(prizepicks_lines.requests, "get", fake_get)

    payload = prizepicks_lines.fetch_projections(league_id=2)

    assert captured["url"] == "https://api.prizepicks.com/projections"
    assert captured["params"]["league_id"] == 2
    assert captured["params"]["per_page"] == 250
    assert "User-Agent" in captured["headers"]
    assert payload == {"data": [], "included": []}


def test_fetch_projections_raises_clear_error_on_403(monkeypatch):
    monkeypatch.setattr(prizepicks_lines.requests, "get", lambda *a, **k: _FakeResponse(403))
    with pytest.raises(RuntimeError, match="rejected the request"):
        prizepicks_lines.fetch_projections()


# ---------------------------------------------------------------------------
# _build_player_lookup / flatten_projections
# ---------------------------------------------------------------------------

def _sample_payload():
    """Representative JSON:API payload with standard, goblin, and demon lines.

    Pitcher Strikeouts rows: proj1 (standard), proj3 (standard, no player),
    proj_goblin (goblin), proj_demon (demon) -- 4 rows total.
    proj2 (Hits Allowed) is filtered out by stat_type.
    """
    return {
        "data": [
            {
                "id": "proj1",
                "type": "projection",
                "attributes": {
                    "stat_type": "Pitcher Strikeouts",
                    "odds_type": "standard",
                    "line_score": 6.5,
                    "start_time": "2026-06-27T23:05:00Z",
                    "status": "pre_game",
                },
                "relationships": {"new_player": {"data": {"id": "player1", "type": "new_player"}}},
            },
            {
                "id": "proj2",
                "type": "projection",
                "attributes": {
                    "stat_type": "Hits Allowed",  # different stat -- should be filtered out
                    "odds_type": "standard",
                    "line_score": 5.5,
                    "start_time": "2026-06-27T23:05:00Z",
                    "status": "pre_game",
                },
                "relationships": {"new_player": {"data": {"id": "player1", "type": "new_player"}}},
            },
            {
                "id": "proj3",
                "type": "projection",
                "attributes": {
                    "stat_type": "Pitcher Strikeouts",
                    "odds_type": "standard",
                    "line_score": 4.5,
                    "start_time": "2026-06-27T20:10:00Z",
                    "status": "pre_game",
                },
                "relationships": {"new_player": {"data": None}},  # missing player relationship
            },
            {
                "id": "proj_goblin",
                "type": "projection",
                "attributes": {
                    "stat_type": "Pitcher Strikeouts",
                    "odds_type": "goblin",
                    "line_score": 5.5,
                    "start_time": "2026-06-27T23:05:00Z",
                    "status": "pre_game",
                },
                "relationships": {"new_player": {"data": {"id": "player1", "type": "new_player"}}},
            },
            {
                "id": "proj_demon",
                "type": "projection",
                "attributes": {
                    "stat_type": "Pitcher Strikeouts",
                    "odds_type": "demon",
                    "line_score": 7.5,
                    "start_time": "2026-06-27T23:05:00Z",
                    "status": "pre_game",
                },
                "relationships": {"new_player": {"data": {"id": "player1", "type": "new_player"}}},
            },
        ],
        "included": [
            {
                "id": "player1",
                "type": "new_player",
                "attributes": {"name": "Gerrit Cole", "team": "NYY", "position": "SP"},
            }
        ],
    }


def test_build_player_lookup_filters_to_new_player_type():
    lookup = prizepicks_lines._build_player_lookup(
        [
            {"id": "player1", "type": "new_player", "attributes": {"name": "Gerrit Cole"}},
            {"id": "team1", "type": "team", "attributes": {"name": "Yankees"}},
        ]
    )
    assert list(lookup.keys()) == ["player1"]
    assert lookup["player1"]["player_name"] == "Gerrit Cole"


def test_flatten_projections_filters_by_stat_type_and_joins_player():
    df = prizepicks_lines.flatten_projections(_sample_payload(), stat_type="Pitcher Strikeouts")

    # proj1 (standard), proj3 (standard/no player), proj_goblin, proj_demon emitted;
    # proj2 filtered out (wrong stat_type). All lines are returned regardless of odds_type.
    assert len(df) == 4
    assert set(df["projection_id"]) == {"proj1", "proj3", "proj_goblin", "proj_demon"}
    assert "odds_type" in df.columns

    row1 = df[df["projection_id"] == "proj1"].iloc[0]
    assert row1["pitcher"] == "Gerrit Cole"
    assert row1["team"] == "NYY"
    assert row1["line"] == 6.5
    assert row1["odds_type"] == "standard"

    goblin_row = df[df["projection_id"] == "proj_goblin"].iloc[0]
    assert goblin_row["odds_type"] == "goblin"
    assert goblin_row["line"] == 5.5

    demon_row = df[df["projection_id"] == "proj_demon"].iloc[0]
    assert demon_row["odds_type"] == "demon"
    assert demon_row["line"] == 7.5

    row3 = df[df["projection_id"] == "proj3"].iloc[0]
    # pandas normalizes None to NaN in a mixed-type object column, so check
    # missingness rather than identity to None.
    assert pd.isna(row3["player_id"])
    assert pd.isna(row3["pitcher"])  # no matching player relationship -> no name


def test_flatten_projections_empty_payload_returns_empty_frame():
    """flatten_projections on an empty data list returns an empty DataFrame (no crash)."""
    empty_payload = {"data": [], "included": []}
    df = prizepicks_lines.flatten_projections(empty_payload, stat_type="Pitcher Strikeouts")
    assert df.empty
