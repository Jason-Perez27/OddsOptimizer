"""
Unit tests for src/data/pitcher_logs.py.

Strategy (see engineering:testing-strategy):
- All pybaseball calls are mocked -- no network access, no real Statcast pulls.
  Those belong in a separate manual/integration smoke test, not CI.
- lookup_pitcher_id: name parsing, multi-match resolution, no-match error.
- get_pitcher_season_logs: correct date range construction, default vs explicit
  end_date, pass-through of the pitcher id.
- save_raw: directory creation, filename convention, content round-trip.
- main(): both branches (empty result vs. populated result) via capsys + monkeypatch,
  with sys.exit behavior checked on the empty path.

Run with: pytest tests/test_pitcher_logs.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import pitcher_logs


# ---------------------------------------------------------------------------
# lookup_pitcher_id
# ---------------------------------------------------------------------------

def test_lookup_pitcher_id_single_match(monkeypatch):
    fake_matches = pd.DataFrame(
        {
            "key_mlbam": [543037],
            "mlb_played_last": [2026],
        }
    )
    monkeypatch.setattr(pitcher_logs, "playerid_lookup", lambda last, first: fake_matches)

    player_id = pitcher_logs.lookup_pitcher_id("Gerrit Cole")
    assert player_id == 543037


def test_lookup_pitcher_id_picks_most_recently_active(monkeypatch):
    # Common name with two players; should pick the one with the latest
    # mlb_played_last rather than the first row returned.
    fake_matches = pd.DataFrame(
        {
            "key_mlbam": [111111, 222222],
            "mlb_played_last": [2015, 2026],
        }
    )
    monkeypatch.setattr(pitcher_logs, "playerid_lookup", lambda last, first: fake_matches)

    player_id = pitcher_logs.lookup_pitcher_id("Chris Young")
    assert player_id == 222222


def test_lookup_pitcher_id_no_match_raises(monkeypatch):
    monkeypatch.setattr(
        pitcher_logs, "playerid_lookup", lambda last, first: pd.DataFrame()
    )

    with pytest.raises(ValueError, match="No player found"):
        pitcher_logs.lookup_pitcher_id("Nobody Real")


@pytest.mark.parametrize("bad_name", ["Cole", ""])
def test_lookup_pitcher_id_rejects_single_token_name(bad_name):
    # split(" ", 1) needs at least one space to produce two parts; a bare
    # last name or empty string should fail fast, before calling pybaseball.
    with pytest.raises(ValueError, match="Expected 'First Last'"):
        pitcher_logs.lookup_pitcher_id(bad_name)


def test_lookup_pitcher_id_three_word_name_uses_remainder_as_last(monkeypatch):
    # split(" ", 1) caps at two parts, so "Gerrit Middle Cole" becomes
    # first="Gerrit", last="Middle Cole" rather than raising. Documenting
    # this as intended behavior, not a bug.
    captured = {}
    monkeypatch.setattr(
        pitcher_logs,
        "playerid_lookup",
        lambda last, first: captured.update(last=last, first=first) or pd.DataFrame(
            {"key_mlbam": [1], "mlb_played_last": [2026]}
        ),
    )

    pitcher_logs.lookup_pitcher_id("Gerrit Middle Cole")

    assert captured == {"first": "Gerrit", "last": "Middle Cole"}


# ---------------------------------------------------------------------------
# get_pitcher_season_logs
# ---------------------------------------------------------------------------

def test_get_pitcher_season_logs_builds_correct_date_range(monkeypatch):
    captured = {}

    monkeypatch.setattr(pitcher_logs, "lookup_pitcher_id", lambda name: 543037)

    def fake_statcast_pitcher(start_dt, end_dt, player_id):
        captured["start_dt"] = start_dt
        captured["end_dt"] = end_dt
        captured["player_id"] = player_id
        return pd.DataFrame({"pitch_type": ["FF"]})

    monkeypatch.setattr(pitcher_logs, "statcast_pitcher", fake_statcast_pitcher)

    df = pitcher_logs.get_pitcher_season_logs("Gerrit Cole", 2026, end_date="2026-06-01")

    assert captured["start_dt"] == "2026-03-01"
    assert captured["end_dt"] == "2026-06-01"
    assert captured["player_id"] == 543037
    assert len(df) == 1


def test_get_pitcher_season_logs_defaults_end_date_to_today(monkeypatch):
    monkeypatch.setattr(pitcher_logs, "lookup_pitcher_id", lambda name: 1)

    captured = {}

    def fake_statcast_pitcher(start_dt, end_dt, player_id):
        captured["end_dt"] = end_dt
        return pd.DataFrame()

    monkeypatch.setattr(pitcher_logs, "statcast_pitcher", fake_statcast_pitcher)

    from datetime import date
    pitcher_logs.get_pitcher_season_logs("Gerrit Cole", 2026)

    assert captured["end_dt"] == date.today().isoformat()


# ---------------------------------------------------------------------------
# get_pitcher_logs_by_id
# ---------------------------------------------------------------------------

def test_get_pitcher_logs_by_id_passes_through_args_with_no_name_lookup(monkeypatch):
    # Must not touch playerid_lookup at all -- by-id pull skips name
    # resolution entirely (task #10 settlement path).
    lookup_called = []
    monkeypatch.setattr(
        pitcher_logs, "playerid_lookup",
        lambda last, first: lookup_called.append(True) or pd.DataFrame(),
    )

    captured = {}

    def fake_statcast_pitcher(start_dt, end_dt, player_id):
        captured["start_dt"] = start_dt
        captured["end_dt"] = end_dt
        captured["player_id"] = player_id
        return pd.DataFrame({"pitcher": [543037], "strikeouts": [7]})

    monkeypatch.setattr(pitcher_logs, "statcast_pitcher", fake_statcast_pitcher)

    df = pitcher_logs.get_pitcher_logs_by_id(543037, "2026-06-26", "2026-06-27")

    assert captured == {
        "start_dt": "2026-06-26", "end_dt": "2026-06-27", "player_id": 543037,
    }
    assert len(df) == 1
    assert not lookup_called


def test_get_pitcher_logs_by_id_returns_empty_frame_passthrough(monkeypatch):
    monkeypatch.setattr(pitcher_logs, "statcast_pitcher", lambda s, e, pid: pd.DataFrame())

    df = pitcher_logs.get_pitcher_logs_by_id(1, "2026-06-26", "2026-06-27")

    assert df.empty


# ---------------------------------------------------------------------------
# save_raw
# ---------------------------------------------------------------------------

def test_save_raw_creates_dir_and_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    out_path = pitcher_logs.save_raw(df, "Gerrit Cole", 2026)

    expected = tmp_path / "data" / "raw" / "gerrit_cole_2026_statcast.csv"
    assert Path(out_path).resolve() == expected.resolve()
    assert Path(out_path).exists()

    round_trip = pd.read_csv(out_path)
    assert len(round_trip) == 2
    assert list(round_trip.columns) == ["a", "b"]


def test_save_raw_filename_uses_lowercase_underscores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"a": [1]})

    out_path = pitcher_logs.save_raw(df, "Shohei Ohtani", 2025)

    assert "shohei_ohtani_2025_statcast.csv" in out_path.replace("\\", "/")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_exits_cleanly_on_empty_dataframe(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pitcher_logs.py", "--name", "Nobody Real", "--season", "2026"])
    monkeypatch.setattr(pitcher_logs, "get_pitcher_season_logs", lambda name, season: pd.DataFrame())

    save_raw_called = []
    monkeypatch.setattr(pitcher_logs, "save_raw", lambda *a, **k: save_raw_called.append(True))

    with pytest.raises(SystemExit) as exc_info:
        pitcher_logs.main()

    assert exc_info.value.code == 0
    assert not save_raw_called
    out = capsys.readouterr().out
    assert "No Statcast rows found" in out


def test_main_saves_and_reports_on_populated_dataframe(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pitcher_logs.py", "--name", "Gerrit Cole", "--season", "2026"])
    fake_df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [98.1, 86.4]})
    monkeypatch.setattr(pitcher_logs, "get_pitcher_season_logs", lambda name, season: fake_df)
    monkeypatch.setattr(pitcher_logs, "save_raw", lambda df, name, season: "data/raw/gerrit_cole_2026_statcast.csv")

    pitcher_logs.main()  # should not raise / exit

    out = capsys.readouterr().out
    assert "Pulled 2 pitch-level rows" in out
    assert "data/raw/gerrit_cole_2026_statcast.csv" in out
