"""
Unit tests for src/data/underdog_ticks.py.

Strategy (matches this project's existing fixture-based testing convention):
no real network calls -- `fetcher` is injected throughout, `now` is an
injected clock. Covers: dedup on an unchanged (over_under_id, over_updated_at,
under_updated_at) triple, a genuinely new row on a changed updated_at, ET
game-date partitioning across the UTC-midnight boundary, an off-day/empty
payload no-op, and idempotency running the poller twice in immediate
succession.

Run with: pytest tests/test_underdog_ticks.py -v
"""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import underdog_ticks


# ---------------------------------------------------------------------------
# Fixture payload builder
# ---------------------------------------------------------------------------

def _line(id_, over_under_id, stat, stat_value, scheduled_at, over_updated_at,
          under_updated_at, over_price="-148", under_price="+124",
          away="NYY", home="BOS", game_id="m-1"):
    return {
        "id": id_,
        "over_under_id": over_under_id,
        "stat_value": stat_value,
        "line_type": "balanced",
        "live_event": False,
        "status": "active",
        "over_under": {
            "appearance_stat": {"appearance_id": f"app-{id_}", "stat": stat, "display_stat": stat},
            "has_alternates": False,
        },
        "options": [
            {"choice": "higher", "american_price": over_price, "decimal_price": 1.68,
             "payout_multiplier": "0.68", "status": "active",
             "selection_header": "Gerrit Cole", "updated_at": over_updated_at},
            {"choice": "lower", "american_price": under_price, "decimal_price": 2.24,
             "payout_multiplier": "1.24", "status": "active",
             "selection_header": "Gerrit Cole", "updated_at": under_updated_at},
        ],
        "_appearance_id": f"app-{id_}",
        "_game_id": game_id,
        "_away": away,
        "_home": home,
        "_scheduled_at": scheduled_at,
    }


def _payload(lines):
    """Build a full over_under_lines payload from a list of `_line(...)` dicts."""
    appearances, players, games = [], [], {}
    out_lines = []
    for ln in lines:
        appearance_id = ln.pop("_appearance_id")
        game_id = ln.pop("_game_id")
        away = ln.pop("_away")
        home = ln.pop("_home")
        scheduled_at = ln.pop("_scheduled_at")
        appearances.append({"id": appearance_id, "player_id": "p-1", "match_id": game_id})
        games[game_id] = {"id": game_id, "sport_id": "MLB", "title": f"{away} @ {home}",
                           "scheduled_at": scheduled_at, "status": "scheduled"}
        out_lines.append(ln)
    players.append({"id": "p-1", "first_name": "Gerrit", "last_name": "Cole"})
    return {
        "over_under_lines": out_lines,
        "appearances": appearances,
        "players": players,
        "games": list(games.values()),
        "solo_games": [],
    }


def _read_ticks(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Dedup on unchanged updated_at / new row on changed updated_at
# ---------------------------------------------------------------------------

def test_poll_once_dedupes_unchanged_updated_at(tmp_path):
    payload = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
    ])
    fetcher = lambda: payload

    first = underdog_ticks.poll_once(str(tmp_path), fetcher=fetcher)
    assert first["n_rows_written"] == 1

    second = underdog_ticks.poll_once(str(tmp_path), fetcher=fetcher)
    assert second["n_rows_written"] == 0

    path = underdog_ticks._log_path(str(tmp_path), "2026-08-23")
    rows = _read_ticks(path)
    assert len(rows) == 1


def test_poll_once_writes_new_row_on_price_move(tmp_path):
    payload_v1 = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
    ])
    payload_v2 = _payload([
        _line("l2", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T15:30:00Z", "2026-08-23T14:00:00Z",  # over side moved
              over_price="-160"),
    ])

    underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: payload_v1)
    summary = underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: payload_v2)
    assert summary["n_rows_written"] == 1

    path = underdog_ticks._log_path(str(tmp_path), "2026-08-23")
    rows = _read_ticks(path)
    assert len(rows) == 2
    assert float(rows[0]["over_american"]) == pytest.approx(-148.0)
    assert float(rows[1]["over_american"]) == pytest.approx(-160.0)


# ---------------------------------------------------------------------------
# ET game-date partitioning across the UTC midnight boundary
# ---------------------------------------------------------------------------

def test_et_partitioning_across_utc_midnight_boundary(tmp_path):
    """
    A 2026-08-24T02:10:00Z first pitch (during US EDT, UTC-4) is
    2026-08-23 22:10 ET -- it must partition under game_date=2026-08-23,
    the GAME's Eastern calendar date, even though the poll happens at
    2026-08-24T03:00:00Z (already the next UTC calendar day).
    """
    payload = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-24T02:10:00Z",
              "2026-08-24T02:00:00Z", "2026-08-24T02:00:00Z"),
    ])
    poll_time = datetime(2026, 8, 24, 3, 0, 0, tzinfo=timezone.utc)

    summary = underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: payload, now=poll_time)
    assert summary["n_rows_written"] == 1

    expected_path = underdog_ticks._log_path(str(tmp_path), "2026-08-23")
    assert os.path.exists(expected_path)
    wrong_path = underdog_ticks._log_path(str(tmp_path), "2026-08-24")
    assert not os.path.exists(wrong_path)


def test_et_date_str_handles_est_and_edt():
    # Mid-summer (EDT, UTC-4): 2026-08-23T23:10:00Z -> 2026-08-23 19:10 ET.
    assert underdog_ticks.et_date_str("2026-08-23T23:10:00Z") == "2026-08-23"
    # Mid-winter (EST, UTC-5): 2026-01-15T04:30:00Z -> 2026-01-14 23:30 ET.
    assert underdog_ticks.et_date_str("2026-01-15T04:30:00Z") == "2026-01-14"


# ---------------------------------------------------------------------------
# Empty slate / off-day no-op
# ---------------------------------------------------------------------------

def test_poll_once_empty_payload_is_a_clean_noop(tmp_path):
    empty_payload = {"over_under_lines": [], "appearances": [], "players": [], "games": [], "solo_games": []}
    summary = underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: empty_payload)

    assert summary["n_lines_seen"] == 0
    assert summary["n_rows_written"] == 0
    assert summary["n_games"] == 0
    assert set(summary.keys()) == {"polled_at", "n_lines_seen", "n_rows_written", "n_games"}
    # Nothing written to disk at all.
    assert list(Path(str(tmp_path)).rglob("*.csv")) == []


# ---------------------------------------------------------------------------
# Idempotency running the poller twice in immediate succession
# ---------------------------------------------------------------------------

def test_running_poller_twice_in_one_minute_does_not_double_write(tmp_path):
    payload = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
        _line("l2", "ou-2", "walks_allowed", "1.5", "2026-08-23T22:10:00Z",
              "2026-08-23T13:45:00Z", "2026-08-23T13:45:00Z", game_id="m-2",
              away="HOU", home="SEA"),
    ])
    same_minute = datetime(2026, 8, 23, 14, 5, 0, tzinfo=timezone.utc)

    first = underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: payload, now=same_minute)
    second = underdog_ticks.poll_once(str(tmp_path), fetcher=lambda: payload, now=same_minute)

    assert first["n_rows_written"] == 2
    assert second["n_rows_written"] == 0

    path = underdog_ticks._log_path(str(tmp_path), "2026-08-23")
    rows = _read_ticks(path)
    assert len(rows) == 2  # not 4 -- the second run wrote nothing new


# ---------------------------------------------------------------------------
# n_lines_seen / n_games fan out across multiple stats from ONE fetch
# ---------------------------------------------------------------------------

def test_poll_once_fans_out_to_multiple_stats_from_a_single_fetch(tmp_path):
    calls = {"n": 0}

    payload = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
        _line("l2", "ou-2", "walks_allowed", "1.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
    ])

    def fetcher():
        calls["n"] += 1
        return payload

    summary = underdog_ticks.poll_once(str(tmp_path), fetcher=fetcher)

    assert calls["n"] == 1  # exactly one HTTP call regardless of stat fan-out
    assert summary["n_lines_seen"] == 2  # one strikeouts line + one walks_allowed line
    assert summary["n_rows_written"] == 2
    assert summary["n_games"] == 1  # both lines are the same game (m-1)


# ---------------------------------------------------------------------------
# Bounded retry (GitHub Actions hardening)
# ---------------------------------------------------------------------------

def test_fetch_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}
    sleeps = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient blip")
        return {"ok": True}

    result = underdog_ticks._fetch_with_retry(flaky, sleep_fn=sleeps.append)

    assert result == {"ok": True}
    assert calls["n"] == 3
    # Two failures before the success -> two backoff sleeps, first two delays.
    assert sleeps == [2, 4]


def test_fetch_with_retry_exhausts_budget_and_raises_last_error():
    calls = {"n": 0}
    sleeps = []

    def always_fails():
        calls["n"] += 1
        raise ConnectionError(f"blip {calls['n']}")

    with pytest.raises(ConnectionError, match="blip 4"):
        underdog_ticks._fetch_with_retry(always_fails, sleep_fn=sleeps.append)

    # 1 initial attempt + 3 retries = 4 total attempts, all three delays used.
    assert calls["n"] == 4
    assert sleeps == [2, 4, 8]


def test_fetch_with_retry_never_retries_a_403():
    calls = {"n": 0}
    sleeps = []

    def rejected():
        calls["n"] += 1
        raise RuntimeError("Underdog rejected the request (403) -- endpoint shape may have changed")

    with pytest.raises(RuntimeError, match="403"):
        underdog_ticks._fetch_with_retry(rejected, sleep_fn=sleeps.append)

    assert calls["n"] == 1  # no retries at all
    assert sleeps == []


def test_fetch_with_retry_never_retries_a_403_style_requests_exception():
    class FakeResponse:
        status_code = 403

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("403 Client Error")
            self.response = FakeResponse()

    calls = {"n": 0}

    def rejected():
        calls["n"] += 1
        raise FakeHTTPError()

    with pytest.raises(Exception):
        underdog_ticks._fetch_with_retry(rejected, sleep_fn=lambda s: None)

    assert calls["n"] == 1


def test_poll_once_retries_transient_fetch_failure_then_succeeds(tmp_path):
    payload = _payload([
        _line("l1", "ou-1", "strikeouts", "6.5", "2026-08-23T23:05:00Z",
              "2026-08-23T14:00:00Z", "2026-08-23T14:00:00Z"),
    ])
    calls = {"n": 0}

    def flaky_fetcher():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient blip")
        return payload

    summary = underdog_ticks.poll_once(
        str(tmp_path), fetcher=flaky_fetcher, sleep_fn=lambda s: None,
    )

    assert calls["n"] == 2
    assert summary["n_rows_written"] == 1


# ---------------------------------------------------------------------------
# poll_at reflects when the fetch actually succeeded, never a pre-fetch time
# ---------------------------------------------------------------------------

def test_poll_at_is_captured_after_fetch_succeeds_not_before():
    """
    With no injected `now`, poll_once must sample the real clock AFTER the
    (possibly retried) fetch returns -- never before it starts. We prove
    ordering, not just a plausible value, by recording the order in which
    the fetch and the clock read happen.
    """
    order = []

    class RecordingDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            order.append("clock_read")
            return super().now(tz)

    def fetcher():
        order.append("fetch")
        return {"over_under_lines": [], "appearances": [], "players": [], "games": [], "solo_games": []}

    original_datetime = underdog_ticks.datetime
    underdog_ticks.datetime = RecordingDatetime
    try:
        underdog_ticks.poll_once("/tmp/unused-does-not-matter", fetcher=fetcher)
    finally:
        underdog_ticks.datetime = original_datetime

    assert order == ["fetch", "clock_read"]
