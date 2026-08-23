"""
Unit tests for scripts/run_cadence.py (cadence automation, task #12 extension).

Design: docs/design/specs/2026-06-29-cadence-automation-design.md
("Tests" section).

Strategy (matches repo's no-network test style):
- dispatch(name, runner) is the import-testable core; every test below injects
  a fake runner that returns (returncode, stdout, stderr) tuples — no real
  subprocess calls, no CLI, no network.
- The EmptySlateError off-day skip is signalled by the fake runner returning
  exit code 2 + the _EMPTY_SLATE_SKIP_MSG sentinel in stderr, exactly as
  _subprocess_runner() does for the real refresh process.

Assertions (per spec):
1. Unknown cadence name raises ValueError.
2. refresh EmptySlate (off-day) → exit 0 + ok=true in last_run.json + SKIP in log.
3. Failing runner → non-zero exit + ok=false in last_run.json.
4. Successful runner → exit 0 + ok=true in last_run.json.

Run with: pytest tests/test_run_cadence.py -v
"""

import json
import os

import pytest

import scripts.run_cadence as rc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runner_ok(argv):
    """Fake runner: always succeeds."""
    return 0, "done\n", ""


def _runner_fail(argv):
    """Fake runner: always fails with exit code 1."""
    return 1, "", "something went wrong\n"


def _runner_empty_slate(argv):
    """
    Fake runner simulating refresh's EmptySlateError on an off-day.
    Uses exit code 2 + sentinel string exactly as _subprocess_runner() does.
    """
    return 2, "", rc._EMPTY_SLATE_SKIP_MSG + "\n"


# ---------------------------------------------------------------------------
# 1. Unknown cadence name raises ValueError
# ---------------------------------------------------------------------------

def test_unknown_cadence_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(rc, "LAST_RUN_PATH", str(tmp_path / "logs" / "last_run.json"))

    with pytest.raises(ValueError, match="Unknown cadence"):
        rc.dispatch("bogus", _runner_ok)


# ---------------------------------------------------------------------------
# 2. refresh EmptySlate (off-day) → exit 0 + ok=true + SKIP in log
# ---------------------------------------------------------------------------

def test_empty_slate_skip_is_exit_zero(tmp_path, monkeypatch):
    logs_dir = str(tmp_path / "logs")
    last_run = str(tmp_path / "logs" / "last_run.json")
    monkeypatch.setattr(rc, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(rc, "LAST_RUN_PATH", last_run)

    exit_code = rc.dispatch("refresh", _runner_empty_slate)

    assert exit_code == 0, "EmptySlateError on an off-day must be a clean skip (exit 0)"

    heartbeat = json.loads(open(last_run).read())
    assert heartbeat["ok"] is True
    assert heartbeat["exit_code"] == 0
    assert heartbeat["cadence"] == "refresh"

    # Log file must contain the SKIP status word
    log_files = list(tmp_path.glob("logs/cadence_refresh_*.log"))
    assert log_files, "Monthly log file must be written"
    log_text = log_files[0].read_text()
    assert "SKIP" in log_text


# ---------------------------------------------------------------------------
# 3. Failing runner → non-zero exit + ok=false in last_run.json
# ---------------------------------------------------------------------------

def test_failing_runner_nonzero_and_ok_false(tmp_path, monkeypatch):
    logs_dir = str(tmp_path / "logs")
    last_run = str(tmp_path / "logs" / "last_run.json")
    monkeypatch.setattr(rc, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(rc, "LAST_RUN_PATH", last_run)

    exit_code = rc.dispatch("settle", _runner_fail)

    assert exit_code != 0, "A failing runner must produce non-zero exit code"

    heartbeat = json.loads(open(last_run).read())
    assert heartbeat["ok"] is False
    assert heartbeat["exit_code"] != 0
    assert heartbeat["cadence"] == "settle"

    # FAIL status must appear in the log
    log_files = list(tmp_path.glob("logs/cadence_settle_*.log"))
    assert log_files, "Monthly log file must be written on failure"
    log_text = log_files[0].read_text()
    assert "FAIL" in log_text


# ---------------------------------------------------------------------------
# 4. Successful runner → exit 0 + ok=true in last_run.json
# ---------------------------------------------------------------------------

def test_success_ok_true(tmp_path, monkeypatch):
    logs_dir = str(tmp_path / "logs")
    last_run = str(tmp_path / "logs" / "last_run.json")
    monkeypatch.setattr(rc, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(rc, "LAST_RUN_PATH", last_run)

    exit_code = rc.dispatch("retrain", _runner_ok)

    assert exit_code == 0

    heartbeat = json.loads(open(last_run).read())
    assert heartbeat["ok"] is True
    assert heartbeat["exit_code"] == 0
    assert heartbeat["cadence"] == "retrain"

    # OK status must appear in the log
    log_files = list(tmp_path.glob("logs/cadence_retrain_*.log"))
    assert log_files, "Monthly log file must be written on success"
    log_text = log_files[0].read_text()
    assert "OK" in log_text


# ---------------------------------------------------------------------------
# 5. All four cadence names are registered (no typo in the dispatch table)
# ---------------------------------------------------------------------------

def test_all_four_cadences_registered():
    for name in ("refresh", "settle", "retrain", "report"):
        assert name in rc._CADENCE_ARGV, f"Cadence {name!r} missing from dispatch table"


# ---------------------------------------------------------------------------
# 6. Heartbeat keys are complete and correct types
# ---------------------------------------------------------------------------

def test_heartbeat_schema(tmp_path, monkeypatch):
    logs_dir = str(tmp_path / "logs")
    last_run = str(tmp_path / "logs" / "last_run.json")
    monkeypatch.setattr(rc, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(rc, "LAST_RUN_PATH", last_run)

    rc.dispatch("report", _runner_ok)

    heartbeat = json.loads(open(last_run).read())
    assert set(heartbeat.keys()) == {"cadence", "started_at", "finished_at", "exit_code", "ok"}
    assert isinstance(heartbeat["ok"], bool)
    assert isinstance(heartbeat["exit_code"], int)
    assert heartbeat["cadence"] == "report"
