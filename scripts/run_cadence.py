"""
Thin wrapper that the OS scheduler invokes for each of the four cadences
(refresh / settle / retrain / report).

Why this exists (spec: docs/design/specs/2026-06-29-cadence-automation-design.md):
- A headless scheduled task has no console; this wrapper captures stdout/stderr
  and appends a timestamped record to logs/cadence_{name}_{YYYY-MM}.log so
  missed/failed runs are visible after the fact.
- It writes logs/last_run.json ({cadence, started_at, finished_at, exit_code,
  ok}) so the dashboard can later surface "last refresh ran at…" and flag a
  stale/failed cadence.
- Exit-code discipline: non-zero on failure so the scheduler records it.
  A refresh EmptySlateError on a true off-day is a clean skip (exit 0), not
  a failure — the distinction is the exception type, not a string match.
- Single source of truth for the CLI commands and flags (keeps ET→local note
  and --window-days / --fit-only details in one reviewed place, not scattered
  across four task definitions).

This module adds NO business logic and never re-runs refresh itself — it runs
exactly the one named action. The five cadences are:
  A  refresh   python -m src.pipeline.refresh
  B  settle    python -m src.pipeline.settle --window-days 4
  C  retrain   python scripts/run_backtest.py --fit-only
  D  report    python -m src.backtest.report
  E  ticks     python -m src.data.underdog_ticks

Cadence E (2026-08, CLV feature) is purely additive: it appends genuinely new
Underdog price ticks to data/raw/underdog_ticks/ and never touches
line_picks.csv, the predictions partition, or the frozen morning snapshot.
Unlike A-D it repeats through the day (see scripts/install_cadences.ps1's
-RepeatInterval) rather than firing once — a single tick log needs many polls
across an evening/day to actually capture line movement.

Design: dependency-injected `runner` so tests call dispatch() with a fake
runner and never invoke a real CLI or touch the network. The CLI path at the
bottom just builds the real subprocess runner and calls dispatch().
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Known cadences — the only place the commands live
# ---------------------------------------------------------------------------

# Each value is the argv list passed to the runner (first element is the
# Python executable; runner replaces it at call time for the venv python).
_CADENCE_ARGV: dict[str, list[str]] = {
    "refresh": [sys.executable, "-m", "src.pipeline.refresh"],
    "settle":  [sys.executable, "-m", "src.pipeline.settle", "--window-days", "4"],
    "retrain": [sys.executable, "scripts/run_backtest.py", "--fit-only"],
    "report":  [sys.executable, "-m", "src.backtest.report"],
    "ticks":   [sys.executable, "-m", "src.data.underdog_ticks"],
}

LOGS_DIR = "logs"
LAST_RUN_PATH = os.path.join(LOGS_DIR, "last_run.json")

# Sentinel string written to the log when refresh exits via EmptySlateError
# (true off-day — clean skip, not a failure).
_EMPTY_SLATE_SKIP_MSG = "SKIP (EmptySlateError: no games today — off-day, clean skip)"


# ---------------------------------------------------------------------------
# Core: import-testable dispatch
# ---------------------------------------------------------------------------

def dispatch(name: str, runner) -> int:
    """
    Run the cadence named `name` via the injected `runner` callable.

    `runner(argv: list[str]) -> (returncode: int, stdout: str, stderr: str)`

    Returns the exit code (0 on success or off-day skip; non-zero on failure).
    Side effects: writes to the monthly log file and updates last_run.json.

    Raises ValueError immediately for unknown cadence names so the error is
    visible at schedule-registration time, not silently swallowed.
    """
    if name not in _CADENCE_ARGV:
        known = ", ".join(sorted(_CADENCE_ARGV))
        raise ValueError(f"Unknown cadence {name!r}. Known: {known}")

    os.makedirs(LOGS_DIR, exist_ok=True)

    argv = _CADENCE_ARGV[name]
    started_at = datetime.now(timezone.utc)
    started_str = started_at.isoformat()

    returncode, stdout, stderr = runner(argv)

    finished_at = datetime.now(timezone.utc)
    finished_str = finished_at.isoformat()

    # Detect the off-day EmptySlateError: the runner signals it by returning
    # the special exit code 2 and a sentinel in stderr (see _subprocess_runner).
    # For test runners, check for the sentinel in stderr directly.
    is_empty_slate_skip = (
        returncode == 2 and _EMPTY_SLATE_SKIP_MSG in (stderr or "")
    )

    if is_empty_slate_skip:
        exit_code = 0
        ok = True
        log_status = "SKIP"
        log_detail = _EMPTY_SLATE_SKIP_MSG
    elif returncode == 0:
        exit_code = 0
        ok = True
        log_status = "OK"
        log_detail = ""
    else:
        exit_code = returncode
        ok = False
        log_status = "FAIL"
        log_detail = f"exit_code={returncode}"

    # --- Monthly log file ---
    month_tag = started_at.strftime("%Y-%m")
    log_path = os.path.join(LOGS_DIR, f"cadence_{name}_{month_tag}.log")
    _append_log(log_path, name, started_str, finished_str, log_status, log_detail, stdout, stderr)

    # --- Heartbeat JSON ---
    heartbeat = {
        "cadence": name,
        "started_at": started_str,
        "finished_at": finished_str,
        "exit_code": exit_code,
        "ok": ok,
    }
    with open(LAST_RUN_PATH, "w", encoding="utf-8") as fh:
        json.dump(heartbeat, fh, indent=2)

    return exit_code


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _append_log(
    log_path: str,
    name: str,
    started_str: str,
    finished_str: str,
    status: str,
    detail: str,
    stdout: str,
    stderr: str,
) -> None:
    sep = "-" * 72
    lines = [
        sep,
        f"cadence={name}  started={started_str}  finished={finished_str}  status={status}",
    ]
    if detail:
        lines.append(f"detail: {detail}")
    if stdout and stdout.strip():
        lines.append("--- stdout ---")
        lines.append(stdout.rstrip())
    if stderr and stderr.strip():
        lines.append("--- stderr ---")
        lines.append(stderr.rstrip())
    lines.append("")  # trailing blank line for readability
    entry = "\n".join(lines) + "\n"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry)


# ---------------------------------------------------------------------------
# Real subprocess runner (used by the CLI path only; never by tests)
# ---------------------------------------------------------------------------

def _subprocess_runner(argv: list[str]):
    """
    Run `argv` as a subprocess, capture output, and return
    (returncode, stdout, stderr).

    EmptySlateError special-casing: refresh's main() catches EmptySlateError
    and exits 0 (it's a clean exit per the runbook). To let dispatch()
    distinguish an off-day skip from a normal success, we inspect stderr for
    the exception name and re-signal with exit code 2 + the sentinel string.
    This keeps the distinction entirely inside the wrapper without touching
    refresh.py's own exit logic.
    """
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
    )
    returncode = result.returncode
    stdout = result.stdout
    stderr = result.stderr

    # EmptySlateError exits 0 from refresh's main(); detect it so dispatch()
    # can log it as a clean skip rather than a success that "did something".
    if returncode == 0 and "EmptySlateError" in stderr:
        returncode = 2
        stderr = _EMPTY_SLATE_SKIP_MSG + "\n" + stderr

    return returncode, stdout, stderr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python scripts/run_cadence.py {refresh|settle|retrain|report|ticks}")
        sys.exit(0)

    name = sys.argv[1]
    try:
        exit_code = dispatch(name, _subprocess_runner)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
