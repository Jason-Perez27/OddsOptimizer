# Cadence Automation (scheduled refresh/settle/retrain/report) — Design Spec

**Date:** 2026-06-29
**Status:** Approved — ready for implementation (Sonnet)
**Related:** `docs/runbook_go_live.md`, `2026-06-29-live-forward-validation-go-live-design.md`
(the four cadences A–D). New: `scripts/run_cadence.py` + scheduled-task definitions;
no change to the existing CLIs.

## Goal

Make the forward-only Track-B record accumulate **hands-off** during the
assessment window. The decision log already defines exactly four cadences and
deliberately deferred the orchestrator; this wires those same CLIs into the OS
scheduler so they run without manual babysitting — the scheduler IS the
orchestrator (no `daily_driver.py`).

## The four cadences (unchanged from the go-live design)

| id | what | when (ET) | command |
|----|------|-----------|---------|
| A | morning `refresh` (authoritative snapshot) | ~10:00, after lines post / before first pitch | `python -m src.pipeline.refresh` |
| B | `settle --window-days 4` | ~12:00, after Statcast finalizes | `python -m src.pipeline.settle --window-days 4` |
| C | weekly retrain | weekly (matches walk-forward step) | `python scripts/run_backtest.py --fit-only` |
| D | weekly live `report` | weekly | `python -m src.backtest.report` (results mode) |

Rules carried from the runbook and enforced by the wrapper:
- **`refresh` is authoritative and morning-only** — the wrapper must NOT re-run
  it later in the day (cadence A fires once). A crash-recovery manual re-run stays
  a human action, pre-first-pitch only.
- `settle` is idempotent/repeatable; safe to run on a schedule.
- Times are **ET-anchored**; the scheduler fires in the machine's local time, so
  the setup step converts ET→local once and documents the offset.

## Design

A single thin wrapper, `scripts/run_cadence.py {refresh|settle|retrain|report}`,
that each scheduled task invokes. Why a wrapper rather than scheduling the CLIs
directly:

- **Logging:** append a timestamped line + captured stdout/stderr to
  `logs/cadence_{name}_{YYYY-MM}.log` so a missed/failed run is visible after the
  fact (a headless scheduled task has no console).
- **Run manifest / heartbeat:** write `logs/last_run.json`
  (`{cadence, started_at, finished_at, exit_code, ok}`) so the dashboard can later
  surface "last refresh ran at…" and flag a stale/failed cadence.
- **Exit-code discipline:** non-zero on failure so the scheduler records it; an
  `EmptySlateError` from `refresh` on a true off-day is logged as a clean skip
  (exit 0), not a failure (distinguish via the exception type).
- **Single source of the commands:** keeps the ET→local and flag details in one
  reviewed place, not duplicated across four task definitions.

`run_cadence.py` shells the existing entry points (subprocess or direct import of
their `main()`); it adds **no business logic** and must never re-run `refresh`
itself (it runs the one action it's named for).

### Scheduling on the user's machine (Windows — primary)

Deliver a PowerShell setup script `scripts/install_cadences.ps1` that registers
four Windows Task Scheduler jobs via `schtasks`/`Register-ScheduledTask`, each
running `python <repo>\scripts\run_cadence.py <name>` with the working directory
set to the repo root and the project venv's python. Document the ET→local
conversion at the top (the operator sets their local times). Include the exact
`schtasks /Create` lines so it's reviewable and reversible (`/Delete`).

### Portable alternatives (documented, not required)

- **cron** (Mac/Linux): a `crontab` block with the four entries.
- **GitHub Actions**: a `schedule:` workflow — only viable if the data lives
  somewhere the runner can read/write (the repo's `data/` is gitignored), so note
  it as future infra, not the default.

Pick Windows Task Scheduler as the shipped path (matches the user's environment);
the others are appendix.

## Tests
- `scripts/run_cadence.py` core is import-testable: a `dispatch(name, runner)` that
  maps name→command and handles exit-code/skip logic, with the actual subprocess
  injected. Assert: unknown name errors; `refresh` EmptySlate → exit 0 + logged
  skip; a failing runner → non-zero + `last_run.json` ok=false; success → ok=true.
- No test invokes the real CLIs/network (consistent with the repo).

## Verification
1. `pytest tests/test_run_cadence.py` green.
2. Dry-run each: `python scripts/run_cadence.py settle` against the local data dir
   writes a log line + updates `last_run.json`.
3. Register tasks with the PS script on the user's machine; confirm Task Scheduler
   shows four jobs and a manual "Run" of the settle task produces a log entry.

## Out of scope
- Alerting/notification on failure (email/Slack) — `last_run.json` + the dashboard
  heartbeat is v1; push alerts deferred.
- A `POST /api/refresh` web trigger (stays out — refresh is morning-authoritative).
- Cloud hosting of the scheduler.

## Decision-log entry to add (newest at top)
> **2026-06-29 — Automated the four cadences.** Wired the existing
> refresh/settle/retrain/report CLIs into the OS scheduler via a thin
> `scripts/run_cadence.py` wrapper (logging + `last_run.json` heartbeat +
> exit-code/skip discipline) and a Windows Task Scheduler install script;
> cron/Actions documented as alternates. No new business logic, no orchestrator
> daemon — the scheduler is the orchestrator, per the go-live design. Enforces the
> morning-authoritative `refresh` rule (fires once; off-day EmptySlate is a clean
> skip, not a failure).
