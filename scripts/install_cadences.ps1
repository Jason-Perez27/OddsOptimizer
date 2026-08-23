# install_cadences.ps1 — Register (or delete) the FOUR locally-scheduled
# OddsOptimizer cadences (A-D) in Windows Task Scheduler.
#
# Cadence E (the Underdog tick poller) is NOT registered here as of 2026-08:
# it moved to GitHub Actions (.github/workflows/tick-poller.yml) so line
# collection continues even when this machine is off. See "Why only cadence
# E is cloud-hosted" in docs/runbook_go_live.md. `run_cadence.py ticks`
# still works for manual/local runs and testing -- it's just no longer
# registered as a scheduled task by this script.
#
# Design: docs/design/specs/2026-06-29-cadence-automation-design.md
# ("Scheduling on the user's machine (Windows — primary)").
#
# ─────────────────────────────────────────────────────────────────────────────
# ET → LOCAL TIME NOTE (read before editing the trigger times below)
# ─────────────────────────────────────────────────────────────────────────────
# The cadence times in the runbook are Eastern Time (ET), which is MLB's
# reference clock. Windows Task Scheduler fires in the machine's LOCAL time.
# Convert before registering:
#
#   ET (standard / EST = UTC-5)   ET (daylight / EDT = UTC-4)
#   10:00 ET  →  10:00 EST / 09:00 EDT   (adjust for your local offset)
#
# Example: if your machine is set to Eastern Time (the common case) you can
# use the ET times directly. If you are on Central Time (UTC-6/UTC-5 DST),
# subtract one hour. Set the START_TIME_* variables below to YOUR LOCAL times.
#
# The constraint that matters (from the runbook):
#   A must fire AFTER StatsAPI has probables (~09:00 ET) and BEFORE first
#     pitches (~13:05 ET). 10:00 ET is the recommended target. StatsAPI
#     probables -- not the line source -- is the binding dependency here:
#     Underdog posts the MLB slate the night before (verified growing from a
#     handful of games around 21:45 ET to the full slate by ~23:00 ET), so by
#     game-day morning its lines are already up; StatsAPI's probable-pitcher
#     hydration is what isn't confirmed until closer to first pitch.
#   B must fire AFTER overnight Statcast finalizes (~11:00 ET). 12:00 ET target.
#   C (weekly) fires before the 7-day staleness warning trips. Monday 08:00 ET.
#   D (weekly) fires after B on the same day. Monday 12:30 ET.
#   E (the tick poller, 2026-08 CLV feature) is NOT scheduled by this script
#     at all -- it runs on GitHub Actions instead (every 15 min, continuous,
#     see .github/workflows/tick-poller.yml), specifically so it keeps
#     running when this machine is off. It has no ordering dependency on
#     A/B/C/D and touches nothing they read or write (see
#     src/data/underdog_ticks.py) -- purely additive either way.
#
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Absolute path to the repo root (no trailing backslash).
$REPO_ROOT = "C:\Users\jdper\Documents\OddsOptimizer"

# Python executable inside the project venv.
$PYTHON    = "$REPO_ROOT\venv\Scripts\python.exe"

# Trigger times in YOUR LOCAL TIME (see ET note above).
# Format: "HH:MM" (24-hour).
$START_TIME_A = "10:00"   # daily   — cadence A: morning refresh
$START_TIME_B = "12:00"   # daily   — cadence B: settle --window-days 4
$START_TIME_C = "08:00"   # weekly  — cadence C: retrain (--fit-only)
$START_TIME_D = "12:30"   # weekly  — cadence D: live report

# Day of week for the weekly tasks (C and D).
$WEEKLY_DAY = "MON"

# (No $START_TIME_E / repeat-interval config here -- cadence E runs on
# GitHub Actions now, not Windows Task Scheduler. See
# .github/workflows/tick-poller.yml.)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the schtasks /Create argument string
# ─────────────────────────────────────────────────────────────────────────────
# -RepeatInterval / -Duration: general support for a repeating trigger,
# originally added (2026-08) for cadence E's tick poller. Cadence E no
# longer runs locally (moved to GitHub Actions -- see above), so nothing
# below currently passes these, but the capability is kept in case a future
# local repeating cadence needs it. schtasks' documented contract for a
# repeating trigger is /RI <minutes> together with /DU <HHHH:MM> (or /ET
# <end-time> — this script always uses /DU); /RI without /DU or /ET
# defaults to a 1-hour repeat window. VERIFY this against `schtasks /Create
# /?` on your own machine before relying on it — Windows version/locale
# differences are exactly the kind of thing worth a 10-second spot check
# before you trust an unattended scheduled task with it.
function Register-Cadence {
    param(
        [string]$TaskName,
        [string]$Cadence,
        [string]$ScheduleType,     # "DAILY" or "WEEKLY"
        [string]$StartTime,
        [string]$Day = "",         # used only for WEEKLY
        [int]$RepeatInterval = 0,  # minutes; 0 = no repetition (cadences A-D)
        [string]$Duration = ""     # HHHH:MM; required when RepeatInterval > 0
    )

    $cmd = "`"$PYTHON`" `"$REPO_ROOT\scripts\run_cadence.py`" $Cadence"

    $schtasksArgs = @(
        "/Create", "/F",
        "/TN", "OddsOptimizer\$TaskName",
        "/TR", $cmd,
        "/SC", $ScheduleType,
        "/ST", $StartTime,
        "/SD", (Get-Date -Format "MM/dd/yyyy"),
        "/RL", "HIGHEST"
    )
    if ($ScheduleType -eq "WEEKLY" -and $Day) {
        $schtasksArgs += "/D", $Day
    }
    if ($RepeatInterval -gt 0) {
        if (-not $Duration) {
            Write-Error "Register-Cadence: -RepeatInterval given without -Duration for $TaskName -- aborting registration to avoid the 1-hour schtasks default."
            return
        }
        $schtasksArgs += "/RI", $RepeatInterval, "/DU", $Duration
    }

    Write-Host "Registering: OddsOptimizer\$TaskName ..."
    & schtasks @schtasksArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED to register $TaskName (exit $LASTEXITCODE)"
    } else {
        Write-Host "  OK: OddsOptimizer\$TaskName"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER — run this block to install all four local tasks
# ─────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Installing OddsOptimizer cadence tasks ==="
Write-Host "Repo root : $REPO_ROOT"
Write-Host "Python    : $PYTHON"
Write-Host ""

Register-Cadence -TaskName "A-refresh" -Cadence "refresh" `
    -ScheduleType "DAILY" -StartTime $START_TIME_A

Register-Cadence -TaskName "B-settle"  -Cadence "settle"  `
    -ScheduleType "DAILY" -StartTime $START_TIME_B

Register-Cadence -TaskName "C-retrain" -Cadence "retrain" `
    -ScheduleType "WEEKLY" -StartTime $START_TIME_C -Day $WEEKLY_DAY

Register-Cadence -TaskName "D-report"  -Cadence "report"  `
    -ScheduleType "WEEKLY" -StartTime $START_TIME_D -Day $WEEKLY_DAY

Write-Host ""
Write-Host "Done. Confirm in Task Scheduler: taskschd.msc -> Task Scheduler Library -> OddsOptimizer"
Write-Host "(Cadence E -- the tick poller -- is not installed by this script; it runs on"
Write-Host " GitHub Actions. See .github/workflows/tick-poller.yml and the runbook.)"
Write-Host ""

# ─────────────────────────────────────────────────────────────────────────────
# EQUIVALENT schtasks /Create LINES (reviewable, copy-pasteable)
# ─────────────────────────────────────────────────────────────────────────────
# Run these manually if you prefer explicit one-liners over the function above.
# Replace <REPO_ROOT> and <PYTHON> with the actual paths.
#
# A — daily morning refresh (~10:00 local, ET-anchored):
#   schtasks /Create /F /TN "OddsOptimizer\A-refresh" /TR "\"<PYTHON>\" \"<REPO_ROOT>\scripts\run_cadence.py\" refresh" /SC DAILY /ST 10:00 /RL HIGHEST
#
# B — daily settle --window-days 4 (~12:00 local, ET-anchored):
#   schtasks /Create /F /TN "OddsOptimizer\B-settle"  /TR "\"<PYTHON>\" \"<REPO_ROOT>\scripts\run_cadence.py\" settle"  /SC DAILY /ST 12:00 /RL HIGHEST
#
# C — weekly retrain --fit-only (Monday ~08:00 local, ET-anchored):
#   schtasks /Create /F /TN "OddsOptimizer\C-retrain" /TR "\"<PYTHON>\" \"<REPO_ROOT>\scripts\run_cadence.py\" retrain" /SC WEEKLY /D MON /ST 08:00 /RL HIGHEST
#
# D — weekly live report (Monday ~12:30 local, ET-anchored):
#   schtasks /Create /F /TN "OddsOptimizer\D-report"  /TR "\"<PYTHON>\" \"<REPO_ROOT>\scripts\run_cadence.py\" report"  /SC WEEKLY /D MON /ST 12:30 /RL HIGHEST
#
# (Cadence E, the tick poller, is intentionally absent here -- it runs on
# GitHub Actions, not schtasks. See .github/workflows/tick-poller.yml.)
#
# ─────────────────────────────────────────────────────────────────────────────
# DELETE / UNREGISTER (reversible)
# ─────────────────────────────────────────────────────────────────────────────
# To remove all four local tasks:
#   schtasks /Delete /TN "OddsOptimizer\A-refresh" /F
#   schtasks /Delete /TN "OddsOptimizer\B-settle"  /F
#   schtasks /Delete /TN "OddsOptimizer\C-retrain" /F
#   schtasks /Delete /TN "OddsOptimizer\D-report"  /F
#
# Or from PowerShell (requires admin):
#   Unregister-ScheduledTask -TaskName "A-refresh" -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "B-settle"  -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "C-retrain" -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "D-report"  -TaskPath "\OddsOptimizer\" -Confirm:$false
#
# To stop the cloud poller (cadence E), disable or delete
# .github/workflows/tick-poller.yml on `main` -- there is nothing to
# unregister locally.

# =============================================================================
# APPENDIX A — cron (macOS / Linux)
# =============================================================================
# Add these lines via `crontab -e`. Adjust the python path to your venv.
# All times are LOCAL; convert from ET as needed.
#
#   REPO=/path/to/OddsOptimizer
#   PYTHON=$REPO/venv/bin/python
#
#   # A — daily refresh ~10:00 local
#   0 10 * * *   cd $REPO && $PYTHON scripts/run_cadence.py refresh >> /tmp/cadence_refresh.log 2>&1
#
#   # B — daily settle ~12:00 local
#   0 12 * * *   cd $REPO && $PYTHON scripts/run_cadence.py settle  >> /tmp/cadence_settle.log  2>&1
#
#   # C — weekly retrain Monday ~08:00 local
#   0  8 * * 1   cd $REPO && $PYTHON scripts/run_cadence.py retrain >> /tmp/cadence_retrain.log 2>&1
#
#   # D — weekly report Monday ~12:30 local
#   30 12 * * 1  cd $REPO && $PYTHON scripts/run_cadence.py report  >> /tmp/cadence_report.log  2>&1
#
# (Cadence E, the tick poller, is intentionally absent here too -- see
# Appendix B. run_cadence.py still writes logs/cadence_*.log regardless of
# the cron redirect above — the redirect is just a belt-and-suspenders
# fallback.)

# =============================================================================
# APPENDIX B — GitHub Actions (cadence E, tick poller — BUILT, 2026-08)
# =============================================================================
# Cadence E (the Underdog tick poller) runs on GitHub Actions, not on this
# machine at all: .github/workflows/tick-poller.yml, polling every 15
# minutes and committing new ticks to the orphan `data-ticks` branch. This
# was the one cadence worth moving off Task Scheduler, because it is the
# only cadence that captures unbackfillable data -- see "Why only cadence E
# is cloud-hosted" in docs/runbook_go_live.md. A-D stayed local; they don't
# need cloud hosting (see that section for the reasoning) and their data/
# directory is gitignored and local-only, which is exactly the constraint
# that ruled out moving A-D themselves: they read/write the predictions
# partition and model artifacts, which are not (and should not become)
# accessible to a GitHub-hosted runner.
#
# One-time setup and verification: docs/runbook_go_live.md, Step 3a/3b.
# Pull cloud ticks down for local analysis:
#   git fetch origin data-ticks && git checkout data-ticks -- data/raw/underdog_ticks
