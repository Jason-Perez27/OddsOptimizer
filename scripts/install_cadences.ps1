# install_cadences.ps1 — Register (or delete) the four OddsOptimizer cadences
# in Windows Task Scheduler.
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
#   A must fire AFTER Underdog posts lines + StatsAPI has probables (~09:00 ET)
#     and BEFORE first pitches (~13:05 ET). 10:00 ET is the recommended target.
#   B must fire AFTER overnight Statcast finalizes (~11:00 ET). 12:00 ET target.
#   C (weekly) fires before the 7-day staleness warning trips. Monday 08:00 ET.
#   D (weekly) fires after B on the same day. Monday 12:30 ET.
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

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the schtasks /Create argument string
# ─────────────────────────────────────────────────────────────────────────────
function Register-Cadence {
    param(
        [string]$TaskName,
        [string]$Cadence,
        [string]$ScheduleType,   # "DAILY" or "WEEKLY"
        [string]$StartTime,
        [string]$Day = ""        # used only for WEEKLY
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

    Write-Host "Registering: OddsOptimizer\$TaskName ..."
    & schtasks @schtasksArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED to register $TaskName (exit $LASTEXITCODE)"
    } else {
        Write-Host "  OK: OddsOptimizer\$TaskName"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER — run this block to install all four tasks
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
# ─────────────────────────────────────────────────────────────────────────────
# DELETE / UNREGISTER (reversible)
# ─────────────────────────────────────────────────────────────────────────────
# To remove all four tasks:
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
# (run_cadence.py still writes logs/cadence_*.log regardless of the cron
# redirect above — the redirect is just a belt-and-suspenders fallback.)

# =============================================================================
# APPENDIX B — GitHub Actions (future infra, not the default)
# =============================================================================
# A schedule: workflow COULD trigger these cadences, but only if the data
# directory (data/) is accessible to the runner — it is gitignored and lives
# only on the local machine. Options to make it viable:
#   - Mount data/ from a cloud storage bucket (S3, GCS) the runner can read/write.
#   - Run a self-hosted GitHub Actions runner on the same machine (then it is
#     effectively the same as Task Scheduler, with more ceremony).
# Until data/ is hosted somewhere the runner can reach, GitHub Actions is not
# a substitute for the local scheduler. Deferred to future infra.
#
# Skeleton for reference (assumes a self-hosted runner or cloud-mounted data/):
#
#   on:
#     schedule:
#       - cron: "0 15 * * *"    # 15:00 UTC = 10:00 ET (standard) / 11:00 EDT
#   jobs:
#     refresh:
#       runs-on: self-hosted
#       steps:
#         - uses: actions/checkout@v4
#         - run: python scripts/run_cadence.py refresh
