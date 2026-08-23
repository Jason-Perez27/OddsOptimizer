# install_cadences.ps1 — Register (or delete) the five OddsOptimizer cadences
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
#   E (daily, repeating -- 2026-08 CLV feature) is NOT a "fire once" cadence
#     like A-D: it polls Underdog's live feed every 20 minutes from 09:00 ET
#     through end of day (24:00 ET), the window covering every first pitch on
#     the slate (first-pitch times spread up to ~5-6 hours across one day) so
#     the tick log can capture each game's actual closing line, not just an
#     early-morning snapshot. It has no ordering dependency on A/B/C/D and
#     touches nothing they read or write (see src/data/underdog_ticks.py) --
#     it is purely additive and safe to add without touching A-D at all.
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
$START_TIME_E = "09:00"   # daily, repeating — cadence E: tick poller (see -RepeatInterval below)

# Day of week for the weekly tasks (C and D).
$WEEKLY_DAY = "MON"

# Cadence E repeats through the day rather than firing once. 20-minute
# interval for 15 hours starting at $START_TIME_E covers 09:00-24:00 local —
# every first pitch on a normal MLB slate.
$TICKS_REPEAT_INTERVAL_MIN = 20
$TICKS_REPEAT_DURATION     = "0015:00"   # HHHH:MM — 15 hours, 0 minutes

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the schtasks /Create argument string
# ─────────────────────────────────────────────────────────────────────────────
# -RepeatInterval / -Duration (2026-08, cadence E): schtasks' documented
# contract for a repeating trigger is /RI <minutes> together with /DU
# <HHHH:MM> (or /ET <end-time> — this script always uses /DU). /RI without
# /DU or /ET defaults to a 1-hour repeat window, which is NOT what cadence E
# wants, so /DU is required whenever /RI is passed. VERIFY this against
# `schtasks /Create /?` on your own machine before relying on it — Windows
# version/locale differences are exactly the kind of thing worth a 10-second
# spot check before you trust an unattended scheduled task with it.
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
# REGISTER — run this block to install all five tasks
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

Register-Cadence -TaskName "E-ticks"   -Cadence "ticks"   `
    -ScheduleType "DAILY" -StartTime $START_TIME_E `
    -RepeatInterval $TICKS_REPEAT_INTERVAL_MIN -Duration $TICKS_REPEAT_DURATION

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
# E — daily, repeating tick poller (09:00-24:00 local, every 20 min, ET-anchored;
#     2026-08 CLV feature -- see src/data/underdog_ticks.py):
#   schtasks /Create /F /TN "OddsOptimizer\E-ticks" /TR "\"<PYTHON>\" \"<REPO_ROOT>\scripts\run_cadence.py\" ticks" /SC DAILY /ST 09:00 /RI 20 /DU 0015:00 /RL HIGHEST
#
# ─────────────────────────────────────────────────────────────────────────────
# DELETE / UNREGISTER (reversible)
# ─────────────────────────────────────────────────────────────────────────────
# To remove all five tasks:
#   schtasks /Delete /TN "OddsOptimizer\A-refresh" /F
#   schtasks /Delete /TN "OddsOptimizer\B-settle"  /F
#   schtasks /Delete /TN "OddsOptimizer\C-retrain" /F
#   schtasks /Delete /TN "OddsOptimizer\D-report"  /F
#   schtasks /Delete /TN "OddsOptimizer\E-ticks"   /F
#
# Or from PowerShell (requires admin):
#   Unregister-ScheduledTask -TaskName "A-refresh" -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "B-settle"  -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "C-retrain" -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "D-report"  -TaskPath "\OddsOptimizer\" -Confirm:$false
#   Unregister-ScheduledTask -TaskName "E-ticks"   -TaskPath "\OddsOptimizer\" -Confirm:$false

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
#   # E — tick poller: every 20 min, 09:00-23:59 local (2026-08 CLV feature)
#   */20 9-23 * * *  cd $REPO && $PYTHON scripts/run_cadence.py ticks >> /tmp/cadence_ticks.log 2>&1
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
