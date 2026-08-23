# Go-live runbook — live forward-validation (Track B)

**Audience:** the operator running this pipeline day to day (not a developer doc).
**Scope:** Task #12. Turns the validated pipeline on against live games and
starts the forward-only Track B (pick profitability) record. No model change,
no new pipeline module — this is the operational checklist for running the
existing `refresh` / `settle` / `run_backtest` / `report` CLIs on a live
cadence.

Design reference: `docs/design/specs/2026-06-29-live-forward-validation-go-live-design.md`.

---

## 0. Before you start — the one rule that matters

**The morning `refresh` is authoritative.** It overwrites a date's partition,
freezing the posted line and `pulled_at` for honest grading.

- Run it **once, in the morning, before first pitch.**
- **Never re-run `refresh` for a date after that date's games start.** A later
  re-run re-freezes a *moved* line and a possibly *changed* slate (late
  scratches) — it corrupts the honest pre-game snapshot, not improves it.
- Manual re-runs are allowed only **before first pitch** (e.g. recovering from
  a crash earlier the same morning).

`settle`, by contrast, is **safely repeatable** — re-settling a date or window
overwrites idempotently and resolves `pending` → `settled`/`void_scratched` as
new Statcast data lands. Run it as often as you like.

---

## Step 0 — Live-data verification gate (must PASS before anything else)

```
python -m src.pipeline.refresh --dry-run --date <today, YYYY-MM-DD>
```

This calls the **real** Underdog and StatsAPI endpoints and writes nothing.
It must report PASS on both checks:

1. `underdog_lines` — `sport_id=MLB` returns over/under lines that parse into
   `pitcher, team, stat_type, line, start_time, over_american, under_american,
   game_status, live_event`.
2. `statsapi_schedule` — schedule hydration returns `probablePitcher.id` +
   team for today's slate.

**If either FAILs on a day with games scheduled: stop.** Fix the `sport_id`
/ parser / hydrate param in the relevant `src/data/` module and re-verify
before any graded run. Do not proceed on an unverified shape — an unofficial
endpoint shape change that slips through corrupts every downstream graded
number.

A FAIL on a genuine off-day (no MLB games) is not a shape break — re-run on a
day games exist before concluding something is broken.

---

## Step 1 — Retrain on 2026 season-to-date (produces the live model)

```
python -m scripts.run_backtest \
    --start <2026 opening week> --end <yesterday> \
    --through-date <yesterday> \
    --fit-only
```

- `--fit-only` skips the walk-forward + report and runs only
  corpus-build → training-table → starter-filter → `fit_production_model`,
  saving to the canonical path `data/models/baseline_model.joblib`.
- The corpus pull is **large and real** (live Statcast network call); it
  caches resumably under `data/raw/statcast/`, so the weekly retrain (cadence
  C below) only re-pulls the newest window.
- After this runs, `model_age_days` reads ~1 and the staleness warning in
  every refresh manifest goes silent.
- The stale 2024 artifact that previously lived at `models/baseline_model.joblib`
  (wrong path, wrong season) has been retired to
  `data/archive/baseline_model_2024_stale.joblib` — it is never loaded by a
  live run.

---

## Step 2 — First live refresh (today's real predictions)

```
python -m src.pipeline.refresh --date <today>
```

Writes the authoritative partition for today under
`data/processed/predictions/game_date=<today>/` (`predictions.csv`,
`threshold_table.csv`, `line_picks.csv`, `diagnostics/`, `run_manifest.json`).
Per the rule in Step 0: run this once, before first pitch, and never again
for this date once games start.

---

## Step 3 — Schedule the unattended daily/weekly loop (four local tasks + one cloud task)

| # | Cadence | Command | Recommended time | Why that time |
|---|---|---|---|---|
| A | Daily, local | `python -m src.pipeline.refresh` | ~10:00 **ET** | After Underdog posts lines + StatsAPI has probables, before nearly all first pitches. |
| B | Daily, local | `python -m src.pipeline.settle --window-days 4` | ~12:00 **ET** | After overnight Statcast finalizes the prior day(s); re-settles the trailing window, resolving `pending` → `settled`/`void`. |
| C | Weekly (Mon), local | `python -m scripts.run_backtest --start <2026 opening week> --end <yesterday> --through-date <yesterday> --fit-only` | ~08:00 **ET** | Keeps the model current as the season evolves; beats the 7-day staleness warning. |
| D | Weekly (Mon, after B), local | `python -m src.backtest.report` | ~12:30 **ET** | Regenerates the live Track-B results report from the accumulated settled partitions, including the CLV section. |
| E | Every 15 min, **GitHub Actions** (cloud) | `.github/workflows/tick-poller.yml` (`python -m src.data.underdog_ticks --output-dir ...`) | ~08:00–02:00 **ET**, continuous | Captures line movement through the day and overnight, including every first pitch on the slate (times can spread 5–6 hours), **even when your laptop is closed**. See Step 3a/3b below. |

**Times are ET-anchored** (MLB's reference clock). Your scheduled tasks fire
in **your machine's local timezone** — convert these ET times to your local
time when you set them up. The constraint that matters is the ordering: A
must fire after lines/probables post and before first pitch; B must fire
after overnight Statcast settles; D must fire after B.

If you already have a daily "refresh" scheduled task, that is cadence A —
confirm it points at the command above (no flags needed; it loads the model
from the now-correct canonical path automatically).

### Why only cadence E is cloud-hosted

Line ticks are the **only unbackfillable data** in this pipeline. Statcast,
boxscores, probable pitchers, and ESPN odds can all be retrieved days later
— `settle --window-days 4` already backfills four days of Statcast, and
retrain (C) / report (D) are weekly, so all four of A–D tolerate a closed
laptop for a day or two. A line movement that was never observed, though,
is gone permanently — there is no "re-fetch yesterday's line history" call.
That asymmetry is the entire reason cadence E moved off the local machine
(2026-08) and onto GitHub Actions while A–D stayed put: A–D don't need to
survive the laptop being closed, and E can't afford not to.

Cadence E is also different in kind from A–D operationally: it's a
**repeating** poll (every 15 minutes, continuous) rather than a once- or
twice-daily run, it has no ordering dependency on A/B/C/D, and it is
**purely additive** — it never touches `line_picks.csv`, the predictions
partition, or the frozen morning snapshot, and never triggers a refresh or
a retrain. Skipping cadence E (or a gap in it) costs you CLV data for that
window, nothing else; it cannot corrupt or interact with any of the other
four cadences' outputs. See the README's "Closing line value (CLV)"
subsection for what it's for.

### Step 3a — One-time setup: bootstrap the `data-ticks` branch

The Actions poller commits tick data to a dedicated orphan branch,
`data-ticks`, so its automated commits never touch `main`'s history and
`main` stays a clean, human-reviewed branch. This is a **one-time, manual**
setup step — deliberately not scripted into the workflow itself:

```
git checkout --orphan data-ticks
git rm -rf .
mkdir -p data/raw/underdog_ticks
printf '# Automated tick data\n\nWritten by .github/workflows/tick-poller.yml.\nDo not edit by hand. Not merged into main.\n' > README.md
git add -A && git commit -m "chore: bootstrap data-ticks branch"
git push -u origin data-ticks
git checkout main
```

Do this once, before the workflow's first run (the workflow's step 2 checks
out `data-ticks` and will fail if the branch doesn't exist yet).

**Repository visibility matters for cost.** GitHub Actions minutes are
unlimited on a **public** repo but capped at 2,000 min/month free on a
**private** one. At 15-minute polling this job runs roughly 64 times/day at
~45s each — about 48 min/day, ~1,450 min/month — which would nearly exhaust
a private repo's free tier on its own, before any other workflow. If this
repository is private, either make it public or widen the cron interval in
`tick-poller.yml` to `*/30 ...` (30 minutes) to stay comfortably under the
cap.

### Step 3b — Merge the workflow, verify it actually runs

Scheduled GitHub Actions workflows **only fire from the repository's
default branch** — a `tick-poller.yml` sitting on a feature branch will
silently never run. After merging `.github/workflows/tick-poller.yml` to
`main`:

1. Trigger it manually first: Actions tab → "tick-poller" → **Run workflow**
   (`workflow_dispatch`). Confirm it completes and either writes a commit to
   `data-ticks` or cleanly no-ops (off-day / no line movement since the last
   poll).
2. **Do not assume the schedule is live just because the manual run worked.**
   Wait for the next scheduled tick and confirm in the Actions run history
   that it actually fired — a workflow can be correctly merged and still
   silently not run if the trigger, cron syntax, or branch is wrong.
3. Once confirmed, ticks accumulate on `data-ticks` continuously, day and
   night, independent of whether your machine is on.

**Pulling cloud ticks down for local analysis** (e.g. to run
`src.evaluation.clv` or `src.backtest.report` locally against the latest
data):

```
git fetch origin data-ticks && git checkout data-ticks -- data/raw/underdog_ticks
```

**Actions cron is best-effort, not a guaranteed fixed-interval clock.**
GitHub documents that scheduled workflows can be delayed 15-60 minutes
under platform load, and that some queued runs are dropped entirely. Never
assume a given cron tick produced a poll. The practical consequence: the
resulting tick log is a **good-faith sample of the market**, not a
guaranteed evenly-spaced time series. `close_quality = "stale"` in
`src.evaluation.clv` exists precisely to let CLV analysis exclude
poorly-captured closes (the last tick observed more than
`STALE_CLOSE_MINUTES` before first pitch) rather than silently averaging in
a low-quality read of a game's closing line. **Any CLV summary you read or
report should state how many closes it excluded as stale**, not just the
number it scored.

**The workflow will auto-disable after 60 days with no commits to `main`.**
This is expected and acceptable — it's designed to happen once the MLB
season ends and `main` goes quiet for the offseason. **Do not** add a
keepalive workflow, a scheduled no-op commit, or any other bot-commit hack
to defeat this behavior; it exists to stop stale automation from running
forever on abandoned repos, and fighting it just adds a second thing that
can silently break. If polling needs to resume next season, re-enable the
workflow (or re-trigger it with a real commit to `main`) at that time.

**A missed local refresh is more recoverable now than before.** Because the
cloud poller captures lines all day regardless of whether A ran, a missed
or late local `refresh` is no longer a total loss for that date — the line
history still exists on `data-ticks` and predictions can, if needed, be
reconstructed offline afterward. If you do this, the reconstruction
**must** grade against the tick nearest the *original* decision time (when
`refresh` should have run), never a later one — grading against a later
tick breaks the frozen-snapshot discipline (Step 0's "one rule that
matters") and makes the backtest dishonest, since it would be scoring a
decision against information that wasn't available when the decision was
supposed to be made.

---

## Step 4 — Accumulate, then read

- The record starts **today** and grows one slate per day. There is **no
  backfill** — Underdog has no historical line source, so Track B cannot be
  seeded from the past. This is expected, not a gap.
- **First "worth reading" checkpoint ≈ 2 weeks.** Low-tier line picks clear
  ~100 settled in roughly 1–2 weeks.
- **High-tier line picks are rare** (a well-calibrated model rarely disagrees
  hard with a near-median line) and may take **a month-plus** to clear 100
  settled. Don't over-read small-sample High-tier ROI swings before then.
- **Tier redefinition stays gated** on the task #10 bar: ≥100 settled picks
  per tier **and** evidence the current probability-only definition is
  failing (e.g. High not separating from Low). Until both hold, the shipped
  tier definition stands — do not redefine tiers off an early read.
- **CLV (2026-08) is readable sooner than outcome-based ROI.** It's a
  continuous quantity measured on every pick regardless of outcome, so it
  doesn't wait on settlement lag or a 100-per-tier floor the way ROI does.
  Like Track B, it starts empty and accumulates forward from whenever
  cadence E was first turned on — there is no backfilled history.

---

## Failure triage

| Signal | What it means | Action |
|---|---|---|
| `EmptySlateError` (refresh) | No starters/games today (off-day). | None — `main()` exits cleanly without writing. This is correct behavior, not an alert-worthy error. |
| `line_source_error` in manifest | Underdog fetch/register failed. | Predictions sweep still runs and is written; `line_picks.csv` is empty for the date. Check the underlying source/network; re-run before first pitch if there's time. |
| `register_error` in manifest | Probable-pitcher registration against the lineup failed for some/all pitchers. | Check `skipped_pitchers` in the manifest; investigate the StatsAPI hydration if it's widespread. |
| Staleness warning (`model_age_days` > `STALE_MODEL_WARNING_DAYS` = 7) | Weekly retrain (cadence C) was skipped or ran late. | Run Step 1's `--fit-only` retrain manually. This is a prompt to retrain, never an auto-retrain inside the daily run. |
| Non-empty `skipped_pitchers` | Individual pitcher's Statcast pull failed, or a debutant has no history. | Expected and surfaced, not silently dropped. No action needed unless the list is unexpectedly large. |
| `pending` settlement status | Outcome not yet resolved (still within the lag/wait window). | None — resolves automatically on the next settle pass. |
| `void_scratched` settlement status | Pitcher never threw a pitch (true scratch/postponement) after the 3-day max wait. | Correctly excluded from grading — not scored as a loss. |
| Pitcher pulled early but did throw | This is a legitimate graded `under`, **not** a void. | None — only a never-thrown start voids. |
| Verification gate FAIL (Step 0) on a game day | A live source's shape changed (Underdog `sport_id`/parser, or StatsAPI hydration). | **Stop.** Fix the relevant `src/data/` module and re-verify before running any graded `refresh`. |
| A day's tick log is missing or thin (`data/raw/underdog_ticks/game_date=.../ticks.csv` absent, or few rows, once pulled down per Step 3b) | The Actions poller (cadence E) was delayed/dropped for that window, the workflow got auto-disabled after 60 days idle, or it was never merged to main. | Not a failure of A–D — nothing else depends on cadence E. That day just contributes less/no CLV; `close_quality` will read `stale` (or the market simply won't appear in the CLV report) for any picks it does resolve. Check the workflow's run history at github.com/<owner>/<repo>/actions/workflows/tick-poller.yml and re-check tomorrow. |

---

## Success criteria + checkpoints

- `verify_live_sources` PASSes against the live endpoints on a game day.
- A 2026-season production model sits at `data/models/baseline_model.joblib`
  with a small `model_age_days` (staleness warning silent).
- Today's predictions partition exists, written by a real `refresh`.
- The four local scheduled tasks (A–D) are running on cadence, and the
  `tick-poller.yml` GitHub Actions workflow (cadence E) is merged to `main`
  and confirmed firing on its schedule (Step 3b) — cadence E is optional in
  the sense that nothing else depends on it, but is needed for CLV to
  accumulate, and unlike A–D it keeps running even when the laptop is
  closed.
- ~2 weeks in: Low-tier line picks should be approaching/past 100 settled —
  first point at which ROI is worth reading, not before.
- Tier redefinition: gated on ≥100 settled/tier **and** evidence of failure
  (e.g. High not separating from Low). Do not redefine early.

---

## What this runbook deliberately does not include

Per the approved spec, the following are named and deferred, not built:

- A `daily_driver.py` orchestrator, Makefile target graph, or CI workflow.
- A run-health/status dashboard over the manifests.
- Retries/alerting infrastructure beyond what the pipeline already does
  (partial-failure degrade, manifest flags).
- Model v2, new features, tier redefinition, prop expansion, or a live
  game-status feed to shorten the pending→void wait.
  (A priced-odds/CLV feed — cadence E, `src/data/underdog_ticks.py` /
  `src/evaluation/clv.py` — was in this original list but has since been
  built; see the CLV row in Step 3 and the README's CLV subsection.)

The existing CLIs and their manifests are the interface; your task scheduler
is the orchestrator.
