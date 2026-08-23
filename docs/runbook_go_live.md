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

## Step 3 — Schedule the unattended daily/weekly loop (four tasks)

| # | Cadence | Command | Recommended time | Why that time |
|---|---|---|---|---|
| A | Daily | `python -m src.pipeline.refresh` | ~10:00 **ET** | After Underdog posts lines + StatsAPI has probables, before nearly all first pitches. |
| B | Daily | `python -m src.pipeline.settle --window-days 4` | ~12:00 **ET** | After overnight Statcast finalizes the prior day(s); re-settles the trailing window, resolving `pending` → `settled`/`void`. |
| C | Weekly (Mon) | `python -m scripts.run_backtest --start <2026 opening week> --end <yesterday> --through-date <yesterday> --fit-only` | ~08:00 **ET** | Keeps the model current as the season evolves; beats the 7-day staleness warning. |
| D | Weekly (Mon, after B) | `python -m src.backtest.report` | ~12:30 **ET** | Regenerates the live Track-B results report from the accumulated settled partitions. |

**Times are ET-anchored** (MLB's reference clock). Your scheduled tasks fire
in **your machine's local timezone** — convert these ET times to your local
time when you set them up. The constraint that matters is the ordering: A
must fire after lines/probables post and before first pitch; B must fire
after overnight Statcast settles; D must fire after B.

If you already have a daily "refresh" scheduled task, that is cadence A —
confirm it points at the command above (no flags needed; it loads the model
from the now-correct canonical path automatically).

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

---

## Success criteria + checkpoints

- `verify_live_sources` PASSes against the live endpoints on a game day.
- A 2026-season production model sits at `data/models/baseline_model.joblib`
  with a small `model_age_days` (staleness warning silent).
- Today's predictions partition exists, written by a real `refresh`.
- The four scheduled tasks above are running on cadence.
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
- Model v2, new features, tier redefinition, prop expansion, a priced-odds/
  CLV feed, or a live game-status feed to shorten the pending→void wait.

The existing CLIs and their manifests are the interface; your task scheduler
is the orchestrator.
