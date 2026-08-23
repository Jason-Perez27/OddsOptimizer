# Task #12 — Live forward-validation go-live design

**Status:** approved spec, ready for Sonnet implementation
**Date:** 2026-06-29
**Author (planning):** Opus
**Prior art:** tasks #1–#11 (full pipeline + Track-A historical backtest, all built and passing).

---

## Goal

Turn the validated pipeline on against **live games**, starting today (2026-06-29),
and accumulate an honest **Track B (pick profitability)** record over time — the
forward-only track the project has deferred since task #10. Track A (model
calibration on historical games) is done and trustworthy (ECE 0.006 on the 2024
walk-forward); this task does **not** touch the model design. It is an
**operational go-live**: stand up the live model, run the first real `refresh`,
schedule the daily/weekly loop, and define the success criteria and failure
triage for running unattended.

The deliverable is small in new code and large in operational correctness. There
are exactly four scheduled cadences, one verification gate that must pass before
anything is trusted, one season-correctness retrain, and a handful of small,
testable code reconciliations. Everything else already exists and is reused
as-is.

### What this task is NOT

- Not a model change (no v2, no new features, no tier redefinition — those stay
  gated behind the task #10 ≥100-settled-picks-per-tier bar).
- Not a new evaluation track. Track B's grading/metrics/report (`settle.py`,
  `evaluation/`, `backtest/roi.py`, `backtest/report.py`) are already built — this
  task *runs* them on a live cadence, it doesn't rebuild them.
- Not prop expansion (strikeouts only, still).

---

## The central correctness insight (build the whole task around this)

**The existing production model artifact is fit on 2024 data and must not make
live 2026 picks.** Every `data/raw/statcast/*` window, the
`walk_forward_oos.csv`, and the `2026-06-29-baseline-backtest.md` report are the
**2024** season — they validated the *method*, not a 2026-ready model. A model
that has never seen a single 2026 pitch (current form, current rosters,
mid-2026 league strikeout environment) is the wrong thing to point at today's
slate.

So **go-live step 1 is a fresh retrain on the 2026 season-to-date corpus**,
fit through yesterday, saved to the canonical model path. This single action
does three jobs at once: (a) produces the live model, (b) optionally refreshes
Track A on 2026 data, and (c) forces the model-path reconciliation below to be
correct, because the retrain writes to the path `refresh` actually loads from.

---

## Inputs (what already exists — read before implementing)

- `src/pipeline/refresh.py` — daily predict run. `run_refresh()` + `main()`,
  `DEFAULT_MODEL_PATH`, `--dry-run` gate (`run_dry_run` → `verify_live_sources`),
  date-partitioned CSV outputs, partial-failure handling, model staleness warning
  (`STALE_MODEL_WARNING_DAYS = 7`).
- `src/pipeline/settle.py` — D+1 grading. `run_settlement()` + `main()` with
  `--date` and `--from/--to` range, trailing re-settle is idempotent (overwrite),
  three settlement states (`settled` / `pending` / `void_scratched`),
  `scratch_void_max_wait_days = 3`, `settlement_lag_hours = 4`.
- `src/pipeline/verify.py` — `verify_live_sources(lines_fetcher, schedule_fetcher,
  game_date)` returning `{"passed": bool, "checks": [...]}`. **Never verified
  against a live call** — that is exactly what go-live step 0 does.
- `scripts/run_backtest.py` — the corpus→walk-forward→report→`fit_production_model`
  pipeline. Flags: `--start --end --through-date --model-path --oos-path
  --no-production-model --cache-dir --window-days --step --min-train-dates
  --min-batters-faced`. Its `DEFAULT_MODEL_PATH` is imported from `refresh.py` to
  prevent path drift (there is already a comment there about a prior drift).
- `src/models/baseline_model.py` — `fit_production_model(corpus, *, through_date,
  fit_fn, save_path=...)`, `save_model`/`load_model`, `model_age_days`.
- `src/backtest/report.py` — `generate_report()` (live Track-B results report,
  `reports/YYYY-MM-DD-results.md`) and `generate_backtest_report()` (Track-A).
- `configs/config.yaml` — `evaluation:` knobs (settle lag, void wait, buckets,
  rolling window); `props.active_prop: Strikeouts`.

---

## The model-path bug to reconcile (do this first, it blocks everything)

Three code call sites agree on `DEFAULT_MODEL_PATH = data/models/baseline_model.joblib`
(refresh default, refresh `--model-path` default, `run_backtest.py` save default).
**But the only artifact on disk is at `models/baseline_model.joblib` (repo root),
and `data/models/` does not exist.** A scheduled `refresh` on defaults would call
`load_model("data/models/baseline_model.joblib")` and fail.

**Resolution (canonical path = `data/models/baseline_model.joblib`):**

- Keep the existing constant — three call sites already point there, and `data/`
  is gitignored, which is the right place for a 1.7 MB binary (matches the
  project's "no large binaries / no raw data committed" convention).
- The go-live retrain (step 1) writes the 2026 model **to that default path**, so
  the reconciliation happens by construction — the live model lands exactly where
  `refresh` loads it.
- The stray `models/baseline_model.joblib` (2024 artifact at the wrong path) is
  retired: delete it, or leave it untracked and ignored. It must never be the
  artifact a live run loads.
- Add a guard test asserting `refresh.DEFAULT_MODEL_PATH ==
  run_backtest.DEFAULT_MODEL_PATH` so the drift the existing comment warns about
  can't silently reappear.

---

## The four moving pieces (go-live sequence)

### Step 0 — Live-data verification gate (must PASS before anything else)

`python -m src.pipeline.refresh --dry-run --date 2026-06-29`

Runs `verify_live_sources` against the **real** PrizePicks and StatsAPI endpoints
and writes nothing. It must report PASS on both checks:

1. `prizepicks_lines` — `league_id=2` returns MLB `"Strikeouts"` projections that
   `flatten_projections` parses into the columns `pitcher, team, stat_type, line,
   start_time`.
2. `statsapi_schedule` — the schedule hydration returns `probablePitcher.id` +
   team for today's slate.

**Failure handling is the whole point of the gate:** if either fails on a day
with games scheduled, **stop** — fix the `league_id` / parser / hydrate param in
the relevant `src/data/` module before any graded run. A FAIL on a genuine off-day
(no MLB games) is not a shape break; re-run on a day games exist before
concluding. **Never run a real graded `refresh` on an unverified shape** — an
unofficial-endpoint shape change that slips through corrupts every downstream
graded number.

### Step 1 — Retrain on 2026 season-to-date (produces the live model)

```
python -m scripts.run_backtest \
    --start 2026-03-26 --end 2026-06-28 \
    --through-date 2026-06-28 \
    --fit-only
```

- `--start` = 2026 season opening week; `--end`/`--through-date` = yesterday.
- `--fit-only` (NEW flag, see Module layout) skips the walk-forward (steps 4–5 of
  `run_backtest`) and runs only corpus-build → training-table → starter-filter →
  `fit_production_model`, saving to the default `data/models/baseline_model.joblib`.
  The weekly walk-forward refresh of Track A is optional and separate (run the
  full `run_backtest` without `--fit-only` when a fresh 2026 calibration read is
  wanted).
- The corpus pull is large and real; it caches resumably under `data/raw/statcast/`
  so the weekly retrain re-pulls only the newest window.
- After this, `model_age_days` reads ~1 and the staleness warning is silent.

### Step 2 — First live refresh (today's predictions)

`python -m src.pipeline.refresh --date 2026-06-29`

Writes the authoritative partition for today under
`data/processed/predictions/game_date=2026-06-29/` (`predictions.csv`,
`threshold_table.csv`, `line_picks.csv`, `diagnostics/`, `run_manifest.json`).
This run **freezes the posted line + `pulled_at`** for honest grading, so it must
run **once, in the morning, before first pitch** and is the authoritative copy —
see "Timing & idempotency."

### Step 3 — The unattended daily/weekly loop (four scheduled tasks)

| # | Cadence | Command | When | Why that time |
|---|---|---|---|---|
| A | Daily | `python -m src.pipeline.refresh` | ~10:00 **ET** | After PrizePicks posts lines + StatsAPI has probables, before nearly all first pitches. |
| B | Daily | `python -m src.pipeline.settle --window-days 4` | ~12:00 **ET** | After overnight Statcast finalizes the prior day(s); re-settles the trailing window to resolve `pending`→`settled`/`void`. |
| C | Weekly | `python -m scripts.run_backtest --start 2026-03-26 --end <yesterday> --through-date <yesterday> --fit-only` | Mon ~08:00 ET | Keep the model current as the season evolves; matches the walk-forward step cadence and beats the 7-day staleness warning. |
| D | Weekly | `python -m src.backtest.report` | Mon ~12:30 ET (after B) | Regenerate the live Track-B results report from the accumulated settled partitions. |

Times are **ET-anchored** (MLB's reference clock); the user's scheduled tasks
fire in **their machine's local timezone**, so the runbook states the constraint
(after lines post / before first pitch; after Statcast finalizes) and the user
converts. The user has already created one "daily refresh" task — task A is that
task, pointed at the right command and the verified model path.

### Step 4 — Accumulate, then read

- The record starts at **today** and grows one slate per day. There is **no
  backfill** — PrizePicks has no historical line source, so the forward track
  cannot be seeded from the past (this is the defining property of Track B).
- **First "worth reading" checkpoint ≈ 2 weeks** (Low-tier line picks clear ~100
  settled in ~1–2 weeks). High-tier line picks are rare (a well-calibrated model
  rarely disagrees hard with a near-median line) and may take **a month-plus** to
  clear 100 — do not over-read small-sample High-tier ROI swings before then.
- The **tier-redefinition decision stays gated** on the task #10 bar: ≥100
  settled picks per tier **and** evidence the current probability-only definition
  is failing (e.g. High not separating from Low). Until both hold, the shipped
  definition stands.

---

## Timing & idempotency (the operational sharp edges)

- **The morning refresh is authoritative.** `refresh` overwrites a date's
  partition by default. Re-running it later in the day re-freezes a *moved* line
  (PrizePicks lines drift) and a possibly *changed* slate (late scratches),
  corrupting the honest pre-game snapshot. Rule: the scheduled morning run is the
  copy of record; manual re-runs are allowed **only before first pitch** (e.g. to
  recover from a crash). Never re-run `refresh` for a date after its games start.
- **Settle is safely repeatable.** Re-settling a date overwrites idempotently;
  `pending` rows resolve to `settled` on a later pass with no upsert bookkeeping.
  The trailing `--window-days 4` covers the 3-day `void_scratched` wait plus a
  one-day buffer, so a scratch/postponement is correctly voided (never silently
  scored as a loss) without any manual date math.
- **Empty slate is fatal-but-clean, by design.** An off-day raises
  `EmptySlateError`; `main()` prints and exits 0-ish without writing — the
  scheduled task simply no-ops that day. This is correct, not an error to alert on.
- **Partial failures degrade, not crash.** PrizePicks/register down → full sweep +
  empty `line_picks` + `prizepicks_error` flagged. A single pitcher's Statcast
  pull failing or a debutant with no history → `skipped_pitchers`, run continues.

---

## Output schema (what go-live produces — all already defined)

No new output schemas. Go-live produces, on a live cadence, the partitions tasks
#9/#10 already specify:

- `data/processed/predictions/game_date=YYYY-MM-DD/` — per the refresh spec.
- `data/processed/outcomes/game_date=YYYY-MM-DD/` — per the settle spec.
- `reports/YYYY-MM-DD-results.md` (+ reliability / cumulative-ROI PNGs) — the live
  Track-B report, distinct from the Track-A `…-baseline-backtest.md`.
- `data/models/baseline_model.joblib` — the live (2026-fit) production model at
  the canonical path.

The **only new artifact this task authors is documentation**:
`docs/runbook_go_live.md` (see Module layout).

---

## Module layout (the minimal new/changed code)

1. **`scripts/run_backtest.py` — add `--fit-only`** (+ test).
   - When set, skip walk-forward + report (steps 4–5) and run corpus → training
     table → starter-filter → `fit_production_model(... save_path=args.model_path)`.
   - Keeps the weekly retrain cheap (no full walk-forward) while reusing the exact
     corpus/feature/fit code path. `--fit-only` and `--no-production-model` are
     mutually exclusive (fitting is the entire point of `--fit-only`) — error if
     both are passed.

2. **`src/pipeline/settle.py` — add `--window-days N`** (+ test).
   - Convenience over the existing `--from/--to`: settle the N days ending
     yesterday (`[today − N, today − 1]`) so the daily scheduled settle needs no
     shell date arithmetic. Mutually exclusive with `--date` / `--from/--to`;
     reuses the existing date-range loop unchanged.

3. **Model-path guard test** (in `tests/test_refresh.py` or `tests/test_run_backtest.py`).
   - Assert `src.pipeline.refresh.DEFAULT_MODEL_PATH ==
     scripts.run_backtest.DEFAULT_MODEL_PATH`, pinning the no-drift invariant the
     existing comment relies on.

4. **`docs/runbook_go_live.md` — NEW operator runbook** (no code, but the
   user-facing deliverable).
   - The ordered go-live sequence (steps 0–4 above) as a literal checklist with
     copy-paste commands.
   - The four scheduled-task definitions: cadence, exact command, recommended
     ET time + the constraint to convert to local time.
   - The failure-triage table (empty slate / prizepicks_error / register_error /
     model_stale / skipped_pitchers / pending vs void) — what each means and
     whether it needs action.
   - The success criteria + checkpoints (≥100 settled/tier, ~2-week first read,
     tier-redefinition gate).

**Deliberately no new pipeline module.** Resist adding a `daily_driver.py`
orchestrator or a status dashboard — the existing CLIs + manifests are the
interface, and the scheduler is the orchestrator. (Both are named in Deferred.)

---

## Edge cases (handle / document, don't silently absorb)

1. **2026 corpus too thin at season edges** — early-season pitchers with <N prior
   starts hit the existing drop-core / `was_imputed` path; coverage is reported in
   `skipped_pitchers`, never silently shrinking the slate.
2. **Verification gate FAIL on a real game day** — stop and fix the source module;
   do not proceed to a graded run. A guard against "it was just an off-day": the
   check message already distinguishes the two; the runbook says re-confirm on a
   day with games before editing parsers.
3. **First-pitch earlier than the scheduled refresh** (rare ~11:00 ET starts /
   getaway days) — those pitchers' lines may already reflect lineup news; documented
   as a known limitation of a single fixed morning run, not engineered around in v1.
4. **Doubleheaders** — already handled by the `(pitcher, game_pk)` key end to end;
   the single-day settle window returns both games together.
5. **A pitcher pulled early** is a legitimate `under`, **not** a void — only a
   never-thrown start voids. (Carried from the task #10 spec; restated so the
   runbook triage is right.)
6. **Weekly retrain run lands mid-week / is skipped** — the 7-day staleness warning
   in every refresh manifest is the safety net; a stale warning is a prompt to run
   retrain, never an auto-retrain inside the daily run.
7. **Re-run of `refresh` after games start** — see Timing; the runbook forbids it
   and explains why (frozen-line corruption).

---

## Deferred (name, don't build)

- **Scheduler/orchestration infra** beyond the four scheduled-task definitions (a
  `daily_driver.py`, a Makefile target graph, a CI workflow, retries/alerting).
- **A run-health/status summarizer** or dashboard over the manifests.
- **Model v2** (exposure offset, regularization/VIF, boosting) and **new features**
  (umpire/weather/bullpen/day-night) — gated behind this live validation.
- **Tier redefinition** — gated on ≥100 settled/tier + evidence of failure.
- **Prop expansion** (pitching outs / earned runs / walks).
- **A priced-odds feed / true-payout ROI / CLV** — still out (free-data bias); the
  even-money flat-bet ROI proxy stays labeled a proxy.
- **Shortening the pending→void wait with a live game-status feed** (task #10's
  flagged risk) — still deferred; the time-threshold inference stands.

---

## Testing approach (numbered — leakage isn't the risk here, *correct wiring* is)

Track A's leakage tests already exist and pass; this task's risk surface is
operational wiring, so the tests pin **flags, paths, and date math**, all
no-network with injected fetchers / hand-built fixtures (matching the repo style).

1. **Model-path no-drift guard** — `refresh.DEFAULT_MODEL_PATH ==
   run_backtest.DEFAULT_MODEL_PATH`; both end in `data/models/baseline_model.joblib`.
2. **`--fit-only` skips the walk-forward** — with injected fetchers/fixtures and a
   fast fake `fit_fn`, assert `run_walk_forward` / `generate_backtest_report` are
   **not** called and `fit_production_model` **is**, saving to `--model-path`.
3. **`--fit-only` + `--no-production-model` is rejected** — parser/`main` errors
   rather than silently doing nothing.
4. **`settle --window-days N` expands to the right date list** — `[today−N,
   today−1]` inclusive, with `now` injected; equals the explicit `--from/--to`
   equivalent.
5. **`settle --window-days` is mutually exclusive** with `--date` and `--from/--to`
   — passing both errors clearly.
6. **`verify_live_sources` PASS/FAIL plumbing** — with stub fetchers returning a
   good payload → `passed: True`; a renamed `line_score` column or a missing
   `probablePitcher.id` → `passed: False` with the specific failing check named.
   (No live call in the test.)
7. **`refresh --dry-run` writes nothing** — assert no partition dir is created for
   the date even when the (stubbed) sources pass.
8. **Empty-slate path stays clean** — `EmptySlateError` → `main()` no-write, no
   traceback to the scheduler.
9. **Trailing settle is idempotent** — settling the same window twice over the same
   fixtures yields identical graded frames (overwrite, no duplication).
10. **Full suite green** — after the changes, `pytest` passes with nothing
    regressed (run it; show output — evidence before assertions).

Gate any scipy-dependent assertions with `pytest.importorskip`. Keep real
statsmodels fits out of the fast tests (fake `fit_fn`).

---

## What "done" looks like

- `verify_live_sources` PASSes against the live endpoints on a game day (or the
  surfaced mismatch is fixed and re-verified).
- A 2026-season production model sits at `data/models/baseline_model.joblib`;
  `model_age_days` is small and the staleness warning is silent.
- Today's `data/processed/predictions/game_date=2026-06-29/` partition exists,
  written by a real refresh.
- The four scheduled tasks are defined (commands + cadence) in
  `docs/runbook_go_live.md`, and task A is wired to the verified model path.
- The small code changes (`--fit-only`, `--window-days`, the path guard) are
  implemented with passing tests and the full suite is green.
- The decision-log entry for task #12 is recorded.
