# Baseline Validation & Go-Live — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (all 2026-06-26/27 entries),
`docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md` (task #7,
which **deferred** walk-forward evaluation to here),
`…-tiered-prop-probabilities-design.md` (task #8),
`…-pre-game-refresh-pipeline-design.md` (task #9),
`…-outcome-tracking-design.md` (task #10).
Implementation targets: `src/backtest/corpus.py` (new),
`src/backtest/walk_forward.py` (new), `src/backtest/report.py` (extend),
`src/evaluation/metrics.py` (extend — PIT), a `fit_production_model` +
`verify_live_sources` path, and the matching `tests/test_*.py`.

## Goal

Establish **trustworthy baseline performance** before any model-v2 optimization or
prop expansion, and define how the system goes live. Tasks #1–#10 built the full
pipeline; nothing has run on real data yet. This task delivers the one missing
build that gives real numbers — a **historical walk-forward backtest** — and
records the **operational go-live decisions** for the live track.

Everything here is organized around one distinction the project's evaluation story
must keep explicit:

- **Track A — model honesty (available now).** Every completed game has a realized
  strikeout count, so a temporal/walk-forward backtest over historical games yields
  MAE/RMSE, calibration/ECE, Brier, log loss, and PIT **today**. No lines, no
  waiting. This is the new build.
- **Track B — pick profitability (forward-only).** PrizePicks has **no historical
  line backfill** (`docs/data_sources.md`), so hit-rate/ROI vs. the posted line
  cannot be backtested — it accrues only forward via the live `refresh`/`settle`
  loop, gated by the ≥100-settled-picks-per-tier bar (task #10). This is an
  operational plan, not a build.

Conflating the two would let historical model calibration be misread as betting
performance. The spec forbids that: **Track A produces no ROI**, by construction.

Scope is **strikeouts only**; the backtest is built prop-agnostic where free, but
only strikeouts is wired and validated.

## Inputs

1. **A historical date range** (season start → yesterday) for the backtest corpus.
2. **The full league pitch-level corpus** for that range — pulled via `pybaseball`
   (`statcast(start, end)` bulk), cached under `data/raw/`, fed through the
   existing `build_features.build_training_table`. The corpus must include **all
   pitchers' games** (not just starters) so opponent/park/league history is
   complete; the *evaluation set* is then filtered to **starts**.
3. **Existing model + metric machinery** — `baseline_model` (`temporal_train_test_split`,
   `fit_baseline_model`, `BaselineModel`, persistence, pure metric helpers),
   `evaluation/metrics.py` (calibration/ECE, Brier/log loss wrappers),
   `evaluation/grading.py` (`grade_threshold_sweep`). Reused, not duplicated.
4. **Live sources for the verification gate** — PrizePicks projections + MLB-StatsAPI
   schedule, only to confirm payload shapes before go-live.
5. **`configs/config.yaml`** — corpus range, walk-forward step, retrain cadence,
   staleness threshold, reliability-bucket count.

## The moving pieces

### 1. Historical corpus assembly (`src/backtest/corpus.py`, new)

A walk-forward needs the whole league's season, not one pitcher — a large, slow
network pull, so it is cached and resumable:

- `build_corpus(start_date, end_date, *, statcast_fetcher, cache_dir) -> pd.DataFrame`
  — iterates the range in **windows** (e.g. weekly), and for each window: if a
  cached CSV exists under `data/raw/statcast/`, load it; else fetch via the
  injected `statcast_fetcher` (default `pybaseball.statcast`) and cache it.
  Concatenate windows into one pitch-level frame. **Resumable** (a failed/rate-
  limited run picks up from cached windows); the injected fetcher makes it
  no-network testable with a fixture corpus.
- The fetched pitch frame is the input to the existing
  `aggregate_pitcher_games` → `build_training_table` path — **no feature code is
  reinvented.** Features built on the full corpus are leakage-safe by construction
  (every rolling/opponent/park builder is strictly-prior), so the feature table is
  built **once, globally**; the walk-forward governs only the train/predict cutoff
  (see below). State this explicitly — it is the key efficiency, and it is correct
  precisely because feature construction never looks forward.
- **Starter filter** for the evaluation set: a start is the pitcher with the most
  batters faced in their team's half of a `game_pk` (the same starter proxy
  `opponent_features` already uses), optionally floored by a batters-faced
  threshold. Openers/bullpen games are a documented limitation of this proxy.

### 2. Walk-forward backtest (`src/backtest/walk_forward.py`, new)

- `run_walk_forward(feature_table, *, step, fit_fn, min_train_dates) -> graded_oos`
  — **expanding window**, stepping by `step` (default **weekly**; daily refit is
  marginal gain for the cost). At each cutoff date `c`: train on starts with
  `game_date < c`, fit via the injected `fit_fn` (default `fit_baseline_model`),
  predict the starts in `[c, c+step)`, and accumulate **out-of-sample** predictions
  (`mu`, `family`, `alpha`, the 1–10 `p_over` sweep) joined to realized
  `strikeouts` on `(pitcher, game_pk)`.
- **Temporal correctness is the highest-stakes property.** No training row may be
  dated on/after its step's cutoff, and an earlier step's OOS predictions must not
  change when later games are appended. Reuse `temporal_train_test_split`
  semantics; the tests pin this directly (items 1–2 below).
- `fit_fn` is **injected** so the windowing/accumulation/leakage logic is testable
  with a fast fake fit (real statsmodels fits are heavy); the real run uses
  `fit_baseline_model`.
- **Track A only — no ROI.** There are no historical lines, so the output carries
  **no line/edge/ROI fields**. A test guards against fabricating betting numbers.
- **Tiers bonus:** because the OOS sweep has `p_over` per threshold, `tier(p_over)`
  can be attached, giving a **first read on whether the sweep tiers are
  calibrated** — useful input to the task #10 tier-validation, available now.
  (This is sweep-tier calibration; live *line-pick* tier hit-rate still needs
  Track B accumulation — keep the distinction.)

### 3. Metrics & reporting (`evaluation/metrics.py` extend; `backtest/report.py` extend)

- Reuse `evaluation/metrics.py` for calibration/ECE, Brier, log loss, MAE/RMSE on
  the OOS predictions; **add a `pit_histogram` helper** (task #7 named PIT; add it
  here if absent). Slice every metric **by threshold**, **by sweep-tier**, and
  **over time** (by walk-forward step) — the over-time view shows whether
  calibration holds as the season progresses.
- `report.py` gains a **historical-backtest report** —
  `reports/YYYY-MM-DD-baseline-backtest.md` plus plots (reliability diagram,
  calibration-by-tier, error-over-time), a **thin layer** over the helpers (which
  return numbers/tables tests assert on). Its header states plainly: *model
  calibration on historical completed games — not betting performance, no lines
  involved.* This is a **sibling** to the live results report
  (`YYYY-MM-DD-results.md`, task #10), sharing helpers, distinct filename, so the
  two tracks never visually merge.
- Persist the graded OOS frame to `data/processed/backtest/` (CSV) for
  reproducibility: `pitcher`, `game_pk`, `game_date`, `wf_step`, `family`, `mu`,
  `alpha`, the threshold sweep `p_over_*`, `realized_strikeouts`, per-threshold
  `over_hit`, `tier`.

### 4. Retrain-or-load for go-live, and retrain cadence

**Decision: refit on full available history before the first live run, and adopt a
weekly retrain cadence.** It is late June 2026 (~half a season of data); the
persisted task-#7 artifact reflects whatever partial data existed when it was fit.
The live `refresh` should run the **most current** fit.

- `fit_production_model(corpus, *, through_date) -> BaselineModel` — fits on **all
  completed starts through `through_date`** (the expanding window's final, full
  fit) and `save_model`s it. This is the artifact `refresh` loads.
- **Cadence:** retrain weekly (matching the walk-forward step) on all data through
  the prior day, overwriting the artifact; the task-#9 staleness warning threshold
  (~14 days) is the safety net if a weekly retrain is missed. Retraining stays a
  deliberate, separate action from the daily predict run (task #9's stance).

### 5. Live-data verification gate (`verify_live_sources`)

Before the first **real** graded run, a dry-run smoke test confirms the live
payloads match what the parsers expect (the PrizePicks `league_id`/shape and the
StatsAPI hydration were **never verified against a live call** — flagged in the
prizepicks and pre-game specs):

- `verify_live_sources(*, lines_fetcher, schedule_fetcher) -> report` and a
  `refresh --dry-run` mode that **writes nothing**. Concrete checks: PrizePicks
  returns MLB projections for `league_id=2` with a `"Strikeouts"` `stat_type` and
  the columns `flatten_projections` expects; the StatsAPI schedule with
  probable-pitcher hydration returns `probablePitcher.id` (MLBAM) + team for at
  least one game. Print a pass/fail summary.
- On mismatch: **stop and fix** the parser/`league_id` (the endpoint is unofficial
  and may have shifted), then re-verify. This gate is non-optional before trusting
  any live output.

### 6. Operational go-live plan (recorded, not coded)

- **Seed/backfill:** Track B cannot be backfilled (no historical lines). Start the
  daily loop fresh: morning `refresh`, D+1 `settle`, `report` weekly / after N
  settled.
- **Timeline to a readable Track-B report:** MLB posts ~10–15 strikeout starters/
  day, and (task #8) line picks **cluster Low**, with High picks being rare
  market-disagreement signals. So **Low** clears the ≥100 bar in ~1–2 weeks while
  **High** may take a month-plus. First read the live report at ~2 weeks; treat
  per-tier ROI/hit-rate as provisional until each tier individually clears 100.
  Meanwhile **Track A gives sweep-tier calibration immediately.**

## Output schema

- **`data/processed/backtest/walk_forward_oos.csv`** — the graded OOS frame above
  (Track A; no line/ROI fields).
- **`reports/YYYY-MM-DD-baseline-backtest.md`** + plot PNGs — the Track-A report.
- **Updated model artifact** (via `fit_production_model` + `save_model`) with
  metadata (`trained_at`, `train_through_date`).
- **`verify_live_sources` summary** — printed/logged pass/fail, written nowhere on
  `--dry-run`.

Canonical key throughout: `(pitcher, game_pk)`.

## Edge cases (all must be handled, none should crash)

- **Early-season walk-forward step with too few training dates** (< `min_train_dates`)
  → skip the step / emit NaN metrics, surfaced in coverage, no crash.
- **Pitcher with insufficient history at a step** → dropped by the model's existing
  core-feature `dropna`; counted in a coverage diagnostic, not silently lost.
- **Corpus window fetch fails / rate-limits** → resume from cache; partial corpus
  flagged in the run summary.
- **Opener / bullpen game** misclassified by the BF starter proxy → documented
  limitation; the BF floor mitigates.
- **No historical lines** → ROI absent **by design**, not an error.
- **Doubleheaders** → keyed by `game_pk`.
- **Live verification mismatch** → fail the gate loudly; never proceed to a real
  run on an unverified shape.
- **Re-running the backtest** → deterministic; cached corpus reused; OOS frame
  overwritten.

## Module layout

- **`src/backtest/corpus.py`** (new) — `build_corpus` (injected `statcast_fetcher`,
  cache/resume) + the starter filter.
- **`src/backtest/walk_forward.py`** (new) — `run_walk_forward` (injected `fit_fn`),
  OOS accumulation.
- **`src/evaluation/metrics.py`** (extend) — add `pit_histogram`; reuse the rest.
- **`src/backtest/report.py`** (extend) — the historical-backtest report (sibling
  to the live report, shared helpers).
- **`src/models/baseline_model.py`** (extend) — `fit_production_model(corpus,
  through_date)`.
- **`src/pipeline/refresh.py`** (extend) — `--dry-run` + `verify_live_sources`
  (or a small `src/pipeline/verify.py`); writes nothing.
- **Tests:** `tests/test_corpus.py`, `tests/test_walk_forward.py`,
  `tests/test_metrics.py` (PIT), `tests/test_report.py` (backtest report),
  `tests/test_verify.py`, + a `fit_production_model` case in
  `tests/test_baseline_model.py`.

No new third-party dependency (matplotlib already pinned; CSV cache, no `pyarrow`).

## Deferred (conscious scope cuts — named, not designed)

- Model-v2 changes (exposure offset, regularized/CV variant, VIF pruning,
  boosting), new features (umpire/weather/bullpen/`day_night`) — justified later by
  *this* backtest, not before it.
- Tier **redefinition** — Track A gives a first calibration read, but the task-#10
  bar (≥100 settled line picks/tier, forward) still governs any redefinition.
- Prop expansion (pitching outs / earned runs / walks).
- Scheduling/automation (cron/CI) and a live game-status feed to shorten the
  pending→void wait — all still deferred.

## Testing approach — what the new `tests/test_*.py` must assert

Deterministic, **no-network**, hand-built fixtures; inject `statcast_fetcher`,
`fit_fn`, and the live fetchers (matching the task #9/#10 convention):

1. **No temporal leakage (front and center):** for each walk-forward step, every
   training row is dated strictly before the step cutoff and every OOS prediction
   is for a start on/after it; a fixture is constructed so a leak would change a
   number, and doesn't.
2. **Leakage invariant under appended data:** an earlier step's OOS predictions are
   unchanged when later games are appended to the corpus (mirrors the
   `rolling_features` leakage test).
3. **Expanding window mechanics:** step boundaries and the growing train set are
   correct; `min_train_dates` skips early steps cleanly.
4. **OOS coverage:** each eval start appears exactly once in the accumulated OOS
   frame, joined to its realized outcome on `(pitcher, game_pk)`; doubleheaders
   stay distinct.
5. **Metric reuse + PIT:** calibration/ECE/Brier/log loss/MAE/RMSE match the
   `evaluation/metrics.py` references on a fixture; `pit_histogram` matches a
   hand-computed reference.
6. **Slices:** by-threshold, by-sweep-tier, and over-time aggregations are correct
   on a fixture.
7. **No-ROI guard:** the backtest output contains **no** line/edge/ROI fields — a
   test asserts their absence so betting performance can't be accidentally
   fabricated from a line-less track.
8. **Corpus cache/resume:** an already-cached window is not re-fetched (assert the
   injected fetcher's call count); concatenation across windows is correct.
9. **No-network end to end:** injected `statcast_fetcher` + `fit_fn` run the whole
   walk-forward on a fixture corpus with no real calls.
10. **Insufficient-history step** surfaces NaN/coverage, never raises.
11. **Verification gate:** a well-formed live-payload fixture passes; a malformed
    one (wrong `league_id` result, missing `stat_type`, missing `probablePitcher.id`)
    is flagged with the specific failure, and `--dry-run` writes nothing.
12. **Production refit:** `fit_production_model` on a fixture corpus returns a model
    that round-trips through `save_model`/`load_model` with correct
    `train_through_date` metadata.
13. **Report helper returns numbers:** the backtest report's metric assembly
    returns the dict/tables tests assert on; markdown/plots are a thin layer
    (smoke-test a file is written to `reports/`).
