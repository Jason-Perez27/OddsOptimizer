# Prediction Outcome Tracking & Backtest — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (2026-06-26, 2026-06-27 entries),
`docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md` (task #7),
`…-tiered-prop-probabilities-design.md` (task #8),
`…-pre-game-refresh-pipeline-design.md` (task #9).
Implementation targets: `src/pipeline/settle.py` (new), `src/evaluation/`
(`grading.py`, `metrics.py`, new), `src/backtest/` (`roi.py`, `report.py`, new),
a by-id outcome pull in `src/data/pitcher_logs.py` (extend), and the matching
`tests/test_*.py`.

## Goal

Close the loop on the pipeline: take the daily picks task #9 writes to disk, pull
each game's **realized** strikeout outcome once the game is final, grade every
prediction against it, and accumulate an honest, growing record of how the model
actually performs — calibration, hit rate, and a flat-bet ROI backtest, sliced by
tier, by threshold/line, and over time. This is the "honestly evaluate those
probabilities against real outcomes over time" half of the README's project goal.

Two design stances frame everything below:

- **Grading is a separate, later pass from prediction.** A game is not final at
  prediction time, so settlement runs *after* the game (next day by default),
  reuses the same free `pybaseball` outcome path the model's training labels come
  from, and is built on the task #9 **dependency-injected fetcher** convention so
  the core is unit-testable with hand-built fixtures and no network.
- **Measure the tiers, don't redefine them — yet.** The decision log flagged the
  probability-only tier definition as a "task #10 revisit." v1 *validates* the
  tiers empirically (hit rate + calibration by tier) and reports the evidence; it
  does **not** redefine them. Redefinition is a separate, future decision with a
  stated bar (see "The confidence-tier revisit").

Scope is **strikeouts only**, consistent with the rest of the project (decision
log, 2026-06-27); the grading/metrics machinery is built prop-agnostic where it is
free to be, but only strikeouts is wired and validated.

## Inputs

1. **A game date (or date range) to settle** — defaults to "the most recent
   unsettled date(s)"; an explicit `--date`/`--from`/`--to` drives backfill.
2. **That date's prediction outputs** — read from task #9's partition
   `{processed_dir}/predictions/game_date=YYYY-MM-DD/`: `predictions.csv`
   (`pitcher` MLBAM id, `pitcher_name`, `pitcher_team`, `opponent_team`,
   `game_date`, **`game_pk`**, `family`, `mu`, `alpha`), `threshold_table.csv`
   (the 1–10 P(over) sweep, carrying `pitcher`, `game_pk`, `threshold`, `p_over`,
   `tier`), and `line_picks.csv` (carrying `pitcher`, `game_pk`, `line`,
   `line_threshold`, `lean`, `tier`, `p_over`, `p_under`, `edge`, `push_mass`,
   `pulled_at`). **Integration checkpoint:** grading joins on `(pitcher,
   game_pk)`, so `game_pk` must be present on all three tables — established by the
   task #9 spec; verify in the task #9 implementation before relying on it.
3. **Realized per-game outcomes** — pulled via `pybaseball`'s
   `statcast_pitcher(start, end, player_id)` for the predicted pitchers/date,
   aggregated by the existing `src.features.game_logs.aggregate_pitcher_games`,
   which already yields `strikeouts` per `(pitcher, game_pk)`. No new data
   provider — the same path that produces training labels produces outcomes.
4. **`configs/config.yaml`** — `data.processed_dir`, and new
   `evaluation`/`backtest` knobs (settlement lag hours, scratch-void max-wait
   days, reliability-bucket count, rolling-window size).

## The moving pieces

### 1. Outcome ingestion & timing (`src/pipeline/settle.py`, new; `pitcher_logs.py` extend)

Realized strikeouts come from Statcast via `pybaseball`. Predictions already carry
the **MLBAM id**, so we pull *by id*, not by name. `pitcher_logs.py` currently
exposes `get_pitcher_season_logs(full_name, ...)` (name → id internally); add a
thin **by-id** sibling so settlement doesn't round-trip through name resolution:

```python
def get_pitcher_logs_by_id(player_id: int, start_date: str, end_date: str) -> pd.DataFrame
```

(thin wrapper over `statcast_pitcher`; the **only** network call on the settle
path, injectable as `outcome_fetcher`, not unit-tested).

**Timing.** Baseball Savant/Statcast for a completed game is reliably available
within a few hours, generally by the next morning. v1 settles **D+1 or later**: a
daily "settle yesterday (and any still-pending older dates)" pass, plus a batch
backfill mode for a date range. A configurable `settlement_lag_hours` guards
against grading a game whose data isn't posted yet.

**Final / pending / void detection** — three outcomes per predicted
`(pitcher, game_pk)`:

- **settled** — Statcast returns completed plate appearances for that
  `(pitcher, game_pk)` → `strikeouts` is real; grade it.
- **pending** — no Statcast rows yet *and* the game is within the lag/age window →
  leave unsettled, retried on a later run (settlement recomputes from scratch each
  pass, so pending naturally resolves once data lands).
- **void_scratched** — no Statcast rows *and* past a `scratch_void_max_wait_days`
  threshold → the probable starter never threw (scratch, postponement,
  rain-shortened DNP). Marked `void_scratched`, **excluded from headline metrics**,
  and kept visible in the ledger. (This is the expected "probable pitcher was
  scratched" case; never a silent drop, never a loss.)

`settle.py` mirrors `refresh.py`: `run_settlement(game_date, *, outcome_fetcher,
out_dir, overwrite, now=...)` with injected fetcher and an injected clock (`now`)
so the pending/void thresholds are testable; pure helpers below do the grading;
`main()` is the CLI (`python -m src.pipeline.settle --date YYYY-MM-DD`,
`--from/--to`, `--no-overwrite`).

### 2. Grading / join logic (`src/evaluation/grading.py`, new — pure)

Pure functions joining predictions to realized outcomes by `(pitcher, game_pk)`:

- `attach_outcomes(pred_df, realized_df) -> DataFrame` — left-join realized
  `strikeouts` onto prediction rows by `(pitcher, game_pk)`; assign
  `settlement_status` (settled / pending / void_scratched) per §1. Left-join (not
  inner) so unresolved predictions stay visible.
- `grade_line_picks(line_picks_df, realized_df) -> DataFrame` — per pick:
  - `over_hit` = `realized >= line_threshold` (= `floor(line)+1`), `NaN` if unsettled.
  - `push` = `True` only for an integer line with `realized == line` (PrizePicks
    uses .5 lines, so push is the defensive case task #8 already records via
    `push_mass`); pushes are refunded, not scored.
  - `pick_correct` = does `lean` match the realized side (`over_hit` for an "over"
    lean, `not over_hit` for "under"); `NaN` on push/unsettled.
  - `pnl_units` = the flat-bet proxy P&L (+1 win / −1 loss / 0 push, `NaN`
    unsettled) — see §4 for the explicit payout convention.
- `grade_threshold_sweep(threshold_table_df, realized_df) -> DataFrame` — per
  pitcher × threshold: `over_hit = realized >= threshold`; this is the frame
  calibration is computed from (every threshold is a P(K≥t) the model committed
  to, line or no line).

All three are deterministic, take frames in / frames out, and never call the
network — the same testability contract as `tiering.py`.

### 3. Storage (graded partitions, mirroring task #9)

Graded results live in a **new partition that mirrors the predictions layout**,
written by `settle.py`:

```
{processed_dir}/outcomes/game_date=YYYY-MM-DD/
  graded_line_picks.csv
  graded_threshold_sweep.csv
  settle_manifest.json
```

**Why per-date CSV partitions and not a single rolling ledger:** it matches task
#9's convention exactly (CSV, no `pyarrow`, `game_date=` Hive partition,
idempotent overwrite), and it avoids a second mutable source of truth that could
drift from the partitions. **Cumulative metrics are computed by globbing the
partitions at report time** — there is no separate append-only ledger to keep in
sync. Re-settling a date **overwrites** that date's graded files (same semantics
as `write_outputs`' `overwrite`; `--no-overwrite` aborts if present). Because each
settle recomputes a date from scratch, a previously-pending row simply becomes
settled on a later run — no upsert bookkeeping required.

### 4. Evaluation metrics (`src/evaluation/metrics.py` + `src/backtest/roi.py`, new)

Two explicitly separated questions, reflecting the decision log's distinction
between **the model's own calibration** and **beating the posted line**:

**(a) Model honesty — `src/evaluation/metrics.py`.** Computed from the graded
threshold sweep (and the per-pick over/under binaries):

- **Calibration / reliability** *(new)* — bucket `p_over` into N reliability bins,
  compare predicted vs. empirical hit frequency, and summarize with **ECE**
  (expected calibration error). Returns both the per-bucket table (for the
  reliability diagram) and the scalar ECE.
- **Brier score** and **log loss** *(reuse)* — `baseline_model.brier_score` /
  `log_loss` applied directly to the over/under binary at the line, and at the
  representative sweep thresholds task #7 named (≈5.5, 6.5).
- **Point accuracy** *(reuse)* — `mean_absolute_error` / `root_mean_squared_error`
  of `mu` vs. realized `strikeouts`.

**(b) Pick profitability — `src/backtest/roi.py`.** Computed from the graded line
picks:

- **Hit rate** *(new)* — wins / (wins + losses), **pushes and unsettled excluded
  from the denominator** (the honest version).
- **Flat-bet ROI** *(new)* — unit stake on each `lean` side. **v1 payout
  convention: even-money / unit flat bet** (win +1, loss −1, push 0), ROI =
  Σ`pnl_units` / number of settled non-push picks. This is a **labeled proxy**:
  PrizePicks is pick'em with parlay-style payout multipliers (e.g. 2-pick power
  play, 6-pick flex), and the decision log put modeling those multipliers — and
  any priced odds feed — **out of scope to keep the project free**. True
  payout-structure ROI and CLV are therefore **deferred** (CLV needs an odds feed
  the project deliberately dropped).

**Slices that actually answer "do the tiers mean anything"** — every (a) and (b)
metric is reported **by tier** (High/Medium/Low), **by threshold / line value**,
and **over time** (cumulative + a rolling window). All metric functions take
graded frames and **return numbers/tables** (no file IO) so tests assert on them
directly, matching the task #7 evaluation-helper convention.

### 5. The confidence-tier revisit (resolve task #8's flag)

**v1 stance: measure, don't redefine.** The tiers stay **probability-only**
(distance of P(over) from 0.5), exactly as task #8 shipped. Task #10's job is to
*validate* them — report empirical hit rate and calibration **by tier** so we can
see whether "High" picks actually separate from "Low." Redefining the tier logic
(promoting `edge`, or a backtest-calibrated cut) is a **separate future decision**,
not taken here, for the same reason task #8 gave: it must not depend on too little
data.

**The bar for revisiting** (stated so it isn't silently re-opened): at least a
**meaningful resolved sample** — on the order of a few hundred settled line picks
with non-trivial counts in each tier (recommend ≥100 settled per tier as a floor)
— **and** evidence the current definition is actually failing (e.g. High-tier
empirical hit rate not separating from Medium/Low, or systematic miscalibration in
a tier). Until both hold, probability-only tiers stand. The report surfaces the
per-tier sample sizes precisely so this trigger is observable.

### 6. Results write-up (`src/backtest/report.py`, new → `reports/`)

"Write up results" produces a **generated, dated markdown report** in `reports/`
(the dir the README reserves for "dated results writeups"):
`reports/YYYY-MM-DD-results.md`, with optional plot PNGs alongside it (a
**reliability diagram** and a **cumulative-ROI curve**, via the already-pinned
`matplotlib`). The report is a **thin presentation layer**: `report.py` calls the
§4 metric helpers (which return numbers/tables), then renders calibration, hit
rate by tier and by threshold, ROI and its time series, and the per-tier sample
sizes for the revisit trigger. The numeric helpers remain independently testable;
the markdown/plots are formatting on top. Cadence is **on-demand / after N graded
games** — a `main()` entry point the operator runs; a human can layer prose on top
of the generated numbers, but the figures are never hand-typed.

## Output schema

**`graded_line_picks.csv`** (one row per settled-or-pending line pick):
`pitcher`, `game_pk`, `pitcher_name`, `team`, `game_date`, `line`,
`line_threshold`, `lean`, `tier`, `p_over`, `p_under`, `edge`, `push_mass`,
`pulled_at`, `realized_strikeouts`, `settlement_status`
(`settled`/`pending`/`void_scratched`), `over_hit` (bool/NaN), `push` (bool),
`pick_correct` (bool/NaN), `pnl_units` (+1/−1/0/NaN).

**`graded_threshold_sweep.csv`** (one row per pitcher × threshold):
`pitcher`, `game_pk`, `pitcher_name`, `game_date`, `threshold`, `p_over`, `tier`,
`realized_strikeouts`, `settlement_status`, `over_hit` (bool/NaN).

**`settle_manifest.json`**: settle timestamp, `game_date`, outcome-pull timestamp,
counts (predicted, settled, pending, void_scratched, picks graded), and any error
flags. Mirrors task #9's `run_manifest.json` discipline.

**Reports** (`reports/YYYY-MM-DD-results.md` + plot PNGs): the rendered summary;
not a machine contract, but stable enough to diff across dates.

Canonical join key throughout: **`(pitcher, game_pk)`** (doubleheader-safe);
`game_date` is the partition.

## Edge cases (all must be handled, none should crash)

- **Game not final / Statcast lag** → `pending`; excluded from metrics; resolves on
  a later settle pass. Never graded as a loss for missing data.
- **Scratched / postponed starter** (no Statcast rows ever) → `void_scratched`
  after the max-wait; excluded from hit rate and ROI; visible in the ledger.
- **Pitcher pulled early** (a short, low-K start) → **not special for strikeouts**:
  fewer Ks is a legitimate `under` outcome and grades normally. Only a start that
  *never happened* voids. (Called out because "no-decision"-style scoring wrinkles
  exist for win/loss props but **do not apply to a strikeout count** — the count is
  whatever was recorded, even if 0.)
- **Doubleheader** (same pitcher, two `game_pk`, same date) → graded independently
  by `game_pk`; never cross-joined.
- **Integer line / push** → `push=True`, `pnl_units=0`, excluded from the hit-rate
  denominator (refunded).
- **Predictions partition missing for a requested date** → clean no-op with a
  manifest note; no raise.
- **Realized outcome present but the pitcher isn't in predictions** (we skipped
  them pre-game) → ignored for grading; nothing to score.
- **Re-settling a date** → overwrite (idempotent); `--no-overwrite` aborts.
- **`game_pk` absent on a prediction table** (task #9 regression) → fail fast with
  a clear message rather than silently mis-joining by `(pitcher, game_date)`
  (which doubleheaders would corrupt).

## Module layout

- **`src/data/pitcher_logs.py`** (extend) — add `get_pitcher_logs_by_id(...)`; no
  change to existing functions.
- **`src/evaluation/grading.py`** (new) — `attach_outcomes`, `grade_line_picks`,
  `grade_threshold_sweep` (pure).
- **`src/evaluation/metrics.py`** (new) — calibration/reliability + ECE; Brier /
  log loss / MAE / RMSE wrappers reusing `baseline_model` helpers (pure, numbers
  out).
- **`src/backtest/roi.py`** (new) — hit rate, flat-bet ROI, by-tier / by-threshold
  / over-time slices (pure).
- **`src/backtest/report.py`** (new) — markdown + plot generation into `reports/`
  (thin layer over the helpers).
- **`src/pipeline/settle.py`** (new) — the settlement orchestrator
  (`run_settlement`, `write_graded`, `main`), injected `outcome_fetcher` and
  `now` clock; mirrors `refresh.py`.
- **Tests:** `tests/test_grading.py`, `tests/test_metrics.py`,
  `tests/test_backtest_roi.py`, `tests/test_settle.py`,
  `tests/test_report.py` (+ a by-id pull stub in `tests/test_pitcher_logs.py`).

This honors the README's reserved dirs: `src/evaluation/` (calibration, Brier/log
loss), `src/backtest/` (ROI/hit-rate), `reports/` (write-ups), with the
network/IO orchestration isolated in `src/pipeline/settle.py` exactly as task #9
isolated it in `refresh.py`.

## Deferred (conscious scope cuts)

- **True PrizePicks payout-structure ROI** (parlay multipliers, power/flex) and
  **CLV** — both need payout modeling or a priced odds feed the project
  deliberately dropped for cost; v1 ships a labeled even-money flat-bet proxy.
- **Tier *redefinition*** (edge-driven or backtest-calibrated tiers) — measured,
  not changed, until the stated sample/evidence bar is met.
- **Non-strikeout props** — machinery is prop-agnostic where free, but only
  strikeouts is wired and validated.
- **A live "is the game final?" status feed** (e.g. MLB-StatsAPI game state) to
  shorten the pending→void wait — v1 infers from Statcast presence + a time
  threshold; a status-feed refinement is a later optimization.
- **Automated cadence** for settle/report (cron / scheduled workflow) — v1 is a
  manually-run script, consistent with task #9's low-infra stance.

## Testing approach — what the new `tests/test_*.py` must assert

Deterministic, **no-network** unit tests on small hand-built fixtures, injecting a
fake `outcome_fetcher` and a fixed `now` clock (the task #9 convention):

1. **`grade_line_picks` correctness:** a fixture of picks + realized counts yields
   the right `over_hit` / `push` / `pick_correct` / `pnl_units`, including an
   "under" win, an "over" loss, and an integer-line **push** (pnl 0, excluded).
2. **Settlement status:** a `(pitcher, game_pk)` with no realized row is `pending`
   within the wait window and `void_scratched` past `scratch_void_max_wait_days`
   — driven by the injected `now`; a present outcome is `settled`.
3. **Join is on `(pitcher, game_pk)`:** a doubleheader (same pitcher, two
   `game_pk`) grades each game independently; no cross-join; a `game_date`-only
   join would be caught failing.
4. **Hit rate excludes pushes and pending** from the denominator; a fixture with
   both confirms the honest rate.
5. **By-tier / by-threshold aggregations** are correct on a hand-checkable fixture
   (counts and rates per tier and per line value).
6. **Calibration + ECE:** reliability buckets and the ECE scalar match a
   hand-computed reference on a tiny fixed set.
7. **Brier / log loss reuse** `baseline_model` helpers and match hand-computed
   values on the over/under binaries.
8. **Flat-bet ROI proxy:** wins/losses/pushes map to the right units and ROI;
   pushes refund (0); unsettled excluded.
9. **Over-time series:** cumulative and rolling ROI/hit-rate series have the right
   length and cumulative values on an ordered fixture.
10. **Idempotent settle:** settling a date twice yields identical graded files
    (overwrite, no duplication); a previously-`pending` row becomes `settled` once
    the fake fetcher returns its outcome.
11. **Injected outcome fetcher, no network:** a fake fetcher returning fixture
    pitch rows drives `run_settlement` to the expected graded partition.
12. **Empty / missing predictions partition** → clean no-op + manifest note, no
    raise.
13. **Report helper returns numbers:** `report.py`'s metric assembly returns the
    metrics dict/tables tests assert on; markdown/plots are a thin layer over it
    (smoke-test that a file is written to `reports/`).
14. **Strikeout-count grading is unconditional on game length:** a short, low-K
    start grades as a normal `under` (not a void) — only a never-thrown start voids.
