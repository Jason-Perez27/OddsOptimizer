# Pre-Game Refresh Pipeline — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (2026-06-26, 2026-06-27 entries),
`docs/design/specs/2026-06-27-strikeout-feature-engineering-design.md` (task #6),
`…-baseline-poisson-nb-model-design.md` (task #7),
`…-tiered-prop-probabilities-design.md` (task #8).
Implementation targets: `src/data/probable_pitchers.py` (new),
`src/features/predict_features.py` (new), `src/models/baseline_model.py`
(add persistence), `src/pipeline/refresh.py` (new), and the matching
`tests/test_*.py`.

## Goal

Produce, for a single game date, the project's actual daily product — the
`threshold_table`, `line_picks`, and `diagnostics` for that day's starting
pitchers — by orchestrating the existing ingestion, feature, model, and tiering
layers end to end. This is the script a human runs each morning (and a future
cron job could call unchanged): *who is starting today → what does the model say →
what does PrizePicks post → write the picks to disk.*

Every existing module either operates on **historical** Statcast data for games
that already happened (`build_features.build_training_table`, the model fit) or
takes a ready-made `predictions_df` as input (`src/predictions/tiering.py`).
Nothing yet answers *which pitchers are starting today, for which team, against
which opponent.* Closing that gap — and computing each starter's pre-game
features **as of today, with no label** — is the substance of this task. Scope is
**strikeouts only**, consistent with the rest of the pipeline (decision log,
2026-06-27); the slate/feature/orchestration machinery is built prop-agnostic so
the later props slot in without redesign.

The pipeline is an **orchestrator**: it owns sequencing, failure handling, and
output, and pushes all real logic into existing tested functions plus a small set
of new **pure, dependency-injected** helpers. Live network calls
(`MLB-StatsAPI`, `pybaseball`, PrizePicks) are kept thin and isolated behind
injectable fetchers so the core is unit-testable with hand-built fixtures and no
network, exactly as `tests/test_tiering.py` and `tests/test_baseline_model.py`
already work.

## Inputs

1. **Game date** — defaults to today (`date.today()`), overridable via
   `--date YYYY-MM-DD` for backfill/replay. All partitioning and "strictly prior"
   feature cutoffs key off this date.
2. **Today's probable starters** — from **MLB-StatsAPI** (`statsapi`, already
   pinned in `requirements.txt`, not yet used by any module). The schedule
   endpoint returns, per game: `gamePk`, game date/time, home/away team, and each
   side's `probablePitcher` (`id` = **MLBAM person id**, `fullName`). Critically,
   StatsAPI gives the **MLBAM id directly**, so the model side needs *no*
   name→id resolution — the join key the model uses (`pitcher` = MLBAM id) is
   handed to us. Pitcher throwing hand (L/R) is read from the player's record
   (or carried over from that pitcher's Statcast history; see "Synthetic row").
3. **Each starter's season-to-date Statcast history** — via existing
   `src.data.pitcher_logs` (`statcast_pitcher` by MLBAM id), aggregated by
   existing `src.features.game_logs.aggregate_pitcher_games`. This is the prior
   history every rolling/opponent/park feature is computed from.
4. **Current PrizePicks lines** — via existing `src.data.prizepicks_lines`
   (`fetch_projections` → `flatten_projections(payload, "Strikeouts")`), unchanged.
5. **A persisted, fitted `BaselineModel`** — loaded from disk, not refit per run
   (see "Model availability"). Persistence does not exist yet; this task adds it.
6. **`configs/config.yaml`** — read for `data.processed_dir`,
   `prizepicks.league_id`, and `project.active_prop` + its `thresholds`, so the
   pipeline has no magic numbers the rest of the repo doesn't already declare.

## The moving pieces

### 1. Today's slate (`src/data/probable_pitchers.py`, new)

A new `src/data/` module, **parallel to `pitcher_logs.py` and
`prizepicks_lines.py`** — one external source, one ingestion module, a thin live
call plus a pure parser. (It is *not* buried inside `src/pipeline/` because it is
a reusable data source in its own right, and isolating the StatsAPI call here is
what lets the pipeline core be tested without it — the same separation
`prizepicks_lines.py` already follows.)

- `fetch_schedule(game_date, *, hydrate="probablePitcher") -> dict` — the **only**
  network call; thin wrapper over `statsapi.schedule(...)` / the schedule
  endpoint with probable-pitcher hydration. Not exercised by tests.
- `parse_probable_starters(schedule_payload, game_date) -> pd.DataFrame` —
  **pure**, the testable core. One row **per probable starting pitcher** (two
  rows per game when both sides are posted, one when only one is, zero when none
  are). Columns:

  | column | source | notes |
  |---|---|---|
  | `pitcher` | `probablePitcher.id` | MLBAM id — the model-side join key |
  | `pitcher_name` | `probablePitcher.fullName` | carried to `threshold_table` |
  | `pitcher_team` | home/away team of this pitcher | **Statcast code** (crosswalk below) |
  | `opponent_team` | the other team | **Statcast code** |
  | `home_away` | `"home"`/`"away"` | needed for park + home/away features |
  | `game_pk` | `gamePk` | real id; the per-game join key (doubleheader-safe) |
  | `game_date` | the run date | |
  | `start_time` | game `gameDate`/time | carried to `line_picks` parity |
  | `pitcher_throws` | player record `pitchHand`, else `None` | filled from history if null |

- **Team-code crosswalk (`STATSAPI_TO_STATCAST_TEAM`).** StatsAPI, Statcast, and
  PrizePicks each use their own team abbreviations. The model's historical
  features key team identity by **Statcast** codes (the `home_team`/`away_team`
  strings `game_logs.py` reads), so the slate's `pitcher_team`/`opponent_team`
  **must be mapped into Statcast space** or every opponent/park lookup silently
  misses. This is a *distinct* crosswalk from tiering's `TEAM_CROSSWALK`
  (PrizePicks→Statcast); both are small static dicts seeded with the known
  mismatches and extended as real data surfaces. Provide `to_statcast_team(code)`
  mirroring tiering's helper. An unmapped code passes through unchanged (and is
  surfaced in diagnostics if it later fails to join), never guessed.

### 2. Pre-game ("as-of-today") feature construction (`src/features/predict_features.py`, new)

This is the trickiest piece, and the key realization is that **almost nothing new
is needed**: the existing feature builders already compute "as of this row's
date" values, because their leakage guardrail *is* an as-of-date computation.
`rolling_features`, `opponent_features`, and `park_factors` all use
`shift(1)` → rolling/cumulative aggregates and strict `game_date <` filters, so a
row dated **today** automatically sees only games strictly before today. We do
not need a "predict mode" branch inside any of them; we need to **hand them a row
dated today** and slice it back out.

**Mechanism — synthetic same-day game rows:**

1. **Build historical game-logs** for the slate's pitchers: for each unique
   `pitcher` in the slate, pull season-to-date Statcast via `pitcher_logs`, run
   `game_logs.aggregate_pitcher_games`, and concatenate. (For opponent and park
   features to be correct, this historical table should also include the
   **opponents'** pitchers' games — see "Opponent/park history coverage" below.)
2. **Build one synthetic row per slate starter** (`build_synthetic_game_rows`):
   a `game_logs`-schema row with the **known pre-game** fields filled
   (`pitcher`, `game_pk`, `game_date`=today, `pitcher_team`, `opponent_team`,
   `home_away`, `pitcher_throws`) and **every same-game-outcome field set to
   `NaN`** (`strikeouts`, `batters_faced`, `pitch_count`, `whiff_rate`,
   `fastball_velo_avg`, `innings_pitched`, the `*_vs_LHB/RHB` counts). These are
   the columns the model spec already forbids as regressors (leakage list); they
   are unknown pre-game and are never read for the synthetic row's *own* features
   (every builder `shift(1)`s the current row out), so `NaN` is safe.
   `day_night` stays `None` as everywhere else. `rest_days` is **recomputed**
   after concatenation (the synthetic row is added post-aggregation, so re-derive
   `rest_days` = `groupby("pitcher")["game_date"].diff().dt.days`, giving
   today − last real start = the correct pre-game rest).
3. **Concatenate** historical rows + synthetic rows into one frame and run the
   existing builders **unmodified**:
   `add_rolling_features → add_opponent_features → add_park_factors`. Same call
   order and whole-table batching as `build_features.build_training_table`
   (opponent/park need the full multi-pitcher table).
4. **Slice out the synthetic (today) rows** as the prediction feature table.

**Why each builder is reused unmodified — verified against the code:**

- `add_rolling_features` is pitcher-grouped and `shift(1)`-based: the synthetic
  today row gets each `*_last5`/`*_season`/split feature from that pitcher's
  strictly-prior real starts only; its own `NaN` outcomes are shifted out and one
  pitcher's synthetic row can't perturb another's (separate groups).
- `add_opponent_features` builds `team_game_logs` from the same frame, so the
  synthetic today team-game exists in `tg` and its `opponent_k_rate_last10` /
  `_home` / `_away` are computed from the opponent's strictly-prior team games,
  then column-joined back on `(opponent_team, game_pk)` using the **real
  gamePk**. `opponent_k_rate_vs_hand_season` is a per-row `game_date <` lookup
  keyed on **this row's** `pitcher_throws` — which we supply on the synthetic row
  — so it resolves correctly. Two opposing starters sharing one gamePk produce
  two distinct team-games (grouped by `opponent_team`), so they don't collide.
- `add_park_factors` keys on `park_team` (= home team) + gamePk, strictly-prior
  park K-rate vs. a strictly-prior league benchmark, with the
  `STATIC_PARK_FACTORS` cold-start fallback. The synthetic today park-game is
  excluded from its own value by the same `shift(1)`/strict-prior logic.

**What is *not* reused:** `build_features.build_training_table()` itself, for two
concrete reasons — it starts from `pitch_df` and re-aggregates (we already have
game-logs), and it **validates `strikeouts` (and other required columns)
non-null**, which a label-less synthetic row violates by design. So
`predict_features.build_prediction_features(historical_game_logs, slate_df)` is a
**sibling** to `build_training_table`, reusing the same three builder functions
but with a no-label contract. (Placement rationale, in the style of the tiering
spec: the *training* join and the *predict* join have genuinely different
contracts — labelled vs. label-less, pitch-level vs. game-level entry — so the
predict path earns its own thin module rather than a `mode=` flag bolted onto the
training join. Fallback if review prefers cohesion: add
`build_prediction_features` *into* `build_features.py` next to its training
sibling — flag in review and it'll be moved.)

**Opponent/park history coverage (call out, don't hand-wave).** Opponent and park
features are only as good as the history fed in. v1 builds historical game-logs
from the **slate pitchers' own** season pulls; this fully covers each pitcher's
rolling features, and covers opponent/park features to the extent those pitchers
have already faced today's opponents / pitched in today's parks. The honest
limitation: a thinly-sampled opponent (few prior games present in the assembled
history) yields a sparse `opponent_k_rate_*`, which the model layer then **imputes
with the train mean and flags `was_imputed`** (existing behavior) rather than
dropping the pitcher. This is acceptable for v1 and explicitly noted; a v2
hardening (deferred) pulls a fuller league-wide team-batting history so opponent
rates rest on complete samples.

### 3. Model availability — persist and load, don't refit (`src/models/baseline_model.py`)

**v1 decision: the model is fit once (offline, by the existing task-#7 path) and
this pipeline *loads* it.** `BaselineModel` currently has **no save/load story** —
that gap is closed here:

- `save_model(model, path)` — `joblib.dump` of the `BaselineModel` (its fitted
  statsmodels `result`, `preprocessor`, `family`, `active_columns`, `alpha`) plus
  a metadata sidecar: `trained_at`, `train_through_date` (max `game_date` in
  training), `family`, and the spec/code version. `joblib` ships with the
  already-pinned `scikit-learn`; pin it explicitly in `requirements.txt`.
- `load_model(path) -> (BaselineModel, metadata)`.

**Why load, not refit each run:** refitting daily is simpler to wire but
re-couples the morning pick run to a full historical training pull (many
`pybaseball` calls) and conflates two cadences that should be separate —
*retraining* (occasional, deliberate, evaluated) vs. *predicting* (daily, cheap).
Loading keeps the daily run fast and reproducible and means a bad data day can't
silently move the model.

**The staleness question this introduces** is handled by surfacing, not hiding:
the run reads the artifact's `trained_at` / `train_through_date`, records **model
age in days** in the run manifest, and **warns** (non-fatal) if the model is older
than a configurable threshold (default ~14 days). It never auto-refits.

- **v2 (deferred):** a scheduled/triggered retrain step (e.g. retrain weekly or
  when model age exceeds a bound) writing a new artifact the pipeline picks up.

### 4. Orchestration & idempotency (`src/pipeline/refresh.py`, new)

The orchestrator lives in `src/pipeline/` exactly as the README reserves it.
`run_refresh(...)` takes **injectable fetchers** (`schedule_fetcher`,
`statcast_fetcher`, `lines_fetcher`, `register_fetcher`, `model_loader`) defaulting
to the real ones, so tests pass fakes — mirroring how `tiering.build_line_picks`
takes an injected `register_df`.

**End-to-end sequence:**

1. Resolve `game_date`; load config; **load the persisted model** (fatal if
   missing — clear "train a model first" error).
2. **Fetch slate** (`probable_pitchers.fetch_schedule` →
   `parse_probable_starters`). **Empty/failed slate → fatal**: no starters means
   nothing to predict (the one genuinely blocking dependency).
3. For each unique slate pitcher, **pull Statcast** and aggregate to game-logs,
   wrapped **per-pitcher in try/except**: a pitcher whose pull fails or who has
   **no resolvable Statcast history this season yet** (debut/first start) is
   **skipped and surfaced in `skipped_pitchers`**, never crashing the slate
   (constraint: this is an expected case).
4. `predict_features.build_prediction_features(history, slate)` → today feature
   rows. A pitcher whose row is dropped by the model's
   `transform_design_matrix` (missing core pitcher-form features → too little
   history) also lands in `skipped_pitchers` (no prediction fabricated).
5. **Predict:** `model.predict_mean` over the design matrix → assemble
   `predictions_df` (exact tiering contract — see Output schema).
6. `tiering.build_threshold_table(predictions_df)` → the sweep (exists for every
   predicted pitcher, line or no line).
7. **Fetch PrizePicks lines** + `fetch_register`, wrapped in try/except:
   **PrizePicks failure is non-fatal.** If lines (or the register) can't be
   pulled, emit the full `threshold_table` for everyone, an **empty
   `line_picks`**, and record `prizepicks_error` in diagnostics — the sweep is
   the product even without lines (consistent with the tiering spec: the sweep is
   line-independent). This is the explicit answer to the prompt's partial-failure
   question: **degrade to a sweep table, do not hard-fail.**
8. `tiering.build_line_picks(predictions_df, lines, register)` → `line_picks` +
   tiering diagnostics (`unmatched_lines`, `predicted_no_line`).
9. **Merge diagnostics** (`skipped_pitchers`, `unmatched_lines`,
   `predicted_no_line`, source pull timestamps, model metadata, error flags) and
   **write outputs** (next section).

**Idempotency / re-running the same day:** outputs for a date live in a
**date-partitioned directory** and are **overwritten** on re-run by default —
each run is a deterministic function of that morning's inputs, so overwrite (not
append) is correct for the processed pick tables; appending would duplicate
pitchers. A `run_manifest.json` timestamps each run (and lines carry `pulled_at`),
so successive snapshots are distinguishable. Raw PrizePicks pulls keep appending
as timestamped files (existing `prizepicks_lines.save_raw` behavior — useful for
later line-movement study). A `--no-overwrite` flag aborts if the partition
already exists. Re-running is therefore always safe.

## Output schema

Written under `{processed_dir}/predictions/game_date=YYYY-MM-DD/` (Hive-style
partition so task #10 can glob/join by date). **Format: CSV** + one JSON manifest
— `pyarrow` is not a project dependency and the repo's `save_raw` convention is
already CSV; staying CSV keeps the project dependency-free and the files
human-inspectable. (Parquet is a deferred optimization once volume warrants,
requiring a `pyarrow` pin.)

Files in the partition:

1. **`predictions.csv`** — the raw model output and the **task-#10 anchor**: one
   row per predicted pitcher with `pitcher` (MLBAM id), `pitcher_name`,
   `pitcher_team`, `opponent_team`, `game_date`, `game_pk`, `family`, `mu`,
   `alpha`. This is **exactly the `predictions_df` contract `tiering` consumes**
   (`pitcher`, `pitcher_name`, `pitcher_team`, `opponent_team`, `game_date`,
   `family`, `mu`, `alpha`), plus `game_pk` for joins.
2. **`threshold_table.csv`** — `tiering.build_threshold_table` output:
   `pitcher`, `pitcher_name`, `team`, `opponent_team`, `game_date`, `threshold`
   (1…10), `p_over`, `tier`. Plus `game_pk` carried through for the join key.
3. **`line_picks.csv`** — `tiering.build_line_picks` output: `pitcher`,
   `pitcher_name`, `team`, `start_time`, `line`, `line_threshold`, `p_over`,
   `p_under`, `tier`, `lean`, `edge`, `push_mass`, `projection_id`, `pulled_at`.
   The `line` and `pulled_at` **freeze the line as posted at prediction time**,
   which task #10 needs to grade honestly (the line may move later).
4. **`diagnostics/`** — `unmatched_lines.csv`, `predicted_no_line.csv`,
   `skipped_pitchers.csv` (id, name, reason).
5. **`run_manifest.json`** — run timestamp, `game_date`, counts (slate size,
   predicted, skipped, lines matched/unmatched), per-source pull timestamps and
   any error flags (`prizepicks_error`, etc.), and model metadata (`trained_at`,
   `train_through_date`, model age days, staleness warning bool).

**Task-#10 join contract (chosen now to avoid a later redesign):** the canonical
key is **`(pitcher, game_pk)`** — `game_pk` is unique per game and
doubleheader-safe, where `(pitcher, game_date)` is not. Task #10 joins realized
strikeout outcomes (pulled the same `pybaseball` way) onto `predictions.csv` /
`line_picks.csv` by `(pitcher, game_pk)`; `game_date` remains the partition for
cheap date-range scans. Every output table carries both.

## Edge cases (all must be handled, none should crash)

- **No games / no probable pitchers today** → fatal-but-clean: empty, well-formed
  partition + manifest noting an empty slate; exit non-error. (Distinct from an
  *errored* fetch, which is fatal-with-error.)
- **Probable pitcher with no Statcast history yet** (debut, or first start of the
  season) → `skipped_pitchers` with reason `no_history`; no fabricated
  prediction. Expected, not exceptional.
- **Pitcher with history but features dropped** by the model's core-feature
  `dropna` (too few prior starts for `*_last5`) → `skipped_pitchers` reason
  `insufficient_features`.
- **PrizePicks endpoint down / 403 / shape change** → non-fatal; full sweep
  emitted, `line_picks` empty, `prizepicks_error` flagged in the manifest.
- **Chadwick register fetch fails** → non-fatal; same as PrizePicks-down (no
  line picks, sweep stands), flagged.
- **Slate pitcher posts no PrizePicks line** → in `threshold_table` +
  `predicted_no_line`; absent from `line_picks` (existing tiering behavior).
- **PrizePicks line with no model prediction** (a starter we skipped) → in
  `unmatched_lines`; never mis-joined (existing tiering resolver guarantees).
- **Team abbreviation not in the StatsAPI→Statcast crosswalk** → passed through;
  if it then fails to match opponent/park history the affected feature imputes
  (model layer) and the case is visible via `was_imputed`/diagnostics — surfaced,
  not silently wrong.
- **Doubleheader** (same pitcher, two `game_pk` same date) → two rows keyed
  distinctly by `game_pk`; nothing collapses.
- **Re-run same day** → overwrite (or abort under `--no-overwrite`); never
  duplicate rows.
- **Model artifact missing/corrupt** → fatal with an explicit "fit & save a model
  first" message.
- **Stale model** (age > threshold) → non-fatal warning recorded in the manifest;
  the run still produces picks.

## Module layout

- **`src/data/probable_pitchers.py`** (new) — `fetch_schedule` (thin, live),
  `parse_probable_starters` (pure), `STATSAPI_TO_STATCAST_TEAM`,
  `to_statcast_team`. Sibling to `pitcher_logs.py` / `prizepicks_lines.py`.
- **`src/features/predict_features.py`** (new) — `build_synthetic_game_rows(slate_df)`,
  `build_prediction_features(historical_game_logs, slate_df)`. Reuses
  `add_rolling_features` / `add_opponent_features` / `add_park_factors`
  **unmodified**. (Fallback: fold into `build_features.py` — flag in review.)
- **`src/models/baseline_model.py`** (extend) — add `save_model` / `load_model`
  (joblib) + artifact metadata. No change to fitting/predicting logic.
- **`src/pipeline/refresh.py`** (new) — `run_refresh(game_date, *fetchers,
  out_dir, overwrite)` orchestrator; pure helpers `assemble_predictions(model,
  feature_rows, slate)` and `write_outputs(results, out_dir, overwrite)`;
  `main()` CLI (`python -m src.pipeline.refresh --date 2026-06-27`). Reads
  `configs/config.yaml` for `processed_dir`, `league_id`, active prop/thresholds.
- **Tests:** `tests/test_probable_pitchers.py`, `tests/test_predict_features.py`,
  `tests/test_refresh.py` (+ a persistence test in `tests/test_baseline_model.py`).

No new third-party dependency beyond an explicit `joblib` pin (transitively
present via scikit-learn). `MLB-StatsAPI` is already pinned.

## Deferred (conscious scope cuts)

- **Non-strikeout props** (pitching outs / earned runs / walks) — the slate/
  feature/output machinery is built prop-agnostic, but only strikeouts is wired
  and validated now (decision log, 2026-06-27).
- **League-wide team-batting history** for fuller opponent/park samples — v1 uses
  the slate pitchers' own pulls; a dedicated batter/team ingestion is the v2
  upgrade (also noted in the feature-engineering spec).
- **Scheduled execution** — v1 is run-manually-each-morning (a script, per the
  README and the project's free/low-infra ethos). OS `cron` / a GitHub Actions
  scheduled workflow is a trivial later wrapper around the same `main()`; no
  scheduler is designed now.
- **Automatic retraining / staleness auto-resolution** — v1 loads a fixed
  artifact and only *warns* on age; a retrain trigger is v2.
- **Intra-day scratch/lineup changes** — a scratched probable or a confirmed
  lineup arriving after the run isn't tracked live; re-running picks up the new
  state. Confirmed-lineup L/R weighting is already deferred upstream (model spec).
- **Parquet/typed processed outputs** — CSV now; revisit with a `pyarrow` pin if
  volume warrants.

## Testing approach — what the new `tests/test_*.py` must assert

Deterministic, **no-network** unit tests on small hand-built fixtures, matching
the existing style (inject fakes; a fake model exposing `predict_mean` /
`family` / `alpha`, like `tests/test_tiering.py` uses plain `(family, mu, alpha)`):

1. **Slate parsing:** a hand-built StatsAPI-shaped payload yields one row per
   probable starter with the right `pitcher` (MLBAM id), `pitcher_name`,
   `home_away`, `game_pk`, and **Statcast-coded** `pitcher_team`/`opponent_team`;
   a game with only one side posted yields one row; none posted → empty frame.
2. **Team crosswalk:** known StatsAPI mismatches map to the right Statcast code;
   an unknown code passes through unchanged (no raise).
3. **Synthetic row shape:** `build_synthetic_game_rows` sets known pre-game fields
   and leaves **every leakage/outcome column `NaN`**; the row carries the real
   `game_pk`, today's `game_date`, and `pitcher_throws`.
4. **As-of-today equals leakage-free history (the core correctness test):** for a
   hand-built history, the synthetic today row's `k_rate_last5` /
   `opponent_k_rate_last10` / `park_k_factor` equal the values the existing
   builders produce for a *real* row dated today over the same history — and
   **appending the synthetic row does not change any historical row's features**
   (the same invariant `tests/test_rolling_features.py` checks for added games).
5. **No-history pitcher is skipped, not crashed:** a slate pitcher with empty/too-
   thin history is absent from predictions and present in `skipped_pitchers` with
   the right reason; the rest of the slate still predicts.
6. **Predictions contract:** `assemble_predictions` emits exactly the columns
   `tiering` consumes (`pitcher`, `pitcher_name`, `pitcher_team`, `opponent_team`,
   `game_date`, `family`, `mu`, `alpha`) plus `game_pk`, with `mu` finite and
   `family`/`alpha` consistent with the (fake) model.
7. **Partial-failure degradation:** with the lines fetcher raising, `run_refresh`
   still writes a full `threshold_table` and `predictions.csv`, an **empty**
   `line_picks.csv`, and a manifest with `prizepicks_error` set — no exception
   escapes. With the **schedule** fetcher raising/empty, the run fails fast with a
   clear error (the one fatal dependency).
8. **Model persistence round-trips:** `save_model` then `load_model` returns a
   model whose `predict_mean` / threshold probabilities match the original on a
   fixed design matrix, and metadata (`trained_at`, `train_through_date`) survives.
9. **Idempotent write:** writing a date twice overwrites (no duplicated rows);
   `--no-overwrite` aborts when the partition exists; the partition path is the
   `game_date=YYYY-MM-DD` Hive form and every table carries `(pitcher, game_pk)`.
10. **Empty slate:** zero probable pitchers yields well-formed empty outputs + a
    manifest noting the empty slate, without raising.
11. **Diagnostics surface coverage:** `skipped_pitchers`, `unmatched_lines`, and
    `predicted_no_line` are all populated from a fixture exercising each path, so
    coverage is visible from the manifest, not inferred from row counts.
