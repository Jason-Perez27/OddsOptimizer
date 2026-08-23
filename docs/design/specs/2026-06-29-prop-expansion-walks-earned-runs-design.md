# Prop Expansion — Walks Allowed + Earned Runs Allowed — Design Spec

**Date:** 2026-06-29
**Status:** Approved — ready for implementation (Sonnet)
**Related:** `2026-06-27` line-source/output-reshape entry (props built prop-agnostic),
task #7 baseline model, task #9 refresh. Builds on the validated strikeout pipeline
(Track-A ECE 0.006 = the approach works). New: a prop registry + per-prop label/feature
wiring; per-prop models.

## Goal

Add **Walks Allowed** and **Earned Runs Allowed** as second/third props end-to-end,
reusing the count-model + threshold-sweep + tiering + refresh + dashboard machinery.
Both PrizePicks stat-types are confirmed live (`Walks Allowed`, `Earned Runs Allowed`;
also `Pitching Outs`, deferred — it's capped by manager pulls, not a clean count
process). Do **walks first** (clean), **earned runs second** (new label source).

## The one real divergence: label source

- **Walks** are Statcast-native: a per-pitcher-game walk count is just
  `count(events == "walk")` — the exact same aggregation path as strikeouts in
  `game_logs.aggregate_pitcher_games`. No new data provider.
- **Earned runs are NOT cleanly in Statcast.** Earned-vs-unearned is an official
  scoring decision Statcast pitch data doesn't carry. The label must come from a
  **boxscore source** — MLB-StatsAPI pitching boxscore (`earnedRuns` per pitcher per
  game) or a pybaseball game-log pull. This is the key design decision: ER reuses
  everything *except* the label aggregation, which needs a new
  `get_pitcher_earned_runs_by_game` ingestion path keyed to the same
  `(pitcher, game_pk)`. (Total `Runs Allowed` would be Statcast-derivable, but
  PrizePicks posts *Earned* Runs, so we match the market.)

## Design — a prop registry, not copy-paste

Introduce `src/props.py` with a `PROP_REGISTRY` mapping a prop key → its config, so
the pipeline is parameterized by prop instead of K-hardcoded:

```
Prop(
  key="walks", prizepicks_stat_type="Walks Allowed",
  label_column="walks", label_source="statcast",   # event aggregation
  statcast_event="walk",
  rate_feature_prefix="bb_rate",                    # bb_rate_last5, bb_rate_season, ...
  thresholds=range(0, 6),                           # over 0.5..4.5 -> 1+..5+ (low counts)
  count_family="poisson",                           # escalate to NB on the same dispersion test
)
Prop(key="strikeouts", ... thresholds=range(1, 11), ...)   # existing, retrofitted
Prop(key="earned_runs", prizepicks_stat_type="Earned Runs Allowed",
     label_source="statsapi_boxscore", label_column="earned_runs",
     rate_feature_prefix="er", thresholds=range(0, 6), count_family="poisson")
```

What each consumer reads from the registry (most are already partly parameterized):
- `prizepicks_lines.flatten_projections(stat_type=...)` — already takes `stat_type`;
  pass the prop's value. Canonical-line / `odds_type` logic is prop-agnostic, no change.
- `game_logs.aggregate_pitcher_games` — generalize the K aggregation to also emit
  `walks` (Statcast). For ER, a separate boxscore label merged on `(pitcher, game_pk)`.
- `rolling_features` — the `k_rate_*` builders become a rate-family factory keyed on
  (numerator events, batters_faced) so `bb_rate_*` / `er_*` come out the same way;
  keep `k_rate_*` names for the strikeout prop (no churn to the validated model).
- `baseline_model` — train a **separate model per prop** (same Poisson→NB-on-evidence
  procedure, same leakage rules). Persist to `data/models/{prop}_model.joblib`.
  Regressor allowlist per prop lives in the registry (walks: bb-rate form + opponent
  walk tendency + park; ER: run-suppression form + opponent + park).
- `tiering.build_threshold_table` — already sweeps thresholds; feed the prop's range.
  `line_to_threshold` / `prob_over_line` are count-generic, no change.
- `refresh.run_refresh(prop="walks")` — loop/parameterize over props; write to
  `data/processed/predictions/game_date=*/prop={key}/` (extend the partition one
  level; strikeouts moves to `prop=strikeouts/` — update settle + serve + report
  globs accordingly, or default-prop to strikeouts for backward-compat).
- `settle` — grading is count-generic; the realized label uses the prop's label
  source (Statcast for walks, boxscore for ER). One settle pass per prop.
- `src/serve` (dashboard) — add a **prop selector**; `data.load_slate(game_date, prop)`
  reads the prop sub-partition. The ladder/line/stats UI is already generic.

## Phasing
1. **Refactor to the registry with strikeouts only** — prove the parameterization is
   behavior-preserving (existing tests still pass; K outputs identical). This is the
   risk-bearing step; do it first and keep it green.
2. **Walks end-to-end** — Statcast label, `bb_rate_*` features, walks model, sweep,
   line picks, settle, dashboard prop toggle.
3. **Earned runs** — add the boxscore label ingestion + ER model; everything else
   reuses step 2's plumbing.

## Tests
- Registry refactor: existing K tests unchanged and green (the contract guard).
- `game_logs` emits a correct `walks` count from a fixture with known walk events.
- ER label ingestion: a fixture boxscore → correct `earned_runs` per `(pitcher, game_pk)`;
  a missing/scratched start → no row (mirrors the void path).
- Per-prop model trains and the dispersion test still chooses a family on each label.
- `refresh(prop="walks")` writes a `prop=walks/` partition with the same file set;
  `settle` grades it against the Statcast walk label.
- Dashboard `load_slate(..., prop="walks")` returns walks ladders/lines.

## Verification
1. Full `pytest` green after each phase.
2. A live `refresh` for walks produces sane lines (walk lines are low — 1.5/2.5) and a
   sweep whose probabilities are monotonic decreasing in threshold.
3. ER label spot-check: a known recent start's `earned_runs` from the boxscore source
   matches the box score.

## Out of scope
- **Pitching Outs** — capped by manager pull decisions on top of the rate; needs a
  pull/leverage model. Deferred until walks + ER validate the multi-prop machinery.
- Hits Allowed / hitter props — different label and feature family; later.
- Tier redefinition (still gated on settled-sample evidence, now per prop).

## Decision-log entry to add (newest at top)
> **2026-06-29 — Prop expansion to Walks Allowed + Earned Runs Allowed.** Introduced
> a `PROP_REGISTRY` so the pipeline is parameterized by prop rather than K-hardcoded;
> refactored strikeouts onto it behavior-preserving first, then added walks
> (Statcast-native walk-count label, `bb_rate_*` features) and earned runs. ER's label
> is **not** Statcast-derivable (earned/unearned is an official scoring call), so it
> uses a new MLB-StatsAPI boxscore `earnedRuns` ingestion keyed on `(pitcher, game_pk)`
> — the one divergence from the K/walks path. Per-prop models persisted separately;
> outputs gain a `prop={key}/` partition level; the dashboard gains a prop selector.
> Pitching Outs stays deferred (manager-pull cap, not a clean count process).
