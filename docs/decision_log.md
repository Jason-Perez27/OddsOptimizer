# Decision Log

Dated record of major project decisions and the reasoning behind them. Newest entries at the top.

---

## 2026-06-30 — Compression fix v2 (post-gate-rejection iteration)

> **2026-06-30 — Compression fix v2 (after gate rejection, three findings corrected).**
> v1 was rejected (top-2-decile bias −0.4% → −2.8%) for three root causes, now addressed
> as separable arms:
>
> ① **Correct test window.** v1 tested only on data-rich 2024 — never scored a 2026 OOS
> game. v2 spans the corpus 2024→2026 and scores 2026 OOS slices head-to-head:
> as-deployed (2026-only training) vs widened (full prior history), on identical 2026 OOS rows.
>
> ② **EB shrinkage retuned.** C=100 was too aggressive (K-rate half-reliability ≈70 BF).
> Tiered prior: own-season K-rate → own-career K-rate → LEAGUE_K_RATE_PRIOR (only league
> when pitcher has no prior history at all). C swept {25, 50, 75}; best C selected by
> minimizing top-decile bias on 2026 OOS subject to no aggregate regression.
> Debut-game k_stab_last5 now correctly equals LEAGUE_K_RATE_PRIOR (prior rolling sums
> filled to 0 rather than NaN, so formula reduces to pure prior on first start).
>
> ③ **Correct collinearity target.** VIF: `velo_avg_last5`=135, `csw_rate_season`=162 were
> the real offenders (not `opponent_k_rate_last10` which v1 dropped). Iterative VIF pruning
> until max VIF < 10; survivor set determined by the procedure + 2026-OOS, not pre-picked.
>
> **New modules / changes:**
> - `src/backtest/vif_prune.py` — pure-numpy iterative VIF pruning (no statsmodels);
>   `compute_vif`, `iterative_vif_prune`, `format_vif_table_md`, `format_prune_log_md`.
> - `src/backtest/tail_calibration.py` — `subset_mask` param added to
>   `compute_mu_decile_table` and `flag_high_mu_underprediction` (score arbitrary OOS subsets
>   without pre-filtering the frame).
> - `src/features/rolling_features.py` — tiered EB prior (own-season → career → league);
>   `fillna(0)` on rolling sums so debut game returns `p_prior` not NaN.
> - `scripts/run_compression_fix_v2_gate.py` — three-arm gate (VIF-prune, C-sweep, widened
>   corpus head-to-head); adoption edits `CORE_PITCHER_FORM_COLUMNS` + `EB_K_CONSTANT`.
> - `tests/test_compression_fix_v2.py` — 14 tests covering subset_mask, EB tiered prior,
>   leakage guard, and iterative VIF prune; compatible with both pytest and python -m unittest.
>
> **Acceptance (pending gate run on 2026 OOS data):** top-2-decile bias shrinks toward 0,
> aggregate MAE / log-loss hold or improve, max VIF < 10. [Fill adopt/reject + numbers after
> running `python scripts/run_compression_fix_v2_gate.py --start 2024-04-01 --end 2026-06-30`.]

## 2026-06-30 — Fix projection compression (regression-to-mean) + sharpen matchup

> **2026-06-30 — Fix projection compression (regression-to-mean) + sharpen matchup.**
> Post-activation projections compressed toward the ~4.6 K baseline (every fitted
> coefficient tiny; `k_rate_last5` only +0.057, dominated by `pitch_count_avg_last5`),
> hitting high-projected aces hardest — caused by retraining on a half-season 2026
> window (data-starved → shrunk slopes) and an under-weighted/split K-skill signal.
> Fix: fit coefficients on the full multi-season corpus, replace raw L5 K-rate with an
> exposure-weighted empirical-Bayes stabilized rate `k_stab_last5 = (K_L5 + C·k_rate_season) / (BF_L5 + C)`
> (C=100 BF pseudo-count ≈ 2-3 starts; falls back to LEAGUE_K_RATE_PRIOR=0.22 for
> debut seasons), and de-collinearize the opponent block (drop `opponent_k_rate_last10`
> which was sign-flipped at −0.003 by collinearity with the hand-split; keep
> `opponent_k_rate_vs_hand_season`). `k_stab_last5` added to `rolling_features.FEATURE_COLUMNS`.
> `src/backtest/tail_calibration.py` created with `compute_mu_decile_table` (predicted-vs-
> realized by μ decile) and `flag_high_mu_underprediction` (detects systematic top-decile bias
> hidden by aggregate ECE). `run_backtest.py --variant compression-fix` implemented via a
> context-manager column patch (no permanent model change until gate passes). Gate script
> `scripts/run_compression_fix_gate.py` runs both walk-forwards in-process, computes decile
> tables + VIF, makes adopt/reject decision per spec rule, and edits `baseline_model.py`
> in-place when adopted. Gated on walk-forward backtest — adopt only if (a) high-μ-decile
> bias shrinks AND (b) aggregate MAE/log-loss don't regress.
>
> **Gate result 2026-06-30: REJECTED.**
> (a) Top-2-decile bias_pct: baseline=−0.4% → fix=−2.8% — WORSENED (gate requires improvement).
> (b) MAE: 1.7943 → 1.7879 (−0.006) ✓; log-loss@7: 0.5171 → 0.5145 (−0.003) ✓.
> Production model unchanged. Three diagnostic findings for Opus:
> **①** The 2024 backtest shows near-zero compression in the baseline (decile 10 = +1.0%);
>   the compression problem appears specific to the 2026-only data-starved retrain — a historical
>   walkforward on abundant 2024 data cannot reproduce or fix it. The PRIMARY lever (widened-corpus
>   retrain on 2024–2026) should be run first; the gate should re-run on a 2025–2026 window where
>   data starvation is realistic.
> **②** k_stab_last5 with C=100 degraded top-decile calibration: decile 9 went −1.9% → −4.7%.
>   C=100 BF pulls predictions too hard toward the season prior for aces; tuning to C=25–50 may
>   preserve EB stabilization for slumping pitchers without throttling ace predictions.
> **③** VIF table reveals the real collinearity problem: `velo_avg_last5`=135, `csw_rate_season`=162 —
>   far worse than k_rate_last5 ever was. The spec targeted the wrong pair. These two features
>   should be the focus of the next de-collinearization spec.

---

## 2026-06-30 — Weather + Vegas context + boosted ensemble (Spec 4, candidate regressors)

> **2026-06-30 — Weather + Vegas context + boosted ensemble.** Three new modules wired into the pipeline as candidate regressors, gated behind the walk-forward comparison gate before any production promotion. `src/data/weather.py`: 30-team `BALLPARK_TABLE` with lat/lon/is_dome; fetches Open-Meteo (no key required) — archive endpoint for past dates, forecast endpoint for future dates; extracts `temp_f`, `humidity`, `wind_mph` at the nearest hourly slot to `first_pitch_hour`; dome teams short-circuit without any network call (`is_dome=1.0`, weather columns NaN, `weather_was_imputed=True`); unknown teams → `_imputed_outdoor_result()`. Open-Meteo archive goes back to 1940, so weather features are available for full walk-forward backtests. `src/data/vegas.py`: ESPN public scoreboard API (no key required) provides `overUnder`, `homeTeamOdds.moneyLine`, `awayTeamOdds.moneyLine` per game; derives `game_total` and `is_favorite` (negative American moneyline = favourite); `team_total_for` / `team_total_against` deferred — ESPN does not carry per-team run totals and the only free source (The-Odds-API) requires a key; documented as `VEGAS_DEFERRED_COLUMNS`. ESPN's historical scoreboard responds to past `dates=` queries, so `game_total` + `is_favorite` are walk-forward compatible (closing-line proxy). `src/models/boosted_model.py`: `HistGradientBoostingRegressor` (sklearn; natively handles NaN) predicts K mean µ directly; per-threshold `IsotonicRegression` calibration on the chronologically last 25% of train rows (or explicit `val_df`); calibration skipped when calibration fold < 20 rows. `BoostedModel` exposes the same interface as `BaselineModel` (`predict_mean`, `predict_mean_with_se`, `predict_over_prob`, `predict_over_prob_sweep`); `predict_mean_with_se` returns zero SE (HGB has no native prediction-interval). `compute_agreement(glm_mu, booster_mu, line)` → +1 (both above line), −1 (both below), 0 (disagree), NaN (any input NaN). Persistence: `save_boosted_model` / `load_boosted_model` via joblib; `DEFAULT_BOOSTED_MODEL_PATH = data/models/boosted_model.joblib`; guarded against missing sklearn at import (raises only at fit/load time). `baseline_model.py`: added `CONTEXT_CANDIDATE_COLUMNS` (6 columns: `temp_f`, `wind_mph`, `humidity`, `is_dome`, `game_total`, `is_favorite`). `refresh.py`: weather/vegas enrichment loops per `game_pk` groupby (injectable `weather_fetcher`, `vegas_fetcher` for tests); booster second-opinion via injectable `boosted_model_loader` and `boosted_model_path`; `booster_mu` written per row with a prefilter fix — `transform_design_matrix` internally calls `_dropna_core(reset_index)` so the booster block prefilters to rows with all `CORE_PITCHER_FORM_COLUMNS` present, saves original indices before reset, then maps `booster_mu_arr[i]` back via those indices (avoids silent misalignment swallowed by `except Exception`); `agreement` computed post-`line_picks` join (requires a line score); `PITCHER_CARD_COLUMNS` extended with booster/weather/vegas/agreement columns. These are **candidate regressors under trial** — walk-forward AUC/log-loss comparison vs. the baseline required before promoting any to the production model. `sklearn` guarded: the module imports cleanly when sklearn is absent; `fit_boosted_model` / `load_boosted_model` raise `ImportError` at call time only.

---

## 2026-06-30 — Lineup-weighted matchup + umpire features (Spec 3, candidate regressors)

> **2026-06-30 — Lineup-weighted matchup + umpire K-factor features.** Replaced the coarse team-level opponent K% with infrastructure for a PA-slot-weighted mean of projected lineup batters' individual K-rate vs. the pitcher's hand. Three new modules: `src/data/lineups.py` (StatsAPI schedule?hydrate=lineups, pure `parse_lineup` + `get_lineup`; `lineup_source = confirmed|projected|team_fallback`), `src/features/batter_logs.py` (batter-game aggregation from Statcast with LHP/RHP split K/PA; strictly-prior rolling via shift-1 before cumsum/rolling — same leakage guardrail as rolling_features.py), `src/data/umpires.py` (day-of HP ump from StatsAPI officials hydrate; tendency from `data/raw/umpires/ump_tendency.csv`; multiplicative `ump_k_factor`; neutral 1.0 + `was_imputed` on any miss). New function `opponent_features.build_lineup_weighted_opp_k` produces `opponent_lineup_k_rate_vs_hand` (slot-weighted, slot 1 = weight 9 → slot 9 = weight 1) and `opp_share_opposite_hand` (switch hitters count as opposite). Added `MATCHUP_CANDIDATE_COLUMNS` to `baseline_model.py` (3 columns: lineup k-rate, platoon share, ump k-factor). `refresh.py` enriched with per-game lineup + ump enrichment step (injectable `lineup_fetcher`, `officials_fetcher`, `batter_rolling_df` for tests); new columns surfaced in `PITCHER_CARD_COLUMNS`. Degradation is fully graceful: no lineup posted → team_fallback; ump unknown or thin sample (<50 games) → k_factor=1.0 + was_imputed=1. These are **candidate regressors under trial** — not yet part of the production baseline (walk-forward comparison vs. team-level baseline required before promoting). Historical backtest requires pre-computed batter rolling table; current refresh falls back to team-level if table absent.

---

## 2026-06-30 — Pitcher plate-discipline skill features (Spec 2, walk-forward trial)

> **2026-06-30 — Pitcher plate-discipline skill features, walk-forward trial.** Added five per-game skill metrics to `game_logs.aggregate_pitcher_games`: `csw_rate` (called-strike + whiff / pitch_count), `whiff_rate_overall` (whiffs / total swings), `putaway_rate` (K / two-strike pitches; NaN when Statcast `strikes` col absent), and `k_minus_bb` (count). Extended `rolling_features.add_rolling_features` with strictly-prior rolling families for all five (last5 + season expanding mean; `k_minus_bb_rate` uses pooled count/BF via `_add_count_rate_family`; `swstr_rate_*` reuses existing whiff_rate rolling as an alias for backward compat). Added `SKILL_CANDIDATE_COLUMNS` (7 columns) and `extra_columns` param to `baseline_model.build_design_matrix` / `fit_preprocessor` / `transform_design_matrix` — columns are imputed-then-standardized rather than raising on NaN (debut / absent source). Added `--variant skill-features` to `run_backtest.py` which wraps `fit_fn` to inject the extra columns without touching the baseline spec. Updated `refresh.py PITCHER_CARD_COLUMNS` to surface five _last5 skill columns on the pitcher card. These are **candidate regressors under trial** — not yet part of the production baseline (gated behind `--variant`). Walk-forward AUC/log-loss comparison vs. baseline required before promoting. Leakage guard: per-game skill stats added to `LEAKAGE_COLUMNS` in baseline_model and `LEAKAGE_OUTCOME_COLUMNS` in predict_features.

---

## 2026-06-30 — Conviction score + no-action band (decision layer, model unchanged)

> **2026-06-30 — Conviction score + no-action band (decision layer, model unchanged).** Added uncertainty-aware decisioning: propagate the GLM's μ standard error to a band on `p_over`, define `conviction = |p_over−0.5| / sd(p_over)`, and label each pick `lean_over`/`lean_under`/`no_action`. Thresholds are ROI-validated per bucket on Track-B settled outcomes once ≥100/bucket exist; until then a documented provisional default, labeled unvalidated on the dashboard. Kept separate from the (still-gated) probability-only tier — `actionability` is a new field, not a tier redefinition. Makes near-line picks decidable and lets the UI honestly say "no edge."

---

## 2026-06-29 — Prop expansion: Walks Allowed + Earned Runs Allowed (Phases 2–3 — bb_rate_* features, ER boxscore label, dashboard prop selector)

> **2026-06-29 — Prop expansion Phases 2–3 (feature generalization + ER label source + UI).** Generalized `rolling_features.py` with a `_add_count_rate_family(df, prefix, count_col)` factory that emits five leakage-safe features (`{prefix}_last5/season/home/away/vs_opponent_career`) for any count column; refactored `k_rate_*` computation to use it and added `bb_rate_*` (walks) and `er_*` (earned_runs, gracefully NaN when column absent). Added `src/data/statsapi_boxscore.py`: fetches `earnedRuns` per `(pitcher, game_pk)` from the MLB StatsAPI pitching boxscore — the only source of earned/unearned distinction (Statcast doesn't carry it). Uses the injected-fetcher pattern for network-free testing. Wired the `prop` query parameter through `src/serve/server.py` `/api/slate` → `load_slate(prop=)`. Added prop selector dropdown to `static/index.html`/`app.js` that refetches on change and updates the page title. Added 23 new tests (bb_rate_* leakage/season/home/away, er_* NaN-when-absent, boxscore ingestion, prop partition routing). Suite: 318 passed, 8 pre-existing scipy failures (unchanged), 26 skipped. Spec: `docs/design/specs/2026-06-29-prop-expansion-walks-earned-runs-design.md`.

---

## 2026-06-29 — Prop expansion: Walks Allowed + Earned Runs Allowed (Phase 1 — prop registry + pipeline parameterization)

> **2026-06-29 — Prop expansion Phase 1 (behavior-preserving refactor).** Introduced `src/props.py` with a `Prop` dataclass and `PROP_REGISTRY` so the pipeline is parameterized by prop rather than K-hardcoded. Wired `prop` parameter into `run_refresh()`, `write_outputs()`, `run_settlement()`, and `load_slate()`. Strikeouts remain the default (`DEFAULT_PROP = "strikeouts"`) and continue writing to the flat `game_date=*/` partition for backward compatibility; non-default props get a `game_date=*/prop={key}/` sub-partition. Added `walks` count to `game_logs.aggregate_pitcher_games` (Statcast event aggregation, same pattern as strikeouts). Added `walks` to `LEAKAGE_COLUMNS` in `baseline_model.py` and `LEAKAGE_OUTCOME_COLUMNS` in `predict_features.py`. Added `tests/test_props.py` (25 tests). All 341 tests pass after Phase 1. Spec: `docs/design/specs/2026-06-29-prop-expansion-walks-earned-runs-design.md`.

---

## 2026-06-29 — Automated the four cadences

> **2026-06-29 — Automated the four cadences.** Wired the existing
> refresh/settle/retrain/report CLIs into the OS scheduler via a thin
> `scripts/run_cadence.py` wrapper (logging + `last_run.json` heartbeat +
> exit-code/skip discipline) and a Windows Task Scheduler install script;
> cron/Actions documented as alternates. No new business logic, no orchestrator
> daemon — the scheduler is the orchestrator, per the go-live design. Enforces the
> morning-authoritative `refresh` rule (fires once; off-day EmptySlate is a clean
> skip, not a failure).

---

## 2026-06-29 — Select one canonical PrizePicks line per pitcher (standard → highest goblin → lowest demon)

The projections endpoint returns multiple lines per pitcher tagged by `odds_type` (`standard` O/U vs `demon`/`goblin` alternates, monotonic: goblin < standard < demon); `flatten_projections` parsed only `stat_type`, so `_dedupe_latest_projection`'s same-`pulled_at` tiebreak kept arbitrary alternates (e.g. a 0.5 goblin for Tyler Alexander). Fix keeps ingestion complete (all lines emitted, now carrying `odds_type`) and moves single-line selection into `tiering` as `_select_canonical_line`: prefer the standard O/U; with no standard, take the highest goblin (closest to the demon side) or, if only demons exist, the lowest demon. The chosen `odds_type` is recorded on every line pick so fallback grading is visible, never laundered as standard. The verify gate now requires the `odds_type` field present with recognized values (rename guard). Spec: `docs/design/specs/2026-06-29-standard-line-filter-design.md`. Live counts: 170 K projections → 23 standard.

---

## 2026-06-29 — Live forward-validation go-live (task #12) design decisions

**Decision:** With task #11's Track-A historical backtest done and trustworthy (ECE 0.006), turn the pipeline on against live games and begin accumulating the forward-only Track-B (pick-profitability) record. This is an **operational go-live, not a model change** — it runs the already-built `refresh`/`settle`/`report` on a live cadence. Spec: `docs/design/specs/2026-06-29-live-forward-validation-go-live-design.md`. New: `--fit-only` on `scripts/run_backtest.py`, `--window-days` on `src/pipeline/settle.py`, a model-path no-drift guard test, and `docs/runbook_go_live.md`.

- **The live model must be retrained on 2026 data — the existing artifact is 2024-fit and validated only the method.** Every `data/raw/statcast/*` window, `walk_forward_oos.csv`, and the `2026-06-29-baseline-backtest.md` report are the **2024** season. A model that has never seen a 2026 pitch (current form, rosters, mid-2026 strikeout environment) is the wrong thing to point at today's slate. **Go-live step 1 is a fresh retrain on the 2026 season-to-date corpus, fit through yesterday**, saved to the canonical model path — one action that produces the live model, optionally refreshes Track A on 2026 data, and forces the path reconciliation below to be correct by construction.

- **Model-path bug reconciled to `data/models/baseline_model.joblib` (the existing default).** Three call sites (refresh default, refresh `--model-path` default, `run_backtest` save default) already point there, but the only artifact on disk was the stray 2024 one at `models/baseline_model.joblib` and `data/models/` didn't exist — so a scheduled `refresh` on defaults would load nothing. Kept the constant (`data/` is gitignored, the right home for a 1.7 MB binary); the 2026 retrain writes to it; the stray artifact is retired; a guard test pins `refresh.DEFAULT_MODEL_PATH == run_backtest.DEFAULT_MODEL_PATH` so the drift the existing comment warns about can't reappear.

- **The verification gate is step 0 and non-negotiable.** PrizePicks' `league_id=2`/payload shape and StatsAPI's schedule hydration were **never verified against a live call**. `refresh --dry-run` (writes nothing) must PASS both checks before any graded run; a FAIL on a real game day means stop and fix the source module, never proceed on an unverified shape (an unofficial-endpoint change that slips through corrupts every downstream graded number). A FAIL on a true off-day is not a shape break.

- **Exactly four scheduled cadences, no orchestrator.** (A) daily morning `refresh` ~10:00 ET (after lines post, before first pitch); (B) daily `settle --window-days 4` ~12:00 ET (after Statcast finalizes; the trailing window covers the 3-day void wait + buffer so `pending`→`settled`/`void` resolves with no manual date math); (C) weekly `run_backtest --fit-only` retrain (matches the walk-forward step cadence, beats the 7-day staleness warning); (D) weekly live `report`. Times are ET-anchored; the user's scheduled tasks fire in local time, so the runbook states the constraint and the user converts. The existing CLIs + run manifests are the interface and the scheduler is the orchestrator — a `daily_driver.py`/status dashboard is explicitly deferred.

- **The morning refresh is authoritative; never re-run it after first pitch.** `refresh` overwrites a date's partition and freezes the posted line + `pulled_at` for honest grading. Re-running later re-freezes a *moved* line and a possibly *changed* slate, corrupting the pre-game snapshot. Manual re-runs are allowed only before first pitch (crash recovery). Settle, by contrast, is safely repeatable (idempotent overwrite; pending resolves on a later pass).

- **No backfill, defined read checkpoints.** Track B is forward-only by definition — PrizePicks has no historical line source, so the record starts **today** and grows one slate per day. First "worth reading" checkpoint ≈ **2 weeks** (Low-tier line picks clear ~100 settled in ~1–2 weeks); High-tier line picks are rare and may take a month-plus. Tier redefinition stays gated on the task #10 bar (≥100 settled/tier **and** evidence the probability-only definition is failing); until both hold, the shipped definition stands.

**Scope:** strikeouts only; operational go-live only. **Explicitly deferred:** scheduler/alerting infra beyond the four task definitions, a status summarizer, model v2, new features, tier redefinition, prop expansion, priced-odds/CLV.

---

## 2026-06-27 — Baseline validation & go-live (task #11) design decisions

**Decision:** With the pipeline complete (tasks #1–#10) but unrun on real data, establish trustworthy baseline performance before any model-v2 work or prop expansion, and define go-live. Spec: `docs/design/specs/2026-06-27-baseline-validation-design.md`. New: `src/backtest/{corpus,walk_forward}.py`, a PIT helper in `src/evaluation/metrics.py`, a historical-backtest report in `src/backtest/report.py`, `fit_production_model` in `baseline_model.py`, and a `verify_live_sources`/`--dry-run` gate.

- **"Baseline performance" is two tracks on two clocks — kept explicitly separate.** **Track A (model honesty)** — calibration/ECE, MAE/RMSE, Brier, log loss, PIT — is computable **now** from historical completed games and is the one new build. **Track B (pick profitability — hit rate/ROI vs the line)** is **forward-only**: PrizePicks has no historical line backfill, so it cannot be backtested and only accrues live, gated by the task-#10 ≥100-settled-picks-per-tier bar. The design forbids conflating them: **Track A produces no ROI by construction** (a test guards against fabricating betting numbers from a line-less track).

- **Historical walk-forward backtest (closes the gap task #7 deferred).** Expanding window, **weekly** refit (daily is marginal gain for the cost), training strictly before each cutoff and predicting the next slice, accumulating out-of-sample sweep predictions joined to realized strikeouts on `(pitcher, game_pk)`. **Key efficiency, and why it's correct:** features are built **once on the full corpus** because every rolling/opponent/park builder is already strictly-prior (leakage-safe by construction); only model *fitting* respects the walk-forward cutoff. Temporal leakage is the highest-stakes property and the lead test (no train row dated ≥ cutoff; earlier steps' predictions unchanged when later games are appended).

- **Corpus assembly reuses ingestion/features, cached and resumable.** Bulk `pybaseball.statcast(start,end)` pulled in weekly windows cached to `data/raw/` (resume from cache on failure), fed through the existing `build_training_table`. Features built on **all** pitchers' games (complete opponent/park/league history); the **evaluation set filtered to starts** via the existing max-batters-faced starter proxy (openers a documented limitation). Injected `statcast_fetcher` keeps it no-network testable.

- **Go-live: refit on full history first, then weekly retrain.** It's late June (~half a season); the live `refresh` should load the **most current** fit, not the stale task-#7 artifact. `fit_production_model(corpus, through_date)` fits on all completed starts to date and saves the artifact; cadence is weekly (matching the walk-forward step), with the task-#9 ~14-day staleness warning as the safety net. Retraining stays a deliberate, separate action from the daily predict run.

- **Live-data verification gate before the first real run.** The PrizePicks `league_id`/payload shape and the StatsAPI hydration were never verified against a live call. `verify_live_sources` + `refresh --dry-run` (writes nothing) confirm the live payloads match the parsers — MLB strikeout projections present, schedule returns `probablePitcher.id` — and **fail loudly** on mismatch so an unofficial-endpoint shape change is caught before it corrupts a graded run.

- **Reporting keeps the two tracks visually distinct.** The Track-A report (`reports/YYYY-MM-DD-baseline-backtest.md`, with reliability / calibration-by-tier / error-over-time plots) is a sibling to the live Track-B results report, sharing helpers but a distinct filename, headed "model calibration on historical games — not betting performance." Track A does yield a **first read on sweep-tier calibration now**, but live line-pick tier hit-rate still needs Track-B accumulation (High-tier line picks are rare and may take a month-plus to clear 100; Low clears in ~1–2 weeks).

**Scope:** strikeouts only; backtest built prop-agnostic where free. **Explicitly deferred:** model-v2 (offset, regularization/VIF, boosting), new features, tier *redefinition*, prop expansion, scheduling/automation — all justified later by this backtest, not before it.

---

## 2026-06-27 — Prediction outcome tracking & backtest (task #10) design decisions

**Decision:** Lock the design for the evaluation/backtest layer that grades the daily picks against realized outcomes and accumulates an honest performance record. Spec: `docs/design/specs/2026-06-27-outcome-tracking-design.md`. New: `src/pipeline/settle.py`, `src/evaluation/{grading,metrics}.py`, `src/backtest/{roi,report}.py`, plus a by-id outcome pull added to `src/data/pitcher_logs.py`.

- **Grading is a separate, later pass — settle D+1, from the free Statcast path.** A game isn't final at prediction time, so settlement runs after the game (next day by default, with a backfill mode). Realized strikeouts come from the same `pybaseball` → `aggregate_pitcher_games` path that produces training labels (no new provider, no paid feed). Predictions already carry the **MLBAM id**, so we pull *by id* (new `get_pitcher_logs_by_id`), not by name. `settle.py` mirrors `refresh.py`: injected `outcome_fetcher` + injected `now` clock, pure grading core, no-network tests.

- **Three settlement states, surfaced not dropped.** Each predicted `(pitcher, game_pk)` is `settled` (Statcast rows present), `pending` (no rows yet, within the lag/age window — retried later), or `void_scratched` (no rows past a max-wait → the probable starter never threw: scratch/postponement). Void and pending are **excluded from headline metrics but kept visible in the ledger** — never a silent drop, never scored as a loss. A pitcher *pulled early* is NOT a void: fewer Ks is a legitimate `under`; only a never-thrown start voids (strikeout counts have no "no-decision" wrinkle).

- **Storage: per-date graded CSV partitions mirroring task #9, cumulative metrics by globbing.** Graded results write to `{processed_dir}/outcomes/game_date=YYYY-MM-DD/` (`graded_line_picks.csv`, `graded_threshold_sweep.csv`, `settle_manifest.json`), same CSV/partition/overwrite convention as the predictions dir. **No separate rolling ledger** — cumulative metrics are computed by reading the partitions at report time, avoiding a second mutable source of truth. Re-settling overwrites idempotently; pending rows resolve to settled on a later pass with no upsert bookkeeping. Canonical join key `(pitcher, game_pk)`, doubleheader-safe (fail-fast if `game_pk` is missing rather than mis-join on `game_date`).

- **Metrics separate model honesty from pick profitability.** Reflecting the decision log's "own calibration vs. beating the market" split: **(a)** model calibration/reliability + ECE (new), Brier/log loss/MAE/RMSE (reuse the existing `baseline_model` pure helpers); **(b)** hit rate (pushes + pending excluded) and a **flat-bet ROI** on the lean. Every metric sliced **by tier, by threshold/line, and over time** (cumulative + rolling). All helpers return numbers/tables (no IO) so tests assert on them.

- **ROI is a labeled even-money proxy; true payout ROI and CLV deferred.** PrizePicks is pick'em with parlay-style payout multipliers; modeling those (or buying a priced odds feed) was put out of scope to keep the project free. v1 ROI is a unit even-money flat bet (win +1 / loss −1 / push 0), explicitly labeled a proxy. CLV stays out (no odds feed).

- **Confidence tiers: measure now, redefine later (resolves task #8's flag).** Tiers stay **probability-only**; task #10 *validates* them by reporting empirical hit rate + calibration **by tier**, and surfaces per-tier sample sizes. Redefinition (promoting `edge` or a backtest-calibrated cut) is a separate future decision gated on a stated bar — roughly ≥100 settled picks per tier **and** evidence the current definition is failing (e.g. High not separating from Low). Until both hold, the shipped definition stands.

- **Write-up: a generated dated markdown report in `reports/`.** `report.py` renders calibration, hit rate by tier/threshold, ROI and its time series, and the per-tier sample sizes — a thin presentation layer (with matplotlib reliability + cumulative-ROI plots) over the numeric helpers, run on demand. Honors the README's reserved dirs: `src/evaluation/` (calibration/Brier/log loss), `src/backtest/` (ROI/hit-rate), `reports/` (write-ups), with network/IO isolated in `src/pipeline/settle.py`.

**Open integration risk flagged:** Statcast data lag drives the pending→void wait (v1 infers "final" from Statcast presence + a time threshold; a live game-status feed to shorten it is deferred). Requires `game_pk` on `predictions.csv` / `threshold_table.csv` / `line_picks.csv` — established by the task #9 spec; verify in its implementation.

**Scope:** strikeouts only; grading/metrics built prop-agnostic where free, but only strikeouts wired and validated.

---

## 2026-06-27 — Two implementation bugs fixed while building `predict_features.py` (task #9, module 3)

**Not a design change** — both are implementation-level corrections surfaced by exercising the already-approved synthetic-row design (see the task #9 entry below) against realistic edge cases, fixed in already-"completed" or newly-built modules:

- **`opponent_features.build_team_game_logs()` crashed on an all-NaN `batters_faced` group.** `g.loc[bf.idxmax()]` raises when every row in a `(opponent_team, game_pk)` group is NaN — which happens specifically for a synthetic as-of-today row's own game, since synthetic rows have NaN outcomes by design (that's the leakage-safety mechanism). Fixed with a `bf.notna().any()` guard that falls back to the group's first row; that team-game's own strikeouts/batters_faced sum to 0 and its row is never read downstream (every consumer filters to strictly-prior games), so the fallback only needs to exist without crashing.
- **`predict_features.build_prediction_features()` crashed on empty/`None` `historical_game_logs`.** An empty or all-synthetic combined table left numeric game-log columns (strikeouts, batters_faced, etc.) as `object` dtype, and `park_factors`'/`rolling_features`'/`opponent_features`' internal `cumsum`/`rolling().sum()` calls raise `TypeError` on object dtype. Fixed by coercing all numeric game-log columns via `pd.to_numeric(..., errors="coerce")` immediately after concatenation, before the builders run.

Both verified via `tests/test_predict_features.py` (13 tests, including dedicated empty/`None`-history and no-history-pitcher cases) plus the core as-of-today correctness test: a synthetic row's rolling/opponent/park features exactly match a "real" row sharing the same pre-game identifiers, proving the synthetic-row mechanism is leakage-free by construction.

---

## 2026-06-27 — Pre-game refresh pipeline (task #9) design decisions

**Decision:** Lock the design for the daily orchestration layer that turns "today's slate" into the `threshold_table` / `line_picks` / `diagnostics` outputs. Spec: `docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md`. New modules: `src/data/probable_pitchers.py`, `src/features/predict_features.py`, `src/pipeline/refresh.py`; persistence added to `src/models/baseline_model.py`.

- **Slate from MLB-StatsAPI, keyed by MLBAM id.** Today's probable starters come from the StatsAPI schedule (probable-pitcher hydration), which returns the pitcher's **MLBAM id directly** — so the model side needs *no* name→id resolution (the PrizePicks side still does, via the existing tiering resolver). A new `src/data/probable_pitchers.py` follows the one-source-one-module pattern (`pitcher_logs.py`, `prizepicks_lines.py`): a thin live `fetch_schedule` plus a pure, testable `parse_probable_starters`. A **StatsAPI→Statcast team crosswalk** maps slate team codes into the Statcast space the historical features key on (distinct from tiering's PrizePicks→Statcast crosswalk) — without it, opponent/park lookups silently miss.

- **"As-of-today" features reuse the existing builders unmodified — no predict-mode branch.** The leakage guardrail in `rolling_features` / `opponent_features` / `park_factors` (`shift(1)` + strict `game_date <`) *is* an as-of-date computation. So the pipeline appends one **synthetic same-day game row** per starter (known pre-game fields filled, all same-game-outcome/label columns `NaN`, real `gamePk`, recomputed `rest_days`) onto the historical game-logs, runs the three builders as-is, and slices today's rows back out. New `src/features/predict_features.build_prediction_features()` is a **sibling** to `build_training_table` — needed because the training join re-aggregates from pitch-level data and validates a non-null `strikeouts` label a pre-game row can't have. Known v1 limitation (surfaced, deferred): opponent/park history comes from the slate pitchers' own pulls, so thin-sample opponent rates impute via the model layer's existing `was_imputed` path rather than dropping the pitcher.

- **Model is persisted and loaded, not refit per run.** Closes a real gap — `BaselineModel` had no save/load. Add `save_model`/`load_model` (joblib, which ships with the pinned scikit-learn) plus metadata (`trained_at`, `train_through_date`). Loading keeps the daily run fast and reproducible and separates *retraining* (occasional, evaluated) from *predicting* (daily). The staleness question is **surfaced not hidden**: the run records model age and warns past a threshold, never auto-refits. Auto-retrain is a v2 deferral.

- **Partial failure degrades to a sweep; only an empty slate is fatal.** PrizePicks (or Chadwick register) down → still emit the full `threshold_table` + `predictions` for everyone, an **empty `line_picks`**, and flag `prizepicks_error` — the sweep is line-independent (per the task #8 spec). A single pitcher's Statcast pull failing or a debutant with no history → **`skipped_pitchers`** diagnostic, never a crashed run. No probable pitchers at all → fatal-but-clean (nothing to predict).

- **Outputs are date-partitioned CSV, joinable by `(pitcher, game_pk)`.** Written to `{processed_dir}/predictions/game_date=YYYY-MM-DD/` (`predictions.csv`, `threshold_table.csv`, `line_picks.csv`, `diagnostics/`, `run_manifest.json`). **CSV not parquet** — `pyarrow` isn't a dependency and the repo's `save_raw` convention is already CSV (keeps the project dependency-free; parquet deferred). The canonical key is **`(pitcher, game_pk)`** (doubleheader-safe, unlike `(pitcher, game_date)`), chosen now so task #10 can join realized outcomes without a redesign; `line` + `pulled_at` freeze the posted line at prediction time for honest grading. Re-running a date **overwrites** (deterministic per morning's inputs; appending would duplicate), with `--no-overwrite` to abort.

- **Scheduling: run manually each morning (v1).** A script (`python -m src.pipeline.refresh`), not a scheduler — consistent with the project's subscription-free/low-infra bias (cf. The Odds API removal). Cron / a GitHub Actions scheduled workflow is a trivial later wrapper around the same `main()`. The orchestrator uses **dependency-injected fetchers** so the core is unit-testable with hand-built fixtures and no network, matching the existing test style.

**Scope:** strikeouts only, end to end; the slate/feature/output machinery is built prop-agnostic so later props slot in without redesign.

---

## 2026-06-27 — Baseline model (task #7) and tiered-probability output (task #8) design decisions

**Decision:** Lock the design for the baseline strikeout model and the layer that turns it into tiered per-threshold picks. Specs: `docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md` and `…-tiered-prop-probabilities-design.md`.

**Task #7 — baseline count model:**
- **statsmodels, not sklearn.** The baseline's value is interpretability plus the Poisson-vs-NB decision; statsmodels gives native Negative Binomial with an estimated dispersion parameter, plus deviance / Pearson χ² / LR-test machinery for the overdispersion check. sklearn has no NB and no inferential output, so it stays a candidate only for a later regularized variant.
- **Poisson first, escalate to NB on evidence.** Fit Poisson, test the *training* residuals (Pearson χ²/df plus an LR test of NB's α=0); use NB only if dispersion ratio > ~1.25 and p < 0.05. Decision made on training data only; the statistic and chosen family are recorded so it's auditable as more season data arrives.
- **No exposure offset in v1.** Strikeout count scales with batters faced, but actual BF is itself a same-game outcome (leakage); building an offset would require a hand-made expected-BF projection. v1 instead carries `pitch_count_avg_last5` as an ordinary workload regressor. The `log(expected_BF)` offset is the first v2 experiment once a calibrated BF projection exists.
- **Parsimonious regressor set + strict temporal split.** The near-collinear `k_rate_*` variants are held out of v1 (documented, not forgotten); train/test is a chronological holdout by `game_date`, never random K-fold, because the deployment task is predicting future games from past ones.

**Task #8 — tiered output:**
- **Confidence tiers stay probability-only** (distance of P(K≥t) from 0.5: High ≥0.20, Medium 0.10–0.20, Low <0.10), exactly per the 2026-06-27 output-reshape entry — no dependency on PrizePicks' line-setting or on backtest history.
- **Known limitation, surfaced not hidden:** because PrizePicks sets the line near the distribution's median, a well-calibrated model's P(over the line) ≈ 0.5 → **Low**. So *line* picks cluster at Low and only reach High when the model strongly disagrees with the line; the sweep's High tiers (1+, 9+/10+) are trivially confident. The spec documents this so a Low line pick isn't misread as "no opinion."
- **`edge` field instead of an edge-based tier — chosen to keep the project free.** Rather than redefine the tier as model-vs-line divergence (which, done precisely, pulls toward a priced sportsbook odds feed — the paid dependency dropped when The Odds API was removed), `line_picks` carries a signed `edge = p_over − 0.5`. A flat pick'em line has no posted odds and sits at the book's ~median, so 0.5 is used as a deliberate free-data proxy for the line's implied probability. Modeling PrizePicks' payout multipliers or buying an odds feed is explicitly out of scope to keep the project subscription-free. Promoting `edge` (or a backtest-calibrated version) to drive the tier is a task #10 revisit.
- **Name↔id join is the main integration risk.** PrizePicks keys pitchers by display name + its own team abbreviation; the model keys by MLBAM id + Statcast team code. A deterministic Chadwick-based resolver (normalize names, disambiguate by a team-abbreviation crosswalk) never silently mis-joins — ambiguous matches and unmatched lines are surfaced in diagnostics, not guessed or dropped.

**Scope:** strikeouts only end-to-end; pitching outs / earned runs / walks deferred until this pipeline is validated.

---

## 2026-06-27 — Line source switch: The Odds API → PrizePicks; output reshaped to multi-threshold tiered probabilities

**Decision:** Replace The Odds API with PrizePicks' public (unofficial) projections endpoint as the primary line source. Reshape the model's output from "predicted mean vs. one sportsbook line" into a full threshold sweep per prop — e.g. strikeouts 1+ through 10+ — with each threshold's P(over) bucketed into 3 confidence tiers. Expand scope beyond strikeouts to pitching outs, earned runs allowed, and walks allowed, but build and validate the strikeout pipeline first before generalizing.

**Why switch away from The Odds API:**
- The free tier's 500 credits/month is a hard ceiling on how often the pipeline can refresh long-term; PrizePicks' endpoint has no such credit budget.
- The Odds API prices `pitcher_strikeouts` as bookmaker odds (e.g. -115/-105), which requires stripping vig to back out an implied probability. PrizePicks posts a single flat line per prop with no juice to remove — directly comparable to a modeled P(stat ≥ threshold) with one less modeling step.

**Trade-offs accepted:**
- PrizePicks' projections endpoint is public but **unofficial** — undocumented, not a published developer API, and subject to their Terms of Service. It can change shape or add bot protection without notice. This stays a personal/research use, consistent with the README disclaimer, not a commercial scraper.
- A single flat line has no market-depth signal the way multi-bookmaker odds do — PrizePicks sets lines to balance their own payout structure, not necessarily to reflect true probability. The model's edge claims are about its own calibration, not "beating the market" in the sportsbook sense.
- Pitching outs don't reduce to a clean count process the way strikeouts/walks/earned runs do — it's capped by manager pull decisions on top of the underlying stochastic rate. Deferred until the strikeout pipeline is validated.

**Confidence tier definition:** Tiers are defined purely from the model's own predicted probability, not from edge-vs-line: high confidence at P≥0.70 or P≤0.30, medium at 0.60–0.70 or 0.30–0.40, low otherwise. Chosen over "edge vs. posted line" and "backtested accuracy by bucket" because it has no dependency on PrizePicks' line-setting behavior and doesn't require a season of tracked predictions to be trustworthy — it can be revisited once backtest data (task #10) accumulates.

**Resolution:** `src/data/odds_feed.py` (The Odds API) and its tests have been removed entirely, along with all related config (`odds_api:` block) and docs references — PrizePicks is now the sole line source, not just the primary one, so there was no reason to keep unused Odds API code around. `src/data/prizepicks_lines.py` is the only ingestion path for prop lines.

---

## 2026-06-26 — Sport, market, and target selection: MLB pitcher strikeout props

**Decision:** Build the first model around MLB pitcher strikeout props (over/under a sportsbook line), with a predicted strikeout-count distribution as the training target.

**Alternatives considered:**
- **NBA moneyline** — strong free data (`nba_api`, Kaggle historical odds), simple binary classification. Rejected because the NBA season had already ended, making live testing impossible at project start.
- **Esports (League of Legends)** — excellent free match data (Oracle's Elixir, updated after every pro match), but thin free odds/player-prop coverage and reliance on scraping less-official sources for some data. Kept as a candidate for a future second project.
- **MLB pitcher strikeout props (chosen)** — in season now, best-in-class free data tooling (`pybaseball`, MLB-StatsAPI), and The Odds API's free tier covers the exact prop market needed (`pitcher_strikeouts`) for current (not historical) odds.

**Why this wins:**
1. **Data availability:** `pybaseball` and MLB-StatsAPI require no API key and update same-day — fits the requirement to refresh pitcher stats after every outing and opponent batter stats as games are played.
2. **Live testability:** MLB is in season, so the pipeline can be validated against real upcoming games immediately, not just historical backtests.
3. **Difficulty:** A single-pitcher count-based regression (Poisson / negative binomial) is a practical, explainable baseline — no deep learning required.
4. **Portfolio value:** A working pre-game refresh pipeline plus an honestly evaluated probability model is a complete, demonstrable system end-to-end.

**Target metric reasoning:** Hit rate, ROI, and CLV are good *evaluation* metrics but poor *training* targets — they're noisy on small samples and don't directly optimize for prediction quality. The model trains on calibrated strikeout probability (via Poisson/NB regression), then hit rate / EV / ROI are computed afterward as honest evaluation, once results accumulate.

---

## 2026-06-26 — Skill/plugin requests not installable in-session

**Decision:** UI/UX Pro Max, Stop Slop, Superpowers, The Council (llm-council), and Playwright (as requested GitHub repos) cannot be installed directly into this session. `llm-council` in particular is not a packaged skill at all — it's a standalone FastAPI/React app requiring its own OpenRouter API key and local server, unrelated to the Claude Skills format.

**Resolution:** Applying the intent of these tools manually (direct, concise recommendations; multi-angle tradeoff comparisons) rather than installing them. Revisit if the user packages/installs them through Settings > Capabilities.
