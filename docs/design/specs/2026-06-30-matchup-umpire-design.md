# Lineup-Weighted Matchup + Umpire Features — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet)
**Related:** task #6 (opponent features — the current team-level opponent K%),
task #8 (the "batter-hand splits held out, needs lineup ingestion" note),
walk-forward harness. Strikeouts only.

## Goal

Replace the coarse team-level opponent K signal with a **batter-level,
projected-lineup-weighted** opponent matchup, and add the **home-plate umpire's
strikeout tendency** — the two factors most likely to be *underweighted by the
line*, hence the best source of honest model-vs-line divergence on near-line picks.

## Part 1 — Lineup-weighted opponent K matchup

Today `opponent_k_rate_vs_hand_season` / `opponent_k_rate_last10` are whole-team
averages. The decision log flagged the real version as deferred because it needs
lineup data. Build it:

1. **Projected lineup ingestion** — a new `src/data/lineups.py` that pulls the
   game's projected/confirmed lineup from MLB-StatsAPI (boxscore/lineups hydrate).
   Lineups post closer to game time than the morning refresh, so:
   - if a confirmed/projected lineup exists, use it;
   - else **fall back to the team-level feature we have today**, and flag
     `lineup_source = {confirmed|projected|team_fallback}` so the dashboard and
     grading know which was used. Never block a prediction on a missing lineup.
2. **Batter K% vs hand** — a batter-side aggregation (new `src/features/batter_logs.py`,
   mirroring `pitcher_logs`): each batter's K-per-PA vs LHP/RHP, season + last-N,
   strictly-prior. (This is the new data work — we currently aggregate only the
   pitcher side.)
3. **Expected-K matchup feature** — `opponent_lineup_k_rate_vs_hand` =
   PA-weighted mean of the projected lineup's batter K% vs *this* pitcher's hand
   (weight by lineup slot → expected PA). Add a hand-split share
   (`opp_share_opposite_hand`) since platoon advantage drives Ks.
4. Add as **candidate regressors** alongside (initially) the team-level ones; the
   backtest decides whether the lineup version replaces or augments them.

## Part 2 — Home-plate umpire K tendency

The plate ump's zone materially moves called strikes → Ks, and the line prices it
weakly. But **the data is the hard part** — flag it explicitly:

- **Historical ump tendency:** a maintained table of each umpire's strikeout impact
  (e.g. K%-above-expected / called-strike-rate above average). There's no official
  free feed; this likely comes from a public umpire-scorecard dataset refreshed
  periodically into `data/raw/umpires/`. Treat the table as an input artifact with a
  documented provenance + refresh cadence, not a live API.
- **Today's assignment:** the home-plate ump for a given game is announced morning-of
  (lineup-card / schedule sources). Ingestion (`src/data/umpires.py`) resolves
  game → HP ump → tendency.
- **Feature:** `ump_k_factor` (multiplicative, like `park_k_factor`), with the **same
  thin-sample / missing-assignment fallback to 1.0 (neutral) + `was_imputed`** the
  park factor already uses. If the assignment isn't known at refresh time, neutral —
  never guess.

**Sourcing gate:** before building Part 2's feature, confirm a workable, legal source
for both the tendency table and the day-of assignment. If neither is reliably
available for free, ship Part 1 alone and leave a stub + the finding documented — do
not fabricate ump factors.

## Wiring
- `opponent_features.py`: add the lineup-weighted builder; keep team-level as
  fallback. New `lineups.py`, `batter_logs.py`, `umpires.py` follow the
  one-source-one-module + injected-fetcher pattern (network-free tests).
- `baseline_model.py`: new features as candidate regressors (VIF-checked vs the
  team-level ones to avoid double-counting opponent signal).
- `refresh.py`: pull lineups + ump assignment per game; record `lineup_source` and
  ump availability in diagnostics (surfaced, never silently defaulted).
- `pitcher_cards.csv` + dashboard: show the lineup-weighted opp K%, `opp_share_opposite_hand`,
  and `ump_k_factor` with their source/imputation flags.

## Evaluation (gate)
Walk-forward out-of-sample vs current model (ECE/PIT/MAE/log loss, by tier). Adopt a
feature only on no-calibration-regression + out-of-sample gain. Because lineups/umps
are only known live, **also** verify the historical backtest uses the
*as-known-pre-game* values (no post-hoc actual lineup), or it overstates the edge —
this is the highest leakage risk in this spec and needs its own test.

## Tests
- `lineups.parse_*` builds the right batter list from a fixture; missing lineup →
  `team_fallback` flag, no crash.
- `batter_logs` K%-vs-hand strictly-prior (leakage test).
- Lineup-weighted opp K% on a known fixture lineup equals the hand-matched PA-weighted
  mean.
- `ump_k_factor` missing assignment → 1.0 + `was_imputed`; thin sample → neutral.
- Backtest uses pre-game-known lineup/ump (a test that post-game actuals never leak
  into a historical feature row).

## Verification
1. `pytest` green.
2. Walk-forward comparison prints adopt/reject per feature.
3. A live refresh shows `lineup_source` per pitcher and a non-neutral `ump_k_factor`
   only where an assignment was actually known.

## Out of scope
- Catcher framing (related; separate, depends on probable-battery data).
- Batter-level modeling beyond K%-vs-hand (full opposing-lineup simulation is later).
- Paid lineup/ump feeds.

## Decision-log entry to add (newest at top)
> **2026-06-30 — Lineup-weighted opponent matchup + umpire K tendency.** Replaced the
> team-level opponent K signal with a projected-lineup-weighted batter-K%-vs-hand
> feature (new lineup + batter-log ingestion; falls back to team-level + a
> `lineup_source` flag when no lineup is posted), and added a multiplicative
> `ump_k_factor` from a maintained umpire-tendency table + day-of assignment (neutral
> + `was_imputed` when unknown, mirroring park factor). Both gated on the walk-forward
> backtest, with an explicit as-known-pre-game leakage test (historical rows must use
> the lineup/ump known before first pitch, not post-game actuals). Umpire part gated
> on confirming a free, legal data source first.
