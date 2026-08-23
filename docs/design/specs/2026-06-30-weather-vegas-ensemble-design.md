# Weather + Vegas Context + Boosted Ensemble — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet), **do last** (most new infra)
**Related:** skill-features / matchup-umpire specs (feature inputs), v2-offset spec
(expected TBF / opportunity), conviction spec (ensemble agreement feeds conviction).
Strikeouts first.

## Goal

Add the remaining underpriced *context* signals (weather, game/team totals) and a
**second model** (gradient-boosted) that can capture the interaction effects the
linear GLM can't — combined into an agreement-based conviction boost. Largest new
infra, so it's sequenced last and split into three independently-shippable parts.

## Part 1 — Weather

Pitch-level K is modestly weather-sensitive (cold/heavy air suppresses contact;
dome = neutral); day-of weather is often weakly priced.

- New `src/data/weather.py`: by ballpark (lat/long table) + first-pitch time, pull
  temperature, wind speed/direction, humidity, precipitation from a **free** API
  (e.g. open-meteo, no key). Domes/retractable-closed → neutral flag, skip weather.
- Features: `temp_f`, `wind_mph` (+ in/out relative to park orientation if cheap),
  `humidity`, `is_dome`. Add as **candidate** regressors; expect small effect — let
  the backtest decide if any survive.
- **Historical backtest needs historical weather** keyed to game date/time, or these
  features can't be walk-forward-validated. Confirm the source provides history; if
  not, weather is live-only (a known limitation, documented) and excluded from the
  Track-A backtest.

## Part 2 — Vegas game/team totals (opportunity context)

A pitcher's K *opportunity* depends on how long he pitches; game state drives that.
Team total / game total / spread proxy the run environment and blowout/length risk.

- Source: a **free** odds source for game total + team totals (the project avoids
  paid feeds — if no free source is reliable, scope this to whatever is freely
  available, or defer Part 2 and document). Confirm before building.
- Features: `game_total`, `team_total_for/against`, `is_favorite`, derived
  `expected_length` proxy. These pair naturally with the v2 expected-TBF offset
  (more expected batters faced = more K chances).
- Same as-known-pre-game leakage discipline: use the line as posted pre-game in the
  historical backtest, never a post-game close.

## Part 3 — Boosted ensemble + recalibration

The GLM is additive; arsenal×matchup, ump×zone, weather×stuff are interactions.

- A second model: `sklearn` HistGradientBoostingRegressor (or xgboost if added to
  deps) predicting the K **rate or mean**, on the full feature set (skill + matchup +
  context). Keep the count distribution: predict μ, retain Poisson/NB around it, OR
  predict per-threshold P(over) and **recalibrate** (isotonic/Platt on a held-out
  fold) so probabilities stay honest — boosting is not calibrated out of the box,
  and calibration is this project's highest-stakes property.
- **Ensemble use:** the GLM stays the interpretable primary. The booster is a
  **second opinion**: when GLM and booster diverge from the line in the *same*
  direction, raise conviction (feeds the conviction spec); when they disagree, lower
  it. Optionally a calibrated blend, but only if it beats the GLM out-of-sample on
  ECE/log loss.
- Guardrails: same walk-forward harness, same leakage rules, recalibration on a
  separate fold, and the adopt-only-on-out-of-sample-gain rule. Interpretability note
  in the report (SHAP/feature importance) so the booster isn't a black box driving
  picks.

## Wiring
- New `src/data/weather.py`, plus a Vegas-totals ingestion module; ballpark lat/long
  + dome table as a static reference (like park factors).
- `baseline_model.py` (or a new `src/models/boosted_model.py`): the booster +
  recalibration, behind a variant flag; the GLM path unchanged.
- `refresh.py`: pull weather + totals per game; carry both models' outputs +
  agreement into `line_picks`/`pitcher_cards`.
- `src/serve`: show weather/totals context and the GLM-vs-booster agreement on the
  card; conviction reflects agreement.

## Evaluation (gate, per part)
Walk-forward out-of-sample vs the current best model on ECE/PIT/MAE/log loss, by tier.
Each part adopted independently and only on a non-trivial out-of-sample gain with no
calibration regression. The booster must additionally show its recalibration holds on
the held-out fold. Null results are logged and dropped.

## Tests
- `weather.py` parses a fixture API response → features; dome → neutral flag, no API
  call. Missing weather → imputed + `was_imputed`.
- Vegas ingestion parses fixture totals; missing → flagged, neutral.
- Booster trains on a fixture, recalibration maps raw→calibrated monotonically;
  walk-forward comparison has no temporal leakage.
- Agreement signal: same-direction divergence → higher conviction; opposite → lower.

## Verification
1. `pytest` green.
2. Walk-forward comparison prints adopt/reject per part; booster's reliability curve
   is plotted post-recalibration.
3. Live slate shows weather/totals + model-agreement on the card; conviction shifts
   sensibly when the two models agree/disagree.

## Out of scope
- Paid odds/weather feeds (free sources only; defer a part if no free source exists).
- Deep learning / full lineup simulation.
- Auto-promoting the booster to primary — the interpretable GLM stays primary unless
  the ensemble decisively and repeatedly wins out-of-sample.

## Decision-log entry to add (newest at top)
> **2026-06-30 — Weather + Vegas context + boosted ensemble (sequenced last).** Added
> free-source weather (temp/wind/humidity/dome by ballpark + first-pitch) and Vegas
> game/team totals (opportunity/length context, pairs with the v2 expected-TBF
> offset) as candidate features, plus a gradient-boosted second model with isotonic
> recalibration to capture interaction effects the additive GLM can't. The GLM stays
> the interpretable primary; the booster is a second opinion whose agreement/
> disagreement modulates the conviction score. Every part walk-forward-gated
> (out-of-sample gain + no calibration regression; recalibration validated on a held-
> out fold), with as-known-pre-game leakage discipline for weather/totals, and
> deferral of any part lacking a free, legal data source.
