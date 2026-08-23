# Pitcher Plate-Discipline Skill Features — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet)
**Related:** task #6/#7 (feature engineering + baseline model), task #11 walk-forward
harness, `2026-06-29-model-v2-exposure-offset-design.md` (same evidence-gating
discipline). Strikeouts only.

## Goal

Sharpen the strikeout model's μ by adding **pitcher swing-and-miss skill** features
that are more stable and more predictive of future Ks than the trailing K *count*
the model leans on now (`k_rate_last5`). This is the lowest-effort accuracy lever:
the inputs are already in the Statcast pitch-level data the pipeline pulls — no new
data source.

## Why this helps the "picks sit on the line" problem

A model whose main signal is recent K rate largely re-derives the market line, so
edge ≈ 0. Plate-discipline rates (whiffs, called+swinging strikes) stabilize faster
than K outcomes and capture *skill* the line may smooth over — giving the model
honest reasons to diverge from the line on specific pitchers, which is the only way
near-line picks separate.

## Features to add (all from pitch-level Statcast already pulled)

Per pitcher-game, computed in `game_logs.aggregate_pitcher_games` from the existing
`description` / `pitch_type` / `events` / count columns, then turned into
strictly-prior rolling/season features in `rolling_features` (same shift(1) leakage
pattern as every other feature):

| feature | definition |
|---------|------------|
| `swstr_rate` | swinging strikes ÷ total pitches (`description ∈ {swinging_strike, swinging_strike_blocked}`) |
| `csw_rate` | (called_strike + swinging_strike*) ÷ total pitches |
| `putaway_rate` | strikeouts ÷ two-strike pitches (pitches thrown in 2-strike counts) |
| `whiff_rate_overall` | swinging strikes ÷ total swings (swings = swinging + foul + hit_into_play) |
| `k_minus_bb_rate` | (strikeouts − walks) ÷ batters_faced |
| `zone_rate` *(optional)* | in-zone pitches ÷ total (`zone` 1–9) — control/aggression context |

Rolling outputs (mirror existing naming): `swstr_rate_last5`, `csw_rate_last5`,
`putaway_rate_last5`, plus `*_season`. (We already have `whiff_rate_last5` from a
coarse per-game whiff_rate; reconcile — keep the precise swings-denominator version
and deprecate the old one, or rename, documented in the module.)

## Wiring

- `game_logs.py`: add the per-game numerators/denominators + emitted rate columns to
  `OUTPUT_COLUMNS`. Keep them out of `LEAKAGE_COLUMNS` consumers' way (they're
  same-game, so the *rolling* versions are the regressors, never the current game's).
- `rolling_features.py`: add the `*_last5` / `*_season` builders (strictly-prior).
- `baseline_model.py`: add the new rolling features to a **candidate** regressor set
  and test them; the final allowlist is decided by the backtest, not assumed. Watch
  collinearity — `swstr_rate`, `csw_rate`, `whiff_rate` are correlated; use VIF and
  likely keep one or two, not all (same parsimony discipline as the held-out
  `k_rate_*` variants).
- `predict_features.py`: the as-of-today synthetic row already runs the rolling
  builders, so new features flow through automatically; confirm no new same-game
  leakage and that debutants impute/flag via the existing `was_imputed` path.
- `pitcher_cards.csv` (dashboard): add the chosen skill features so they show in the
  decision panel.

## Evaluation (the gate — same as v2)

Run the candidate feature set through the **walk-forward out-of-sample harness** and
compare to the current model on ECE/PIT, MAE/RMSE, Brier, log loss — overall and by
tier. **Adopt only if** calibration doesn't regress and log loss/MAE improve
out-of-sample by a non-trivial margin. A feature that doesn't pay off out-of-sample
is dropped, not kept because it's theoretically sound. Record the decision + numbers
in a short `reports/<date>-skill-features.md`.

## Tests
- `game_logs` emits correct `swstr_rate` / `csw_rate` / `putaway_rate` from a
  hand-built pitch fixture with known descriptions and counts.
- Rolling versions are strictly-prior (reuse the leakage test: appending a later game
  doesn't change earlier rows).
- Debutant (no history) → NaN skill features → imputed + `was_imputed`, never NaN
  into the fit.
- Walk-forward comparison runner emits both feature-sets' metrics with no temporal
  leakage.

## Verification
1. `python -m pytest -q` green.
2. `python scripts/run_backtest.py --variant skill-features` (or equivalent) prints
   the side-by-side metrics and the adopt/reject call.
3. If adopted, a deliberate retrain writes the new production model; the dashboard
   cards show the new skill stats.

## Out of scope
- Pitch-arsenal × opponent-vulnerability interactions (needs the boosted model —
  separate spec) — these features feed it later.
- Spin/movement/release features (add only if the swing-miss rates prove their worth
  first).
- Other props (skill features generalize, but validate on strikeouts first).

## Decision-log entry to add (newest at top)
> **2026-06-30 — Pitcher plate-discipline skill features (evidence-gated).** Added
> SwStr%, CSW%, putaway%, overall whiff%, and K−BB% per pitcher-game from the
> existing Statcast pitch data, with strictly-prior `*_last5`/`*_season` rolling
> versions, as candidate regressors to sharpen μ with more stable signal than the
> trailing K rate. Final allowlist decided by the walk-forward backtest (VIF-pruned
> for collinearity; adopt only on no-calibration-regression + out-of-sample log-loss/
> MAE gain). Surfaced on the dashboard cards. [Fill adopt/reject after the run.]
