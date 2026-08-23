# Conviction Score + No-Action Band — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet)
**Related:** task #8 tiering (`edge = p_over − 0.5`, probability-only tiers), task #10
outcome tracking (Track B ROI by bucket), task #7 baseline (statsmodels GLM).
Strikeouts first; prop-agnostic where free.

## Goal

Make near-line picks **decidable** without changing the model. Today `edge = p_over −
0.5` ignores how *confident* the model is, and tiers are pure distance-from-0.5, so a
slate of picks sitting on their lines all look the same. Add (1) an
uncertainty-aware **conviction score** and (2) a validated **no-action band** so the
dashboard can say "lean over / lean under / no edge" honestly — surfacing the few
spots worth acting on and explicitly passing on the rest.

This is the fastest win for the stated problem: it operates on the current model and
turns "everything's a coin flip" into a ranked, thresholded shortlist.

## Part 1 — uncertainty-aware conviction

The current pipeline uses only the point estimate μ. statsmodels GLM exposes the
estimate's uncertainty:

- From the fitted model, `get_prediction(X)` gives the standard error of the linear
  predictor → a CI on μ for each pitcher (model/parameter uncertainty; note this is
  **estimation** uncertainty, not the count distribution's spread — keep the two
  distinct and documented).
- Propagate to the line: compute `p_over` at μ and at the μ-CI bounds, giving a band
  on `p_over` (or a delta-method SE on `p_over`). 
- **Conviction** = a single interpretable number, e.g. `conviction = |p_over − 0.5| /
  sd(p_over)` (a z-like ratio: how many SEs the model's lean is from a coin flip).
  Tight estimate + real divergence → high conviction; wide estimate or μ≈line → low.
- Persist `mu_se`, `p_over_lo`/`p_over_hi`, and `conviction` on `line_picks` and the
  dashboard cards.

Implementation: add a `predict_mean_with_se` (or `get_prediction` wrapper) to
`baseline_model`; thread the SE through `tiering.build_line_picks` /
`assemble_predictions`. Pure functions return numbers; no IO.

## Part 2 — the no-action band

A pick only matters if its lean is large enough *and* confident enough to have paid
off historically. Add an **actionability label** per pick: `lean_over` /
`lean_under` / `no_action`, decided by thresholds on `conviction` (and/or `|edge|`).

- **Threshold source:** validate on Track-B settled outcomes — pick the conviction
  cutoff where realized hit-rate/ROI actually clears break-even, **per bucket**
  (tier, threshold). This reuses `src/backtest/roi.py`.
- **Until ≥100 settled picks/bucket exist** (the standing task-#10 gate), thresholds
  are a documented **provisional default** (e.g. require `conviction ≥ 1` and
  `|edge| ≥ 0.05`), clearly labeled "provisional — not yet ROI-validated" on the
  dashboard. Do **not** present an unvalidated band as proven edge.
- This is **not** a tier redefinition (tiers stay probability-only, still gated).
  `actionability` is a new, separate field layered on top — the honest "should I bet
  this" view, distinct from the calibration-confidence tier.

## Wiring
- `baseline_model.py`: expose prediction SE (`get_prediction`).
- `tiering.py`: compute `conviction`, `p_over_lo/hi`, `actionability`; add to
  `LINE_PICKS_COLUMNS`. Add a `no_action_threshold` config (provisional defaults).
- `refresh.py`: carry the new fields through to `line_picks.csv` + `pitcher_cards`.
- `src/backtest/`: a `calibrate_no_action_band(graded)` helper that, given settled
  outcomes, returns the ROI-optimal conviction cutoff per bucket (used once data
  accrues; emits the table into the weekly report).
- `src/serve`: dashboard shows conviction + the `lean/no-action` badge; add a "show
  actionable only" filter and sort-by-conviction. Make "no edge" a first-class,
  unembarrassed state in the UI.

## Evaluation
- Sanity: conviction is high only when divergence is large AND `mu_se` is small;
  μ≈line ⇒ conviction≈0 ⇒ `no_action`.
- Once Track B has samples: the validated band's flagged picks should out-hit the
  passed ones; report it. If they don't separate, the band's threshold is wrong —
  re-fit, don't ship a band that doesn't discriminate.

## Tests
- `predict_mean_with_se` returns sensible SE on a fitted fixture model; wider with
  less training data.
- `conviction` math: large edge + small SE → high; μ at line → ~0; monotonic.
- `actionability` labels respect the thresholds; a μ≈line pick → `no_action`.
- `calibrate_no_action_band` on a fixture graded set returns the break-even cutoff;
  with too-few samples it returns the provisional default + a "not validated" flag.
- New `line_picks` columns present; serve renders them and the actionable filter.

## Verification
1. `pytest` green.
2. On the live slate, near-line picks get low conviction → `no_action`; only genuine
   divergences surface as leans. Confirm the dashboard "actionable only" view is
   short and sane.
3. The band is labeled provisional until Track B validates it; the weekly report
   prints the per-bucket conviction-vs-ROI table as data accrues.

## Out of scope
- Redefining the confidence tier (still gated on ≥100 settled/tier + evidence).
- Kelly/stake sizing (this labels actionability, not bet size; sizing is later and
  needs payout modeling the project currently avoids).
- Model changes (this layer is model-agnostic; it'll wrap whatever model wins the
  feature/v2 experiments).

## Decision-log entry to add (newest at top)
> **2026-06-30 — Conviction score + no-action band (decision layer, model
> unchanged).** Added uncertainty-aware decisioning: propagate the GLM's μ standard
> error to a band on `p_over`, define `conviction = |p_over−0.5| / sd(p_over)`, and
> label each pick `lean_over`/`lean_under`/`no_action`. Thresholds are ROI-validated
> per bucket on Track-B settled outcomes once ≥100/bucket exist; until then a
> documented provisional default, labeled unvalidated on the dashboard. Kept separate
> from the (still-gated) probability-only tier — `actionability` is a new field, not a
> tier redefinition. Makes near-line picks decidable and lets the UI honestly say "no
> edge."
