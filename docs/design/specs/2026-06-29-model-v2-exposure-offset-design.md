# Model v2 — Exposure Offset (+ regularization/VIF) — Design Spec

**Date:** 2026-06-29
**Status:** Approved — ready for implementation (Sonnet)
**Related:** task #7 baseline (the "no exposure offset in v1" decision), task #11
walk-forward backtest (the evaluation harness). This is an **evidence-gated
experiment**, not an automatic production swap.

## Goal

Test the v2 experiment the baseline decision explicitly queued: model the
strikeout **rate per batter** and scale by expected batters faced, via a
`log(expected_BF)` exposure offset — now unblocked because we added `bf_avg_last5`
(a prior-only expected-BF estimate; the decision log said this offset was "the
first v2 experiment once a calibrated BF projection exists"). Adopt it **only if**
the walk-forward backtest shows it beats the current model without degrading
calibration.

## Why this is principled now

The baseline omitted the offset because actual batters-faced is a same-game
outcome (leakage) and no prior-only BF estimate existed; it carried
`pitch_count_avg_last5` as a workload proxy instead. `bf_avg_last5` (avg BF over
the prior 5 starts, strictly-prior shifted — leakage-safe, already shipped) is
that missing estimate. Strikeouts mechanically scale with batters faced, so an
exposure term is the theoretically right structure: `K_i ~ Poisson(μ_i)`,
`log(μ_i) = log(bf_expected_i) + Xβ` — i.e. `μ_i = bf_expected_i · exp(Xβ)`, where
`exp(Xβ)` is the modeled per-batter K rate.

## Experiment design

1. **Offset model.** Fit Poisson/NB (same dispersion-test family choice) with
   `statsmodels` `exposure=bf_avg_last5` (equivalently `offset=log(bf_avg_last5)`,
   coefficient fixed at 1). Drop `pitch_count_avg_last5` from the regressors in
   this variant — it was standing in for exposure; keeping both double-counts
   workload (confirm via VIF). Keep all other regressors.
2. **Regularization / VIF arm (secondary).** Compute VIF across the continuous
   regressors; if the held-out near-collinear `k_rate_*` variants are to be tested,
   add them only under `fit_regularized` (elastic net) and report whether they help
   out-of-sample. Keep this arm separate from the offset arm so each effect is
   attributable.
3. **Guard for `bf_avg_last5` missingness.** A pitcher with no prior starts has NaN
   `bf_avg_last5`; the offset is undefined there. Reuse the existing impute/`was_imputed`
   path (impute the league-mean expected BF, flag it) — never drop silently, never
   feed NaN to the offset.

## Evaluation (the gate)

Run both variants through the existing **walk-forward out-of-sample** harness
(`src/backtest/walk_forward.py`) on the same expanding-window splits as the
baseline, and compare on the Track-A metrics the project already trusts:
ECE/reliability, MAE, RMSE, Brier, log loss, PIT histogram — **per tier and
overall**. No ROI/Track-B here (Track A only, by construction).

**Adoption rule (honest, pre-registered):** the offset model replaces the baseline
as the production artifact **only if** it (a) does not worsen ECE/PIT
(calibration is the highest-stakes property) and (b) improves log loss and MAE
out-of-sample by a non-trivial margin. If it ties or loses, the baseline stands
and the result is recorded — a null result is a valid, logged outcome, not a
reason to ship complexity. The decision and the numbers go in the decision log and
a short `reports/YYYY-MM-DD-model-v2-offset.md`.

## Deliverables
- A v2 fit path in `baseline_model` (offset/exposure support + optional
  `fit_regularized`), behind a config/flag so the baseline remains fittable
  unchanged.
- A comparison runner (extends `scripts/run_backtest.py`, e.g. `--variant offset`)
  that emits the side-by-side metric table + reliability/PIT plots.
- The dated report + decision-log entry with the adopt/reject call.

## Tests
- Offset fit on a fixture: `μ` scales ~linearly with `bf_avg_last5` holding `Xβ`
  fixed (the exposure structure holds).
- NaN `bf_avg_last5` → imputed + `was_imputed` set, no NaN reaches the fit.
- Walk-forward comparison runner produces both variants' metrics on a fixture
  corpus with no temporal leakage (reuse the existing leakage test).
- Dropping `pitch_count_avg_last5` in the offset variant is reflected in its design
  matrix columns; baseline variant unchanged.

## Verification
1. `pytest` green (new v2 tests + unchanged baseline tests).
2. `python scripts/run_backtest.py --variant offset` produces the comparison report;
   confirm the adoption rule is evaluated explicitly (printed adopt/reject).
3. If adopted, the production model path is rewritten by a deliberate retrain run
   (not auto-swapped mid-experiment).

## Out of scope
- Boosting / GBM models (a separate later experiment; baseline stays interpretable
  GLM for now).
- A learned expected-BF model (v2 uses the simple `bf_avg_last5`; a regression for
  expected BF — opponent, park, lineup — is a follow-up if the offset proves its
  worth).
- Applying the offset to other props before it's validated on strikeouts.

## Decision-log entry to add (newest at top)
> **2026-06-29 — Model v2 experiment: exposure offset (evidence-gated).** Tested the
> queued `log(expected_BF)` offset using the newly added `bf_avg_last5` as the
> prior-only expected-BF estimate (`μ = bf_expected · exp(Xβ)`), with
> `pitch_count_avg_last5` dropped from that variant to avoid double-counting workload,
> plus a separate regularized/VIF arm for the near-collinear `k_rate_*` family.
> Evaluated out-of-sample on the walk-forward harness against the baseline on
> ECE/PIT/MAE/log loss; pre-registered adoption rule (no calibration regression +
> non-trivial log-loss/MAE gain) decides whether it ships. Result + numbers: see
> `reports/…-model-v2-offset.md`. [Fill adopt/reject after the run.]
