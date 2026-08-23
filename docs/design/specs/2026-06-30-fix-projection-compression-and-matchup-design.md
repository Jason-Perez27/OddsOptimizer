# Fix Projection Compression + Sharpen Matchup — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet). One retrain-and-validate cycle.
**Related:** task #7 baseline, task #11 walk-forward harness, `2026-06-30-pitcher-skill-features`
and `2026-06-30-matchup-umpire` specs. Strikeouts only.

## Problem (observed)

After the activation retrain, projections fell — and disproportionately for the
**highest-projected** pitchers. Inspecting the fitted standardized coefficients:

```
const                          1.5324   -> baseline mu = exp(1.53) ≈ 4.6 K
pitch_count_avg_last5         +0.1095   (largest driver = exposure proxy)
velo_avg_last5                +0.0722
is_home                       +0.0626
k_rate_last5                  +0.0574   (pitcher's own K rate — under-weighted)
park_k_factor                 +0.0422
whiff_rate_last5              +0.0311
opponent_k_rate_vs_hand_season +0.0248  (matchup: correct sign, weak)
rest_days                     +0.0056
opponent_k_rate_last10        -0.0031   (redundant, sign-flipped by collinearity)
```

The coefficients are uniformly tiny, so μ barely departs from the ~4.6 baseline for
anyone — **regression to the mean**: high-K aces are pulled down toward league
average, weak arms pushed up. This biases the model to lean *under* on every ace
(matches the reported symptom) and is independent of the matchup question.

## Root causes

1. **Data-starved fit.** The production model was retrained on 2026-to-date only
   (~half a season). Limited data → the GLM can't confidently separate aces from
   average → it shrinks all slopes toward zero → compressed μ range. The earlier,
   higher projections came from a fuller-season fit.
2. **Pitcher K-skill under-weighted and split.** `k_rate_last5` is weaker than the
   workload/velocity terms, and the strikeout signal is spread across correlated
   features (k_rate / whiff / velo / csw), so "elite swing-and-miss" barely moves μ.
   The L5 window also adds noise (see the stabilization point below).
3. **Redundant opponent feature.** `opponent_k_rate_last10` (−0.003) is collinear
   with the hand-split and dilutes the (correctly-signed) matchup term.

## Fixes (one cycle, all gated on the walk-forward backtest)

1. **Fit coefficients on the full multi-season corpus, not 2026-only.** Train on all
   cached history (at minimum 2024 + 2026; pull 2025 if feasible) so the regression
   has enough signal to separate the high end — this is the primary lever against
   compression. Note the distinction the pipeline already respects: *features* are
   always as-of-date (leakage-safe), so widening the training span changes only how
   well coefficients are estimated, not the per-row feature values. Keep the
   go-live `--fit-only` production fit pointed at the widened span.
2. **Strengthen + consolidate the pitcher K-skill signal.** Replace raw
   `k_rate_last5` with an exposure-weighted empirical-Bayes blend toward season/league
   (`k_stab = (K_recent + C·p_prior)/(BF_recent + C)`) so a real ace isn't shrunk by
   one or two soft starts, and VIF-prune the correlated K cluster (k_rate / whiff /
   velo / csw) down to a non-redundant set — promoting a stable skill rate
   (`csw_rate_season`) if it earns it. Goal: let genuine strikeout skill widen the μ
   spread.
3. **De-collinearize the opponent block.** Drop `opponent_k_rate_last10`; keep
   `opponent_k_rate_vs_hand_season` (and trial the backtestable `opponent_k_rate_vs_hand_last10`)
   as the single matchup regressor, undiluted. Keep the lineup-weighted version
   gated (no historical lineup data to validate it).

## Evaluation (the gate — with a tail check, not just aggregate)

Run the candidate model through the walk-forward harness and compare to the current
production model on the usual OOS metrics (ECE/PIT, MAE/RMSE, Brier, log loss) **plus
an explicit tail-calibration check that's the whole point here:** predicted-vs-realized
strikeouts **bucketed by μ decile** (and a reliability curve restricted to high-μ
starts). The compression bug lives in the top deciles, where aggregate ECE can look
fine while high-μ starts are systematically under-predicted.

**Adoption rule:** adopt the changes only if (a) high-μ-decile bias shrinks (aces no
longer systematically under-predicted) **and** (b) aggregate calibration doesn't
regress and OOS log loss/MAE improve or hold. Record the before/after decile table +
the new coefficients in `reports/<date>-compression-matchup-fix.md`.

## Deliverables
- Widened-corpus production fit (update the retrain command / `--start`).
- EB-stabilized K-rate feature + VIF-pruned K-skill regressor set in `baseline_model`.
- Opponent block de-collinearized (drop `opponent_k_rate_last10`; matchup hand-split retained).
- A `--variant` (or extension) that runs the candidate set through walk-forward, and a
  decile/tail-calibration report helper in `src/backtest/`.
- Decision-log entry + the dated report with the decile table and new coefficients.

## Tests
- EB-stabilized rate: a pitcher with a 2-start slump but strong season is pulled
  toward season, not tanked; strictly-prior (leakage test).
- Dropping `opponent_k_rate_last10` leaves the matchup term positive and present.
- Tail-calibration helper: on a fixture OOS frame, returns predicted-vs-actual by μ
  decile correctly; flags systematic high-μ under-prediction.
- Full `pytest -q` green.

## Verification
1. `pytest -q` green.
2. Walk-forward comparison prints the μ-decile bias table before/after and the
   adopt/reject call; high-μ under-prediction visibly reduced if adopted.
3. After a widened-corpus `--fit-only` retrain + `refresh`, spot-check 3 aces: their
   μ should rise back toward sane levels, and the coefficient on the K-skill term
   should be materially larger than the current +0.057.

## Out of scope
- Lineup-weighted matchup promotion (still needs historical lineup data).
- Umpire / weather / Vegas in μ (umpire stays out by your call; weather/Vegas remain
  decision-layer/conviction inputs until their historical backtest path exists).
- Switching model families (GLM stays; boosting is a separate later experiment).

## Decision-log entry to add (newest at top)
> **2026-06-30 — Fix projection compression (regression-to-mean) + sharpen matchup.**
> Post-activation projections compressed toward the ~4.6 K baseline (every fitted
> coefficient tiny; `k_rate_last5` only +0.057, dominated by `pitch_count_avg_last5`),
> hitting high-projected aces hardest — caused by retraining on a half-season 2026
> window (data-starved → shrunk slopes) and an under-weighted/split K-skill signal.
> Fix: fit coefficients on the full multi-season corpus, replace raw L5 K-rate with an
> exposure-weighted empirical-Bayes stabilized rate (VIF-pruned K cluster), and
> de-collinearize the opponent block (drop `opponent_k_rate_last10`, keep the
> correctly-signed hand-split matchup). Gated on the walk-forward backtest with an
> explicit μ-decile tail-calibration check (aggregate ECE alone hid the high-end bias).
> [Fill adopt/reject + decile numbers after the run.]
