# Compression Fix v2 (gate-feedback iteration) — Design Spec

**Date:** 2026-06-30
**Status:** Approved — ready for implementation (Sonnet)
**Supersedes the execution of:** `2026-06-30-fix-projection-compression-and-matchup-design.md`
(v1 was correctly **rejected** by the gate: top-2-decile bias went −0.4% → −2.8%).
Same goal — un-compress the high end — with the three gate findings fixed.

## Why v1 failed (gate findings, all valid)

1. **Tested in the wrong regime.** The walk-forward ran on **2024 only**, whose
   expanding window is data-rich by midseason and therefore never reproduces the
   compression — which is specifically a *thin-2026-sample* artifact. The fix
   couldn't show improvement because the test set didn't have the disease, and it
   was scored on **zero 2026 OOS games**.
2. **EB shrinkage too aggressive (C=100).** It pulled aces too hard toward the
   prior (decile 9 bias −1.9% → −4.7%).
3. **Wrong collinearity target.** VIF: `velo_avg_last5`=135, `csw_rate_season`=162 —
   catastrophic, far worse than the `opponent_k_rate_last10` (VIF-tiny) that v1
   dropped. Extreme collinearity makes the K-skill coefficients unstable and
   individually shrunk, which is itself a **co-cause** of the compression.

## Fix 1 — evaluate in the production regime (finding ①)

The walk-forward must reproduce how production is actually trained and must be scored
on the period where compression appears.

- **Span the corpus 2024 → 2026** (pull/extend cache as needed) and run the
  expanding-window walk-forward across it, so each 2026 step is trained on the full
  prior history — exactly what production should do.
- **Compute the μ-decile bias on the 2026 OOS slices specifically** (and overall),
  not on 2024. The headline acceptance metric is top-decile bias **on 2026 OOS**.
- **Head-to-head, same OOS games:** compare (a) the *as-deployed* production fit
  (2026-only training) vs (b) the *widened* fit (full prior history) — both scored on
  the identical 2026 OOS rows. The widened fit should show larger K-skill coefficients
  and smaller high-decile under-prediction. This is the actual test of "more data
  un-compresses the high end," which v1 never ran.

(If extending the Statcast cache to 2024–2026 is too heavy in one pass, the minimum
viable version is: fit both models through an early-2026 cutoff and score on the
remaining 2026 starts — the key is **2026 OOS scoring**, never 2024-only.)

## Fix 2 — empirical-Bayes shrinkage, retuned (finding ②)

- **Shrink toward the pitcher's own baseline, not the league:** prior tier =
  own-season K-rate → own-career → league mean (only fall to league when the pitcher's
  own sample is genuinely empty). Shrinking an ace toward his own (high) season rate
  must not cap him; capping means the prior leaked to league — verify it doesn't.
- **Sweep C ∈ {25, 50, 75}** (batters-faced units) instead of hardcoding 100, and
  select C by **minimizing top-decile bias on 2026 OOS subject to no aggregate
  regression**. Report the full sweep, not just the winner. (Context: K-rate's
  ~half-reliability point is ≈70 BF, so C above that over-shrinks a full recent
  sample — consistent with C=100 being too strong.)

## Fix 3 — de-collinearize the K-skill cluster, correct target (finding ③)

- **Iterative VIF pruning:** drop the highest-VIF feature, refit, re-VIF, repeat until
  **max VIF < 10**. `velo_avg_last5` (135) and `csw_rate_season` (162) are the first
  to go; do **not** prescribe the survivor set in advance — let the procedure + OOS
  pick which single swing-miss representative stays (likely the stabilized K-rate plus
  one whiff/CSW measure).
- **Confirm the mechanism:** after pruning, the surviving K-skill coefficient should be
  **materially larger** than the current +0.057, and the model's μ spread should widen
  — that's the collinearity-driven compression being released. Report the before/after
  coefficient + VIF table.

## Order of operations (so effects are attributable)

Run as separable arms and report each independently, then combined:
1. VIF-prune (Fix 3) on the current data — does releasing collinearity alone widen μ?
2. + EB shrinkage retuned (Fix 2).
3. + widened-corpus fit (Fix 1) — the head-to-head on 2026 OOS.
Attribute the decile-bias change to each so we learn what actually un-compresses
the high end, rather than shipping a bundle and guessing.

## Acceptance (unchanged bar, correct measurement)

Adopt only if, **on 2026 OOS**: (a) top-2-decile under-prediction shrinks toward 0
(the v1 failure metric), (b) aggregate ECE/PIT not worse, (c) OOS log loss/MAE
hold or improve, and (d) max VIF < 10. Write `reports/<date>-compression-fix-v2.md`
with: the C sweep, the iterative-VIF table, before/after coefficients, and the
2026-OOS decile-bias table for each arm.

## Tests
- Tail-calibration helper computes decile bias on a **filtered OOS subset** (e.g.
  2026 rows) correctly.
- EB prior-tiering: an ace shrinks toward his own high season rate (not league); a
  pitcher with empty own-history falls to league; strictly-prior (leakage test).
- VIF helper returns per-feature VIF and the iterative-prune routine terminates at
  max VIF < 10 on a collinear fixture.
- Full `pytest -q` green.

## Verification
1. `pytest -q` green.
2. The walk-forward run prints **2026-OOS** decile-bias tables for each arm + the C
   sweep + the VIF prune log; adopt/reject evaluated on 2026 OOS.
3. If adopted: widened-corpus `--fit-only` retrain → `refresh`; spot-check 3 aces' μ
   rises and the surviving K-skill coefficient ≫ +0.057.

## Out of scope
- Lineup-weighted matchup promotion, umpire/weather/Vegas in μ (unchanged).
- Model-family change (GLM stays; boosting separate).
- The expected-batters exposure offset — still the recommended *next* lever if, even
  after Fixes 1–3, the widened fit can't separate the top (count = rate × batters).

## Decision-log entry to add (newest at top)
> **2026-06-30 — Compression fix v2 (after gate rejection).** v1 was rejected (top-2
> decile bias −0.4%→−2.8%) for three reasons, now corrected: (①) evaluate the
> walk-forward across 2024→2026 and score the **2026 OOS** slices head-to-head
> (2026-only vs widened fit) — v1 tested only on data-rich 2024 and never scored a
> 2026 game; (②) EB shrinkage retuned — prior tiers to the pitcher's own season/career
> before league, with C swept {25,50,75} instead of 100; (③) VIF-prune the real
> offenders (`velo_avg_last5`=135, `csw_rate_season`=162) via iterative pruning to max
> VIF<10, not the tiny-VIF opponent feature v1 dropped. Arms run separably and
> attributed; adopt only on improved 2026-OOS top-decile bias + no aggregate
> regression + max VIF<10. [Fill results after the run.]
