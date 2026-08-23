# Alternate-Ladder Market-Anchor Diagnostic — Design Spec

**Date:** 2026-06-29
**Status:** Approved (scoped down from "implied distribution") — ready for implementation
**Related:** `2026-06-29-standard-line-filter-design.md` (we now retain the full
goblin/standard/demon ladder in raw snapshots), task #8 tiering (`edge = p_over − 0.5`
free proxy).

## Reality check that reshapes this work

The original idea was to back out a **market-implied K distribution** from the
goblin/demon ladder and use it as a free CLV-like edge signal. A live inspection of
the public projections endpoint (2026-06-29) shows that's **not identifiable from
this feed**: each projection's attributes carry only `odds_type`
(standard/goblin/demon) and `adjusted_odds: true/false` — **no payout multiplier and
no implied probability**. Without payouts you cannot convert an alternate line into a
probability, so there is no per-pitcher implied CDF to extract here.

What the ladder *does* give, for one pitcher, is a set of threshold anchor points:
goblin lines (low), the standard line (≈ the book's central estimate — for a pick'em
the standard line sits near the implied median, which is exactly why the project uses
`edge = p_over − 0.5`), and demon lines (high). So the honest, useful version is a
**market-anchor diagnostic**, not a distribution.

## Step 0 — data-availability gate (do this first)

Before building anything, probe whether payout multipliers are obtainable elsewhere:
inspect `GET /projections/{id}`, any `included` records of payout/odds type, and the
flash-sale fields, for a numeric multiplier or implied probability. Document the
finding.
- **If multipliers exist somewhere** (unlikely on the free feed): a true implied
  distribution becomes possible — pause and re-spec the richer version.
- **If not** (expected): ship the anchor diagnostic below. Do **not** fabricate
  probabilities from `odds_type` alone.

## Deliverable — the anchor diagnostic

Overlay the model's own implied CDF against the line ladder, per pitcher, and surface
the divergence:

1. `src/analysis/ladder_anchor.py` (pure, testable): given a pitcher's fitted count
   distribution (family, mu, alpha — already on `predictions`) and that pitcher's full
   ladder of `(threshold, odds_type)` points (from the retained raw lines / a ladder
   file), compute:
   - the **model median** K (smallest k with `P(K ≤ k) ≥ 0.5`),
   - **model P(over)** at every ladder threshold (standard, each goblin, each demon),
   - **anchor gap** = model median − standard line (signed; the closest free
     model-vs-market disagreement signal), and
   - flags: standard line far from model median; a goblin threshold the model rates
     as near-certain over, or a demon the model rates as near-impossible (i.e. the
     model thinks the book's "hard"/"easy" framing is mispriced — directional only,
     not a probability claim).
2. Persist a per-pitcher `ladder_anchor.csv` in the partition (needs the full ladder,
   so either keep all `odds_type` rows in a raw ladder file at refresh time or read
   the already-retained raw snapshot).
3. **Dashboard overlay** (extends the existing ladder panel, which already marks the
   standard line): also mark the goblin/demon thresholds on the 1+→10+ bars and draw
   the model-median tick, with the anchor-gap shown as a chip. This turns the ladder
   into a model-vs-line read at a glance.

## Honest framing (carried into UI + log)

This is a **model-vs-line divergence diagnostic**, explicitly **not** CLV and **not**
"beating the market": there are no odds, so no implied probability and no expected
value. It extends the existing `edge = p_over − 0.5` proxy with the alternate-line
geometry as context. The README/disclaimer stance is unchanged.

## Tests
- `ladder_anchor` on a fixture: model median computed correctly from a known
  Poisson μ; P(over) at each ladder threshold matches the count distribution; anchor
  gap signs correct; the "goblin the model loves / demon the model hates" flags fire
  on constructed cases.
- A pitcher with only a standard line (no alternates) → gap computed, no alternate
  flags, no crash.
- A pitcher with no standard (fallback case) → uses the canonical line per the
  existing rule; diagnostic still computes against whatever line exists, labeled.

## Verification
1. `pytest tests/test_ladder_anchor.py` green.
2. Spot-check one pitcher: model median, standard line, and the bar overlay agree with
   `predictions`/`threshold_table`.
3. Confirm Step 0's finding is written down (so a future reader doesn't re-chase
   payout data that isn't there).

## Out of scope
- Any probability/EV/CLV derived from `odds_type` alone (not identifiable — the whole
  point of Step 0).
- Buying a priced-odds feed (the paid dependency the project deliberately avoids).
- Using the anchor gap to redefine tiers — tier redefinition stays gated on Track-B
  settled-sample evidence.

## Decision-log entry to add (newest at top)
> **2026-06-29 — Ladder work scoped to a market-anchor diagnostic (no implied
> distribution).** A live check confirmed the public projections feed exposes no
> payout multiplier or implied probability — only `odds_type` + `adjusted_odds` — so a
> market-implied K distribution isn't identifiable from it. Reframed the work as a
> model-vs-line diagnostic: overlay the model's implied CDF and median on the retained
> goblin/standard/demon ladder, surface the model-median-vs-standard-line gap and
> directional "the model disagrees with the book's hard/easy framing" flags. Explicitly
> not CLV/EV (no odds). A Step-0 probe for payout data elsewhere gates whether the
> richer version is ever revisited.
