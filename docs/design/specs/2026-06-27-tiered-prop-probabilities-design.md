# Tiered Per-Threshold Prop Probabilities — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (2026-06-27 entry — output reshape + tier
definitions), `docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md`
(task #7), `src/data/prizepicks_lines.py`, task #8. Implementation target:
`src/predictions/tiering.py`, `tests/test_tiering.py`.

## Goal

Turn the baseline model's predicted strikeout-count distribution into the
project's actual product output: for each starting pitcher, a P(over) for every
strikeout threshold 1+ through 10+, each bucketed into one of three confidence
tiers; and, where a PrizePicks line exists, a single line-specific pick row
(P(over the posted line), its tier, and which side the model leans). Scope is
**strikeouts only** — pitching outs / earned runs / walks are deferred until the
strikeout pipeline is validated end-to-end (decision log, 2026-06-27).

This is the layer that consumes the model; it does not re-fit or re-evaluate it.
It is deliberately thin and deterministic so it can be unit-tested without a
network call or a fitted model.

## Inputs

1. **Model distribution per pitcher-game** — from task #7's `BaselineModel`:
   a predicted mean μ (Poisson) or (μ, α) (Negative Binomial) per pitcher, and
   the `predict_over_prob_sweep()` helper that returns P(K ≥ t) for t = 1…10.
   For line comparison this layer calls the model's exact survival function for
   an arbitrary integer threshold rather than only indexing the 1…10 sweep (so
   lines above 9.5 are handled — see "Line → threshold").
2. **Current PrizePicks lines** — from `src.data.prizepicks_lines.flatten_projections`:
   one row per player/prop with `pitcher` (player *name* string), `team`
   (PrizePicks abbreviation), `stat_type`, `line` (float, e.g. 6.5),
   `start_time`, `status`, `projection_id`, `pulled_at`. Filtered to
   `stat_type == "Strikeouts"`.

The two sides do **not** share a join key: the model keys pitchers by MLBAM
numeric id (`pitcher`) with Statcast team codes; PrizePicks keys by display name
+ its own team abbreviation. Reconciling them is the main integration risk and is
specced explicitly below.

## The four pieces

### 1. Per-threshold P(over) sweep

For each pitcher with a model prediction, compute P(K ≥ t) for t = 1…10 from the
fitted distribution (task #7 already guarantees this is in [0,1] and
monotonically non-increasing in t). This is the decision-log-mandated "threshold
sweep, 1+ through 10+." It exists for every predicted pitcher regardless of
whether PrizePicks posts a line for them.

### 2. Line → threshold conversion

PrizePicks posts half-integer lines (e.g. 6.5). "Over 6.5" means the pitcher
records **7 or more** strikeouts, so:

```
P(over line L) = P(K ≥ floor(L) + 1) = model survival function at floor(L)+1
```

- L = 6.5 → P(K ≥ 7); L = 0.5 → P(K ≥ 1); L = 9.5 → P(K ≥ 10).
- **Integer lines** (e.g. L = 6.0) can in principle push (exactly 6 refunds).
  PrizePicks pick'em effectively always uses .5 lines, so this shouldn't occur,
  but handle it defensively: treat over as K ≥ L + 1, record the push mass
  P(K = L) in the output, and do **not** crash. Document that a push is possible.
- **Lines beyond the 1…10 sweep** (L ≥ 10.5 → threshold ≥ 11): compute P(over)
  directly from the model's survival function, not from the fixed sweep. This is
  why line comparison uses the distribution directly rather than a sweep lookup.

### 3. Confidence tiering (the core of task #8)

Tiers are a function of the predicted probability **only** — never of the line,
the edge vs. the line, or any backtest history. Defining them by **distance from
50%** captures the decision log's rule exactly. Let `d = |P − 0.5|`:

| Tier   | Rule (decision log)            | Equivalent distance `d` |
|--------|--------------------------------|-------------------------|
| High   | P ≥ 0.70 **or** P ≤ 0.30       | `d ≥ 0.20`              |
| Medium | 0.60–0.70 **or** 0.30–0.40     | `0.10 ≤ d < 0.20`       |
| Low    | otherwise (0.40 < P < 0.60)    | `d < 0.10`              |

Boundary inclusivity (made explicit so the test can pin it): 0.70 and 0.30 are
**High**; 0.60 and 0.40 are **Medium**. So `d ≥ 0.20 → High`,
`0.10 ≤ d < 0.20 → Medium`, `d < 0.10 → Low`. The single `tier(p)` function is
applied identically to every sweep threshold and to the line-specific
probability — the line never changes a tier, it only selects which threshold's
probability gets surfaced as "the line's pick."

**Why probability-only, not edge-vs-line** (decision log rationale, restated so
the spec stands alone): edge-based tiers would depend on PrizePicks' line-setting
behavior — which is tuned to their payout structure, not to true probability —
and on a season of tracked results to trust. Probability-distance tiers depend on
nothing but the model's own calibration, so they're usable on day one and can be
revisited once task #10 accumulates real outcome data.

**What the tier actually means on the line pick — read this before trusting a
"High."** The tier is a property of a *probability*, P(K ≥ some number) vs. a coin
flip. Applied across the sweep it forms a gradient: the tails (1+, 2+ ≈ 1.0; 9+,
10+ ≈ 0.0) are trivially **High** — true but useless, since every starter clears
1 and nobody's a lock for 10 — while the thresholds near the pitcher's projected
mean sit near 0.5 and read **Low**. Now the important consequence: PrizePicks sets
the line near the distribution's *median*, so a well-calibrated model's P(over the
line) ≈ 0.5, which under these bands is **Low** (a 45–60% true hit rate maps almost
entirely to Low, not Medium). So **line picks cluster at Low, and only reach High
when the model strongly disagrees with the posted line.** A High-tier *line* pick
is therefore rare and meaningful — it is, in effect, a disagreement-with-the-market
signal. This is by design, not a bug, but it must be documented so a "Low" line
pick isn't misread as "the model has no opinion" (it usually means the line is
efficient).

**The `edge` field (free-data actionable signal).** To make the bettable signal
explicit without re-opening the probability-only tiering decision, `line_picks`
carries a signed `edge = p_over − 0.5`. Because a flat pick'em line has no posted
odds and sits at the book's ~median, 0.5 is used as the line's implied
probability — a deliberate **free-data proxy**. Computing a precise implied
probability would require modeling PrizePicks' payout multipliers or pulling a
priced sportsbook odds feed; the latter is the paid dependency this project
dropped for cost (decision log, The Odds API), so it is **explicitly out of scope**
to keep the project subscription-free. `edge` is the continuous magnitude behind
the line pick's tier; promoting it (or a backtest-calibrated version) to drive the
tier is a task #10 revisit, once real outcomes exist to justify it.

### 4. Joining model predictions to PrizePicks lines (name ↔ id)

Because the two sides don't share a key, the layer needs a **deterministic
name→MLBAM-id resolver**:

- Build the resolver from pybaseball's player-id register (Chadwick) — normalize
  both sides' names (case-fold, strip accents/punctuation/suffixes like "Jr."),
  and disambiguate same-name players by team (mapping PrizePicks' team
  abbreviation to the Statcast team code via a small static crosswalk, since the
  two abbreviation sets differ, e.g. WSH/WAS, CWS/CHW).
- The resolver is the riskiest part of this task. Two non-negotiable rules:
  1. **Never silently mis-join.** An ambiguous match (same normalized name, no
     unique team match) is treated as *unmatched*, not guessed.
  2. **Surface, don't drop.** Every PrizePicks line that fails to resolve to a
     model prediction goes into an explicit `unmatched_lines` output, and every
     predicted pitcher with no posted line is simply absent from the pick table
     (present in the sweep). Nothing is dropped without a trace.

## Output schema

Two related tables plus a diagnostics frame:

1. **`threshold_table`** (long format, one row per pitcher × threshold) —
   `pitcher` (id), `pitcher_name`, `team`, `opponent_team`, `game_date`,
   `threshold` (1…10), `p_over`, `tier`. Exists for every predicted pitcher.
2. **`line_picks`** (one row per pitcher with a resolved PrizePicks line) —
   `pitcher`, `pitcher_name`, `team`, `start_time`, `line`, `line_threshold`
   (= floor(line)+1), `p_over`, `p_under` (= 1 − p_over − push_mass), `tier`,
   `lean` ("over" if p_over ≥ 0.5 else "under"), `edge` (signed,
   = p_over − 0.5; the free-data bet signal — see tiering section),
   `push_mass` (0 for half-integer lines), `projection_id`, `pulled_at`. The
   actionable pick view.
3. **`diagnostics`** — `unmatched_lines` (PrizePicks lines that didn't resolve to
   a prediction) and `predicted_no_line` (pitchers predicted but with no posted
   line), so the operator can see coverage rather than infer it from row counts.

`lean` gives direction, `tier` gives confidence; a pick is only actionable when a
line exists, so direction lives on `line_picks`, not the bare sweep.

## Edge cases (all must be handled, none should crash)

- Pitcher predicted, no PrizePicks line → in `threshold_table` and
  `predicted_no_line`; absent from `line_picks`.
- PrizePicks line, no model prediction (e.g. a debut not in the feature table) →
  in `unmatched_lines`; no fabricated probability.
- Integer line / push possible → `push_mass` recorded, no crash.
- Line ≥ 10.5 → P(over) from the survival function directly.
- `status` not pre-game (started/suspended) → filtered out before tiering; only
  active pre-game lines produce picks.
- Duplicate projections for one pitcher (re-posts / variant projections) →
  dedupe to the latest `pulled_at` for the standard projection.
- Empty predictions or empty lines → empty, well-formed outputs.

## Module layout

`src/predictions/tiering.py` — a new module (new `src/predictions/` package).
Justification: this is the *produce-live-picks* concern, distinct from
`src/evaluation/` (which measures model quality against known outcomes) and from
`src/backtest/` (ROI/hit-rate over time). Pure functions: `tier(p)`,
`line_to_threshold(line)`, `prob_over_line(model, line)`, `build_threshold_table`,
`resolve_pitcher_ids`, `build_line_picks`. If you'd rather not add a package, the
fallback is `src/models/tiering.py` — flag it in review and I'll adjust.

## Deferred (conscious scope cuts)

- Non-strikeout props (pitching outs / earned runs / walks) — until the
  strikeout pipeline is validated.
- Edge-vs-line and EV/ROI tiering — intentionally excluded now (decision log);
  belongs with task #10 once outcomes accumulate.
- A live name-resolver fed by MLB-StatsAPI probable-pitcher ids for the day
  (more robust than the static Chadwick + crosswalk approach) — a v2 hardening
  once the pre-game pipeline exists.

## Testing approach — what `tests/test_tiering.py` must assert

Deterministic, no-network unit tests on small hand-built inputs, matching the
existing `tests/test_*` style (a fake model object exposing a known
survival/`predict_over_prob_sweep` is fine — no real fit needed):

1. **Tier boundaries are exact and symmetric** (table-driven): P = 0.70 → High,
   0.699 → Medium, 0.60 → Medium, 0.599 → Low, 0.50 → Low, and the mirror image
   0.30 → High, 0.301 → Medium, 0.40 → Medium, 0.401 → Low.
2. **Tier depends only on P:** same P with two different lines → identical tier.
3. **Line → threshold conversion:** 6.5 → 7, 0.5 → 1, 9.5 → 10, 10.5 → 11; an
   integer line 6.0 → over = K ≥ 7 with `push_mass` = P(K = 6) recorded.
4. **P(over line) matches the model distribution:** for a known μ (Poisson) and a
   known (μ, α) (NB), `prob_over_line` equals the `scipy.stats` survival-function
   reference.
5. **Resolver correctness:** a known name+team resolves to the right MLBAM id; an
   accented/suffix variant still resolves; an ambiguous same-name/no-team-match
   case goes to `unmatched_lines` rather than mis-joining.
6. **Coverage is surfaced, not dropped:** a predicted pitcher with no line appears
   in `predicted_no_line`; a line with no prediction appears in `unmatched_lines`.
7. **Sweep integrity at this layer:** `threshold_table` covers t = 1…10 inclusive,
   `p_over` is non-increasing in t and in [0,1], every row has a tier.
8. **`lean` direction and `edge`:** p_over ≥ 0.5 → "over", else "under"; `edge`
   equals p_over − 0.5 (signed), and an efficient-line case (p_over ≈ 0.5) yields
   `edge` ≈ 0 with a Low tier.
9. **Status filter + dedupe:** non-pre-game lines are excluded; duplicate
   projections collapse to the latest `pulled_at`.
10. **Edge cases:** empty predictions and empty lines both yield empty,
    correctly-typed output frames without raising.
