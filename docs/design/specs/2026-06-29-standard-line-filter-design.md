# Canonical-Line Selection for PrizePicks Projections — Design Spec

**Date:** 2026-06-29
**Status:** Approved — ready for implementation (Sonnet)
**Related:** `docs/decision_log.md` (2026-06-27 line-source switch), task #9 (refresh
pipeline), task #12 (go-live). Touches `src/data/prizepicks_lines.py`,
`src/pipeline/verify.py`, `src/predictions/tiering.py`.

## Problem

When matching modeled K projections to the posted PrizePicks line, the pipeline
sometimes grades against an **alternate** line rather than the standard
over/under. Example: a 0.5 line for Tyler Alexander — a discounted "goblin"
alternate, not his real O/U.

## Root cause (verified against a live call, 2026-06-29)

PrizePicks posts **multiple projections per pitcher per stat**, distinguished by
the `odds_type` field inside each projection's `attributes`:

| `odds_type` | meaning | line shape |
|-------------|---------|------------|
| `standard`  | the real over/under line | one per pitcher/stat |
| `demon`     | alternate — higher line, boosted payout | a ladder above standard |
| `goblin`    | alternate — lower line, discounted payout | a ladder below standard |

A live pull (`league_id=2`, `per_page=1000`) returned **170 `Pitcher Strikeouts`
projections** but only **23 `standard`** ones — the other 147 were `demon` (101)
and `goblin` (46). Per pitcher the ladder is monotonic in line value:
goblins (low) < standard (mid) < demons (high). Concrete example:

```
Sean Burke   9.5 demon  8.5 demon  7.5 demon  6.5 demon
             5.5 standard
             4.5 goblin  3.5 goblin
```

`odds_type` is **not parsed today** — `flatten_projections` filters only on
`stat_type`, so every alternate flows downstream.

### Why an alternate currently wins the join

`flatten_projections` returns all 170 rows. Downstream,
`src/predictions/tiering.py::_dedupe_latest_projection` collapses to one row per
`(pitcher, team)` via:

```python
df = df.sort_values("_pulled_at_dt", ascending=False)
df = df.drop_duplicates(subset=["pitcher", "team"], keep="first")
```

But `flatten_projections` stamps a **single `pulled_at`** on every row in the
pull, so all of a pitcher's standard/demon/goblin rows share an identical
timestamp. The "keep latest" tiebreak therefore degenerates to *payload order*,
which is effectively arbitrary — and routinely keeps a demon or goblin. That is
the bug.

## Decision — select one canonical line per pitcher, biased to the standard/goblin side

Replace the arbitrary `keep="first"` dedupe with an explicit **canonical-line
selection** applied per `(pitcher, team)`:

1. **`standard` if present** — the real O/U; this is the normal, common case.
2. **else, if any `goblin`** → the **highest-line goblin** (the goblin "closest to
   the demon" — the top of the goblin ladder, the nearest available proxy to where
   the standard would sit). This covers both the bracketed case (goblins below,
   demons above, no standard) and the goblin-only case — both resolve to the
   highest goblin.
3. **else (only `demon` lines)** → the **lowest-line demon** (the bottom of the
   demon ladder, the nearest demon to where the standard would sit).

This is a total rule: every pitcher with any usable line gets exactly one. A
pitcher only loses a line pick if they have **no** standard/goblin/demon line at
all after the pre-game status filter — in which case they still get the full
(line-independent) threshold sweep and land in the existing `predicted_no_line`
diagnostic, never silently dropped.

The selected row's `odds_type` is carried onto the line pick so the ledger always
shows **which kind of line each pick was graded against** — a `goblin`/`demon`
fallback is recorded as such, never laundered into looking like a standard O/U.
This preserves honest grading (decision log, 2026-06-29: the morning refresh
snapshot is frozen and authoritative) and lets fallback picks be filtered or
down-weighted in later analysis.

### Where the change lives — and what supersedes the earlier sketch

The selection needs the **whole ladder** grouped per pitcher, so it cannot be a
row-by-row drop inside `flatten_projections` (an earlier draft of this spec
proposed exactly that; the fallback requirement supersedes it). Instead:

- **Ingestion stays complete and "dumb":** `flatten_projections` keeps emitting
  **every** line for the stat, now carrying `odds_type`. Raw snapshots in
  `data/raw/` therefore retain the full ladder — better for audit and for any
  future "alternate ladder as implied distribution" work.
- **The betting decision lives in the betting layer:** the single-line choice
  moves into `tiering.py`, replacing `_dedupe_latest_projection`, which is already
  the point that reduces to one row per pitcher and the only consumer that needs a
  single line.

## Changes

### 1. `src/data/prizepicks_lines.py` — `flatten_projections`

- Keep the `stat_type` filter. **Do not** filter on `odds_type` — emit all lines.
- Add `"odds_type": attrs.get("odds_type")` to each emitted row dict.
- Update the module docstring to document the `odds_type` field (standard vs
  demon/goblin) and state that all lines are emitted, with canonical selection done
  downstream in `tiering.build_line_picks`.

### 2. `src/predictions/tiering.py` — replace dedupe with canonical selection

Rename/replace `_dedupe_latest_projection` with `_select_canonical_line(prizepicks_df)`:

```python
_FALLBACK_RANK = {"standard": 0, "goblin": 1, "demon": 2}

def _select_canonical_line(prizepicks_df):
    """One line per (pitcher, team): prefer standard; else highest goblin; else
    lowest demon (see 2026-06-29 spec). Within a pitcher, the most recent pull
    wins first, so a re-pull can't resurrect a stale alternate."""
    if prizepicks_df.empty:
        return prizepicks_df
    df = prizepicks_df.copy()
    df["_pulled_at_dt"] = pd.to_datetime(df["pulled_at"], errors="coerce", utc=True)
    df["_rank"] = df["odds_type"].map(_FALLBACK_RANK)
    # Drop unrecognized odds_type (defensive: a renamed/new type isn't a line we
    # know how to grade) -- surfaced via the verify gate, not silently bet.
    df = df[df["_rank"].notna()]
    # Within goblin we want the HIGHEST line; within demon the LOWEST. Encode both
    # as a single sortable key: standard ignores line; goblin sorts by -line;
    # demon sorts by +line. Pick the first row after ordering by
    # (latest pull, category rank, tie-break key).
    df["_line"] = pd.to_numeric(df["line"], errors="coerce")
    df["_tiebreak"] = df.apply(
        lambda r: -r["_line"] if r["odds_type"] == "goblin" else r["_line"], axis=1
    )
    df = df.sort_values(
        ["_pulled_at_dt", "_rank", "_tiebreak"], ascending=[False, True, True]
    )
    df = df.drop_duplicates(subset=["pitcher", "team"], keep="first")
    return df.drop(columns=["_pulled_at_dt", "_rank", "_line", "_tiebreak"]).reset_index(drop=True)
```

Note the pull-recency handling: dedupe must pick within the **latest pull** for a
pitcher before applying the category rank, so sort by `_pulled_at_dt` first. (In a
single refresh all rows share one `pulled_at`, so this only matters if a frame ever
combines pulls — preserved defensively, as the old function did.)

`build_line_picks` calls `_select_canonical_line` where it previously called
`_dedupe_latest_projection` — **after** `_filter_pre_game`, so only active pre-game
lines are eligible (if a standard line is suspended while alternates are live, the
fallback correctly applies among the live lines).

Add `"odds_type"` to `LINE_PICKS_COLUMNS` and populate it from the selected row in
the `build_line_picks` output loop, so every graded pick records the line kind.

### 3. `src/pipeline/verify.py` — verification gate

- Add `"odds_type"` to `REQUIRED_LINE_COLUMNS` (it is now always emitted).
- Add a check that the field carries **recognized values**: at least one returned
  row has `odds_type in {"standard", "goblin", "demon"}`. This is what catches a
  PrizePicks rename/retyping of `odds_type` (which would otherwise make
  `_select_canonical_line` silently drop every line as unrecognized). Fail with a
  clear detail naming the `odds_type` field.
- The gate's report should print the **`odds_type` distribution** for the target
  stat and the **distinct-pitcher count** (the actionable board size), e.g.
  "47 projections for 23 pitchers — 23 standard, 0 needing fallback." A FAIL stays
  reserved for: fetch error, missing `data`, empty after `stat_type` filter, or no
  recognized `odds_type` present.

## Tests

Existing fixtures have **no `odds_type`** and must be updated; that update pins the
new contract.

**`tests/test_prizepicks_lines.py`**
1. `flatten_projections` now returns **all** lines for the stat (standard + demon +
   goblin), each with `odds_type` populated. Update
   `test_flatten_projections_filters_by_stat_type_and_joins_player` and
   `_sample_payload()` accordingly (add `odds_type` to every projection; add a
   demon and a goblin row for a pitcher; assert all are returned and the wrong-stat
   row is still filtered out).
2. Update `test_main_saves_and_reports_on_success` fixtures to include `odds_type`.

**`tests/test_tiering.py` — `_select_canonical_line`**
3. Standard present (with goblins + demons) → standard row selected.
4. No standard, goblins below + demons above → **highest goblin** selected.
5. No standard, only goblins → **highest goblin** selected.
6. No standard, only demons → **lowest demon** selected.
7. Unrecognized/missing `odds_type` only → that pitcher yields no line (dropped),
   does not raise.
8. Two pulls for one pitcher (different `pulled_at`) → selection comes from the
   **latest** pull.
9. `build_line_picks` output carries `odds_type` matching the selected line; a
   fallback pick shows `goblin`/`demon`, a normal pick shows `standard`. The
   Tyler-Alexander-style case (only a 0.5 goblin) produces a line pick on the 0.5
   **goblin** (not skipped), correctly labeled.

**`tests/test_verify.py`**
10. Payload with recognized `odds_type` values passes; report names the standard /
    fallback split.
11. Payload whose matching-stat rows all have missing/renamed `odds_type` fails
    with the `odds_type`-naming detail (rename guard).

Run `pytest tests/test_prizepicks_lines.py tests/test_tiering.py tests/test_verify.py -v`,
then the full suite to confirm no regression in `test_refresh.py`.

## Verification before "done"

1. Full `pytest` green.
2. `python -m scripts.diagnose_live_sources --date <today>` — confirm the live
   `attributes` dump shows `odds_type` and that `standard`/`goblin`/`demon` appear
   among `Pitcher Strikeouts` values. (Optionally extend the diagnostic to print
   the `odds_type` distribution for the target stat — that is what made this bug
   obvious. Hit the endpoint sparingly: it 429-rate-limits after a few calls.)
3. `python -m src.pipeline.refresh --dry-run` — gate PASSes and reports a pitcher
   count ≈ one per probable starter, with the standard-vs-fallback split shown.
4. Spot-check a real `line_picks.csv`: each row's `odds_type` is `standard` for the
   common case; any fallback row is labeled `goblin`/`demon` and its `line` matches
   the highest-goblin / lowest-demon rule for that pitcher.

## Out of scope

- Treating the full alternate ladder as an implied distribution / using demon-goblin
  spread as signal. The complete ladder is now retained in raw snapshots, enabling
  this later; explicitly deferred.
- Any change to tier definition, the threshold sweep, or the model.
- Other stats (pitching outs, ERA, walks) — selection is stat-agnostic and applies
  automatically when those props are wired; only strikeouts is validated here.

## Decision-log entry to add (newest at top)

> **2026-06-29 — Select one canonical PrizePicks line per pitcher (standard →
> highest goblin → lowest demon).** The projections endpoint returns multiple lines
> per pitcher tagged by `odds_type` (`standard` O/U vs `demon`/`goblin` alternates,
> monotonic: goblin < standard < demon); `flatten_projections` parsed only
> `stat_type`, so `_dedupe_latest_projection`'s same-`pulled_at` tiebreak kept
> arbitrary alternates (e.g. a 0.5 goblin for Tyler Alexander). Fix keeps ingestion
> complete (all lines emitted, now carrying `odds_type`) and moves single-line
> selection into `tiering` as `_select_canonical_line`: prefer the standard O/U;
> with no standard, take the highest goblin (closest to the demon side) or, if only
> demons exist, the lowest demon. The chosen `odds_type` is recorded on every line
> pick so fallback grading is visible, never laundered as standard. The verify gate
> now requires the `odds_type` field present with recognized values (rename guard).
> Live counts: 170 K projections → 23 standard.
