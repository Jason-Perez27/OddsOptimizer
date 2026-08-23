# Pitcher Strikeout Decision Dashboard (live web deployment) — Design Spec

**Date:** 2026-06-29
**Status:** Approved — ready for implementation (Sonnet)
**Related:** task #9 (refresh pipeline), `2026-06-29-standard-line-filter-design.md`
(canonical line + `odds_type`). New: `src/serve/` package; one additive output file
from `src/pipeline/refresh.py`. Approved design mockup rendered in chat 2026-06-29.

## Goal

A live, auto-refreshing web page that turns each morning's `refresh` output into a
per-pitcher decision view: the **1+→10+ over probabilities**, **projected K (mu)**,
the **canonical PrizePicks line** (with `odds_type`), and the **supporting form
stats** (pitcher K-rate, opponent K-rate vs handedness, park factor, etc.) — all
read from the date-partitioned CSVs the pipeline already writes, plus one new
features file this spec adds.

This is a **read-only presentation layer**. It never re-runs `refresh` (the
morning snapshot is authoritative and frozen — decision log 2026-06-29) and never
writes to the predictions partitions.

## Scope decisions (confirmed with Jason, 2026-06-29)

- **Delivery:** live web deployment (not a static report or one-off HTML).
- **Decision stats:** persist the features first, then build the dashboard on real
  data — the features are computed every refresh but currently dropped before
  `predictions.csv` is written.
- **Stat windows:** use the **existing** feature windows — pitcher `k_rate_last5`
  (there is no last-10), opponent `opponent_k_rate_vs_hand_season` (vs the
  starter's hand) plus team `opponent_k_rate_last10` (not hand-split). No new
  feature-engineering math in this task.

---

## Part A — Persist decision features (pipeline change)

### Problem

`refresh.assemble_predictions` computes `feature_rows` (full feature table) but
writes only `["pitcher","game_pk","pitcher_team","opponent_team","game_date",
"pitcher_name","family","mu","alpha"]` to `predictions.csv`. Every stat the
dashboard needs already exists on `kept_rows` at that point and is then discarded.

### Change

Emit one **additional** file per partition — do **not** widen `predictions.csv`
(its column contract is consumed by `tiering` and joined by `settle`; keep it
stable). New file:

```
data/processed/predictions/game_date=YYYY-MM-DD/pitcher_cards.csv
```

Keyed by the canonical `(pitcher, game_pk)` (doubleheader-safe), one row per
**predicted** pitcher (same set as `predictions.csv`). Columns (all already present
on `kept_rows` / `feature_rows`; carry through unchanged — round only at display
time, not on disk):

| group | columns |
|-------|---------|
| keys / identity | `pitcher`, `game_pk`, `game_date`, `pitcher_name`, `pitcher_team`, `opponent_team`, `pitcher_throws`, `is_home` |
| projection | `mu` (join from predictions) |
| pitcher form | `k_rate_last5`, `k_rate_season`, `k_rate_vs_LHB`, `k_rate_vs_RHB`, `k_rate_home`, `k_rate_away`, `k_rate_vs_opponent_career`, `whiff_rate_last5`, `velo_avg_last5`, `pitch_count_avg_last5`, `ip_avg_last5` |
| opponent | `opponent_k_rate_vs_hand_season`, `opponent_k_rate_last10`, `opponent_k_rate_home`, `opponent_k_rate_away` |
| context / quality | `park_k_factor`, `was_imputed` |

Implementation notes:
- In `assemble_predictions`, build the card frame from `kept_rows` **before** it is
  sliced to `PREDICTIONS_COLUMNS`. Return it alongside `predictions` (extend the
  return tuple, or add a `build_pitcher_cards(kept_rows, mu, name_by_id)` helper).
  Keep `mu` consistent by attaching the same `mu` array used for predictions.
- Thread the card frame through `run_refresh`'s results dict
  (`results["pitcher_cards"]`) and `write_outputs` (write `pitcher_cards.csv`).
- Define a `PITCHER_CARD_COLUMNS` constant (single source of truth) and select to
  it, so a feature-layer column rename surfaces as a clear KeyError in tests, not a
  silently missing dashboard stat.
- Empty/degenerate slate: write `pitcher_cards.csv` with headers and zero rows
  (mirror how the other outputs handle the empty case).
- `settle.py` / grading are **untouched** — they read `predictions.csv` /
  `line_picks.csv` only. Add the new filename to any "expected partition contents"
  assertion if one exists.

### Tests (`tests/test_refresh.py`)
- `pitcher_cards.csv` is written with `PITCHER_CARD_COLUMNS` and one row per
  predicted pitcher; `(pitcher, game_pk)` matches `predictions.csv`.
- A persisted stat (e.g. `k_rate_last5`, `park_k_factor`) equals the value on the
  source `feature_rows` for that pitcher (no transform drift).
- A pitcher dropped for no usable features appears in neither `predictions.csv` nor
  `pitcher_cards.csv` (still surfaced in `skipped_pitchers`).

---

## Part B — Live web deployment (`src/serve/`)

Small FastAPI app serving a single-page dashboard plus a JSON API, reading the
partitions read-only. FastAPI + uvicorn are the only new runtime deps (add to
`requirements.txt`); both are small and standard. Keep the data/join logic in a
pure, network-free module so it is unit-testable exactly like the rest of the repo.

### Modules

```
src/serve/
  __init__.py
  data.py        # pure: read a partition -> assembled slate dict (NO web, NO network)
  app.py         # FastAPI: routes + static file serving (thin shell over data.py)
  static/
    index.html   # the dashboard (matches the approved mockup)
    app.js        # fetch + render + sort + auto-refresh
    styles.css
```

### `data.py` (the testable core)

- `list_game_dates(processed_dir) -> list[str]` — sorted `game_date=*` partitions.
- `latest_game_date(processed_dir) -> str | None`.
- `load_slate(processed_dir, game_date) -> dict` — read and **join on
  `(pitcher, game_pk)`**:
  - `predictions.csv` (mu, family, alpha, teams),
  - `threshold_table.csv` → nested `ladder: [{threshold, p_over, tier}]` (1..10,
    sorted) per pitcher,
  - `line_picks.csv` → canonical `line`, `line_threshold`, `odds_type`, `lean`,
    `edge`, `p_over`, `p_under`, `push_mass`, `tier`, `start_time`,
  - `pitcher_cards.csv` → the decision stats,
  - `diagnostics/*.csv` → counts + lists (`skipped_pitchers`, `unmatched_lines`,
    `predicted_no_line`),
  - `run_manifest.json` → `model_age_days`, `model_stale`, `prizepicks_error`,
    counts.
  - Returns `{game_date, generated_at, manifest, kpis, pitchers:[...],
    diagnostics:{...}}`. Pitchers with a sweep but **no line** are included with
    `line: null` (sweep-only) — never dropped. Each pitcher carries its
    `odds_type` so the UI can label `standard` / goblin / demon fallbacks.
  - Missing `pitcher_cards.csv` (pre-Part-A partition) → degrade gracefully:
    stats `null`, everything else still renders.

### `app.py` routes
- `GET /` → `static/index.html`.
- `GET /api/dates` → `{dates:[...], latest:...}`.
- `GET /api/slate?date=YYYY-MM-DD` (default latest) → `load_slate(...)`; 404 if the
  partition doesn't exist, with a clear message.
- `GET /healthz` → `{status:"ok"}`.
- Static mount for `static/`. Bind `127.0.0.1` by default (personal research tool;
  document how to expose it deliberately). **No** write endpoints; `refresh` stays a
  separate scheduled CLI. (If a manual trigger is ever wanted, it is a guarded,
  default-off `POST /api/refresh` — out of scope here, and must respect the
  never-re-run-after-first-pitch rule.)

### Frontend (matches the approved mockup)
- KPI row: pitchers on slate, standard lines, fallback lines, no-line(sweep-only);
  model age + staleness warning; data-source/disclaimer line ("research only").
- Slate list: one row per pitcher — name, matchup, projected K, line + `odds_type`
  badge, lean/edge, tier chip. **Sortable** (projected K, edge, P(over line),
  tier). Sorting is pure JS over the fetched JSON — never refetches, never repaints
  identity colors.
- Detail panel (selected pitcher): the **1+→10+ probability ladder** (horizontal
  bars, the posted-line threshold marked), the line block (line, lean, edge,
  P(over), push, tier), and the decision-stats grid (projected K, K% L5, K% season,
  K% vs LHB/RHB, opp K% vs hand (season), opp K% L10, park K factor, whiff% L5,
  velo L5). Label the windows honestly (L5 / season) per the scope decision.
- Auto-refresh: `setInterval` re-`GET /api/slate` every ~5 min (cheap disk re-read)
  + a manual refresh button + a date picker from `/api/dates`. Show "last loaded"
  time.
- Honest framing baked in: fallback (`goblin`/`demon`) picks visibly badged and
  never styled as standard; diagnostics (skipped pitchers, unmatched lines)
  reachable; if `prizepicks_error` is set, banner that lines are unavailable and
  only the sweep is shown.
- Styling: round every displayed number; dark-mode safe; no secrets in the client.

### Deployment
- **Primary:** local — `uvicorn src.serve.app:app --port 8000`, documented in a new
  `docs/runbook_dashboard.md` (and referenced from `docs/runbook_go_live.md`). It
  reads whatever the morning refresh wrote; run it alongside the existing cadences.
- **Optional hosting:** a minimal `Dockerfile` (python-slim, `pip install -r
  requirements.txt`, `CMD uvicorn ...`) for Render/Fly/Railway. The app is stateless
  and read-only over the partition files; a hosted instance needs the
  `data/processed/predictions/` tree mounted/synced. Keep this optional to honor the
  project's low-infra/free bias — note it, don't require it.

### Tests
- `tests/test_serve_data.py` — build a fixture partition (tiny predictions /
  threshold / line_picks / pitcher_cards / manifest) and assert `load_slate`
  assembles correct KPIs, the nested 1..10 ladder, a sweep-only (line=null)
  pitcher, a goblin/demon fallback carrying its `odds_type`, and graceful handling
  of a missing `pitcher_cards.csv`.
- `tests/test_serve_app.py` — FastAPI `TestClient` smoke tests: `/healthz`,
  `/api/dates`, `/api/slate` happy path + 404 for a missing date. No network.

---

## Verification before "done"

1. Full `pytest` green (refresh + serve).
2. Re-run a live refresh so the partition has the **post-fix** line picks
   (canonical `odds_type`, standard-preferred) **and** the new `pitcher_cards.csv`:
   `python -m src.pipeline.refresh` (the on-disk 2026-06-29 partition predates the
   `odds_type` fix — Sean Burke shows a 9.5 demon — so it must be regenerated).
3. `uvicorn src.serve.app:app` → load `/`, confirm: KPIs match the manifest; a
   known pitcher's ladder matches `threshold_table.csv`; the line badge shows
   `standard` for the common case and a labeled goblin/demon only on genuine
   fallbacks; sorting works without refetch; auto-refresh picks up a re-run.
4. Spot-check one detail card's stats against `pitcher_cards.csv` for that pitcher.

## Out of scope
- Triggering `refresh`/`settle` from the web app (scheduling stays CLI + the four
  cadences).
- New features / new stat windows (pitcher last-10, hand-split opponent last-10) —
  revisit only if the existing windows prove insufficient in use.
- Auth / multi-user / public hosting hardening beyond the localhost default + the
  optional Dockerfile note.
- Live Track-B performance metrics on the dashboard (that's the `report` layer);
  this view is the pre-game decision board, not the results ledger.

## Decision-log entry to add (newest at top)

> **2026-06-29 — Pre-game decision dashboard (live web deployment).** Added a
> read-only FastAPI app (`src/serve/`) that renders each morning's refresh partition
> as a per-pitcher board: 1+→10+ over probabilities, projected K, the canonical
> PrizePicks line (with `odds_type` badge + goblin/demon fallback labeling), and
> supporting form stats. Required persisting the decision features the pipeline
> already computes but dropped — added an additive `pitcher_cards.csv` per partition
> (keyed `(pitcher, game_pk)`), leaving `predictions.csv`/`settle` contracts
> untouched. Used existing feature windows (pitcher K% last-5, opponent K% vs-hand
> season + team last-10); no new feature math. The app is presentation-only — it
> never re-runs refresh (morning snapshot stays authoritative) and binds localhost
> by default, with an optional Dockerfile for hosting. Design approved via mockup.
