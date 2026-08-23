"""
Live-data verification gate (task #11 spec section 5, "Live-data
verification gate (`verify_live_sources`)").

Both live sources' actual payload shape were flagged in their own modules'
docstrings as NEVER verified against a live call:
- src/data/underdog_lines.py: the Underdog over_under_lines payload's join
  path (over_under_lines -> appearances -> players -> games) and stat-key
  contract were verified live once (2026-08 migration) but nothing guards
  against Underdog changing shape again.
- src/data/probable_pitchers.py: the StatsAPI schedule's `probablePitcher`
  hydration shape is documented from public usage, also unverified live.

This module is the gate that closes that gap before trusting any REAL graded
run: fetch each live source's RAW payload, run it through the SAME parser
the refresh pipeline uses (`flatten_lines` / `parse_probable_starters`),
and check the result actually has the shape those parsers -- and everything
downstream of them -- expect. A mismatch is reported, never raised past this
module: that's the whole point of a gate that's meant to be run and read,
not a function whose failure crashes a script.

`verify_live_sources`'s `lines_fetcher` / `schedule_fetcher` take the same
RAW-payload contract as `underdog_lines.fetch_over_under_lines` /
`probable_pitchers.fetch_schedule` (dict in, no pre-parsing) -- NOT
`src.pipeline.refresh`'s `default_lines_fetcher`/`default_schedule_fetcher`,
which already parse. This module needs the raw shape to verify the parse
step itself, not its output.
"""

import datetime as _dt

from src.data.underdog_lines import flatten_lines, DEFAULT_STAT, DEFAULT_SPORT_ID
from src.data.probable_pitchers import parse_probable_starters

# Columns assemble_predictions / build_line_picks (src/predictions/tiering.py)
# actually read off the flattened lines frame downstream of this pipeline's
# join -- a verification that only checked "non-empty" would miss a partial
# shape change (e.g. Underdog renaming `stat_value` -> `line`). `game_status`
# and `live_event` are required because tiering._filter_pre_game filters on
# them directly; a rename there would silently drop every line.
REQUIRED_LINE_COLUMNS = [
    "pitcher", "stat_type", "line", "start_time",
    "over_american", "under_american", "game_status", "live_event",
]


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_underdog_lines(lines_fetcher, *, stat: str = DEFAULT_STAT,
                           sport_id: str = DEFAULT_SPORT_ID) -> dict:
    """
    Call `lines_fetcher()` (no args, returns the RAW Underdog
    over_under_lines payload -- the dict `underdog_lines.fetch_over_under_lines`
    returns, not an already-flattened DataFrame) and confirm it actually
    contains `stat` lines in the shape `flatten_lines` expects.

    Never raises: a fetch failure, a malformed payload, an empty result
    after filtering to `stat`, or a missing expected column are each
    reported as a failed check, not propagated -- they're exactly what this
    gate exists to catch.
    """
    try:
        payload = lines_fetcher()
    except Exception as exc:
        return _check("underdog_lines", False, f"fetch failed: {exc}")

    if not isinstance(payload, dict) or "over_under_lines" not in payload:
        got = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        return _check(
            "underdog_lines", False,
            f"payload missing top-level 'over_under_lines' list -- got {got}. "
            "Check sport_id and the endpoint response shape.",
        )

    try:
        flattened = flatten_lines(payload, stat=stat, sport_id=sport_id)
    except Exception as exc:
        return _check("underdog_lines", False, f"flatten_lines raised: {exc}")

    # Check emptiness before missing-columns: when no line matches `stat`,
    # flatten_lines legitimately returns pd.DataFrame([]) with NO columns at
    # all (not an empty frame carrying the right schema), since it builds the
    # frame from an empty `rows` list. Checking missing-columns first would
    # catch that as a generic "missing column(s)" failure and never surface
    # the actually-useful, specific "no '{stat}' lines found" detail -- which
    # is what a wrong sport_id or a renamed/mistyped stat key actually looks
    # like.
    if flattened.empty:
        return _check(
            "underdog_lines", False,
            f"no {stat!r} lines found in payload -- sport_id or stat key may be "
            "wrong (Underdog posts the slate progressively through the morning, "
            "so an early-morning empty result is also possible; re-check later "
            "before concluding the shape is broken).",
        )

    missing_cols = [c for c in REQUIRED_LINE_COLUMNS if c not in flattened.columns]
    if missing_cols:
        return _check(
            "underdog_lines", False,
            f"flattened frame missing expected column(s): {missing_cols}",
        )

    priced_ct = int(flattened["over_american"].notna().sum())
    pitcher_ct = flattened["pitcher"].nunique()
    return _check(
        "underdog_lines", True,
        f"{len(flattened)} {stat!r} line(s) for {pitcher_ct} pitcher(s) -- "
        f"{priced_ct} with a parsed over price.",
    )


def verify_schedule_hydration(schedule_fetcher, game_date) -> dict:
    """
    Call `schedule_fetcher(game_date)` (returns the RAW StatsAPI schedule
    payload -- the dict `probable_pitchers.fetch_schedule` returns) and
    confirm at least one game carries probable-pitcher hydration
    (`probablePitcher.id` + team) in the shape `parse_probable_starters`
    expects. Never raises -- see module docstring.
    """
    try:
        payload = schedule_fetcher(game_date)
    except Exception as exc:
        return _check("statsapi_schedule", False, f"fetch failed: {exc}")

    try:
        slate = parse_probable_starters(payload, game_date)
    except Exception as exc:
        return _check("statsapi_schedule", False, f"parse_probable_starters raised: {exc}")

    if slate.empty:
        return _check(
            "statsapi_schedule", False,
            f"no probable starters with hydrated probablePitcher.id found for {game_date} "
            "-- the hydrate param, date, or payload shape may be wrong (a true off-day "
            "with no MLB games is also possible; re-check on a day games are scheduled "
            "before concluding the shape is broken).",
        )

    missing_id = slate["pitcher"].isna().any()
    missing_team = slate["pitcher_team"].isna().any() or (slate["pitcher_team"] == "").any()
    if missing_id or missing_team:
        return _check(
            "statsapi_schedule", False,
            "parsed slate has row(s) missing pitcher id or team despite a non-empty result.",
        )

    return _check(
        "statsapi_schedule", True,
        f"{len(slate)} probable starter row(s) found for {game_date}, all with "
        "hydrated pitcher id + team.",
    )


def verify_live_sources(
    *, lines_fetcher, schedule_fetcher, game_date=None,
    stat: str = DEFAULT_STAT, sport_id: str = DEFAULT_SPORT_ID,
) -> dict:
    """
    Run both live-source checks and return a combined report:
    {"passed": bool, "checks": [...]}. `passed` is True only if every
    individual check passed. This is the function `refresh --dry-run` calls;
    it never raises and never writes anything -- callers are responsible for
    printing/logging the report and deciding what to do with a failure (per
    the spec: "stop and fix the parser/sport_id... non-optional before
    trusting any live output").
    """
    game_date = game_date or _dt.date.today().isoformat()

    checks = [
        verify_underdog_lines(lines_fetcher, stat=stat, sport_id=sport_id),
        verify_schedule_hydration(schedule_fetcher, game_date),
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
