"""
Overnight/pre-game tick log for Underdog Fantasy pitcher prop lines.

Feeds the closing-line resolver and CLV metrics in src/evaluation/clv.py
(2026-08 CLV feature). This module is PURELY ADDITIVE: it never reads,
writes, or otherwise touches line_picks.csv, the predictions partition
(`data/processed/predictions/...`), or the frozen morning decision snapshot.
It has no opinion about, and never triggers, a model run.

Verified facts (trust these, do not re-research):
- GET https://api.underdogfantasy.com/beta/v6/over_under_lines?sport_id=MLB
  returns EVERY MLB stat in one ~0.5s / ~1.4MB response. `poll_once` calls
  it exactly once per invocation and fans out to every stat in
  MLB_PITCHER_STATS locally -- never one HTTP call per stat.
- Each of a line's two `options` entries carries its own `updated_at`
  timestamp (src/data/underdog_lines.flatten_lines surfaces these as
  `over_updated_at` / `under_updated_at`). A price move shows up as a NEW
  updated_at on one or both sides -- this is the change-detection signal,
  never a full-snapshot diff.
- Measured 2026-08-23 at T-12h to first pitch: 26 strikeout lines across 15
  games, option `updated_at` ages ranging 16-442 minutes (median ~132).
  Lines move overnight and movement accelerates near first pitch.
- First-pitch times can spread ~5-6 hours across one slate. "Closing" is
  therefore per GAME, never per slate (see src/evaluation/clv.py).
- `live_event` (bool, line-level) and `games[].status` (game-level, exposed
  here as `game_status`) mark when a game goes live.

Design:
- Reuses src.data.underdog_lines.fetch_over_under_lines / flatten_lines for
  every byte of Underdog payload parsing -- this module adds NO second
  parser. Its own logic is: fan-out across stats from one fetched payload,
  ET game-date partitioning, dedup/append, and idempotent CSV writing.
- One row per genuinely NEW (over_under_id, over_updated_at, under_updated_at)
  triple. Running poll_once() twice against an unchanged market writes
  nothing the second time -- the log only grows on real price movement, so a
  day's log stays on the order of a few hundred rows, not one row per poll.
- Off-day / no games: poll_once() still calls the feed once (that's how you
  find out it's an off-day), writes nothing, and returns a summary dict with
  n_rows_written=0. This mirrors src.pipeline.refresh's EmptySlateError
  discipline in spirit -- an empty slate is a clean no-op, not a failure --
  but there is no exception type here at all: 0 games is never fatal for a
  ticker, only for a same-day prediction run.
- CSV, not parquet, to match this repo's existing raw/processed file
  convention (every other module in src/data + src/pipeline writes CSV).
- Partition key is the GAME's date in US Eastern civil time, derived from
  each line's `start_time` (Underdog's `scheduled_at`, UTC) -- NOT the date
  the poll happened to run on. A 23:10Z first pitch and a 03:00Z poll (next
  UTC calendar day) both belong to the same ET game date; see `et_date_str`
  below. Eastern time is computed by hand (see the DST helpers) rather than
  via `zoneinfo`/`tzdata`: `zoneinfo` is stdlib but needs an IANA tz database
  that Windows -- this project's primary deployment target, see
  scripts/install_cadences.ps1 -- does not ship; pulling in the `tzdata`
  PyPI package just to look up one long-stable, well-documented US civil
  rule (2nd Sunday of March 02:00 local -> 1st Sunday of November 02:00
  local, in effect since 2007) would be a new dependency for no real benefit.

Hardening for the GitHub Actions poller (2026-08, cloud-hosted cadence E --
see .github/workflows/tick-poller.yml and docs/runbook_go_live.md):
- `poll_at` is the actual wall-clock time the fetch SUCCEEDED, captured
  AFTER `fetch_over_under_lines` returns, never before it and never derived
  from a cron schedule time. Actions cron fires 15-60 minutes late under
  load (documented GitHub behavior) -- recording an assumed/scheduled time
  instead of the real one would silently corrupt every downstream
  `minutes_before_first_pitch` / `close_quality` computation in
  src/evaluation/clv.py.
- The fetch is wrapped in a bounded retry (`_fetch_with_retry`): up to 3
  retries beyond the first attempt (4 attempts total), exponential backoff
  2s/4s/8s between them. A dropped poll is unrecoverable -- unlike
  Statcast/boxscores/probables/Vegas odds, a line movement not observed is
  gone forever, so it's worth spending a few seconds retrying a transient
  network blip. A 403, however, means Underdog's endpoint shape or bot
  protection changed (the same permanent failure `fetch_over_under_lines`
  already raises a clear RuntimeError for) -- that is NOT retried; it needs
  a human, and retrying it would just burn Actions minutes for nothing.
"""

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.data.underdog_lines import (
    MLB_PITCHER_STATS,
    american_to_prob,
    fetch_over_under_lines,
    flatten_lines,
    no_vig_two_way,
)

# ---------------------------------------------------------------------------
# Paths / schema
# ---------------------------------------------------------------------------

# Default output root. Overridable via poll_once(output_dir=...) / the CLI's
# --output-dir flag -- the GitHub Actions poller (see .github/workflows/
# tick-poller.yml) points this at a checked-out `data-ticks` branch instead
# of the local data/raw/ tree. Local runs never need to pass it.
DEFAULT_TICKS_ROOT = os.path.join("data", "raw", "underdog_ticks")

TICK_COLUMNS = [
    "poll_at", "over_under_id", "projection_id", "player_id", "pitcher",
    "stat_type", "line", "over_american", "under_american",
    "over_payout_multiplier", "under_payout_multiplier",
    "p_over_implied", "p_under_implied", "p_market",
    "over_updated_at", "under_updated_at",
    "game_id", "game_title", "start_time", "game_status", "live_event", "status",
]

# The dedupe key: a row is "genuinely new" only if this triple hasn't been
# written for today's game date before. Deliberately excludes poll_at, so
# running the poller twice within the same cycle -- or twice in one minute,
# per the idempotency requirement -- writes zero new rows the second time.
DEDUPE_KEY_COLUMNS = ["over_under_id", "over_updated_at", "under_updated_at"]


# ---------------------------------------------------------------------------
# US Eastern civil-time helpers (stdlib only -- see module docstring)
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th occurrence (1-indexed) of `weekday` (Mon=0..Sun=6) in year/month."""
    d = date(year, month, 1)
    first = d + timedelta(days=(weekday - d.weekday()) % 7)
    return first + timedelta(days=7 * (n - 1))


def _us_dst_start_utc(year: int) -> datetime:
    """2nd Sunday of March, 02:00 EST (UTC-5) -> the UTC instant clocks spring forward."""
    d = _nth_weekday_of_month(year, 3, 6, 2)
    return datetime(d.year, d.month, d.day, 7, 0, tzinfo=timezone.utc)  # 02:00 EST = 07:00 UTC


def _us_dst_end_utc(year: int) -> datetime:
    """1st Sunday of November, 02:00 EDT (UTC-4) -> the UTC instant clocks fall back."""
    d = _nth_weekday_of_month(year, 11, 6, 1)
    return datetime(d.year, d.month, d.day, 6, 0, tzinfo=timezone.utc)  # 02:00 EDT = 06:00 UTC


def _et_utc_offset_hours(utc_dt: datetime) -> int:
    """-4 (EDT) if `utc_dt` falls within US daylight saving, else -5 (EST)."""
    year = utc_dt.year
    if _us_dst_start_utc(year) <= utc_dt < _us_dst_end_utc(year):
        return -4
    return -5


def _parse_utc(ts) -> datetime:
    """Parse an ISO-8601 UTC timestamp (Underdog's 'Z'-suffixed style, or with an offset)."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_eastern(ts) -> datetime:
    """Convert a UTC timestamp (str or datetime) to a naive US-Eastern local datetime."""
    utc_dt = _parse_utc(ts)
    offset = _et_utc_offset_hours(utc_dt)
    return (utc_dt + timedelta(hours=offset)).replace(tzinfo=None)


def et_date_str(ts) -> str:
    """The US-Eastern calendar date (YYYY-MM-DD) a UTC timestamp falls on."""
    return to_eastern(ts).date().isoformat()


# ---------------------------------------------------------------------------
# Tick log paths / dedupe
# ---------------------------------------------------------------------------

def _log_path(ticks_root: str, game_date: str) -> str:
    return os.path.join(ticks_root, f"game_date={game_date}", "ticks.csv")


def _existing_dedupe_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            keys.add(tuple((row.get(c) or "") for c in DEDUPE_KEY_COLUMNS))
    return keys


def _append_new_ticks(ticks_root: str, game_date: str, ticks: list) -> int:
    """Append only the ticks whose dedupe key is new (today's file + this batch). Returns count written."""
    path = _log_path(ticks_root, game_date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_keys = _existing_dedupe_keys(path)

    new_rows = []
    seen_this_call = set()
    for tick in ticks:
        key = tuple(
            ("" if tick.get(c) is None or (isinstance(tick.get(c), float) and pd.isna(tick.get(c)))
             else str(tick.get(c)))
            for c in DEDUPE_KEY_COLUMNS
        )
        if key in existing_keys or key in seen_this_call:
            continue
        seen_this_call.add(key)
        new_rows.append(tick)

    if not new_rows:
        return 0

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TICK_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


# ---------------------------------------------------------------------------
# Bounded retry (GitHub Actions hardening -- see module docstring)
# ---------------------------------------------------------------------------

# Delays (seconds) before retry attempts 1, 2, 3 -- i.e. up to 3 retries
# beyond the first attempt, 4 attempts total. Three delay values is a
# deliberate signal of "3 retries", not "3 attempts" (which would only need
# two gaps) -- every stated delay is actually used.
_RETRY_DELAYS = (2, 4, 8)


def _is_permanent_403(exc: Exception) -> bool:
    """
    True if `exc` represents a 403 -- Underdog's endpoint shape or bot
    protection changed, which needs a human, not a retry.

    Covers both cases seen in this codebase:
    - fetch_over_under_lines raises a plain RuntimeError whose message
      contains "403" (its own permanent-failure convention).
    - A defensive fallback for a `requests`-style exception carrying a
      `.response.status_code == 403` (in case a future `fetcher=` override
      raises requests.HTTPError directly instead).
    """
    if isinstance(exc, RuntimeError) and "403" in str(exc):
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 403:
        return True
    return False


def _fetch_with_retry(fetch, *, retry_delays=_RETRY_DELAYS, sleep_fn=None):
    """
    Call `fetch()` (zero-arg callable) and return its result, retrying
    transient failures up to len(retry_delays) times with the given
    backoff. A 403 (`_is_permanent_403`) is never retried -- it's re-raised
    immediately, since it means the endpoint needs a human look, not
    another attempt.

    `sleep_fn`: injectable sleep (defaults to time.sleep) so tests never
    actually wait out the backoff.
    """
    _sleep = sleep_fn or time.sleep
    last_exc = None
    for attempt in range(len(retry_delays) + 1):
        try:
            return fetch()
        except Exception as exc:
            if _is_permanent_403(exc):
                raise
            last_exc = exc
            if attempt < len(retry_delays):
                _sleep(retry_delays[attempt])
    raise last_exc


# ---------------------------------------------------------------------------
# The poller
# ---------------------------------------------------------------------------

def _build_tick(row: "pd.Series", poll_at: str) -> dict:
    over_am = row.get("over_american")
    under_am = row.get("under_american")
    p_over_implied = p_under_implied = p_market = None
    if pd.notna(over_am) and pd.notna(under_am):
        p_over_implied = american_to_prob(over_am)
        p_under_implied = american_to_prob(under_am)
        p_market = no_vig_two_way(p_over_implied, p_under_implied)

    return {
        "poll_at":                 poll_at,
        "over_under_id":           row.get("over_under_id"),
        "projection_id":           row.get("projection_id"),
        "player_id":               row.get("player_id"),
        "pitcher":                 row.get("pitcher"),
        "stat_type":               row.get("stat_type"),
        "line":                    row.get("line"),
        "over_american":           over_am,
        "under_american":          under_am,
        "over_payout_multiplier":  row.get("over_payout_multiplier"),
        "under_payout_multiplier": row.get("under_payout_multiplier"),
        "p_over_implied":          p_over_implied,
        "p_under_implied":         p_under_implied,
        "p_market":                p_market,
        "over_updated_at":         row.get("over_updated_at"),
        "under_updated_at":        row.get("under_updated_at"),
        "game_id":                 row.get("game_id"),
        "game_title":              row.get("game_title"),
        "start_time":              row.get("start_time"),
        "game_status":             row.get("game_status"),
        "live_event":              row.get("live_event"),
        "status":                  row.get("status"),
    }


def poll_once(output_dir: str = DEFAULT_TICKS_ROOT, stats=None, now=None, *, fetcher=None, sleep_fn=None) -> dict:
    """
    Poll Underdog's over_under_lines feed exactly ONCE (through a bounded
    retry, see `_fetch_with_retry`), flatten it for every stat key in
    `stats` (default: every key in MLB_PITCHER_STATS), and append any
    genuinely new ticks to each affected game date's
    `{output_dir}/game_date=YYYY-MM-DD/ticks.csv`.

    `output_dir`: the ticks-log root (default DEFAULT_TICKS_ROOT =
    "data/raw/underdog_ticks"); tests point this at a tmp_path. The GitHub
    Actions poller overrides this (via the CLI's --output-dir) to a
    checked-out `data-ticks` branch working copy -- see
    .github/workflows/tick-poller.yml.
    `stats`: iterable of Underdog stat keys to fan out to; None = all of
    MLB_PITCHER_STATS.
    `now`: injectable clock (a datetime); None = real UTC now. Matches this
    repo's dependency-injected-clock testing convention (src.evaluation.grading).
    IMPORTANT: `now`/real-clock time is sampled AFTER the fetch (through all
    retries) succeeds, never before -- `poll_at` must reflect when the data
    was actually observed, not when polling started or was scheduled. This
    matters most for the Actions poller, where cron can fire 15-60 minutes
    late and a retry can add several more seconds; using a pre-fetch or
    assumed/schedule time would silently corrupt every downstream
    `minutes_before_first_pitch` / `close_quality` computation in
    src/evaluation/clv.py.
    `fetcher`: injectable zero-arg callable returning the raw payload dict;
    None = the real fetch_over_under_lines(). Tests never touch the network.
    `sleep_fn`: injectable sleep for the retry backoff; None = real
    time.sleep. Tests never actually wait.

    Returns {"polled_at", "n_lines_seen", "n_rows_written", "n_games"}.
    Off-day / no games: still returns this summary (n_rows_written=0) and
    never raises -- a poller has no "fatal empty slate" case, unlike a
    same-day prediction run. A permanent 403 (`_is_permanent_403`) or an
    exhausted retry budget both propagate as a raised exception -- that IS
    fatal for this invocation, since it means no data was observed at all.
    """
    stat_keys = list(stats) if stats else list(MLB_PITCHER_STATS)
    _fetch = fetcher or fetch_over_under_lines
    payload = _fetch_with_retry(_fetch, sleep_fn=sleep_fn)

    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    poll_at = now_dt.isoformat()

    n_lines_seen = 0
    rows_by_date = {}
    game_ids_seen = set()

    for stat in stat_keys:
        df = flatten_lines(payload, stat=stat)
        if df.empty:
            continue
        n_lines_seen += len(df)
        for _, row in df.iterrows():
            start_time = row.get("start_time")
            if start_time is None or (isinstance(start_time, float) and pd.isna(start_time)):
                continue  # can't partition without a game date -- skip, never guess
            try:
                game_date = et_date_str(start_time)
            except (ValueError, TypeError):
                continue

            game_id = row.get("game_id")
            if game_id is not None:
                game_ids_seen.add(game_id)

            rows_by_date.setdefault(game_date, []).append(_build_tick(row, poll_at))

    n_rows_written = 0
    for game_date, ticks in rows_by_date.items():
        n_rows_written += _append_new_ticks(output_dir, game_date, ticks)

    return {
        "polled_at": poll_at,
        "n_lines_seen": n_lines_seen,
        "n_rows_written": n_rows_written,
        "n_games": len(game_ids_seen),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Poll Underdog once and append genuinely new pitcher-prop "
                     "line ticks to today's game-date tick log(s)."
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_TICKS_ROOT,
        help="Ticks-log root to write under (default: data/raw/underdog_ticks). "
             "The GitHub Actions poller points this at a data-ticks branch checkout.",
    )
    parser.add_argument(
        "--stats", nargs="*", default=None,
        help="Underdog stat keys to poll (default: all of MLB_PITCHER_STATS)",
    )
    args = parser.parse_args()

    summary = poll_once(args.output_dir, stats=args.stats)
    print(
        f"Polled at {summary['polled_at']}: saw {summary['n_lines_seen']} line(s) "
        f"across {summary['n_games']} game(s); wrote {summary['n_rows_written']} new tick row(s)."
    )
    # Machine-readable line for the GitHub Actions poller's run-summary step
    # (see .github/workflows/tick-poller.yml) -- avoids parsing the
    # human-readable line above. n_rows_written IS "lines whose updated_at
    # changed since the previous poll": a row is only ever written when the
    # dedupe key (over_under_id, over_updated_at, under_updated_at) is new.
    print(f"SUMMARY_JSON:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
