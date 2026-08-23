"""
Historical corpus assembly for the walk-forward backtest (task #11).

Design: docs/design/specs/2026-06-27-baseline-validation-design.md,
section 1 "Historical corpus assembly". A walk-forward needs the WHOLE
league's pitch-level history for a date range, not one pitcher -- a large,
slow network pull, so it's windowed (default weekly), cached under
data/raw/statcast/ (gitignored, CSV -- no pyarrow), and resumable: a window
already cached on disk is loaded from disk, never re-fetched.

Resumability is deliberately simple: if a window's fetch raises (network
failure / rate limit), this function lets the exception propagate rather
than swallowing it -- but every window fetched *before* the failure is
already written to its cache file. Re-running build_corpus with the same
range picks up exactly where it left off, because those cache files are
found and loaded instead of re-fetched. No separate "resume" bookkeeping is
needed; the cache directory IS the resume state.

The fetched pitch-level frame is the input to the EXISTING
src.features.build_features.build_training_table path (which itself calls
aggregate_pitcher_games -> add_rolling_features -> add_opponent_features ->
add_park_factors) -- no feature code is reinvented here. Features are built
once on the full corpus; every rolling/opponent/park builder is strictly-
prior (.shift(1) before any cumulative/rolling aggregation), so building
them on the complete history is leakage-safe by construction. The
walk-forward (src/backtest/walk_forward.py) governs only the train/predict
CUTOFF, not feature re-derivation per step.

Starter filter: the evaluation set for the backtest is STARTS, not every
relief appearance. A start is the pitcher with the most batters faced in
their own team's half of a game_pk -- the same max-batters-faced proxy
src.features.opponent_features.build_team_game_logs() already uses to pick
a team-game's "starter" (there: the opposing team's representative pitcher
hand; here: the pitcher's OWN team/game). Openers/bullpen games can fool
this proxy (a documented limitation, mitigated by an optional
batters-faced floor).
"""

import os
from datetime import timedelta

import pandas as pd

DEFAULT_CACHE_DIR = os.path.join("data", "raw", "statcast")
DEFAULT_WINDOW_DAYS = 7

# Raw Statcast pitch-level rows have no fixed schema we want to assert here
# (pybaseball's column set has drifted over seasons) -- corpus.py stays
# schema-agnostic and lets build_features.build_training_table validate
# what it actually needs.


def _date_windows(start_date, end_date, window_days: int = DEFAULT_WINDOW_DAYS) -> list:
    """
    Split [start_date, end_date] (inclusive) into consecutive, non-overlapping
    (window_start, window_end) date-string pairs, each window_days long (the
    final window may be shorter). Both bounds are ISO date strings.
    """
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        return []

    windows = []
    cur = start
    while cur <= end:
        win_end = min(cur + timedelta(days=window_days - 1), end)
        windows.append((cur.isoformat(), win_end.isoformat()))
        cur = win_end + timedelta(days=1)
    return windows


def _cache_path(cache_dir: str, win_start: str, win_end: str) -> str:
    return os.path.join(cache_dir, f"{win_start}_{win_end}.csv")


def build_corpus(
    start_date,
    end_date,
    *,
    statcast_fetcher,
    cache_dir: str = DEFAULT_CACHE_DIR,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Pull the full league's pitch-level Statcast rows for [start_date,
    end_date], windowed (default weekly) with cache + resume, and
    concatenated into one pitch-level frame.

    `statcast_fetcher(window_start, window_end) -> pd.DataFrame` is injected
    (the real caller passes a thin wrapper over pybaseball.statcast) so this
    runs no-network in tests against a fixture fetcher.

    A window already cached under `cache_dir` is loaded from disk and the
    fetcher is NOT called for it. If `statcast_fetcher` raises for an
    uncached window, the exception propagates (every window fetched so far
    is already on disk -- re-running picks up from there; see module
    docstring).
    """
    os.makedirs(cache_dir, exist_ok=True)
    windows = _date_windows(start_date, end_date, window_days)

    frames = []
    for win_start, win_end in windows:
        path = _cache_path(cache_dir, win_start, win_end)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            continue

        df = statcast_fetcher(win_start, win_end)
        df.to_csv(path, index=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_starters(game_df: pd.DataFrame, min_batters_faced: int = None) -> pd.DataFrame:
    """
    Filter a per-pitcher-game table (one row per (pitcher, game_pk), as
    produced by src.features.game_logs.aggregate_pitcher_games or the full
    build_training_table) down to the evaluation set: one row per
    (pitcher_team, game_pk), the pitcher with the most batters_faced in that
    team's half of that game (the starter proxy).

    `min_batters_faced`, if given, additionally drops rows below that floor
    -- a guard against an opener/short-relief appearance being mistaken for
    a start by the max-BF proxy (documented limitation, not eliminated).
    """
    if game_df.empty:
        return game_df.copy()

    df = game_df.copy()
    idx = df.groupby(["pitcher_team", "game_pk"])["batters_faced"].idxmax()
    starters = df.loc[idx].sort_values(["game_date", "pitcher_team"]).reset_index(drop=True)

    if min_batters_faced is not None:
        starters = starters[starters["batters_faced"] >= min_batters_faced].reset_index(drop=True)

    return starters
