"""
Projected/confirmed lineup ingestion from MLB StatsAPI (spec 3, 2026-06-30).

Returns one row per batter in the game's projected lineup, with bat_side and
lineup_slot.  If no lineup is posted yet, returns an empty DataFrame and the
caller falls back to team-level opponent features.

API used: MLB StatsAPI schedule endpoint with hydrate=lineups:
    GET https://statsapi.mlb.com/api/v1/schedule
        ?gamePks={game_pk}&hydrate=lineups

Payload shape (relevant sub-tree):
    {"dates": [{"games": [{"gamePk": ..., "lineups": {
        "homePlayers": [{"id": 123, "fullName": "...",
                          "batSide": {"code": "R"},
                          "lineupPosition": 1}],
        "awayPlayers": [...]
    }}]}]}

lineup_source values:
    "confirmed"     -- both home and away lineups have >= MIN_LINEUP_SIZE batters
    "projected"     -- at least one side has batters but not both complete
    "team_fallback" -- no lineup data; caller uses team-level features

Injected-fetcher pattern (same as statsapi_boxscore.py): tests pass a
lambda that returns a pre-built dict instead of hitting the network.
"""

import json
from typing import Callable, Optional
from urllib.request import urlopen

import pandas as pd

_SCHEDULE_LINEUP_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?gamePks={game_pk}&hydrate=lineups"
)

LINEUP_COLUMNS = [
    "game_pk", "team_side", "batter_id", "batter_name",
    "bat_side", "lineup_slot", "lineup_source",
]

# Minimum number of batters per side we treat as a "full" lineup.
MIN_LINEUP_SIZE = 7


def _default_fetcher(game_pk: int) -> dict:
    """Fetch the raw schedule+lineups JSON for game_pk from statsapi.mlb.com."""
    url = _SCHEDULE_LINEUP_URL.format(game_pk=game_pk)
    with urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def parse_lineup(raw: dict, game_pk: int) -> pd.DataFrame:
    """
    Parse a StatsAPI schedule+lineups response (already fetched) into a
    batter-list DataFrame.  Returns an empty DataFrame (correct columns,
    zero rows, lineup_source absent) if no lineup data is present -- callers
    check .empty to detect the team_fallback case.

    Pure function (no network calls) -- testable with any hand-built dict.
    """
    rows = []
    try:
        game = raw["dates"][0]["games"][0]
    except (KeyError, IndexError):
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    lineups = game.get("lineups") or {}
    if not lineups:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    for side_key, side_label in (("homePlayers", "home"), ("awayPlayers", "away")):
        players = lineups.get(side_key) or []
        for p in players:
            batter_id = p.get("id")
            if batter_id is None:
                continue
            bat_side_raw = (p.get("batSide") or {}).get("code", "R")
            # StatsAPI sometimes returns lineupPosition, sometimes battingOrder
            # (encoded as 100, 200, ...) -- normalise both to 1-9.
            lineup_slot = p.get("lineupPosition") or p.get("battingOrder")
            if isinstance(lineup_slot, (int, float)) and lineup_slot > 9:
                lineup_slot = int(lineup_slot) // 100
            rows.append({
                "game_pk": int(game_pk),
                "team_side": side_label,
                "batter_id": int(batter_id),
                "batter_name": p.get("fullName", ""),
                "bat_side": bat_side_raw,
                "lineup_slot": int(lineup_slot) if lineup_slot else None,
                "lineup_source": "confirmed",  # overwritten below
            })

    if not rows:
        return pd.DataFrame(columns=LINEUP_COLUMNS)

    df = pd.DataFrame(rows)
    home_count = (df["team_side"] == "home").sum()
    away_count = (df["team_side"] == "away").sum()
    if home_count >= MIN_LINEUP_SIZE and away_count >= MIN_LINEUP_SIZE:
        source = "confirmed"
    elif home_count > 0 or away_count > 0:
        source = "projected"
    else:
        source = "team_fallback"
    df["lineup_source"] = source
    return df[LINEUP_COLUMNS]


def get_lineup(
    game_pk: int,
    fetcher: Optional[Callable[[int], dict]] = None,
) -> pd.DataFrame:
    """
    Fetch and parse the lineup for game_pk.

    Returns a batter-list DataFrame (LINEUP_COLUMNS).  If the lineup is not
    yet posted or the fetch fails, returns an empty DataFrame -- the caller
    treats this as lineup_source='team_fallback' and uses team-level features.
    Never raises.
    """
    if fetcher is None:
        fetcher = _default_fetcher
    try:
        raw = fetcher(game_pk)
    except Exception:
        return pd.DataFrame(columns=LINEUP_COLUMNS)
    return parse_lineup(raw, game_pk)
