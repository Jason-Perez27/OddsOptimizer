"""
Home-plate umpire K-tendency ingestion (spec 3, 2026-06-30).

Two data sources:

1. Historical tendency table -- data/raw/umpires/ump_tendency.csv
   Columns: ump_id (int), ump_name (str), k_factor (float, multiplicative
   1.0 = league-neutral), games_sampled (int, thin-sample guard).
   Maintained artifact -- refreshed periodically off-line, not a live call.
   Suitable sources: Baseball Reference umpire season pages, umpscorecard.com
   season summaries, or computed from box-score strikeout totals.

2. Day-of HP ump assignment -- MLB StatsAPI schedule with hydrate=officials.
   GET https://statsapi.mlb.com/api/v1/schedule
       ?gamePks={game_pk}&hydrate=officials

   Payload shape (relevant sub-tree):
     {"dates": [{"games": [{"officials": [
         {"official": {"id": 123, "fullName": "..."},
          "officialType": "Home Plate"}
     ]}]}]}

Degradation rules (identical to park_k_factor):
    * HP assignment not yet posted (fetch returns None) --> neutral + was_imputed
    * Ump not in tendency table                         --> neutral + was_imputed
    * games_sampled < MIN_GAMES (thin sample)           --> neutral + was_imputed

Injected-fetcher pattern: tests pass a lambda returning a pre-built dict.
"""

import json
import os
from typing import Callable, Optional, Tuple
from urllib.request import urlopen

import pandas as pd

_SCHEDULE_OFFICIALS_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?gamePks={game_pk}&hydrate=officials"
)

# Default path relative to this source file (src/data/umpires.py → ../../data/raw/)
_HERE = os.path.dirname(__file__)
DEFAULT_TENDENCY_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "data", "raw", "umpires", "ump_tendency.csv")
)

TENDENCY_COLUMNS = ["ump_id", "ump_name", "k_factor", "games_sampled"]

# Minimum number of games to treat a ump's k_factor as reliable.
MIN_GAMES = 50
NEUTRAL_K_FACTOR = 1.0


# ---------------------------------------------------------------------------
# Source 1: historical tendency CSV
# ---------------------------------------------------------------------------

def load_ump_tendency(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the umpire tendency CSV.

    Returns a DataFrame with TENDENCY_COLUMNS.  Returns an empty DataFrame
    (correct columns, zero rows) if the file does not exist -- the live
    refresh degrades gracefully to ump_k_factor=1.0 in that case.
    """
    path = path or DEFAULT_TENDENCY_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=TENDENCY_COLUMNS)

    df = pd.read_csv(path)
    for col in TENDENCY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["ump_id"] = pd.to_numeric(df["ump_id"], errors="coerce").astype("Int64")
    df["k_factor"] = pd.to_numeric(df["k_factor"], errors="coerce")
    df["games_sampled"] = pd.to_numeric(df["games_sampled"], errors="coerce").fillna(0)
    return df[TENDENCY_COLUMNS]


# ---------------------------------------------------------------------------
# Source 2: day-of HP ump assignment via StatsAPI
# ---------------------------------------------------------------------------

def _default_officials_fetcher(game_pk: int) -> dict:
    """Fetch the raw schedule+officials JSON from statsapi.mlb.com."""
    url = _SCHEDULE_OFFICIALS_URL.format(game_pk=game_pk)
    with urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def fetch_hp_umpire(
    game_pk: int,
    fetcher: Optional[Callable[[int], dict]] = None,
) -> Optional[int]:
    """
    Return the HP umpire's MLBAM official ID for game_pk, or None if the
    assignment is not yet posted or the fetch fails.

    Never raises: any network error, missing key, or bad payload → None.
    """
    if fetcher is None:
        fetcher = _default_officials_fetcher
    try:
        raw = fetcher(game_pk)
    except Exception:
        return None

    try:
        officials = raw["dates"][0]["games"][0].get("officials") or []
    except (KeyError, IndexError, TypeError):
        return None

    for entry in officials:
        office_type = (entry.get("officialType") or "").strip().lower()
        if office_type in ("home plate", "hp"):
            return entry.get("official", {}).get("id")
    return None


# ---------------------------------------------------------------------------
# Combined: k_factor lookup
# ---------------------------------------------------------------------------

def get_ump_k_factor(
    game_pk: int,
    tendency_df: pd.DataFrame,
    fetcher: Optional[Callable[[int], dict]] = None,
) -> Tuple[float, bool]:
    """
    Return ``(k_factor, was_imputed)`` for the HP umpire of game_pk.

    k_factor is a multiplicative modifier (1.0 = neutral).
    was_imputed is True when:
        - HP assignment not yet posted (fetch returns None)
        - Ump ID not found in tendency_df
        - Ump has < MIN_GAMES sample (too thin to trust)
        - tendency_df is empty or NaN k_factor

    Parameters
    ----------
    game_pk : int
        MLBAM game primary key.
    tendency_df : DataFrame
        Output of load_ump_tendency().
    fetcher : callable | None
        Injected fetcher for tests (lambda game_pk -> dict).
    """
    ump_id = fetch_hp_umpire(game_pk, fetcher=fetcher)
    if ump_id is None:
        return NEUTRAL_K_FACTOR, True

    if tendency_df is None or tendency_df.empty:
        return NEUTRAL_K_FACTOR, True

    match = tendency_df[tendency_df["ump_id"] == ump_id]
    if match.empty:
        return NEUTRAL_K_FACTOR, True

    row = match.iloc[0]
    games = float(row.get("games_sampled", 0) or 0)
    kf = row.get("k_factor")
    if games < MIN_GAMES or pd.isna(kf):
        return NEUTRAL_K_FACTOR, True

    return float(kf), False
