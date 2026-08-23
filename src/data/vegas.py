"""
src/data/vegas.py — game total / odds context features.

Part 2 of Spec ④ (2026-06-30): weather + Vegas context + boosted ensemble.

Source: ESPN's public scoreboard API (no API key required).
  https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard
  ?dates={YYYYMMDD}

The response includes an `odds` array per competition with:
  - overUnder   : game total (e.g. 8.5)
  - homeTeamOdds.moneyLine : home team moneyline (negative = favourite)
  - awayTeamOdds.moneyLine : away team moneyline

Features (VEGAS_CANDIDATE_COLUMNS):
  game_total       – over/under for the game (e.g. 8.5 runs)
  is_favorite      – 1.0 if this team's pitcher is on the favoured side,
                     0.0 if underdog, NaN if moneyline unavailable
  team_total_for   – NaN (no free source provides per-team totals without
                     a paid feed; documented limitation below)
  team_total_against – NaN (same)

Limitation note (documented per spec's "if no free source is reliable,
defer and document" rule):
  Per-team run totals are a distinct betting market from game totals and are
  not carried in ESPN's public scoreboard endpoint.  The-Odds-API offers
  team-totals but requires a (free-tier) API key.  Because the project uses
  no-key-required free sources, team_total_for and team_total_against are
  left as NaN and excluded from the walk-forward backtest until a keyless
  source is identified.  The spec documents this as an acceptable scope
  reduction; game_total + is_favorite still carry the opportunity/length
  context the spec targets.

Historical data availability:
  ESPN's scoreboard API does return historical dates' data (the `dates=`
  parameter accepts past YYYYMMDD values), so game_total can be used in
  walk-forward backtests.  However, ESPN's odds represent the CLOSING line,
  not the opening or pre-game morning line.  For a pre-game discipline, the
  closing line is the best freely available proxy; in production the refresh
  runs in the morning and will pick up opening lines (or whatever ESPN has
  posted at that hour).

vegas_was_imputed=True when:
  - the ESPN fetch fails or returns no matching game
  - odds are not yet posted for the requested date
  - the home_team abbreviation cannot be matched to an ESPN event
"""

import json
from urllib.request import urlopen

import numpy as np

# ---------------------------------------------------------------------------
# Public column names
# ---------------------------------------------------------------------------

VEGAS_CANDIDATE_COLUMNS = ["game_total", "is_favorite"]

# Columns present in the returned dict but always NaN (see docstring).
VEGAS_DEFERRED_COLUMNS = ["team_total_for", "team_total_against"]

# ---------------------------------------------------------------------------
# ESPN team-abbreviation mapping
# ---------------------------------------------------------------------------
# ESPN uses slightly different abbreviations in some cases.  This maps our
# system's abbreviations (matching park_factors.py) to ESPN's.

_OUR_TO_ESPN = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD":  "SD",  "SEA": "SEA",
    "SF":  "SF",  "STL": "STL", "TB":  "TB",  "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH",
}

_ESPN_API_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    "?dates={date_nodash}"
)

# ---------------------------------------------------------------------------
# Return shapes
# ---------------------------------------------------------------------------

def _imputed_result() -> dict:
    return {
        "game_total": np.nan,
        "is_favorite": np.nan,
        "team_total_for": np.nan,
        "team_total_against": np.nan,
        "vegas_was_imputed": True,
    }


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def _default_fetcher(url: str) -> dict:
    with urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def fetch_espn_scoreboard(game_date: str, *, fetcher=None) -> dict:
    """
    Fetch the ESPN MLB scoreboard JSON for game_date ("YYYY-MM-DD").

    Returns the raw dict on success, empty dict on any failure.
    Injected fetcher(url) -> dict for tests.
    """
    _fetch = fetcher or _default_fetcher
    date_nodash = game_date.replace("-", "")
    url = _ESPN_API_URL.format(date_nodash=date_nodash)
    try:
        return _fetch(url)
    except Exception:
        return {}


def _parse_odds_from_competition(competition: dict) -> tuple:
    """
    Extract (over_under, home_ml, away_ml) from a competition dict.
    Returns (None, None, None) if odds are not present.
    """
    odds_list = competition.get("odds", [])
    if not odds_list:
        return None, None, None
    odds = odds_list[0]  # use the first provider
    over_under = odds.get("overUnder")
    home_ml = None
    away_ml = None
    home_team_odds = odds.get("homeTeamOdds", {})
    away_team_odds = odds.get("awayTeamOdds", {})
    if home_team_odds:
        home_ml = home_team_odds.get("moneyLine")
    if away_team_odds:
        away_ml = away_team_odds.get("moneyLine")
    return over_under, home_ml, away_ml


def get_game_odds(
    home_team: str,
    game_date: str,
    pitcher_home_away: str = "home",
    *,
    fetcher=None,
) -> dict:
    """
    Return Vegas odds context for a specific game.

    Matches the correct ESPN event by home_team abbreviation within the
    given game_date's scoreboard response.

    Parameters
    ----------
    home_team : str
        Home-team abbreviation (e.g. "NYY").
    game_date : str
        ISO date "YYYY-MM-DD".
    pitcher_home_away : str
        "home" or "away" — determines is_favorite from the perspective of
        the pitcher's team.
    fetcher : callable, optional
        Injected for tests: ``fetcher(url) -> dict``.

    Returns
    -------
    dict with keys: game_total, is_favorite, team_total_for,
    team_total_against, vegas_was_imputed.
    """
    espn_home = _OUR_TO_ESPN.get(home_team, home_team)
    raw = fetch_espn_scoreboard(game_date, fetcher=fetcher)
    if not raw:
        return _imputed_result()

    events = raw.get("events", [])
    for event in events:
        competitions = event.get("competitions", [])
        for comp in competitions:
            competitors = comp.get("competitors", [])
            home_comp = next(
                (c for c in competitors if c.get("homeAway") == "home"),
                None,
            )
            if home_comp is None:
                continue
            team_abbr = (
                home_comp.get("team", {}).get("abbreviation", "")
            )
            if team_abbr != espn_home:
                continue

            # Found the matching game
            over_under, home_ml, away_ml = _parse_odds_from_competition(comp)
            if over_under is None:
                return _imputed_result()

            # is_favorite: negative moneyline = favourite in American odds
            if pitcher_home_away == "home":
                ml = home_ml
            else:
                ml = away_ml

            if ml is not None:
                try:
                    is_fav = 1.0 if float(ml) < 0 else 0.0
                except (TypeError, ValueError):
                    is_fav = np.nan
            else:
                is_fav = np.nan

            return {
                "game_total": float(over_under),
                "is_favorite": is_fav,
                "team_total_for": np.nan,      # not available from ESPN
                "team_total_against": np.nan,  # not available from ESPN
                "vegas_was_imputed": False,
            }

    # No matching event found
    return _imputed_result()
