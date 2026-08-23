"""
Today's probable starting pitchers, via MLB-StatsAPI's schedule endpoint with
probable-pitcher hydration.

Design: docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md
(task #9). One external source, one ingestion module -- parallel to
src/data/pitcher_logs.py and src/data/underdog_lines.py: a thin live
fetch_schedule() plus a pure, testable parse_probable_starters().

Why this module exists: every other module in the project either operates on
historical Statcast data (training) or takes a ready-made predictions_df
(tiering). Nothing yet answers "who is starting today, for which team,
against which opponent" -- that's the genuine gap this module closes.

StatsAPI gives the pitcher's MLBAM id DIRECTLY (probablePitcher.id), so unlike
the PrizePicks side (which still needs tiering's name-based resolver), the
model side needs no name->id resolution here -- the join key the model uses
(`pitcher` = MLBAM id) is handed to us straight from the schedule payload.

Schedule payload shape (raw MLB Stats API `schedule` endpoint, NOT the
statsapi package's simplified statsapi.schedule() helper):
    {
      "dates": [
        {
          "games": [
            {
              "gamePk": 12345,
              "gameDate": "2026-06-27T23:05:00Z",
              "teams": {
                "away": {
                  "team": {"abbreviation": "NYY", ...},
                  "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole",
                                       "pitchHand": {"code": "R"}}
                },
                "home": {
                  "team": {"abbreviation": "BOS", ...},
                  "probablePitcher": {"id": 999999, "fullName": "Some Guy"}
                }
              }
            }
          ]
        }
      ]
    }
A side with no probablePitcher posted yet is simply absent from "teams.<side>"
-- never fabricated. `pitchHand` is only present when the schedule's
hydration happens to include it; when absent, `pitcher_throws` is None here
and is meant to be backfilled from that pitcher's own Statcast history
downstream (predict_features.py), never guessed in this module.
"""

import pandas as pd

try:
    import statsapi
except ImportError:  # MLB-StatsAPI needs network access; not available everywhere
    statsapi = None


# StatsAPI team abbreviations that differ from the Statcast/MLBAM team code
# the model's historical features key team identity by (the same
# home_team/away_team strings game_logs.py reads). This is a DISTINCT
# crosswalk from tiering.TEAM_CROSSWALK (PrizePicks->Statcast) -- different
# source, possibly different mismatches. Seeded with the known cases;
# extended as real data surfaces. An unmapped code passes through unchanged
# (never guessed) -- if it then fails to join downstream, that's visible via
# diagnostics/imputation, not a silent miss.
STATSAPI_TO_STATCAST_TEAM = {
    "WSH": "WAS",
    "CWS": "CHW",
}

# Fallback: the live StatsAPI schedule endpoint (with hydrate=probablePitcher)
# returns team blocks with only {id, name, link} -- no abbreviation field.
# When abbreviation is absent, look up by MLBAM team id. Confirmed 2026-06-29:
# id=145 -> Chicago White Sox (CWS), id=110 -> Baltimore Orioles (BAL).
# IDs are stable even across franchise renames/relocations.
MLBAM_TEAM_ID_TO_ABBREV = {
    108: "LAA",   # Los Angeles Angels
    109: "ARI",   # Arizona Diamondbacks
    110: "BAL",   # Baltimore Orioles
    111: "BOS",   # Boston Red Sox
    112: "CHC",   # Chicago Cubs
    113: "CIN",   # Cincinnati Reds
    114: "CLE",   # Cleveland Guardians
    115: "COL",   # Colorado Rockies
    116: "DET",   # Detroit Tigers
    117: "HOU",   # Houston Astros
    118: "KC",    # Kansas City Royals
    119: "LAD",   # Los Angeles Dodgers
    120: "WSH",   # Washington Nationals (crosswalk -> WAS)
    121: "NYM",   # New York Mets
    133: "OAK",   # Oakland Athletics
    134: "PIT",   # Pittsburgh Pirates
    135: "SD",    # San Diego Padres
    136: "SEA",   # Seattle Mariners
    137: "SF",    # San Francisco Giants
    138: "STL",   # St. Louis Cardinals
    139: "TB",    # Tampa Bay Rays
    140: "TEX",   # Texas Rangers
    141: "TOR",   # Toronto Blue Jays
    142: "MIN",   # Minnesota Twins
    143: "PHI",   # Philadelphia Phillies
    144: "ATL",   # Atlanta Braves
    145: "CWS",   # Chicago White Sox (crosswalk -> CHW)
    146: "MIA",   # Miami Marlins
    147: "NYY",   # New York Yankees
    158: "MIL",   # Milwaukee Brewers
}

SLATE_COLUMNS = [
    "pitcher", "pitcher_name", "pitcher_team", "opponent_team", "home_away",
    "game_pk", "game_date", "start_time", "pitcher_throws",
]


def to_statcast_team(code) -> str:
    """Map a StatsAPI team abbreviation to its Statcast/model-side code."""
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return ""
    code = str(code).upper()
    return STATSAPI_TO_STATCAST_TEAM.get(code, code)


def fetch_schedule(game_date, *, hydrate: str = "probablePitcher") -> dict:
    """
    The only network call in this module: a thin wrapper over the MLB Stats
    API schedule endpoint for a single date, hydrated with probable-pitcher
    info. Not exercised by tests -- pass a hand-built payload to
    parse_probable_starters() instead.
    """
    if statsapi is None:
        raise ImportError("MLB-StatsAPI is required to fetch a live schedule (pip install MLB-StatsAPI)")
    date_str = game_date.isoformat() if hasattr(game_date, "isoformat") else str(game_date)
    return statsapi.get("schedule", {"sportId": 1, "date": date_str, "hydrate": hydrate})


def _extract_pitch_hand(probable_pitcher: dict):
    hand = probable_pitcher.get("pitchHand")
    if isinstance(hand, dict):
        return hand.get("code")
    return None


def parse_probable_starters(schedule_payload: dict, game_date) -> pd.DataFrame:
    """
    Pure, testable core: turn a raw StatsAPI schedule payload into one row
    per probable starting pitcher (two rows per game when both sides are
    posted, one when only one is, zero when none are).

    Columns: pitcher (MLBAM id), pitcher_name, pitcher_team (Statcast code),
    opponent_team (Statcast code), home_away, game_pk, game_date, start_time,
    pitcher_throws (None if not present in the payload -- filled from history
    downstream, never guessed here).
    """
    rows = []
    dates = (schedule_payload or {}).get("dates", [])
    for date_entry in dates:
        for game in date_entry.get("games", []):
            game_pk = game.get("gamePk")
            start_time = game.get("gameDate")
            teams = game.get("teams", {})

            sides = {"home": teams.get("home", {}), "away": teams.get("away", {})}
            for home_away, side in sides.items():
                probable = side.get("probablePitcher")
                if not probable or probable.get("id") is None:
                    continue

                other_side = sides["away"] if home_away == "home" else sides["home"]
                # Live schedule: team block has {id, name, link} but no
                # abbreviation. Fall back to MLBAM_TEAM_ID_TO_ABBREV when
                # abbreviation is absent (confirmed 2026-06-29 diagnostic).
                team_block = side.get("team") or {}
                other_block = other_side.get("team") or {}
                pitcher_team_raw = (
                    team_block.get("abbreviation")
                    or MLBAM_TEAM_ID_TO_ABBREV.get(team_block.get("id"))
                )
                opponent_team_raw = (
                    other_block.get("abbreviation")
                    or MLBAM_TEAM_ID_TO_ABBREV.get(other_block.get("id"))
                )

                rows.append({
                    "pitcher": probable.get("id"),
                    "pitcher_name": probable.get("fullName"),
                    "pitcher_team": to_statcast_team(pitcher_team_raw),
                    "opponent_team": to_statcast_team(opponent_team_raw),
                    "home_away": home_away,
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "start_time": start_time,
                    "pitcher_throws": _extract_pitch_hand(probable),
                })

    if not rows:
        return pd.DataFrame(columns=SLATE_COLUMNS)
    return pd.DataFrame(rows, columns=SLATE_COLUMNS)
