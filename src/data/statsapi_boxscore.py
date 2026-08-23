"""
Earned-runs label ingestion from the MLB StatsAPI pitching boxscore.

WHY a separate source: Statcast pitch data records plate-appearance outcomes
(strikeout, walk, field_out, ...) but does NOT distinguish earned from
unearned runs -- that determination is made by the official scorer after the
game and is only exposed in box-score endpoints.  PrizePicks posts *Earned*
Runs Allowed, so `game_logs.aggregate_pitcher_games` cannot supply the label;
we must hit the boxscore API.

API endpoint used:
    https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore
    -> liveData.boxscore.teams.{home|away}.pitchers  (list of player IDs)
    -> liveData.boxscore.teams.{home|away}.players["ID{player_id}"]
           .stats.pitching.earnedRuns

The function returns one row per starting/relief pitcher who appears in the
boxscore for that game, keyed on (pitcher, game_pk) to match the partition
contract used throughout the rest of the pipeline.

Injected fetcher pattern (same as src/data/pitcher_logs.py and
src/data/probable_pitchers.py): the `fetcher` argument defaults to a real
HTTP GET against statsapi.mlb.com; tests inject a lambda that returns a
pre-built dict, so no network calls are needed in CI.
"""

import json
from typing import Callable, Optional
from urllib.request import urlopen

import pandas as pd

_BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

OUTPUT_COLUMNS = ["pitcher", "game_pk", "earned_runs"]


def _default_fetcher(game_pk: int) -> dict:
    """Fetch the raw boxscore JSON for game_pk from statsapi.mlb.com."""
    url = _BOXSCORE_URL.format(game_pk=game_pk)
    with urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def get_pitcher_earned_runs_by_game(
    game_pk: int,
    fetcher: Optional[Callable[[int], dict]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per pitcher in `game_pk` who has an
    earnedRuns entry in the pitching boxscore.

    Columns: pitcher (int MLBAM ID), game_pk (int), earned_runs (int).

    If the boxscore is missing, malformed, or the game has not yet been
    played (no pitching stats present), an empty DataFrame with the correct
    columns is returned -- the caller treats this the same as a void/scratch
    start.
    """
    if fetcher is None:
        fetcher = _default_fetcher

    try:
        data = fetcher(game_pk)
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []
    teams = data.get("teams", {})
    for side in ("home", "away"):
        team = teams.get(side, {})
        players = team.get("players", {})
        pitcher_ids = team.get("pitchers", [])
        for pid in pitcher_ids:
            player_key = f"ID{pid}"
            player = players.get(player_key, {})
            pitching = player.get("stats", {}).get("pitching", {})
            if "earnedRuns" not in pitching:
                continue
            rows.append({
                "pitcher": int(pid),
                "game_pk": int(game_pk),
                "earned_runs": int(pitching["earnedRuns"]),
            })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def main():
    """CLI entry point: print earned-runs boxscore for a game_pk."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch earnedRuns per pitcher from the MLB StatsAPI boxscore."
    )
    parser.add_argument("game_pk", type=int, help="MLB game_pk (e.g. 747066)")
    args = parser.parse_args()

    df = get_pitcher_earned_runs_by_game(args.game_pk)
    if df.empty:
        print(f"No earned-runs data found for game_pk={args.game_pk}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
