"""
park_k_factor: a ballpark strikeout index -- the park's K-rate relative to
league average, computed across ALL pitchers who pitched in that park (both
the home team's and the visiting team's pitchers), built on top of
game_logs.py's per-pitcher-game table.

Park is keyed by team abbreviation (the home team for a given game), not
stadium name -- an acceptable simplification since every current MLB team
has its own home park (no shared-park edge cases as of this writing).

Leakage guardrail: like rolling_features.py/opponent_features.py, a game's
park_k_factor uses only games played at that park strictly before this
game's date (shift-before-aggregate), and is benchmarked against the
league-wide K-rate over that same strictly-prior period (so early in a
season, both the park-specific and league-wide denominators are small,
which is exactly when we fall back to the static table below instead).

Two-tier approach:
1. Once at least MIN_PARK_GAMES_FOR_COMPUTED games have been played at a
   park so far this season (strictly before today), park_k_factor is
   COMPUTED: 100 * (park's prior season-to-date K-rate) / (league-wide
   prior season-to-date K-rate). 100 = league average that season to date;
   >100 = more strikeouts than average at that park, <100 = fewer.
2. Before that threshold is reached (cold start -- e.g. the first couple of
   weeks of a new season, where neither the park-specific nor league-wide
   sample is reliable yet), park_k_factor falls back to STATIC_PARK_FACTORS,
   a small reference table of approximate park strikeout indices.

STATIC_PARK_FACTORS source: illustrative multi-year-blended K-park-factor
index values (100 = league average), in the style of the park factor guts
tables published by outlets like FanGraphs / Baseball Savant. These are
placeholder values for cold-start fallback ONLY -- not fetched live -- and
should be refreshed from an authoritative public source each offseason.
Teams not present in the table (e.g. a future relocation/expansion club)
fall back to 100 (league average) rather than raising.
"""

import numpy as np
import pandas as pd

MIN_PARK_GAMES_FOR_COMPUTED = 15  # roughly 2-3 homestands worth of games

STATIC_PARK_FACTORS = {
    "ARI": 99, "ATL": 101, "BAL": 98, "BOS": 97, "CHC": 100,
    "CWS": 102, "CIN": 100, "CLE": 101, "COL": 94, "DET": 99,
    "HOU": 101, "KC": 98, "LAA": 100, "LAD": 103, "MIA": 102,
    "MIL": 100, "MIN": 100, "NYM": 101, "NYY": 99, "OAK": 103,
    "PHI": 100, "PIT": 99, "SD": 102, "SEA": 104, "SF": 103,
    "STL": 99, "TB": 102, "TEX": 97, "TOR": 99, "WSH": 99,
}
DEFAULT_STATIC_PARK_FACTOR = 100

FEATURE_COLUMNS = ["park_k_factor"]


def _prior_cumsum(series: pd.Series) -> pd.Series:
    return series.shift(1).cumsum()


def _build_park_game_log(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (park_team, game_pk): totals strikeouts/batters_faced summed
    across every pitcher who pitched in that game (both the home and away
    team's pitchers), since park_k_factor describes the park itself, not
    either team specifically.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["park_team"] = np.where(
        df["home_away"] == "home", df["pitcher_team"], df["opponent_team"]
    )

    pg = (
        df.groupby(["park_team", "game_pk"], sort=False)
        .agg(
            game_date=("game_date", "first"),
            strikeouts=("strikeouts", "sum"),
            batters_faced=("batters_faced", "sum"),
        )
        .reset_index()
    )
    pg = pg.sort_values(["park_team", "game_date"]).reset_index(drop=True)
    pg["_season"] = pg["game_date"].dt.year
    return pg


def add_park_factors(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add park_k_factor to the per-pitcher-game table. Each row's value
    describes the park that game was played in (the home team's park),
    using only games at that park strictly before this row's game_date.
    """
    if game_df.empty:
        out = game_df.copy()
        out["park_k_factor"] = pd.Series(dtype="float64")
        return out

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["park_team"] = np.where(
        df["home_away"] == "home", df["pitcher_team"], df["opponent_team"]
    )

    pg = _build_park_game_log(game_df)

    # ---- park-specific prior season-to-date K-rate ----
    park_grp = pg.groupby(["park_team", "_season"], sort=False)
    pg["_prior_park_k"] = park_grp["strikeouts"].transform(_prior_cumsum)
    pg["_prior_park_bf"] = park_grp["batters_faced"].transform(_prior_cumsum)
    pg["_prior_park_games"] = park_grp.cumcount()

    # ---- league-wide prior season-to-date K-rate (all parks, same dates) ----
    daily = (
        pg.groupby(["_season", "game_date"], sort=False)
        .agg(strikeouts=("strikeouts", "sum"), batters_faced=("batters_faced", "sum"))
        .reset_index()
        .sort_values(["_season", "game_date"])
        .reset_index(drop=True)
    )
    season_grp = daily.groupby("_season", sort=False)
    daily["_cum_k"] = season_grp["strikeouts"].cumsum()
    daily["_cum_bf"] = season_grp["batters_faced"].cumsum()
    # Strictly-prior cumulative (excludes today's own games at every park):
    # shifting the running cumulative by one row works here because each
    # (_season, game_date) pair is unique in `daily`.
    daily["_prior_league_k"] = season_grp["_cum_k"].shift(1)
    daily["_prior_league_bf"] = season_grp["_cum_bf"].shift(1)

    league_lookup = daily.set_index(["_season", "game_date"])[
        ["_prior_league_k", "_prior_league_bf"]
    ]
    pg = pg.join(league_lookup, on=["_season", "game_date"])

    with np.errstate(invalid="ignore", divide="ignore"):
        park_rate = pg["_prior_park_k"].astype(float) / pg["_prior_park_bf"].astype(float)
        league_rate = pg["_prior_league_k"].astype(float) / pg["_prior_league_bf"].astype(float)
        computed_index = 100.0 * park_rate / league_rate.replace(0, np.nan)

    static_fallback = pg["park_team"].map(STATIC_PARK_FACTORS).fillna(DEFAULT_STATIC_PARK_FACTOR)

    enough_sample = pg["_prior_park_games"] >= MIN_PARK_GAMES_FOR_COMPUTED
    pg["park_k_factor"] = computed_index.where(enough_sample & computed_index.notna(), static_fallback)

    pg_lookup = pg.set_index(["park_team", "game_pk"])["park_k_factor"]
    df["park_k_factor"] = df.join(pg_lookup, on=["park_team", "game_pk"])["park_k_factor"]

    return df.drop(columns=["park_team"])
