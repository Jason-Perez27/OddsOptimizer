"""
Opponent-side features built on top of src/features/game_logs.py's
per-pitcher-game table (the FULL table across all pitchers/teams, not just
one pitcher -- we need every pitcher's games to reconstruct each opposing
team's batting history).

Per docs/design/specs/2026-06-27-strikeout-feature-engineering-design.md:
- opponent_k_rate_last10: team K-rate, last 10 team games (teams play
  near-daily, so a wider window than the pitcher's last-5-starts is needed
  for a comparable sample size).
- opponent_k_rate_vs_hand_season: opponent's season-to-date K-rate
  specifically against the handedness of today's starting pitcher.
- opponent_k_rate_home, opponent_k_rate_away: opponent's season-to-date
  K-rate split by whether they're home or away today.

Two-step approach:
1. build_team_game_logs() collapses the per-pitcher-game table (one row per
   pitcher per game) into one row per (batting team, game) by summing across
   every pitcher that team faced in that game. This is a different axis from
   game_logs.py's strikeouts_vs_LHB/RHB (which splits by *batter* stand) --
   here we care about which hand the *pitcher* threw, via the
   pitcher_throws column game_logs.py now carries. The pitcher who faced the
   most batters in that team-game (the starter, in the common case) is used
   as that game's representative pitcher hand -- a reasonable v1
   approximation when a team also faced relievers of a different hand.
2. add_opponent_features() takes the original per-pitcher-game table plus
   the team-game log and, for every (pitcher, game) row, looks up that row's
   opponent_team's history strictly before that row's game_date (leakage
   guardrail -- same shift-before-aggregate pattern as rolling_features.py).
   opponent_k_rate_vs_hand_season needs a per-row lookup (not a simple
   column join) because the hand it filters on is THIS row's own pitcher's
   hand, which varies row to row even for the same opponent.
"""

import numpy as np
import pandas as pd

LAST10_WINDOW = 10

TEAM_GAME_COLUMNS = [
    "team", "game_pk", "game_date", "team_home_away",
    "strikeouts", "batters_faced", "opponent_pitcher_hand",
]

FEATURE_COLUMNS = [
    "opponent_k_rate_last10", "opponent_k_rate_vs_hand_season",
    "opponent_k_rate_vs_hand_last10",
    "opponent_k_rate_home", "opponent_k_rate_away",
]


def _shifted_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.astype(float).replace(0, np.nan)
    return numerator.astype(float) / denom


def _prior_cumsum(series: pd.Series) -> pd.Series:
    return series.shift(1).cumsum()


def _prior_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).sum()


def build_team_game_logs(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive one row per (batting team, game) from the full per-pitcher-game
    table, by summing across every pitcher that team faced in that game.
    """
    if game_df.empty:
        return pd.DataFrame(columns=TEAM_GAME_COLUMNS)

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    # The batting team's site is the opposite of the pitcher's site.
    df["team_home_away"] = df["home_away"].map({"home": "away", "away": "home"})

    rows = []
    for (team, game_pk), g in df.groupby(["opponent_team", "game_pk"], sort=False):
        bf = g["batters_faced"]
        if bf.notna().any():
            starter_row = g.loc[bf.idxmax()]
        else:
            starter_row = g.iloc[0]
        rows.append({
            "team": team,
            "game_pk": game_pk,
            "game_date": g["game_date"].iloc[0],
            "team_home_away": g["team_home_away"].iloc[0],
            "strikeouts": int(g["strikeouts"].sum()),
            "batters_faced": int(g["batters_faced"].sum()),
            "opponent_pitcher_hand": starter_row["pitcher_throws"],
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["team", "game_date"]).reset_index(drop=True)
    return out[TEAM_GAME_COLUMNS]


def add_opponent_features(game_df: pd.DataFrame, team_game_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add opponent-side features to the per-pitcher-game table. Each row's
    features describe that row's opponent_team's history strictly before
    that row's game_date.
    """
    if game_df.empty:
        out = game_df.copy()
        for col in FEATURE_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    if team_game_df is None:
        team_game_df = build_team_game_logs(game_df)
    tg = team_game_df.copy()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)
    tg["_season"] = tg["game_date"].dt.year

    # ---- last-10-team-games rolling K-rate ----
    team_grp = tg.groupby("team", sort=False)
    prior_k_last10 = team_grp["strikeouts"].transform(lambda s: _prior_rolling_sum(s, LAST10_WINDOW))
    prior_bf_last10 = team_grp["batters_faced"].transform(lambda s: _prior_rolling_sum(s, LAST10_WINDOW))
    tg["_k_rate_last10"] = _shifted_rate(prior_k_last10, prior_bf_last10)

    # ---- home/away splits, season-to-date ----
    tg["_k_if_home"] = tg["strikeouts"].where(tg["team_home_away"] == "home", 0)
    tg["_bf_if_home"] = tg["batters_faced"].where(tg["team_home_away"] == "home", 0)
    tg["_k_if_away"] = tg["strikeouts"].where(tg["team_home_away"] == "away", 0)
    tg["_bf_if_away"] = tg["batters_faced"].where(tg["team_home_away"] == "away", 0)
    season_grp = tg.groupby(["team", "_season"], sort=False)
    tg["_k_rate_home"] = _shifted_rate(
        season_grp["_k_if_home"].transform(_prior_cumsum),
        season_grp["_bf_if_home"].transform(_prior_cumsum),
    )
    tg["_k_rate_away"] = _shifted_rate(
        season_grp["_k_if_away"].transform(_prior_cumsum),
        season_grp["_bf_if_away"].transform(_prior_cumsum),
    )

    tg_lookup = tg.set_index(["team", "game_pk"])[["_k_rate_last10", "_k_rate_home", "_k_rate_away"]]

    # ---- join the team-keyed features back onto the per-pitcher-game table ----
    joined = df.join(tg_lookup, on=["opponent_team", "game_pk"])
    df["opponent_k_rate_last10"] = joined["_k_rate_last10"]
    df["opponent_k_rate_home"] = joined["_k_rate_home"]
    df["opponent_k_rate_away"] = joined["_k_rate_away"]

    df["opponent_k_rate_vs_hand_season"] = _opponent_k_rate_vs_hand(df, tg)
    df["opponent_k_rate_vs_hand_last10"] = _opponent_k_rate_vs_hand_last10(df, tg)

    return df


def _opponent_k_rate_vs_hand(df: pd.DataFrame, tg: pd.DataFrame) -> pd.Series:
    """
    For each row in df, compute opponent_team's season-to-date pooled
    K-rate against pitchers throwing the SAME hand as this row's own
    pitcher_throws, using only opponent games strictly before this row's
    game_date (and within the same season).
    """
    results = np.full(len(df), np.nan)
    tg_by_team = {team: g for team, g in tg.groupby("team", sort=False)}

    for i, row in enumerate(df.itertuples(index=False)):
        team = getattr(row, "opponent_team")
        hand = getattr(row, "pitcher_throws")
        game_date = getattr(row, "game_date")
        if team not in tg_by_team or pd.isna(hand):
            continue
        team_games = tg_by_team[team]
        prior = team_games[
            (team_games["game_date"] < game_date)
            & (team_games["opponent_pitcher_hand"] == hand)
            & (team_games["_season"] == pd.Timestamp(game_date).year)
        ]
        if prior.empty:
            continue
        bf = prior["batters_faced"].sum()
        if bf == 0:
            continue
        results[i] = prior["strikeouts"].sum() / bf

    return pd.Series(results, index=df.index)


def _opponent_k_rate_vs_hand_last10(df: pd.DataFrame, tg: pd.DataFrame) -> pd.Series:
    """
    For each row in df, compute opponent_team's K-rate against pitchers
    throwing the SAME hand as this row's pitcher_throws, using the last
    LAST10_WINDOW games of that hand type strictly before this row's
    game_date.  No season boundary — the 10-game window is tight enough
    to be meaningful regardless of season cross.
    """
    results = np.full(len(df), np.nan)
    tg_by_team = {team: g for team, g in tg.groupby("team", sort=False)}

    for i, row in enumerate(df.itertuples(index=False)):
        team = getattr(row, "opponent_team")
        hand = getattr(row, "pitcher_throws")
        game_date = getattr(row, "game_date")
        if team not in tg_by_team or pd.isna(hand):
            continue
        team_games = tg_by_team[team]
        prior = team_games[
            (team_games["game_date"] < game_date)
            & (team_games["opponent_pitcher_hand"] == hand)
        ].tail(LAST10_WINDOW)
        if prior.empty:
            continue
        bf = prior["batters_faced"].sum()
        if bf == 0:
            continue
        results[i] = prior["strikeouts"].sum() / bf

    return pd.Series(results, index=df.index)


# ---------------------------------------------------------------------------
# Spec 3 (2026-06-30): lineup-weighted matchup features
# ---------------------------------------------------------------------------

MATCHUP_FEATURE_COLUMNS = [
    "opponent_lineup_k_rate_vs_hand",  # PA-slot-weighted batter K% vs this hand
    "opp_share_opposite_hand",         # fraction of lineup with platoon advantage
]

# Slot-based PA weight: slot 1 bats most often in a game.
_SLOT_PA_WEIGHT = {slot: 10 - slot for slot in range(1, 10)}  # {1:9, ..., 9:1}


def build_lineup_weighted_opp_k(
    opp_lineup_df: pd.DataFrame,
    batter_rolling_df: pd.DataFrame,
    pitcher_throws: str,
) -> dict:
    """
    Compute lineup-weighted opponent K-rate vs this pitcher's hand and the
    platoon share of the opposing lineup.

    Parameters
    ----------
    opp_lineup_df : DataFrame
        Batters from the OPPOSING team's lineup only (pre-filtered by caller).
        Columns needed: batter_id, bat_side, lineup_slot.
    batter_rolling_df : DataFrame
        Pre-game per-batter rolling K-rates (shift-1 already applied).
        Columns needed: batter, k_rate_vs_lhp_season, k_rate_vs_rhp_season,
        game_date.
    pitcher_throws : str
        "L" or "R".

    Returns
    -------
    dict with keys:
        opponent_lineup_k_rate_vs_hand : float | NaN
        opp_share_opposite_hand : float | NaN
    """
    _nan = {"opponent_lineup_k_rate_vs_hand": np.nan, "opp_share_opposite_hand": np.nan}

    if opp_lineup_df is None or opp_lineup_df.empty:
        return _nan

    lu = opp_lineup_df[opp_lineup_df["batter_id"].notna()].copy()
    if lu.empty:
        return _nan

    if pitcher_throws == "L":
        k_rate_col = "k_rate_vs_lhp_season"
        opposite_hands = frozenset(("R", "S"))
    else:
        k_rate_col = "k_rate_vs_rhp_season"
        opposite_hands = frozenset(("L", "S"))

    if batter_rolling_df is not None and not batter_rolling_df.empty:
        latest = (
            batter_rolling_df
            .sort_values("game_date")
            .groupby("batter")
            .last()
            .reset_index()[["batter", k_rate_col]]
            .rename(columns={"batter": "batter_id"})
        )
        lu = lu.merge(latest, on="batter_id", how="left")
    else:
        lu[k_rate_col] = np.nan

    lu["_slot"] = pd.to_numeric(lu.get("lineup_slot"), errors="coerce")
    lu["_weight"] = lu["_slot"].map(_SLOT_PA_WEIGHT).fillna(5.0)

    lu["_k_rate"] = pd.to_numeric(lu[k_rate_col], errors="coerce")
    valid = lu[lu["_k_rate"].notna()]
    if valid.empty:
        opp_k = np.nan
    else:
        opp_k = float((valid["_k_rate"] * valid["_weight"]).sum() / valid["_weight"].sum())

    bat_sides = lu["bat_side"].fillna("R")
    opp_share = float(bat_sides.isin(opposite_hands).mean())

    return {
        "opponent_lineup_k_rate_vs_hand": opp_k,
        "opp_share_opposite_hand": opp_share,
    }
