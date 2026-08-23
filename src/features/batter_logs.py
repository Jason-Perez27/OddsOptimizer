"""
Batter-side K-rate features (spec 3, 2026-06-30).

Mirrors src/features/game_logs.py / rolling_features.py but from the
BATTER's perspective:
  1. aggregate_batter_games() -- collapse pitch-level Statcast rows into
     one row per (batter, game_pk), splitting K and PA by opposing hand
     (p_throws).
  2. add_batter_rolling_features() -- compute strictly-prior (shift-1)
     rolling K-rates vs LHP and RHP, both season-expanding and last-10-game
     windows.

These per-batter rolling rates feed build_lineup_weighted_opp_k() in
src/features/opponent_features.py to produce the lineup-weighted matchup
features.

Leakage guardrail: identical to rolling_features.py -- shift(1) is applied
before every cumsum / rolling call, so the current game's outcomes never
pollute that same game's pre-game features.
"""

import numpy as np
import pandas as pd

# Columns returned by aggregate_batter_games()
BATTER_GAME_COLUMNS = [
    "batter",
    "game_pk",
    "game_date",
    "strikeouts_vs_lhp",
    "pa_vs_lhp",
    "strikeouts_vs_rhp",
    "pa_vs_rhp",
    "bat_side",
]

# New columns added by add_batter_rolling_features()
ROLLING_BATTER_COLUMNS = [
    "k_rate_vs_lhp_season",
    "k_rate_vs_rhp_season",
    "k_rate_vs_lhp_last10",
    "k_rate_vs_rhp_last10",
]

STRIKEOUT_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
LAST10_WINDOW = 10


# ---------------------------------------------------------------------------
# Step 1: aggregate pitch-level Statcast → batter-game table
# ---------------------------------------------------------------------------

def aggregate_batter_games(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse pitch-level Statcast rows into one row per (batter, game_pk),
    tracking K and PA split by opposing pitcher hand (p_throws column).

    Expected input columns (same Statcast schema as game_logs.py):
        batter, game_pk, game_date, events (str | NaN), stand, p_throws

    Returns a DataFrame with BATTER_GAME_COLUMNS, sorted by (batter, game_date).
    Empty if pitch_df is empty or missing required columns.
    """
    if pitch_df.empty:
        return pd.DataFrame(columns=BATTER_GAME_COLUMNS)

    required = {"batter", "game_pk", "game_date"}
    missing = required - set(pitch_df.columns)
    if missing:
        raise ValueError(f"aggregate_batter_games: missing columns {missing}")

    df = pitch_df.copy()
    has_p_throws = "p_throws" in df.columns
    has_stand = "stand" in df.columns

    rows = []
    for (batter, game_pk), g in df.groupby(["batter", "game_pk"], sort=False):
        # PA = rows with non-null events (plate-appearance outcomes)
        pa = g[g["events"].notna()] if "events" in g.columns else pd.DataFrame()

        # Dominant bat-side from `stand` column
        if has_stand and not g["stand"].isna().all():
            bat_side = g["stand"].mode().iloc[0]
        else:
            bat_side = None

        if has_p_throws and not pa.empty:
            pa_lhp = pa[pa["p_throws"] == "L"]
            pa_rhp = pa[pa["p_throws"] == "R"]
            k_lhp = int(pa_lhp["events"].isin(STRIKEOUT_EVENTS).sum())
            k_rhp = int(pa_rhp["events"].isin(STRIKEOUT_EVENTS).sum())
            n_lhp = len(pa_lhp)
            n_rhp = len(pa_rhp)
        else:
            k_lhp = k_rhp = 0
            n_lhp = n_rhp = 0

        rows.append({
            "batter": int(batter),
            "game_pk": int(game_pk),
            "game_date": g["game_date"].iloc[0],
            "strikeouts_vs_lhp": k_lhp,
            "pa_vs_lhp": n_lhp,
            "strikeouts_vs_rhp": k_rhp,
            "pa_vs_rhp": n_rhp,
            "bat_side": bat_side,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=BATTER_GAME_COLUMNS)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out = out.sort_values(["batter", "game_date"]).reset_index(drop=True)
    return out[BATTER_GAME_COLUMNS]


# ---------------------------------------------------------------------------
# Step 2: strictly-prior rolling K-rates
# ---------------------------------------------------------------------------

def _prior_season_rate(
    group: pd.DataFrame, num_col: str, den_col: str
) -> pd.Series:
    """Shift-1 cumulative K-rate within a (batter, season) group."""
    num_cs = group[num_col].shift(1).cumsum()
    den_cs = group[den_col].shift(1).cumsum().replace(0, np.nan)
    return (num_cs / den_cs).values


def _prior_rolling_rate(
    group: pd.DataFrame, num_col: str, den_col: str, window: int
) -> pd.Series:
    """Shift-1 rolling K-rate (batter-level, crosses season boundary)."""
    num_r = group[num_col].shift(1).rolling(window, min_periods=1).sum()
    den_r = group[den_col].shift(1).rolling(window, min_periods=1).sum().replace(0, np.nan)
    return (num_r / den_r).values


def add_batter_rolling_features(batter_game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly-prior rolling K-rate features to each batter-game row.

    New columns (see ROLLING_BATTER_COLUMNS):
        k_rate_vs_lhp_season    season-expanding K/PA vs LHP (shift-1)
        k_rate_vs_rhp_season    season-expanding K/PA vs RHP (shift-1)
        k_rate_vs_lhp_last10   rolling 10-game K/PA vs LHP  (shift-1)
        k_rate_vs_rhp_last10   rolling 10-game K/PA vs RHP  (shift-1)

    Season is defined as calendar year (game_date.dt.year).  Season-rate
    resets at year boundary; last-10 window does NOT reset (cross-season is
    intentional for the rolling window).

    Input must have BATTER_GAME_COLUMNS; returns input + 4 new columns.
    Returns input with all-NaN new columns if batter_game_df is empty.
    """
    if batter_game_df.empty:
        out = batter_game_df.copy()
        for col in ROLLING_BATTER_COLUMNS:
            out[col] = np.nan
        return out

    df = batter_game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["_season"] = df["game_date"].dt.year
    df = df.sort_values(["batter", "game_date"]).reset_index(drop=True)

    # Initialise new columns
    for col in ROLLING_BATTER_COLUMNS:
        df[col] = np.nan

    # Season-expanding rates -- group by (batter, season)
    for (batter, season), grp_idx in df.groupby(["batter", "_season"]).groups.items():
        idx = list(grp_idx)
        g = df.loc[idx].copy()
        df.loc[idx, "k_rate_vs_lhp_season"] = _prior_season_rate(g, "strikeouts_vs_lhp", "pa_vs_lhp")
        df.loc[idx, "k_rate_vs_rhp_season"] = _prior_season_rate(g, "strikeouts_vs_rhp", "pa_vs_rhp")

    # Rolling 10-game rate -- group by batter only (cross-season window)
    for batter, grp_idx in df.groupby("batter").groups.items():
        idx = list(grp_idx)
        g = df.loc[idx].copy()
        df.loc[idx, "k_rate_vs_lhp_last10"] = _prior_rolling_rate(
            g, "strikeouts_vs_lhp", "pa_vs_lhp", LAST10_WINDOW
        )
        df.loc[idx, "k_rate_vs_rhp_last10"] = _prior_rolling_rate(
            g, "strikeouts_vs_rhp", "pa_vs_rhp", LAST10_WINDOW
        )

    df = df.drop(columns=["_season"])
    return df
