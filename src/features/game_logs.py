"""
Aggregate pitch-level Statcast rows (from src/data/pitcher_logs.py) into one
row per pitcher-game.

This is the first feature-engineering step (task #6, see
docs/design/specs/2026-06-27-strikeout-feature-engineering-design.md).
Statcast gives one row per pitch; the rolling/opponent features built on top
of this (rolling_features.py, opponent_features.py) need one row per
pitcher-game with the per-game totals/rates those later steps roll up.

day_night is NOT derivable from Statcast pitch data alone (there's no
time-of-day field) -- it's left null here and gets filled in later from an
MLB-StatsAPI schedule lookup (see src/features/build_features.py). rest_days
*is* derivable here since it's just the gap between consecutive game dates
for the same pitcher.

pitcher_throws (L/R) comes from Statcast's p_throws column when present, and
is None otherwise (kept optional so older callers/tests that don't supply
p_throws don't break). It's what opponent_features.py uses to compute a
team's K-rate specifically against left- vs right-handed pitching -- a
different axis than strikeouts_vs_LHB/RHB below, which split by the *batter's*
stand, not the pitcher's throwing hand.
"""

import pandas as pd

STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WALK_EVENTS = {"walk"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
FASTBALL_TYPES = {"FF", "FT", "SI"}

# Plate-discipline skill metrics (spec 2, 2026-06-30).
# CALLED_STRIKE_DESCRIPTIONS: pitches graded called strike (batter took).
# SWING_DESCRIPTIONS: all pitches the batter offered at, used as the
# denominator for whiff_rate_overall (swinging strikes / total swings).
# `whiff_rate` (existing column) = swinging_strikes / pitch_count = SwStr%.
# `swstr_rate` is the rolling feature name for the same concept; see
# rolling_features.py note.
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
SWING_DESCRIPTIONS = {
    # whiffs
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
    # fouls
    "foul", "foul_tip", "foul_bunt",
    # contact
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}

# How many outs each plate-appearance-ending event produces. Used to estimate
# innings pitched from play-by-play events (outs_recorded / 3) since Statcast
# doesn't expose a direct per-pitcher-game IP field.
OUT_EVENTS = {
    "strikeout": 1,
    "field_out": 1,
    "force_out": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "strikeout_double_play": 2,
    "triple_play": 3,
    "sac_fly": 1,
    "sac_bunt": 1,
    "fielders_choice_out": 1,
}

OUTPUT_COLUMNS = [
    "pitcher", "game_pk", "game_date", "pitcher_team", "opponent_team",
    "home_away", "strikeouts", "walks", "batters_faced", "pitch_count", "whiff_rate",
    "fastball_velo_avg", "innings_pitched", "pitcher_throws", "strikeouts_vs_LHB",
    "batters_faced_vs_LHB", "strikeouts_vs_RHB", "batters_faced_vs_RHB",
    "rest_days", "day_night",
    # plate-discipline skill metrics (spec 2, 2026-06-30)
    "csw_rate",           # (called_strike + swinging_strike) / pitch_count
    "putaway_rate",       # K / two-strike pitches (NaN if `strikes` col absent)
    "whiff_rate_overall", # swinging_strikes / total_swings
    "k_minus_bb",         # strikeouts - walks (count; rolling rate in rolling_features)
]


def aggregate_pitcher_games(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse pitch-level Statcast rows into one row per pitcher-game.

    Expects columns: pitcher, game_pk, game_date, home_team, away_team,
    inning_topbot, events, description, pitch_type, release_speed, stand.
    p_throws is optional -- if absent, pitcher_throws is None in the output.

    Returns one row per (pitcher, game_pk), sorted by pitcher/game_date, with
    rest_days computed as the day-gap from that pitcher's previous game
    (null for a pitcher's first game in the input).
    """
    if pitch_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pitch_df.copy()

    # Pitcher's team is home when top of inning (away team is batting), away
    # when bottom of inning (home team is batting).
    is_home = df["inning_topbot"].eq("Top")
    df["pitcher_team"] = df["home_team"].where(is_home, df["away_team"])
    df["opponent_team"] = df["away_team"].where(is_home, df["home_team"])
    df["home_away"] = is_home.map({True: "home", False: "away"})

    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_fastball"] = df["pitch_type"].isin(FASTBALL_TYPES)
    df["is_called_strike"] = df["description"].isin(CALLED_STRIKE_DESCRIPTIONS)
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)

    has_p_throws = "p_throws" in df.columns
    has_strikes = "strikes" in df.columns  # Statcast count before pitch (for putaway)

    rows = []
    for (pitcher, game_pk), g in df.groupby(["pitcher", "game_pk"], sort=False):
        pa = g.dropna(subset=["events"])  # one row per completed plate appearance
        strikeouts = pa["events"].isin(STRIKEOUT_EVENTS).sum()
        walks = pa["events"].isin(WALK_EVENTS).sum()
        batters_faced = len(pa)

        lhb = pa[pa["stand"] == "L"]
        rhb = pa[pa["stand"] == "R"]
        k_lhb = lhb["events"].isin(STRIKEOUT_EVENTS).sum()
        k_rhb = rhb["events"].isin(STRIKEOUT_EVENTS).sum()

        pitch_count = len(g)
        whiff_count = g["is_whiff"].sum()
        # whiff_rate = swStr% = swinging_strikes / total_pitches (legacy name kept
        # for backward compat; rolling_features names it swstr_rate_*)
        whiff_rate = whiff_count / pitch_count if pitch_count else float("nan")
        fb_speeds = g.loc[g["is_fastball"], "release_speed"]
        fastball_velo_avg = fb_speeds.mean() if not fb_speeds.empty else float("nan")

        outs_recorded = pa["events"].map(OUT_EVENTS).fillna(0).sum()
        innings_pitched = outs_recorded / 3.0

        pitcher_throws = g["p_throws"].iloc[0] if has_p_throws else None

        # --- plate-discipline skill metrics (spec 2, 2026-06-30) ---
        csw_count = whiff_count + g["is_called_strike"].sum()
        csw_rate = csw_count / pitch_count if pitch_count else float("nan")

        total_swings = g["is_swing"].sum()
        whiff_rate_overall = whiff_count / total_swings if total_swings > 0 else float("nan")

        if has_strikes:
            two_strike_pitches = int((g["strikes"] == 2).sum())
            putaway_rate = int(strikeouts) / two_strike_pitches if two_strike_pitches > 0 else float("nan")
        else:
            putaway_rate = float("nan")

        k_minus_bb = int(strikeouts) - int(walks)

        rows.append({
            "pitcher": pitcher,
            "game_pk": game_pk,
            "game_date": g["game_date"].iloc[0],
            "pitcher_team": g["pitcher_team"].iloc[0],
            "opponent_team": g["opponent_team"].iloc[0],
            "home_away": g["home_away"].iloc[0],
            "strikeouts": int(strikeouts),
            "walks": int(walks),
            "batters_faced": int(batters_faced),
            "pitch_count": int(pitch_count),
            "whiff_rate": whiff_rate,
            "fastball_velo_avg": fastball_velo_avg,
            "innings_pitched": innings_pitched,
            "pitcher_throws": pitcher_throws,
            "strikeouts_vs_LHB": int(k_lhb),
            "batters_faced_vs_LHB": int(len(lhb)),
            "strikeouts_vs_RHB": int(k_rhb),
            "batters_faced_vs_RHB": int(len(rhb)),
            "day_night": None,
            "csw_rate": csw_rate,
            "putaway_rate": putaway_rate,
            "whiff_rate_overall": whiff_rate_overall,
            "k_minus_bb": k_minus_bb,
        })

    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"])
    out = out.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    out["rest_days"] = out.groupby("pitcher")["game_date"].diff().dt.days
    return out[OUTPUT_COLUMNS]
