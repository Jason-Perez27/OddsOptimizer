"""
Pitcher-side rolling/season/split features built on top of
src/features/game_logs.py's per-pitcher-game table.

Leakage guardrail: every feature for game N is computed using ONLY that
pitcher's games strictly before game N's date. The pattern used throughout
is shift(1) before cumsum/rolling, so a feature value for game N never sees
game N's own stats. tests/test_rolling_features.py asserts this directly.

The per-prop rate families (k_rate_*, bb_rate_*, er_*) share the same
rolling/season/home/away/vs-opponent-career pattern via
_add_count_rate_family. The strikeout family additionally carries
handedness (LHB/RHB) splits. If the underlying count column is absent
(e.g. earned_runs before a boxscore merge), the feature columns are created
but filled with NaN -- no error is raised.
"""

import numpy as np
import pandas as pd

LAST5_WINDOW = 5
OPPONENT_CAREER_MIN_STARTS = 3

# Empirical-Bayes K-rate stabilization constants (spec: fix-projection-
# compression, 2026-06-30). C is a BF-equivalent pseudo-count (~2-3 starts'
# worth of exposure) that blends the recent L5 sample toward the season prior.
# A real ace who had one rough start is pulled back toward their season rate
# rather than tanked. Tune C via run_compression_fix_gate.py.
EB_K_CONSTANT = 100.0          # BF pseudo-count for the season-rate prior
LEAGUE_K_RATE_PRIOR = 0.22     # fallback prior for debut season (no season rate yet)

_RATE_FAMILY_SUFFIXES = [
    "_last5", "_season", "_home", "_away", "_vs_opponent_career",
]

FEATURE_COLUMNS = [
    # strikeout rate family (+ handedness splits unique to k_rate)
    "k_rate_last5", "k_rate_season", "k_rate_vs_LHB", "k_rate_vs_RHB",
    "k_rate_home", "k_rate_away", "k_rate_vs_opponent_career",
    # walk rate family (bb_rate)
    "bb_rate_last5", "bb_rate_season",
    "bb_rate_home", "bb_rate_away", "bb_rate_vs_opponent_career",
    # earned-run rate family (er) -- NaN when earned_runs column absent
    "er_last5", "er_season",
    "er_home", "er_away", "er_vs_opponent_career",
    # volume/stuff metrics (not prop-rate features)
    "ip_avg_last5", "pitch_count_avg_last5", "bf_avg_last5",
    "whiff_rate_last5", "velo_avg_last5",
    # plate-discipline skill features (spec 2, 2026-06-30) -- candidate regressors.
    # swstr_rate_* uses the existing whiff_rate column (swinging_strikes/pitch_count);
    # the same stat as SwStr%, renamed for clarity in the rolling output names.
    "swstr_rate_last5", "swstr_rate_season",
    "csw_rate_last5", "csw_rate_season",
    "putaway_rate_last5", "putaway_rate_season",
    "whiff_rate_overall_last5", "whiff_rate_overall_season",
    "k_minus_bb_rate_last5", "k_minus_bb_rate_season",
    "k_minus_bb_rate_home", "k_minus_bb_rate_away", "k_minus_bb_rate_vs_opponent_career",
    # EB-stabilized K rate (spec: fix-projection-compression, 2026-06-30)
    # k_stab_last5 = (K_last5 + C * k_rate_season) / (BF_last5 + C)
    # Replaces raw k_rate_last5 as the primary K-skill regressor in the
    # compression-fix variant; candidate for baseline promotion after gate.
    "k_stab_last5",
]


def _shifted_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise numerator/denominator as floats, NaN where denominator is 0/NaN."""
    denom = denominator.astype(float).replace(0, np.nan)
    return numerator.astype(float) / denom


def _prior_cumsum(series: pd.Series) -> pd.Series:
    """Cumulative sum of all values strictly before the current row."""
    return series.shift(1).cumsum()


def _prior_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Rolling sum over the window rows strictly before the current row."""
    return series.shift(1).rolling(window, min_periods=1).sum()


def _prior_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over the window rows strictly before the current row."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _prior_expanding_mean(series: pd.Series) -> pd.Series:
    """Expanding mean over all rows strictly before the current row (season-to-date when grouped by season)."""
    return series.shift(1).expanding().mean()


def _add_count_rate_family(df: pd.DataFrame, prefix: str, count_col: str) -> pd.DataFrame:
    """
    Add five leakage-safe rate features for any per-game count column:
        {prefix}_last5              -- pooled count/BF over the prior 5 starts
        {prefix}_season             -- season-to-date count/BF
        {prefix}_home               -- season-to-date rate in home games
        {prefix}_away               -- season-to-date rate in away games
        {prefix}_vs_opponent_career -- career rate vs today's opponent
                                       (falls back to {prefix}_season when
                                        fewer than OPPONENT_CAREER_MIN_STARTS)

    df must already have _season set (int year column). If count_col is not
    present in df the five columns are added as NaN.
    """
    feature_names = [f"{prefix}{s}" for s in _RATE_FAMILY_SUFFIXES]

    if count_col not in df.columns:
        for col in feature_names:
            df[col] = float("nan")
        return df

    pitcher_grp = df.groupby("pitcher", sort=False)

    # --- last-5-starts pooled rate ---
    prior_cnt_last5 = pitcher_grp[count_col].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    )
    prior_bf_last5 = pitcher_grp["batters_faced"].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    )
    df[f"{prefix}_last5"] = _shifted_rate(prior_cnt_last5, prior_bf_last5)

    # --- season-to-date ---
    season_grp = df.groupby(["pitcher", "_season"], sort=False)
    prior_cnt_season = season_grp[count_col].transform(_prior_cumsum)
    prior_bf_season = season_grp["batters_faced"].transform(_prior_cumsum)
    df[f"{prefix}_season"] = _shifted_rate(prior_cnt_season, prior_bf_season)

    # --- home/away splits (season-to-date) ---
    _home_col = f"_{prefix}_if_home"
    _home_bf  = f"_{prefix}_bf_if_home"
    _away_col = f"_{prefix}_if_away"
    _away_bf  = f"_{prefix}_bf_if_away"

    df[_home_col] = df[count_col].where(df["home_away"] == "home", 0)
    df[_home_bf]  = df["batters_faced"].where(df["home_away"] == "home", 0)
    df[_away_col] = df[count_col].where(df["home_away"] == "away", 0)
    df[_away_bf]  = df["batters_faced"].where(df["home_away"] == "away", 0)

    season_grp2 = df.groupby(["pitcher", "_season"], sort=False)
    df[f"{prefix}_home"] = _shifted_rate(
        season_grp2[_home_col].transform(_prior_cumsum),
        season_grp2[_home_bf].transform(_prior_cumsum),
    )
    df[f"{prefix}_away"] = _shifted_rate(
        season_grp2[_away_col].transform(_prior_cumsum),
        season_grp2[_away_bf].transform(_prior_cumsum),
    )
    df = df.drop(columns=[_home_col, _home_bf, _away_col, _away_bf])

    # --- career vs this specific opponent, with small-sample fallback ---
    opp_grp = df.groupby(["pitcher", "opponent_team"], sort=False)
    prior_cnt_opp = opp_grp[count_col].transform(_prior_cumsum)
    prior_bf_opp  = opp_grp["batters_faced"].transform(_prior_cumsum)
    prior_starts_vs_opp = opp_grp.cumcount()

    raw_opp_rate = _shifted_rate(prior_cnt_opp, prior_bf_opp)
    enough_sample = prior_starts_vs_opp >= OPPONENT_CAREER_MIN_STARTS
    df[f"{prefix}_vs_opponent_career"] = raw_opp_rate.where(
        enough_sample, df[f"{prefix}_season"]
    )

    return df


def add_rolling_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pitcher-side rolling/season/split features to a per-pitcher-game
    table (the output of game_logs.aggregate_pitcher_games).

    Returns a new DataFrame with the same rows as the input (re-sorted by
    pitcher/game_date) plus the FEATURE_COLUMNS columns added.
    """
    if game_df.empty:
        out = game_df.copy()
        for col in FEATURE_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    df["_season"] = df["game_date"].dt.year

    pitcher_grp = df.groupby("pitcher", sort=False)

    # ---- last-5-starts rolling averages (volume/stuff metrics) ----
    df["ip_avg_last5"] = pitcher_grp["innings_pitched"].transform(
        lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
    )
    df["pitch_count_avg_last5"] = pitcher_grp["pitch_count"].transform(
        lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
    )
    df["bf_avg_last5"] = pitcher_grp["batters_faced"].transform(
        lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
    )
    df["whiff_rate_last5"] = pitcher_grp["whiff_rate"].transform(
        lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
    )
    df["velo_avg_last5"] = pitcher_grp["fastball_velo_avg"].transform(
        lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
    )

    # ---- strikeout rate family (k_rate_*) ----
    df = _add_count_rate_family(df, "k_rate", "strikeouts")

    season_grp = df.groupby(["pitcher", "_season"], sort=False)
    prior_k_lhb  = season_grp["strikeouts_vs_LHB"].transform(_prior_cumsum)
    prior_bf_lhb = season_grp["batters_faced_vs_LHB"].transform(_prior_cumsum)
    df["k_rate_vs_LHB"] = _shifted_rate(prior_k_lhb, prior_bf_lhb)

    prior_k_rhb  = season_grp["strikeouts_vs_RHB"].transform(_prior_cumsum)
    prior_bf_rhb = season_grp["batters_faced_vs_RHB"].transform(_prior_cumsum)
    df["k_rate_vs_RHB"] = _shifted_rate(prior_k_rhb, prior_bf_rhb)

    # ---- EB-stabilized K rate (spec: compression-fix v2, 2026-06-30) ----
    # k_stab_last5 = (K_recent + C * p_prior) / (BF_recent + C)
    # K_recent / BF_recent: strictly-prior rolling sums (shift-1 -> no same-game leakage)
    #
    # Tiered prior (compression-fix v2 — finding ②):
    #   Tier 1: own season-to-date K rate (k_rate_season) — pitcher's own recent baseline
    #   Tier 2: own career K rate (cumulative all seasons) — when season not yet started
    #   Tier 3: LEAGUE_K_RATE_PRIOR — only when pitcher has no prior history at all
    #
    # This prevents shrinking aces toward the league mean (C=100 v1 bug) while still
    # pulling genuinely slumping pitchers toward their true career level. Sweep C ∈
    # {25,50,75} via module-level EB_K_CONSTANT (see run_compression_fix_v2_gate.py).
    prior_k_sum_last5 = pitcher_grp["strikeouts"].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    ).fillna(0.0)  # debut game: 0 prior K (no games yet)
    prior_bf_sum_last5 = pitcher_grp["batters_faced"].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    ).fillna(0.0)  # debut game: 0 prior BF → formula reduces to pure prior

    # Career K-rate: cumulative across ALL prior seasons (pitcher-level, no season grouping)
    # Same shift-1 leakage guard as k_rate_season but not reset at season boundary
    prior_k_career = pitcher_grp["strikeouts"].transform(_prior_cumsum)
    prior_bf_career = pitcher_grp["batters_faced"].transform(_prior_cumsum)
    k_rate_career = _shifted_rate(prior_k_career, prior_bf_career)

    # Tiered: own-season → own-career → league mean
    p_prior = df["k_rate_season"].fillna(k_rate_career).fillna(LEAGUE_K_RATE_PRIOR)

    # When prior_k=0 and prior_bf=0 (debut start), k_stab = (0 + C*p_prior)/(0 + C) = p_prior
    df["k_stab_last5"] = (
        (prior_k_sum_last5 + EB_K_CONSTANT * p_prior)
        / (prior_bf_sum_last5 + EB_K_CONSTANT)
    )

    # ---- walk rate family (bb_rate_*) ----
    df = _add_count_rate_family(df, "bb_rate", "walks")

    # ---- earned-run rate family (er_*) ----
    # game_logs cannot supply earned_runs; when training the ER model the
    # caller merges the boxscore label first. For other props, fills NaN.
    df = _add_count_rate_family(df, "er", "earned_runs")

    # ---- plate-discipline skill features (spec 2, 2026-06-30) ----
    # Rate features (per-game rate column -> last-5 rolling mean + season expanding
    # mean). swstr_rate uses the existing whiff_rate column (same formula).
    # NaN when the source column is absent (e.g. old cached feature tables).
    _skill_rate_pairs = [
        ("swstr_rate",         "whiff_rate"),
        ("csw_rate",           "csw_rate"),
        ("putaway_rate",       "putaway_rate"),
        ("whiff_rate_overall", "whiff_rate_overall"),
    ]
    for feat_prefix, src_col in _skill_rate_pairs:
        if src_col in df.columns:
            df[f"{feat_prefix}_last5"] = pitcher_grp[src_col].transform(
                lambda s: _prior_rolling_mean(s, LAST5_WINDOW)
            )
            skill_season_grp = df.groupby(["pitcher", "_season"], sort=False)
            df[f"{feat_prefix}_season"] = skill_season_grp[src_col].transform(
                _prior_expanding_mean
            )
        else:
            df[f"{feat_prefix}_last5"] = float("nan")
            df[f"{feat_prefix}_season"] = float("nan")
    # k_minus_bb_rate: count-based family (pooled like k_rate, BF denominator)
    df = _add_count_rate_family(df, "k_minus_bb_rate", "k_minus_bb")

    df = df.drop(columns=["_season"])
    return df
