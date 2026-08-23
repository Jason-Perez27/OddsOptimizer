"""
Joins game_logs, rolling_features, opponent_features, and park_factors into
one final training table: one row per pitcher-game, with `strikeouts` as
the label (task #6 finale, see
docs/design/specs/2026-06-27-strikeout-feature-engineering-design.md).

Pipeline:
1. game_logs.aggregate_pitcher_games(pitch_df) -> per-pitcher-game base table
2. rolling_features.add_rolling_features(game_df) -> + pitcher-side features
3. opponent_features.add_opponent_features(game_df) -> + opponent-side features
4. park_factors.add_park_factors(game_df) -> + park_k_factor

Order note: opponent_features and park_factors both need the FULL
per-pitcher-game table (every pitcher/team) to reconstruct team/park
history, so this must run on the complete game_df in one batch, not per
pitcher one at a time. rolling_features is pitcher-scoped and composes
fine either way; it's run right after game_logs for readability.

day_night note: game_logs.py leaves day_night null (Statcast has no
time-of-day field). Per the design spec it's meant to be backfilled from an
MLB-StatsAPI schedule lookup, which isn't built yet -- out of scope here,
so it's passed through as-is (null).

Nulls ARE expected in the rolling/opponent/park feature columns for
early games -- a pitcher's first start has no prior history, and a park's
first ~15 games fall back to the static table in park_factors.py (not a
null). REQUIRED_COLUMNS below are the identifier/label columns that must
never be null regardless of how little history exists yet; those are what
build_training_table() validates.
"""

import pandas as pd

from src.features.game_logs import aggregate_pitcher_games
from src.features.rolling_features import add_rolling_features
from src.features.opponent_features import add_opponent_features
from src.features.park_factors import add_park_factors

REQUIRED_COLUMNS = [
    "pitcher", "game_pk", "game_date", "pitcher_team", "opponent_team",
    "home_away", "strikeouts",
]


def build_training_table(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final per-pitcher-game training table from pitch-level
    Statcast rows: one row per pitcher-game, every rolling/opponent/park
    feature attached, `strikeouts` as the label.

    Raises ValueError if any REQUIRED_COLUMNS end up null -- that would
    indicate a real bug upstream, not expected cold-start sparsity.
    """
    game_df = aggregate_pitcher_games(pitch_df)
    game_df = add_rolling_features(game_df)
    game_df = add_opponent_features(game_df)
    game_df = add_park_factors(game_df)

    _validate_required_columns(game_df)
    return game_df


def _validate_required_columns(df: pd.DataFrame) -> None:
    if df.empty:
        return
    missing = [col for col in REQUIRED_COLUMNS if df[col].isna().any()]
    if missing:
        raise ValueError(
            f"build_training_table produced unexpected nulls in required columns: {missing}"
        )
