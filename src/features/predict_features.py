"""
Pre-game ("as-of-today") feature construction for the slate's probable
starters -- task #9, module 2.

Design: docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md
("Pre-game (as-of-today) feature construction" section).

Key idea: rolling_features.add_rolling_features, opponent_features.
add_opponent_features, and park_factors.add_park_factors all compute "as of
this row's date" values via shift(1)-before-aggregate + strict
`game_date <` filters -- that IS their leakage guardrail. So no "predict
mode" branch is needed inside any of them: hand them one extra row dated
today per slate starter (a "synthetic" game_logs-schema row with every
same-game outcome column NaN, since those are genuinely unknown pre-game)
and slice the synthetic rows back out after running the same three
builders, UNMODIFIED, in the same order build_features.build_training_table
uses.

Deliberately NOT reusing build_features.build_training_table itself: it
starts from pitch-level Statcast (we already have game-logs here) and
validates `strikeouts` non-null, which a label-less synthetic row violates
by design. This module is a sibling to build_features.py, not a mode= flag
bolted onto it (see spec's "Placement rationale" -- flagged there as movable
into build_features.py in review if cohesion is preferred).
"""

import numpy as np
import pandas as pd

from src.features.game_logs import OUTPUT_COLUMNS
from src.features.rolling_features import add_rolling_features
from src.features.opponent_features import add_opponent_features
from src.features.park_factors import add_park_factors

# Same-game outcome/leakage columns -- genuinely unknown pre-game, and never
# read for the synthetic row's OWN features (every builder shift(1)s the
# current row out), so NaN is safe here. Kept independent of (not imported
# from) baseline_model.LEAKAGE_COLUMNS -- feature construction shouldn't
# depend on which model consumes its output.
LEAKAGE_OUTCOME_COLUMNS = [
    "strikeouts", "walks", "batters_faced", "pitch_count", "whiff_rate",
    "fastball_velo_avg", "innings_pitched", "strikeouts_vs_LHB",
    "batters_faced_vs_LHB", "strikeouts_vs_RHB", "batters_faced_vs_RHB",
    # spec ② skill metrics (2026-06-30) -- same-game, never used as regressors
    "csw_rate", "putaway_rate", "whiff_rate_overall", "k_minus_bb",
]

# Known pre-game fields carried straight from the slate (src.data.probable_
# pitchers.SLATE_COLUMNS) into the game_logs schema.
SLATE_TO_GAME_LOG_FIELDS = [
    "pitcher", "game_pk", "game_date", "pitcher_team", "opponent_team",
    "home_away", "pitcher_throws",
]

# Internal marker used only to slice the synthetic rows back out after the
# builders run on the combined table; always dropped before returning.
_SYNTHETIC_FLAG = "_is_synthetic_prediction_row"

# Numeric game_logs columns that must be true numeric dtype (not object)
# before being handed to the rolling/opponent/park builders -- they use
# cumsum/rolling-sum internally, which raise on object dtype. An empty
# `historical_game_logs` (pd.DataFrame(columns=OUTPUT_COLUMNS), no rows) and
# the synthetic rows' NaN-filled leakage columns both default to object
# dtype until coerced, so this is applied unconditionally after
# concatenation rather than trusted to either side beforehand.
_NUMERIC_GAME_LOG_COLUMNS = [
    "pitcher", "game_pk", "rest_days",
] + LEAKAGE_OUTCOME_COLUMNS


def build_synthetic_game_rows(slate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one game_logs-schema row per slate starter: known pre-game fields
    filled from the slate, every same-game outcome/leakage column NaN,
    `day_night` None (never derivable pre-game, same as game_logs.py).
    `rest_days` is left NaN here -- it's recomputed in
    build_prediction_features AFTER concatenation with history, since a
    pitcher's true pre-game rest is today minus their last REAL start, which
    this function alone (with no history in scope) can't see.
    """
    if slate_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame(index=slate_df.index, columns=OUTPUT_COLUMNS)
    for col in SLATE_TO_GAME_LOG_FIELDS:
        out[col] = slate_df[col].values
    for col in LEAKAGE_OUTCOME_COLUMNS:
        out[col] = np.nan
    out["day_night"] = None
    out["rest_days"] = np.nan
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def build_prediction_features(historical_game_logs: pd.DataFrame, slate_df: pd.DataFrame) -> pd.DataFrame:
    """
    The label-less sibling to build_features.build_training_table(): build
    today's feature rows for the slate's probable starters by concatenating
    their season-to-date history with one synthetic today-row each, running
    the existing rolling/opponent/park builders UNMODIFIED on the combined
    table (they need the full multi-pitcher table for opponent/park
    history -- same whole-table batching build_training_table uses), then
    slicing the synthetic rows back out.

    Returns one row per slate starter (same row count as `slate_df`), with
    every rolling/opponent/park feature column attached plus the slate's own
    identifier columns. A starter with no rows in `historical_game_logs` at
    all still gets a row here, with all-NaN rolling/opponent/park features --
    whether that's enough to predict on is the model layer's call
    (transform_design_matrix's dropna), not this function's; see
    src/pipeline/refresh.py's skipped_pitchers handling.
    """
    synthetic = build_synthetic_game_rows(slate_df)
    if synthetic.empty:
        return synthetic

    if historical_game_logs is None or historical_game_logs.empty:
        history = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        history = historical_game_logs.copy()

    history[_SYNTHETIC_FLAG] = False
    synthetic = synthetic.copy()
    synthetic[_SYNTHETIC_FLAG] = True

    combined = pd.concat([history, synthetic], ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    for col in _NUMERIC_GAME_LOG_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    # Recompute rest_days post-concatenation: today's synthetic row now sees
    # its pitcher's true last REAL start, so rest_days = today - that date.
    combined["rest_days"] = combined.groupby("pitcher")["game_date"].diff().dt.days

    combined = add_rolling_features(combined)
    combined = add_opponent_features(combined)
    combined = add_park_factors(combined)

    predicted = combined[combined[_SYNTHETIC_FLAG]].drop(columns=[_SYNTHETIC_FLAG])
    return predicted.reset_index(drop=True)
