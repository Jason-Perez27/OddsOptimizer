"""
Grading / join logic for task #10 (prediction outcome tracking & backtest).

Design: docs/design/specs/2026-06-27-outcome-tracking-design.md (section
2, "Grading / join logic"). Pure functions only -- frames in, frames out, no
network, no file IO, no wall-clock reads (the "clock" is always an injected
`now`) -- the same testability contract as src/predictions/tiering.py.

Canonical join key throughout: (pitcher, game_pk) -- doubleheader-safe. Never
join on game_date alone (the spec's edge case: "a game_date-only join would be
caught failing" on a doubleheader, since one pitcher can have two game_pk rows
on the same date).

Design choice (not spelled out as exact signatures in the spec, only as
prose + a bare `(pred_df, realized_df) -> DataFrame` bullet): assigning the
3-way `settlement_status` (settled / pending / void_scratched) requires a
clock (`now`) and a wait-day threshold (`scratch_void_max_wait_days`) -- per
the spec's own testing-approach item #2 ("driven by the injected now"). Only
`attach_outcomes` does that timing classification, because only the frames it
is meant to run against (`predictions_df` / `threshold_table_df`) carry a
`game_date` column to measure elapsed time from. `line_picks_df` (see
src/predictions/tiering.py LINE_PICKS_COLUMNS) carries `start_time`, not
`game_date` -- it is not the frame timing decisions are made from.

`grade_line_picks` / `grade_threshold_sweep` therefore do NOT assign the 3-way
`settlement_status` themselves -- they only attach `realized_strikeouts` and
the grading-specific derived columns, treating "no realized row" generically
as unsettled (every derived column NaNs out the same way regardless of
whether the true reason is "pending" or "void_scratched" -- the grading math
genuinely does not care which). The orchestrator (src/pipeline/settle.py)
calls `attach_outcomes` once against the date's predictions/threshold_table
(which has `game_date`) to get the authoritative 3-way status per
`(pitcher, game_pk)`, then merges that status (and the partition's
`game_date`, which it already knows -- it's the partition key) onto the
graded line-picks/threshold-sweep frames before writing them to disk, to
match the `graded_line_picks.csv` / `graded_threshold_sweep.csv` schemas in
the spec exactly.
"""

import numpy as np
import pandas as pd

SETTLED = "settled"
PENDING = "pending"
VOID_SCRATCHED = "void_scratched"


def _require_game_pk(df: pd.DataFrame, label: str) -> None:
    """
    Edge case (spec): `game_pk` absent on a prediction-shaped table is a task
    #9 regression -- fail fast with a clear message rather than silently
    falling back to a `(pitcher, game_date)` join, which a doubleheader would
    corrupt (two real games collapsed into one).
    """
    if "game_pk" not in df.columns:
        raise ValueError(
            f"{label} is missing 'game_pk' -- cannot join safely on "
            f"(pitcher, game_pk). Joining on (pitcher, game_date) alone would "
            f"silently corrupt doubleheaders. This indicates a task #9 "
            f"regression upstream; fix the producer, don't work around it here."
        )


def _empty_realized() -> pd.DataFrame:
    return pd.DataFrame(columns=["pitcher", "game_pk", "realized_strikeouts"])


def _realized_lookup(realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Narrow `realized_df` (the output of aggregate_pitcher_games, with many
    columns) down to just the join key + the one stat task #10 grades:
    strikeouts, scoped to strikeouts-only per the spec's stated scope.
    """
    if realized_df is None or realized_df.empty:
        return _empty_realized()
    cols = realized_df[["pitcher", "game_pk", "strikeouts"]].copy()
    cols = cols.rename(columns={"strikeouts": "realized_strikeouts"})
    # Defensive: realized rows should be unique per (pitcher, game_pk) -- a
    # duplicate would silently fan out the left join. Keep the first and
    # don't crash; settle.py's own data path (aggregate_pitcher_games) never
    # produces duplicates, but hand-built test fixtures could.
    cols = cols.drop_duplicates(subset=["pitcher", "game_pk"], keep="first")
    return cols


def attach_outcomes(
    pred_df: pd.DataFrame,
    realized_df: pd.DataFrame,
    *,
    now,
    scratch_void_max_wait_days: float,
    settlement_lag_hours: float = 0.0,
) -> pd.DataFrame:
    """
    Left-join realized strikeouts onto `pred_df` by (pitcher, game_pk) and
    assign `settlement_status` (settled / pending / void_scratched).

    `pred_df` must carry `pitcher`, `game_pk`, and `game_date` (predictions.csv
    or threshold_table.csv shaped). Left join (not inner), so unresolved
    predictions stay visible -- never silently dropped.

    Timing (spec section 1):
    - settled: a realized row exists.
    - void_scratched: no realized row, and more than
      `scratch_void_max_wait_days` days have elapsed since `game_date` as of
      `now` -- the probable starter never threw (scratch/postponement/DNP).
    - pending: no realized row, still within the wait window -- resolves on a
      later settle pass.

    `settlement_lag_hours` is accepted for completeness/injection (the spec's
    "lag / max-wait" pairing) but, since `scratch_void_max_wait_days` is
    always >= the lag in any sane config, it does not change the 3-way
    classification here -- `settle.py` is the one that actually uses
    `settlement_lag_hours` to decide *which dates to even attempt* (skip a
    date that's too recent to expect Statcast data for at all). Kept as a
    parameter so callers/tests can inject it without it silently doing the
    wrong thing if a future config ever sets lag > max-wait.
    """
    _require_game_pk(pred_df, "pred_df")

    out = pred_df.copy()
    if out.empty:
        out["realized_strikeouts"] = pd.Series(dtype="float64")
        out["settlement_status"] = pd.Series(dtype="object")
        return out

    realized = _realized_lookup(realized_df)
    out = out.merge(realized, on=["pitcher", "game_pk"], how="left")

    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_localize(None)
    game_dates = pd.to_datetime(out["game_date"])
    if getattr(game_dates.dt, "tz", None) is not None:
        game_dates = game_dates.dt.tz_localize(None)
    elapsed_days = (now_ts - game_dates).dt.total_seconds() / 86400.0

    statuses = []
    for realized_val, elapsed in zip(out["realized_strikeouts"], elapsed_days):
        if pd.notna(realized_val):
            statuses.append(SETTLED)
        elif elapsed > scratch_void_max_wait_days:
            statuses.append(VOID_SCRATCHED)
        else:
            statuses.append(PENDING)
    out["settlement_status"] = statuses
    return out


def grade_line_picks(line_picks_df: pd.DataFrame, realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per pick (spec section 2):
    - over_hit = realized_strikeouts >= line_threshold (NaN if unsettled).
    - push = True only for an integer line with realized == line (always a
      real bool, never NaN -- "no realized row" simply isn't a push).
    - pick_correct = does `lean` match the realized side; NaN on push or
      unsettled.
    - pnl_units = +1 win / -1 loss / 0 push / NaN unsettled (flat-bet proxy).

    Does not assign the 3-way `settlement_status` -- see module docstring.
    Doubleheader-safe: joins on (pitcher, game_pk) only, never game_date.
    """
    _require_game_pk(line_picks_df, "line_picks_df")

    out = line_picks_df.copy()
    if out.empty:
        for col in ("realized_strikeouts", "over_hit", "pick_correct", "pnl_units"):
            out[col] = pd.Series(dtype="float64" if col != "over_hit" and col != "pick_correct" else "object")
        out["push"] = pd.Series(dtype="bool")
        return out

    realized = _realized_lookup(realized_df)
    out = out.merge(realized, on=["pitcher", "game_pk"], how="left")

    over_hits = []
    pushes = []
    pick_corrects = []
    pnls = []
    for _, row in out.iterrows():
        realized_val = row["realized_strikeouts"]
        unsettled = pd.isna(realized_val)

        is_push = (
            not unsettled
            and float(row["line"]).is_integer()
            and realized_val == row["line"]
        )

        if unsettled:
            over_hit = np.nan
        else:
            over_hit = bool(realized_val >= row["line_threshold"])

        if unsettled:
            pick_correct = np.nan
        elif is_push:
            pick_correct = np.nan
        elif row["lean"] == "over":
            pick_correct = bool(over_hit)
        else:  # "under"
            pick_correct = bool(not over_hit)

        if unsettled:
            pnl = np.nan
        elif is_push:
            pnl = 0
        elif pick_correct:
            pnl = 1
        else:
            pnl = -1

        over_hits.append(over_hit)
        pushes.append(bool(is_push))
        pick_corrects.append(pick_correct)
        pnls.append(pnl)

    out["over_hit"] = over_hits
    out["push"] = pushes
    out["pick_correct"] = pick_corrects
    out["pnl_units"] = pnls
    return out


def grade_threshold_sweep(threshold_table_df: pd.DataFrame, realized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per pitcher x threshold (spec section 2): over_hit = realized_strikeouts
    >= threshold (NaN if unsettled). This is the frame calibration is computed
    from -- every threshold is a P(K>=t) the model committed to, line or no
    line.

    Does not assign the 3-way `settlement_status` -- see module docstring.
    Doubleheader-safe: joins on (pitcher, game_pk) only, never game_date.
    """
    _require_game_pk(threshold_table_df, "threshold_table_df")

    out = threshold_table_df.copy()
    if out.empty:
        out["realized_strikeouts"] = pd.Series(dtype="float64")
        out["over_hit"] = pd.Series(dtype="object")
        return out

    realized = _realized_lookup(realized_df)
    out = out.merge(realized, on=["pitcher", "game_pk"], how="left")

    out["over_hit"] = [
        np.nan if pd.isna(realized_val) else bool(realized_val >= threshold)
        for realized_val, threshold in zip(out["realized_strikeouts"], out["threshold"])
    ]
    return out
