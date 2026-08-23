"""
Walk-forward backtest for the baseline model (task #11).

Design: docs/design/specs/2026-06-27-baseline-validation-design.md,
section 2 "Walk-forward backtest". Expanding window, stepping by `step` days
(default weekly -- daily refit is marginal gain for the cost). At each cutoff
date `c`: train on starts with `game_date < c`, fit via the injected
`fit_fn` (default `fit_baseline_model`), predict the starts in
`[c, c + step)`, and accumulate out-of-sample predictions (`mu`, `family`,
`alpha`, the 1-10 `p_over` sweep) joined to realized `strikeouts` on
`(pitcher, game_pk)`.

Temporal correctness is the highest-stakes property here: no training row
may be dated on/after its step's cutoff, and an earlier step's OOS
predictions must not change when later games are appended to the corpus.
tests/test_walk_forward.py pins this directly (testing items #1-#2) --
those tests were written FIRST, before this module, per the project's TDD
mandate for this task.

`fit_fn` contract (matches src.models.baseline_model.fit_baseline_model's
actual behavior, documented here since it's an integration point this
module depends on but that isn't spelled out as a public contract
elsewhere): `fit_fn(train_df, test_df) -> (model, diagnostics, (X_test,
y_test))`, where `model` exposes `.family`, `.alpha`, `.predict_mean(X)`,
and `.predict_over_prob_sweep(X, thresholds)`; `X_test`/`y_test` are
row-aligned to `test_df.dropna(subset=CORE_PITCHER_FORM_COLUMNS).reset_index
(drop=True)` -- the exact row selection/order
`baseline_model.transform_design_matrix` produces, since dropping null core
features never reorders rows. This module recomputes that same survivor set
directly from `test_df` (via `CORE_PITCHER_FORM_COLUMNS`, a public constant)
to attach identifiers (`pitcher`, `game_pk`, `game_date`) to each OOS
prediction -- it does not reach into `fit_baseline_model`'s private
`_dropna_core` to do this, but reproduces the identical, public-constant-
driven selection.

Track A only -- no historical lines exist for this corpus, so the output
schema below carries no line/edge/ROI fields. test_no_roi_fields_in_output
guards against fabricating betting numbers here.

Tiers bonus: because the OOS sweep carries `p_over` per threshold,
`tier(p_over)` (src.predictions.tiering.tier) is attached per threshold --
a first read on whether the sweep tiers are calibrated, useful input to the
task #10 tier-validation work.

Output schema (one row per evaluated pitcher-start, wide -- matches the
spec's literal column list; `data/processed/backtest/walk_forward_oos.csv`):
pitcher, game_pk, game_date, wf_step, family, mu, alpha, p_over_<t> for each
t in `thresholds`, realized_strikeouts, over_hit_<t> for each t, tier_<t>
for each t.

`melt_oos_sweep` reshapes that wide frame into the long, one-row-per-
(pitcher-game, threshold) shape that src/evaluation/metrics.py's
threshold-based helpers (reliability_table, brier_and_log_loss_at_thresholds)
and src/evaluation/grading.grade_threshold_sweep expect -- used by
src/backtest/report.py (task #11 step 6) so those metric helpers are reused,
never reimplemented, against walk-forward output.
"""

import numpy as np
import pandas as pd

from src.models.baseline_model import (
    CORE_PITCHER_FORM_COLUMNS,
    THRESHOLDS,
    fit_baseline_model,
)
from src.predictions.tiering import tier

DEFAULT_STEP_DAYS = 7
DEFAULT_MIN_TRAIN_DATES = 14


def _oos_columns(thresholds) -> list:
    return (
        ["pitcher", "game_pk", "game_date", "wf_step", "family", "mu", "alpha"]
        + [f"p_over_{t}" for t in thresholds]
        + ["realized_strikeouts"]
        + [f"over_hit_{t}" for t in thresholds]
        + [f"tier_{t}" for t in thresholds]
    )


def run_walk_forward(
    feature_table: pd.DataFrame,
    *,
    step: int = DEFAULT_STEP_DAYS,
    fit_fn=fit_baseline_model,
    min_train_dates: int = DEFAULT_MIN_TRAIN_DATES,
    thresholds=THRESHOLDS,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward backtest over `feature_table` (one row per
    pitcher-start, as produced by src.features.build_features.
    build_training_table + src.backtest.corpus.filter_starters -- features
    are already built on the FULL corpus, leakage-safe by construction; this
    function governs only the train/predict cutoff per step, never feature
    re-derivation).

    `step` is the step size in days (default 7 -- weekly). `min_train_dates`
    is the minimum number of distinct training `game_date` values required
    before a step is even attempted; steps that don't meet it are skipped
    entirely (no row emitted for that window, no exception raised) -- this
    is what keeps the early season from being scored against a model fit on
    almost nothing.

    Returns a wide OOS frame (see module docstring for the exact schema).
    Empty input returns an empty frame with the correct columns.
    """
    columns = _oos_columns(thresholds)
    if feature_table.empty:
        return pd.DataFrame(columns=columns)

    df = feature_table.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    start = df["game_date"].min()
    end = df["game_date"].max()
    step_td = pd.Timedelta(days=step)

    rows = []
    cutoff = pd.Timestamp(start)
    while cutoff <= pd.Timestamp(end):
        window_end = cutoff + step_td  # exclusive upper bound

        train_df = df[df["game_date"] < cutoff]
        predict_df = df[(df["game_date"] >= cutoff) & (df["game_date"] < window_end)]

        n_train_dates = train_df["game_date"].nunique()
        if n_train_dates < min_train_dates or predict_df.empty:
            cutoff = window_end
            continue

        # The same row selection/order transform_design_matrix produces (see
        # module docstring) -- used to attach identifiers to X_test/y_test,
        # which fit_fn returns with no identifier columns of their own.
        survivors = predict_df.dropna(subset=CORE_PITCHER_FORM_COLUMNS).reset_index(drop=True)
        if survivors.empty:
            cutoff = window_end
            continue

        model, _diagnostics, (X_test, y_test) = fit_fn(train_df, predict_df)

        if X_test is None or len(X_test) == 0:
            cutoff = window_end
            continue

        mu = model.predict_mean(X_test)
        sweep = model.predict_over_prob_sweep(X_test, thresholds)

        for i in range(len(survivors)):
            ident = survivors.iloc[i]
            realized = float(y_test.iloc[i]) if not pd.isna(y_test.iloc[i]) else np.nan
            row = {
                "pitcher": ident["pitcher"],
                "game_pk": ident["game_pk"],
                "game_date": ident["game_date"],
                "wf_step": cutoff.date().isoformat(),
                "family": model.family,
                "mu": float(mu[i]),
                "alpha": model.alpha if model.alpha is not None else np.nan,
                "realized_strikeouts": realized,
            }
            for t in thresholds:
                p_over = float(sweep.iloc[i][t])
                row[f"p_over_{t}"] = p_over
                row[f"over_hit_{t}"] = (
                    np.nan if pd.isna(realized) else bool(realized >= t)
                )
                row[f"tier_{t}"] = tier(p_over)
            rows.append(row)

        cutoff = window_end

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def melt_oos_sweep(oos_df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """
    Reshape the wide `run_walk_forward` output into the long, one-row-per-
    (pitcher-game, threshold) shape src/evaluation/metrics.py's threshold-
    based helpers and src/evaluation/grading.grade_threshold_sweep expect:
    a `threshold` column alongside `p_over` / `over_hit` (instead of the wide
    `p_over_<t>` / `over_hit_<t>` columns).
    """
    long_columns = [
        "pitcher", "game_pk", "game_date", "wf_step", "realized_strikeouts",
        "threshold", "p_over", "over_hit", "tier",
    ]
    if oos_df.empty:
        return pd.DataFrame(columns=long_columns)

    id_cols = ["pitcher", "game_pk", "game_date", "wf_step", "realized_strikeouts"]
    rows = []
    for _, row in oos_df.iterrows():
        base = {c: row[c] for c in id_cols}
        for t in thresholds:
            rows.append({
                **base,
                "threshold": t,
                "p_over": row[f"p_over_{t}"],
                "over_hit": row[f"over_hit_{t}"],
                "tier": row[f"tier_{t}"],
            })
    return pd.DataFrame(rows, columns=long_columns)
