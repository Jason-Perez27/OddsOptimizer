"""
Tests for src/backtest/walk_forward.py (task #11).

Per the project's TDD mandate for this task, the leakage tests (spec testing
items #1-#2) are written FIRST, ahead of every other assertion in this file,
and are the ones this whole module is trusted against.

A deterministic fake `fit_fn` (mu = mean of training strikeouts; constant
0.5 across the threshold sweep) stands in for the real (heavy) statsmodels
fit -- it honors the exact contract run_walk_forward depends on:
`fit_fn(train_df, test_df) -> (model, diagnostics, (X_test, y_test))`, with
X_test/y_test row-aligned to `test_df.dropna(subset=CORE_PITCHER_FORM_COLUMNS
).reset_index(drop=True)`. This keeps every test here no-network and fast.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.walk_forward import run_walk_forward, melt_oos_sweep
from src.models.baseline_model import CORE_PITCHER_FORM_COLUMNS, THRESHOLDS


CORE_FILL = {
    "k_rate_last5": 0.25,
    "whiff_rate_last5": 0.30,
    "velo_avg_last5": 95.0,
    "pitch_count_avg_last5": 90.0,
}


def _row(pitcher, game_pk, game_date, strikeouts, **overrides):
    row = {"pitcher": pitcher, "game_pk": game_pk, "game_date": game_date, "strikeouts": strikeouts}
    row.update(CORE_FILL)
    row.update(overrides)
    return row


class _FakeModel:
    """Deterministic stand-in for BaselineModel -- no statsmodels/scipy."""

    def __init__(self, mu_value: float, family: str = "poisson", alpha=None):
        self.family = family
        self.alpha = alpha
        self._mu_value = mu_value

    def predict_mean(self, X):
        return np.full(len(X), self._mu_value, dtype=float)

    def predict_over_prob_sweep(self, X, thresholds=THRESHOLDS):
        # Constant filler probability -- these tests are about the
        # windowing/leakage/accumulation plumbing, not sweep math (that's
        # already covered by tests/test_baseline_model.py and
        # tests/test_tiering.py against the real survival functions).
        data = {t: np.full(len(X), 0.5, dtype=float) for t in thresholds}
        return pd.DataFrame(data, index=range(len(X)))


def make_fake_fit_fn():
    """Returns (fit_fn, calls) -- calls records every (train_df, test_df) pair
    fit_fn was invoked with, so tests can inspect exactly what each step saw."""
    calls = []

    def fit_fn(train_df, test_df=None):
        calls.append((train_df.copy(), test_df.copy() if test_df is not None else None))
        mu_value = float(train_df["strikeouts"].mean()) if len(train_df) else 5.0
        model = _FakeModel(mu_value)

        if test_df is None:
            return model, {"n_train": len(train_df)}, (None, None)

        survivors = test_df.dropna(subset=CORE_PITCHER_FORM_COLUMNS).reset_index(drop=True)
        X_test = survivors
        y_test = survivors["strikeouts"] if "strikeouts" in survivors.columns else pd.Series([], dtype=float)
        return model, {"n_train": len(train_df)}, (X_test, y_test)

    return fit_fn, calls


# ---------------------------------------------------------------------------
# 1. No temporal leakage (spec testing item #1) -- written FIRST.
# ---------------------------------------------------------------------------

def test_no_training_row_is_dated_on_or_after_its_step_predict_window():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
        _row(1, 103, "2026-04-22", 8),
    ]
    df = pd.DataFrame(rows)
    fit_fn, calls = make_fake_fit_fn()

    run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=1)

    assert calls  # at least one step actually ran
    for train_df, test_df in calls:
        if test_df is None or test_df.empty or train_df.empty:
            continue
        assert train_df["game_date"].max() < test_df["game_date"].min(), (
            "a training row was dated on/after the predict window's earliest date"
        )


def test_every_oos_row_is_dated_on_or_after_its_wf_step():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
    ]
    df = pd.DataFrame(rows)
    fit_fn, _ = make_fake_fit_fn()

    oos = run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=1)

    assert not oos.empty
    wf_step = pd.to_datetime(oos["wf_step"])
    game_date = pd.to_datetime(oos["game_date"])
    assert (game_date >= wf_step).all()


def test_a_constructed_leak_would_change_mu_but_the_real_run_does_not_leak():
    # Pitcher 1: a quiet early start (sk=2) on day0, then a huge start
    # (sk=100) on day7 that falls inside the SECOND window's predict range.
    # If walk_forward leaked day7's own row into its training set, the fake
    # fit_fn's mu (mean of training strikeouts) for that step would jump to
    # include 100. The correct, non-leaking mu is just the prior game's
    # value, 2.0.
    rows = [
        _row(1, 100, "2026-04-01", 2),
        _row(1, 101, "2026-04-08", 100),
    ]
    df = pd.DataFrame(rows)
    fit_fn, _ = make_fake_fit_fn()

    oos = run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=1)

    row_for_game101 = oos[oos["game_pk"] == 101].iloc[0]
    leaked_mu = pd.Series([2, 100]).mean()  # what mu WOULD be if day7 leaked into training
    correct_mu = 2.0  # what mu should be: only day0's start is strictly prior

    assert row_for_game101["mu"] == pytest.approx(correct_mu)
    assert row_for_game101["mu"] != pytest.approx(leaked_mu)


# ---------------------------------------------------------------------------
# 2. Leakage invariant under appended data (spec testing item #2) -- written
#    FIRST, mirrors tests/test_rolling_features.py's leakage-invariant test.
# ---------------------------------------------------------------------------

def test_earlier_step_oos_predictions_unchanged_when_later_games_appended():
    base_rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
        _row(1, 103, "2026-04-22", 8),
    ]
    later_rows = [
        _row(1, 104, "2026-04-29", 9),
        _row(1, 105, "2026-05-06", 11),
    ]

    fit_fn_a, _ = make_fake_fit_fn()
    oos_before = run_walk_forward(pd.DataFrame(base_rows), step=7, fit_fn=fit_fn_a, min_train_dates=1)

    fit_fn_b, _ = make_fake_fit_fn()
    oos_after = run_walk_forward(pd.DataFrame(base_rows + later_rows), step=7, fit_fn=fit_fn_b, min_train_dates=1)

    before_by_pk = oos_before.set_index("game_pk")
    after_by_pk = oos_after.set_index("game_pk")

    for game_pk in before_by_pk.index:
        for col in ["mu", "family", "wf_step", "p_over_1", "p_over_10"]:
            a = before_by_pk.loc[game_pk, col]
            b = after_by_pk.loc[game_pk, col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == b, f"game_pk {game_pk} col {col} changed when later games were appended"


# ---------------------------------------------------------------------------
# 3. Expanding window mechanics / min_train_dates (spec testing item #3)
# ---------------------------------------------------------------------------

def test_training_set_grows_across_consecutive_steps():
    rows = [
        _row(1, 100 + i, pd.Timestamp("2026-04-01") + pd.Timedelta(days=7 * i), 5)
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    fit_fn, calls = make_fake_fit_fn()

    run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=1)

    train_sizes = [len(train_df) for train_df, test_df in calls if test_df is not None and not test_df.empty]
    assert train_sizes == sorted(train_sizes)  # non-decreasing -- expanding window
    assert train_sizes[-1] > train_sizes[0]


def test_min_train_dates_skips_early_steps_cleanly():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
    ]
    df = pd.DataFrame(rows)
    fit_fn, calls = make_fake_fit_fn()

    oos = run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=2)

    # The first step (train dates={} ) and second step (train dates={4/1})
    # both have fewer than 2 distinct training dates -- skipped, no crash,
    # no row for game_pk 100 or 101.
    assert set(oos["game_pk"]) == {102}


# ---------------------------------------------------------------------------
# 4. OOS coverage, doubleheader-safe (spec testing item #4)
# ---------------------------------------------------------------------------

def test_each_eval_start_appears_exactly_once_and_doubleheaders_stay_distinct():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),   # game 1 of a same-day doubleheader
        _row(1, 102, "2026-04-08", 9),   # game 2 of the same doubleheader
        _row(1, 103, "2026-04-15", 7),
    ]
    df = pd.DataFrame(rows)
    fit_fn, _ = make_fake_fit_fn()

    oos = run_walk_forward(df, step=7, fit_fn=fit_fn, min_train_dates=1)

    assert oos["game_pk"].is_unique
    assert {101, 102}.issubset(set(oos["game_pk"]))


# ---------------------------------------------------------------------------
# 5. Metric reuse + PIT (spec testing item #5)
# ---------------------------------------------------------------------------

def _scipy_stats():
    return pytest.importorskip("scipy.stats")


def test_point_accuracy_reused_directly_on_oos_frame():
    from src.evaluation import metrics

    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)

    result = metrics.point_accuracy(oos)
    assert result["n"] == len(oos)
    assert not np.isnan(result["mae"])


def test_pit_histogram_reused_directly_on_oos_frame():
    _scipy_stats()
    from src.evaluation import metrics

    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
        _row(1, 103, "2026-04-22", 8),
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)

    hist = metrics.pit_histogram(oos)
    assert list(hist.columns) == metrics.PIT_HISTOGRAM_COLUMNS
    assert hist["n"].sum() == len(oos)


def test_melt_oos_sweep_is_grade_threshold_sweep_shaped_for_metric_reuse():
    from src.evaluation import metrics

    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)

    long_df = melt_oos_sweep(oos)
    assert {"threshold", "p_over", "over_hit", "tier"}.issubset(long_df.columns)
    assert len(long_df) == len(oos) * len(THRESHOLDS)

    table = metrics.reliability_table(long_df, n_buckets=2)
    assert len(table) == 2


# ---------------------------------------------------------------------------
# 6. Slices: by threshold, by sweep-tier, over time (spec testing item #6)
# ---------------------------------------------------------------------------

def test_slices_by_threshold_tier_and_wf_step_are_groupable():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6),
        _row(1, 102, "2026-04-15", 7),
        _row(1, 103, "2026-04-22", 8),
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)
    long_df = melt_oos_sweep(oos)

    by_threshold = long_df.groupby("threshold")["p_over"].mean()
    assert set(by_threshold.index) == set(THRESHOLDS)

    by_tier = long_df.groupby("tier")["over_hit"].mean()
    assert len(by_tier) >= 1

    by_step = oos.groupby("wf_step")["mu"].mean()
    assert len(by_step) >= 1


# ---------------------------------------------------------------------------
# 7. No-ROI guard (spec testing item #7)
# ---------------------------------------------------------------------------

def test_no_roi_line_or_edge_fields_in_output():
    rows = [_row(1, 100, "2026-04-01", 5), _row(1, 101, "2026-04-08", 6)]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)

    forbidden_substrings = ["roi", "line", "edge", "pnl"]
    lowered_cols = [c.lower() for c in oos.columns]
    for forbidden in forbidden_substrings:
        assert not any(forbidden in c for c in lowered_cols), (
            f"OOS output unexpectedly contains a {forbidden!r}-named column -- "
            f"Track A has no historical lines, no ROI/line/edge fields allowed"
        )


# ---------------------------------------------------------------------------
# 9. No-network end-to-end (spec testing item #9)
# ---------------------------------------------------------------------------

def test_runs_end_to_end_no_network_with_injected_fit_fn():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(2, 200, "2026-04-01", 4),
        _row(1, 101, "2026-04-08", 6),
        _row(2, 201, "2026-04-08", 9),
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)
    assert not oos.empty


# ---------------------------------------------------------------------------
# 10. Insufficient-history step never raises (spec testing item #10)
# ---------------------------------------------------------------------------

def test_insufficient_history_never_raises_and_returns_well_formed_empty_frame():
    rows = [_row(1, 100, "2026-04-01", 5)]
    fit_fn, _ = make_fake_fit_fn()

    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=100)

    assert oos.empty
    assert "pitcher" in oos.columns
    assert "p_over_1" in oos.columns


def test_empty_feature_table_returns_empty_frame_with_columns():
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(), step=7, fit_fn=fit_fn, min_train_dates=1)
    assert oos.empty
    assert "wf_step" in oos.columns


def test_core_feature_nulls_drop_the_row_without_raising():
    rows = [
        _row(1, 100, "2026-04-01", 5),
        _row(1, 101, "2026-04-08", 6, k_rate_last5=np.nan),  # missing core feature
    ]
    fit_fn, _ = make_fake_fit_fn()
    oos = run_walk_forward(pd.DataFrame(rows), step=7, fit_fn=fit_fn, min_train_dates=1)
    assert 101 not in set(oos["game_pk"])
