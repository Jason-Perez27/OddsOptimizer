"""
Unit tests for src/evaluation/metrics.py (task #10, spec section 4(a) +
"Testing approach" items 6-7).

Covers:
  6. Calibration + ECE: reliability buckets and the ECE scalar match a
     hand-computed reference on a tiny fixed set.
  7. Brier / log loss reuse baseline_model helpers and match hand-computed
     values on the over/under binaries.

Run with: pytest tests/test_metrics.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import metrics


# ---------------------------------------------------------------------------
# 6. Calibration + ECE
# ---------------------------------------------------------------------------

def test_reliability_table_two_buckets_hand_computed():
    # n_buckets=2 -> edges [0, 0.5, 1.0].
    # bucket0 (p_over < 0.5): rows (0.2, False), (0.3, True)
    #   mean_predicted = 0.25, empirical_rate = 0.5
    # bucket1 (p_over >= 0.5): rows (0.8, True), (0.9, True)
    #   mean_predicted = 0.85, empirical_rate = 1.0
    sweep = pd.DataFrame({
        "p_over": [0.2, 0.3, 0.8, 0.9],
        "over_hit": [False, True, True, True],
    })

    table = metrics.reliability_table(sweep, n_buckets=2)

    assert len(table) == 2
    bucket0, bucket1 = table.iloc[0], table.iloc[1]
    assert bucket0["n"] == 2
    assert bucket0["mean_predicted"] == pytest.approx(0.25)
    assert bucket0["empirical_rate"] == pytest.approx(0.5)
    assert bucket1["n"] == 2
    assert bucket1["mean_predicted"] == pytest.approx(0.85)
    assert bucket1["empirical_rate"] == pytest.approx(1.0)


def test_expected_calibration_error_hand_computed():
    sweep = pd.DataFrame({
        "p_over": [0.2, 0.3, 0.8, 0.9],
        "over_hit": [False, True, True, True],
    })
    table = metrics.reliability_table(sweep, n_buckets=2)

    ece = metrics.expected_calibration_error(table)

    # ECE = (2/4)*|0.25-0.5| + (2/4)*|0.85-1.0| = 0.5*0.25 + 0.5*0.15 = 0.2
    assert ece == pytest.approx(0.2)


def test_reliability_table_excludes_unsettled_rows():
    sweep = pd.DataFrame({
        "p_over": [0.2, 0.9],
        "over_hit": [False, np.nan],
    })

    table = metrics.reliability_table(sweep, n_buckets=2)

    assert table["n"].sum() == 1


def test_reliability_table_empty_buckets_are_nan_not_dropped():
    # All rows fall in bucket0 -- bucket1 must still appear, with n=0 and NaN
    # rates, never silently dropped from the table.
    sweep = pd.DataFrame({"p_over": [0.1, 0.2], "over_hit": [True, False]})

    table = metrics.reliability_table(sweep, n_buckets=2)

    assert len(table) == 2
    assert table.iloc[1]["n"] == 0
    assert pd.isna(table.iloc[1]["mean_predicted"])
    assert pd.isna(table.iloc[1]["empirical_rate"])


def test_expected_calibration_error_all_unsettled_is_nan():
    sweep = pd.DataFrame({"p_over": [0.2, 0.9], "over_hit": [np.nan, np.nan]})
    table = metrics.reliability_table(sweep, n_buckets=2)

    assert math.isnan(metrics.expected_calibration_error(table))


# ---------------------------------------------------------------------------
# 7. Brier / log loss reuse
# ---------------------------------------------------------------------------

def test_brier_and_log_loss_at_line_hand_computed():
    # y = [1, 0, 0], p = [0.7, 0.3, 0.6]; one extra unsettled row excluded.
    picks = pd.DataFrame({
        "p_over": [0.7, 0.3, 0.6, 0.5],
        "over_hit": [True, False, False, np.nan],
    })

    result = metrics.brier_and_log_loss_at_line(picks)

    expected_brier = ((0.7 - 1) ** 2 + (0.3 - 0) ** 2 + (0.6 - 0) ** 2) / 3
    expected_log_loss = -(
        (1 * math.log(0.7) + 0 * math.log(0.3))
        + (0 * math.log(0.7) + 1 * math.log(1 - 0.3))
        + (0 * math.log(0.6) + 1 * math.log(1 - 0.6))
    ) / 3

    assert result["n"] == 3
    assert result["brier_score"] == pytest.approx(expected_brier)
    assert result["log_loss"] == pytest.approx(expected_log_loss)


def test_brier_and_log_loss_at_thresholds_separates_by_threshold():
    sweep = pd.DataFrame({
        "threshold": [6, 6, 7, 7],
        "p_over": [0.6, 0.4, 0.7, 0.5],
        "over_hit": [True, False, True, True],
    })

    result = metrics.brier_and_log_loss_at_thresholds(sweep, thresholds=(6, 7))

    assert result[6]["n"] == 2
    assert result[7]["n"] == 2
    expected_brier_6 = ((0.6 - 1) ** 2 + (0.4 - 0) ** 2) / 2
    assert result[6]["brier_score"] == pytest.approx(expected_brier_6)


def test_brier_and_log_loss_at_line_empty_is_nan():
    picks = pd.DataFrame({"p_over": [0.5], "over_hit": [np.nan]})
    result = metrics.brier_and_log_loss_at_line(picks)
    assert result["n"] == 0
    assert math.isnan(result["brier_score"])
    assert math.isnan(result["log_loss"])


# ---------------------------------------------------------------------------
# Point accuracy (MAE/RMSE reuse)
# ---------------------------------------------------------------------------

def test_point_accuracy_hand_computed():
    attached = pd.DataFrame({
        "mu": [5.0, 6.0, 7.0],
        "realized_strikeouts": [6.0, 6.0, 5.0],
    })

    result = metrics.point_accuracy(attached)

    # errors: |5-6|=1, |6-6|=0, |7-5|=2 -> MAE = 1.0
    assert result["mae"] == pytest.approx(1.0)
    # squared errors: 1, 0, 4 -> mean=5/3 -> rmse = sqrt(5/3)
    assert result["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert result["n"] == 3


def test_point_accuracy_excludes_unsettled():
    attached = pd.DataFrame({
        "mu": [5.0, 6.0],
        "realized_strikeouts": [6.0, np.nan],
    })

    result = metrics.point_accuracy(attached)

    assert result["n"] == 1
    assert result["mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PIT (task #11, testing item #5)
# ---------------------------------------------------------------------------

# Function-scoped (not module-level) importorskip: a module-level skip would
# skip every test in this file, including the scipy-independent calibration/
# brier/point_accuracy tests above. Each PIT test pulls scipy_stats itself so
# only the PIT tests are skipped when scipy is unavailable.

def _scipy_stats():
    return pytest.importorskip("scipy.stats")


def test_pit_values_poisson_hand_computed():
    scipy_stats = _scipy_stats()
    # mu=4, realized=4: mid-PIT = P(K<=3) + 0.5*P(K=4).
    cdf_below = scipy_stats.poisson.cdf(3, 4)
    pmf_at = scipy_stats.poisson.pmf(4, 4)
    expected = cdf_below + 0.5 * pmf_at

    graded = pd.DataFrame({
        "family": ["poisson"],
        "mu": [4.0],
        "alpha": [np.nan],
        "realized_strikeouts": [4.0],
    })

    pit = metrics.pit_values(graded)

    assert len(pit) == 1
    assert pit.iloc[0] == pytest.approx(expected)


def test_pit_values_negative_binomial_hand_computed():
    scipy_stats = _scipy_stats()
    mu, alpha, y = 5.0, 0.3, 6.0
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    expected = scipy_stats.nbinom.cdf(y - 1, n, p) + 0.5 * scipy_stats.nbinom.pmf(y, n, p)

    graded = pd.DataFrame({
        "family": ["negative_binomial"],
        "mu": [mu],
        "alpha": [alpha],
        "realized_strikeouts": [y],
    })

    pit = metrics.pit_values(graded)

    assert pit.iloc[0] == pytest.approx(expected)


def test_pit_values_excludes_unsettled_rows():
    _scipy_stats()
    graded = pd.DataFrame({
        "family": ["poisson", "poisson"],
        "mu": [4.0, 5.0],
        "alpha": [np.nan, np.nan],
        "realized_strikeouts": [4.0, np.nan],
    })

    pit = metrics.pit_values(graded)

    assert len(pit) == 1


def test_pit_values_mixed_families_in_one_frame():
    _scipy_stats()
    graded = pd.DataFrame({
        "family": ["poisson", "negative_binomial"],
        "mu": [4.0, 5.0],
        "alpha": [np.nan, 0.3],
        "realized_strikeouts": [4.0, 6.0],
    })

    pit = metrics.pit_values(graded)

    assert len(pit) == 2
    assert ((pit >= 0) & (pit <= 1)).all()


def test_pit_histogram_hand_computed_buckets():
    _scipy_stats()
    # Two poisson rows whose mid-PIT values fall in different halves of [0,1]
    # when bucketed n_buckets=2.
    graded = pd.DataFrame({
        "family": ["poisson", "poisson"],
        "mu": [10.0, 1.0],
        "alpha": [np.nan, np.nan],
        # mu=10, y=2 -> deep left tail -> low PIT (bucket 0).
        # mu=1, y=5 -> deep right tail -> high PIT (bucket 1).
        "realized_strikeouts": [2.0, 5.0],
    })

    hist = metrics.pit_histogram(graded, n_buckets=2)

    assert list(hist.columns) == metrics.PIT_HISTOGRAM_COLUMNS
    assert len(hist) == 2
    assert hist["n"].sum() == 2
    assert hist.iloc[0]["n"] == 1
    assert hist.iloc[1]["n"] == 1


def test_pit_histogram_empty_buckets_not_dropped():
    _scipy_stats()
    graded = pd.DataFrame({
        "family": ["poisson"],
        "mu": [10.0],
        "alpha": [np.nan],
        "realized_strikeouts": [2.0],  # low PIT -> lands in bucket 0 only
    })

    hist = metrics.pit_histogram(graded, n_buckets=4)

    assert len(hist) == 4
    assert hist["n"].sum() == 1


# ---------------------------------------------------------------------------
# Slices: by-sweep-tier and over-time (task #11, testing item #6)
# ---------------------------------------------------------------------------

def test_by_sweep_tier_hand_computed():
    sweep = pd.DataFrame({
        "tier": ["low", "low", "high"],
        "p_over": [0.2, 0.3, 0.8],
        "over_hit": [False, True, True],
    })

    table = metrics.by_sweep_tier(sweep)

    assert list(table.columns) == ["tier", "n", "mean_predicted", "empirical_rate", "brier_score", "log_loss"]
    low = table[table["tier"] == "low"].iloc[0]
    high = table[table["tier"] == "high"].iloc[0]
    assert low["n"] == 2
    assert low["mean_predicted"] == pytest.approx(0.25)
    assert low["empirical_rate"] == pytest.approx(0.5)
    assert high["n"] == 1
    assert high["mean_predicted"] == pytest.approx(0.8)
    assert high["empirical_rate"] == pytest.approx(1.0)


def test_by_sweep_tier_excludes_unsettled_rows():
    sweep = pd.DataFrame({
        "tier": ["low", "low"],
        "p_over": [0.2, 0.9],
        "over_hit": [False, np.nan],
    })

    table = metrics.by_sweep_tier(sweep)

    assert table.set_index("tier").loc["low", "n"] == 1


def test_by_sweep_tier_empty_input_returns_empty_correctly_columned_frame():
    sweep = pd.DataFrame({"tier": [], "p_over": [], "over_hit": []})
    table = metrics.by_sweep_tier(sweep)
    assert table.empty
    assert list(table.columns) == ["tier", "n", "mean_predicted", "empirical_rate", "brier_score", "log_loss"]


def test_over_time_hand_computed():
    sweep = pd.DataFrame({
        "wf_step": ["2026-04-07", "2026-04-07", "2026-04-14"],
        "p_over": [0.2, 0.3, 0.8],
        "over_hit": [False, True, True],
    })

    table = metrics.over_time(sweep)

    assert list(table.columns) == ["wf_step", "n", "mean_predicted", "empirical_rate", "brier_score", "log_loss"]
    step1 = table[table["wf_step"] == "2026-04-07"].iloc[0]
    step2 = table[table["wf_step"] == "2026-04-14"].iloc[0]
    assert step1["n"] == 2
    assert step1["mean_predicted"] == pytest.approx(0.25)
    assert step1["empirical_rate"] == pytest.approx(0.5)
    assert step2["n"] == 1
    assert step2["empirical_rate"] == pytest.approx(1.0)


def test_over_time_sorted_chronologically():
    sweep = pd.DataFrame({
        "wf_step": ["2026-04-14", "2026-04-07"],
        "p_over": [0.5, 0.6],
        "over_hit": [True, False],
    })

    table = metrics.over_time(sweep)

    assert list(table["wf_step"]) == ["2026-04-07", "2026-04-14"]


def test_over_time_all_unsettled_returns_empty_frame():
    sweep = pd.DataFrame({"wf_step": ["2026-04-07"], "p_over": [0.5], "over_hit": [np.nan]})
    table = metrics.over_time(sweep)
    assert table.empty
    assert list(table.columns) == ["wf_step", "n", "mean_predicted", "empirical_rate", "brier_score", "log_loss"]


def test_pit_histogram_all_unsettled_is_all_empty_buckets():
    _scipy_stats()
    graded = pd.DataFrame({
        "family": ["poisson"],
        "mu": [10.0],
        "alpha": [np.nan],
        "realized_strikeouts": [np.nan],
    })

    hist = metrics.pit_histogram(graded, n_buckets=3)

    assert len(hist) == 3
    assert (hist["n"] == 0).all()
