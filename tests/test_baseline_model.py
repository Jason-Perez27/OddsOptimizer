"""
Unit tests for src/models/baseline_model.py.

Design: docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md
(task #7, "Testing approach" section -- this file's 10 numbered test groups
mirror that list).

Anything that needs an actual statsmodels GLM fit or scipy.stats is gated
with `pytest.importorskip` so this file still collects (and the
preprocessing/split/metric tests still run) in an environment where those
optional deps aren't installed; install both (`pip install -r
requirements.txt`) to run the full suite.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import baseline_model as bm


def _row(**overrides):
    base = {
        "pitcher": "P1",
        "game_pk": 1,
        "game_date": "2026-04-01",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "strikeouts": 6,
        "k_rate_last5": 0.25,
        "whiff_rate_last5": 0.28,
        "velo_avg_last5": 95.0,
        "pitch_count_avg_last5": 95.0,
        "opponent_k_rate_last10": 0.22,
        "opponent_k_rate_vs_hand_season": 0.23,
        "park_k_factor": 100.0,
        "rest_days": 5.0,
        "batters_faced": 24,
        "pitch_count": 95,
        "whiff_rate": 0.27,
        "fastball_velo_avg": 95.0,
        "innings_pitched": 6.0,
        "strikeouts_vs_LHB": 3,
        "batters_faced_vs_LHB": 12,
        "strikeouts_vs_RHB": 3,
        "batters_faced_vs_RHB": 12,
    }
    base.update(overrides)
    return base


def _frame(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Temporal split is leak-free
# ---------------------------------------------------------------------------

def test_temporal_split_respects_chronological_boundary():
    rows = [_row(game_pk=i, game_date=d) for i, d in enumerate(
        ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04",
         "2026-04-05", "2026-04-06", "2026-04-07", "2026-04-08"]
    )]
    df = _frame(rows)
    train, test = bm.temporal_train_test_split(df, test_fraction=0.25)

    assert len(train) > 0 and len(test) > 0
    assert train["game_date"].max() < test["game_date"].min()
    assert (test["game_date"] >= train["game_date"].max()).all()


def test_temporal_split_is_shuffle_invariant():
    rows = [_row(game_pk=i, game_date=d) for i, d in enumerate(
        ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04",
         "2026-04-05", "2026-04-06", "2026-04-07", "2026-04-08"]
    )]
    df = _frame(rows)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_a, test_a = bm.temporal_train_test_split(df, test_fraction=0.25)
    train_b, test_b = bm.temporal_train_test_split(shuffled, test_fraction=0.25)

    assert sorted(train_a["game_pk"].tolist()) == sorted(train_b["game_pk"].tolist())
    assert sorted(test_a["game_pk"].tolist()) == sorted(test_b["game_pk"].tolist())


def test_temporal_split_explicit_cutoff_date():
    rows = [_row(game_pk=i, game_date=d) for i, d in enumerate(
        ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"]
    )]
    df = _frame(rows)
    train, test = bm.temporal_train_test_split(df, cutoff_date="2026-04-03")

    assert train["game_date"].max() < pd.Timestamp("2026-04-03")
    assert test["game_date"].min() >= pd.Timestamp("2026-04-03")


# ---------------------------------------------------------------------------
# 2. Regressor matrix excludes leakage columns
# ---------------------------------------------------------------------------

def test_design_matrix_matches_allowlist_and_excludes_leakage_columns():
    rows = [_row(game_pk=i) for i in range(5)]
    df = _frame(rows)
    preprocessor = bm.fit_preprocessor(df)
    X = bm.transform_design_matrix(df, preprocessor)

    expected_columns = set(bm.REGRESSOR_COLUMNS) | set(bm.DESIGN_MATRIX_EXTRA_COLUMNS)
    assert set(X.columns) == expected_columns
    assert set(X.columns).isdisjoint(set(bm.LEAKAGE_COLUMNS))


# ---------------------------------------------------------------------------
# 3. No NaNs reach the fitter
# ---------------------------------------------------------------------------

def test_core_pitcher_form_nulls_are_dropped():
    rows = [_row(game_pk=0), _row(game_pk=1, k_rate_last5=np.nan)]
    df = _frame(rows)
    preprocessor = bm.fit_preprocessor(df)
    # Drop the null core row from the frame the preprocessor itself was fit on
    # to mirror "fit on train" then "transform train" using the same rows.
    X = bm.transform_design_matrix(df, preprocessor)
    assert len(X) == 1
    assert not X.isna().any().any()


def test_opponent_park_nulls_are_imputed_with_flag_set():
    rows = [
        _row(game_pk=0, opponent_k_rate_last10=0.20),
        _row(game_pk=1, opponent_k_rate_last10=np.nan),
    ]
    df = _frame(rows)
    preprocessor = bm.fit_preprocessor(df)
    X = bm.transform_design_matrix(df, preprocessor)

    assert len(X) == 2
    assert not X.isna().any().any()
    assert X.loc[0, "was_imputed"] == 0.0
    assert X.loc[1, "was_imputed"] == 1.0


# ---------------------------------------------------------------------------
# 4. Scaler is fit on train only
# ---------------------------------------------------------------------------

def test_scaler_statistics_come_from_train_only():
    train_rows = [_row(game_pk=i, park_k_factor=100.0) for i in range(4)]
    train_df = _frame(train_rows)
    # Test set has a wildly different park_k_factor -- if the scaler were
    # refit on test data this would change the standardization stats.
    test_rows = [_row(game_pk=10, park_k_factor=250.0)]
    test_df = _frame(test_rows)

    preprocessor = bm.fit_preprocessor(train_df)
    mean, std = preprocessor["scale_stats"]["park_k_factor"]
    assert mean == pytest.approx(100.0)

    X_test = bm.transform_design_matrix(test_df, preprocessor)
    expected_z = (250.0 - mean) / std
    assert X_test.loc[0, "park_k_factor"] == pytest.approx(expected_z)


# ---------------------------------------------------------------------------
# 5 & 9. Model fits/predicts; reproducibility (require statsmodels)
# ---------------------------------------------------------------------------

def _synthetic_training_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-04-01", periods=30, freq="D")
    rows = []
    for i in range(n):
        k_rate = rng.uniform(0.15, 0.35)
        mu = 4 + 10 * k_rate
        rows.append(_row(
            game_pk=i,
            game_date=str(dates[i % len(dates)].date()),
            home_away="home" if i % 2 == 0 else "away",
            k_rate_last5=k_rate,
            whiff_rate_last5=rng.uniform(0.2, 0.35),
            velo_avg_last5=rng.uniform(92, 97),
            pitch_count_avg_last5=rng.uniform(80, 105),
            opponent_k_rate_last10=rng.uniform(0.18, 0.28),
            opponent_k_rate_vs_hand_season=rng.uniform(0.18, 0.28),
            park_k_factor=rng.uniform(90, 110),
            rest_days=float(rng.integers(3, 8)),
            strikeouts=int(rng.poisson(mu)),
        ))
    return _frame(rows)


def test_model_fits_and_predicts_finite_nonnegative_mean():
    statsmodels = pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")

    df = _synthetic_training_frame()
    train, test = bm.temporal_train_test_split(df, test_fraction=0.25)
    model, diagnostics, (X_test, y_test) = bm.fit_baseline_model(train, test)

    mu = model.predict_mean(X_test)
    assert np.all(np.isfinite(mu))
    assert np.all(mu >= 0)
    assert len(mu) == len(X_test)
    assert diagnostics["chosen_family"] in ("poisson", "negative_binomial")


def test_fitting_twice_on_identical_data_is_reproducible():
    pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")

    df = _synthetic_training_frame()
    train, test = bm.temporal_train_test_split(df, test_fraction=0.25)

    model_a, diag_a, _ = bm.fit_baseline_model(train, test)
    model_b, diag_b, _ = bm.fit_baseline_model(train, test)

    np.testing.assert_allclose(model_a.result.params.values, model_b.result.params.values)
    assert diag_a["chosen_family"] == diag_b["chosen_family"]


# ---------------------------------------------------------------------------
# Persistence (task #9, spec testing item 8): save_model / load_model
# round-trip a fitted model without changing its predictions, and metadata
# survives the round trip.
# ---------------------------------------------------------------------------

def test_save_and_load_model_round_trips_predictions_and_metadata(tmp_path):
    pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")
    pytest.importorskip("joblib")

    df = _synthetic_training_frame()
    train, test = bm.temporal_train_test_split(df, test_fraction=0.25)
    model, _, (X_test, y_test) = bm.fit_baseline_model(train, test)

    path = tmp_path / "model.joblib"
    bm.save_model(model, path, train_through_date="2026-04-30")

    loaded_model, metadata = bm.load_model(path)

    # Predictions on a fixed design matrix match the original exactly.
    np.testing.assert_allclose(
        model.predict_mean(X_test), loaded_model.predict_mean(X_test),
    )
    np.testing.assert_allclose(
        model.predict_over_prob(X_test, 5), loaded_model.predict_over_prob(X_test, 5),
    )

    # Reconstruction fields and metadata survive intact.
    assert loaded_model.family == model.family
    assert loaded_model.active_columns == model.active_columns
    assert loaded_model.alpha == model.alpha
    assert metadata["family"] == model.family
    assert metadata["train_through_date"] == "2026-04-30"
    assert "trained_at" in metadata
    assert metadata["artifact_version"] == bm.MODEL_ARTIFACT_VERSION


# ---------------------------------------------------------------------------
# fit_production_model (task #11, spec testing item #12): refits on ALL
# completed starts through through_date, with no holdout, and (if save_path
# is given) round-trips through save_model/load_model with correct
# train_through_date metadata.
# ---------------------------------------------------------------------------

def test_fit_production_model_trains_on_everything_through_date_inclusive():
    calls = []

    def fake_fit_fn(train_df, test_df):
        calls.append((train_df.copy(), test_df))
        return bm.BaselineModel("poisson", result=None, preprocessor={}), {}, (None, None)

    df = _synthetic_training_frame(n=20)
    df["game_date"] = pd.to_datetime(df["game_date"])
    through_date = df["game_date"].quantile(0.5, interpolation="nearest")

    bm.fit_production_model(df, through_date=through_date, fit_fn=fake_fit_fn)

    assert len(calls) == 1
    train_df, test_df = calls[0]
    assert test_df is None  # no holdout for the production fit
    assert (train_df["game_date"] <= through_date).all()
    # Inclusive boundary: rows dated exactly on through_date are included.
    if (df["game_date"] == through_date).any():
        assert (train_df["game_date"] == through_date).any()
    # Rows strictly after through_date are excluded.
    assert not (train_df["game_date"] > through_date).any()


def test_fit_production_model_raises_clear_error_when_nothing_to_train_on():
    def fake_fit_fn(train_df, test_df):
        raise AssertionError("fit_fn should never be called with an empty training set")

    df = _synthetic_training_frame(n=5)
    df["game_date"] = pd.to_datetime(df["game_date"])
    too_early = df["game_date"].min() - pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="No completed starts"):
        bm.fit_production_model(df, through_date=too_early, fit_fn=fake_fit_fn)


def test_fit_production_model_round_trips_through_save_and_load(tmp_path):
    pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")
    pytest.importorskip("joblib")

    df = _synthetic_training_frame()
    through_date = pd.Timestamp("2026-04-30")

    path = tmp_path / "production_model.joblib"
    model = bm.fit_production_model(df, through_date=through_date, save_path=path)

    loaded_model, metadata = bm.load_model(path)

    assert loaded_model.family == model.family
    assert loaded_model.active_columns == model.active_columns
    assert metadata["train_through_date"] == through_date.isoformat()
    assert "trained_at" in metadata


def test_load_model_missing_file_raises_clear_actionable_error(tmp_path):
    pytest.importorskip("joblib")
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="train and save a model first"):
        bm.load_model(missing_path)


def test_load_model_corrupt_file_raises_value_error_not_a_bare_exception(tmp_path):
    pytest.importorskip("joblib")
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_text("not a real joblib artifact")
    with pytest.raises(ValueError):
        bm.load_model(corrupt_path)


def test_model_age_days_computed_from_trained_at():
    metadata = {"trained_at": "2026-06-01T00:00:00+00:00"}
    age = bm.model_age_days(metadata, as_of="2026-06-15T00:00:00+00:00")
    assert age == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# 6. Family selector logic (requires statsmodels + scipy)
# ---------------------------------------------------------------------------

def test_dispersion_ratio_hand_computed():
    # y = [2, 4, 6], mu = [4, 4, 4], n_params = 1
    # pearson resid = (y-mu)/sqrt(mu) = (2-4)/2=-1, (4-4)/2=0, (6-4)/2=1
    # chi2 = 1 + 0 + 1 = 2 ; dof = 3-1=2 ; ratio = 1.0
    y = [2, 4, 6]
    mu = [4, 4, 4]
    ratio = bm.dispersion_ratio(y, mu, n_params=1)
    assert ratio == pytest.approx(1.0)


def test_family_selector_picks_negative_binomial_for_overdispersed_data():
    pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 300
    # Negative-binomial-distributed counts with substantial overdispersion.
    mu_true = 6.0
    alpha_true = 0.8
    p = 1 / (1 + alpha_true * mu_true)
    nbn = 1 / alpha_true
    y = rng.negative_binomial(nbn, p, size=n)

    rows = [_row(game_pk=i, strikeouts=int(y[i])) for i in range(n)]
    df = _frame(rows)
    X, y_train, _, _, _ = bm.build_design_matrix(df)
    chosen, _, _, diagnostics = bm.select_family(X, y_train)
    assert chosen == "negative_binomial"
    assert diagnostics["dispersion_ratio"] > 1.0


def test_family_selector_picks_poisson_for_clean_poisson_data():
    pytest.importorskip("statsmodels")
    pytest.importorskip("scipy")

    rng = np.random.default_rng(2)
    n = 300
    y = rng.poisson(6.0, size=n)
    rows = [_row(game_pk=i, strikeouts=int(y[i])) for i in range(n)]
    df = _frame(rows)
    X, y_train, _, _, _ = bm.build_design_matrix(df)
    chosen, _, _, diagnostics = bm.select_family(X, y_train)
    assert chosen == "poisson"


# ---------------------------------------------------------------------------
# 7. Threshold probabilities are well-formed (requires scipy)
# ---------------------------------------------------------------------------

def test_poisson_over_prob_matches_scipy_reference_and_is_monotonic():
    scipy_stats = pytest.importorskip("scipy.stats")

    mu = 6.0
    probs = [bm.poisson_over_prob(mu, t) for t in range(0, 11)]
    assert probs[0] == pytest.approx(1.0)
    for a, b in zip(probs, probs[1:]):
        assert a >= b  # non-increasing
    for t in range(1, 11):
        assert probs[t] == pytest.approx(float(scipy_stats.poisson.sf(t - 1, mu)))
        assert 0.0 <= probs[t] <= 1.0


def test_nbinom_over_prob_matches_scipy_reference():
    scipy_stats = pytest.importorskip("scipy.stats")

    mu, alpha = 6.0, 0.5
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    for t in range(1, 11):
        got = bm.nbinom_over_prob(mu, alpha, t)
        expected = float(scipy_stats.nbinom.sf(t - 1, n, p))
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 8. Metric helpers are correct
# ---------------------------------------------------------------------------

def test_mean_absolute_error_and_rmse():
    y_true = [4, 6, 8]
    y_pred = [5, 5, 5]
    # errors: -1, 1, 3 -> MAE = (1+1+3)/3 = 5/3 ; RMSE = sqrt((1+1+9)/3)=sqrt(11/3)
    assert bm.mean_absolute_error(y_true, y_pred) == pytest.approx(5 / 3)
    assert bm.root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(11 / 3))


def test_brier_score_and_log_loss():
    y_true = [1, 0, 1]
    p_pred = [0.8, 0.3, 0.6]
    # brier = mean((p-y)^2) = ((0.8-1)^2+(0.3-0)^2+(0.6-1)^2)/3
    expected_brier = ((0.8 - 1) ** 2 + (0.3 - 0) ** 2 + (0.6 - 1) ** 2) / 3
    assert bm.brier_score(y_true, p_pred) == pytest.approx(expected_brier)

    expected_ll = -np.mean([
        np.log(0.8), np.log(1 - 0.3), np.log(0.6),
    ])
    assert bm.log_loss(y_true, p_pred) == pytest.approx(expected_ll)


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_handled_gracefully():
    empty = _frame([])
    train, test = bm.temporal_train_test_split(empty)
    assert train.empty and test.empty


def test_single_row_input_handled_gracefully():
    df = _frame([_row(game_pk=0)])
    train, test = bm.temporal_train_test_split(df)
    # With only one distinct date, everything lands in test, train is empty.
    assert len(train) + len(test) == 1

    preprocessor = bm.fit_preprocessor(df)
    X = bm.transform_design_matrix(df, preprocessor)
    assert len(X) == 1
    assert not X.isna().any().any()


def test_fit_baseline_model_raises_clear_error_on_no_usable_training_rows():
    pytest.importorskip("statsmodels")
    df = _frame([_row(game_pk=0, k_rate_last5=np.nan)])
    with pytest.raises(ValueError):
        bm.fit_baseline_model(df)
