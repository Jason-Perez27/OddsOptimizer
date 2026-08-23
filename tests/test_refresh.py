"""
Unit tests for src/pipeline/refresh.py (task #9, module 4).

Design: docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md
("Outputs", "Partial failure" sections); decision log, 2026-06-27 "Pre-game
refresh pipeline" entry.

Strategy (matches the rest of this repo's no-network test style):
- run_refresh() takes every fetcher and the model loader as injected
  callables, so every test below uses hand-built fixtures/stubs -- no
  network, no real fitted statsmodels model, no real joblib artifact.
- A FakeModel stands in for BaselineModel: it exposes exactly the surface
  assemble_predictions()/transform_design_matrix() need (.family, .alpha,
  .preprocessor, .predict_mean(X)) without requiring statsmodels to fit one.
  Its preprocessor is a hand-built dict matching baseline_model.
  fit_preprocessor()'s output shape (impute_means/scale_stats for the real
  IMPUTE_COLUMNS/CONTINUOUS_REGRESSOR_COLUMNS), not a real fitted one.
- build_threshold_table() (called unconditionally inside run_refresh) goes
  through tiering.poisson_over_prob/nbinom_over_prob, which need scipy.
  Every test that calls run_refresh() end-to-end is gated with
  pytest.importorskip("scipy"); assemble_predictions()/write_outputs() unit
  tests and the empty-slate test (which raises before reaching the model)
  do not need scipy and run unconditionally.

Run with: pytest tests/test_refresh.py -v
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import src.models.baseline_model as bm
from src.pipeline.refresh import (
    EmptySlateError,
    PREDICTIONS_COLUMNS,
    PITCHER_CARD_COLUMNS,
    SKIPPED_PITCHERS_COLUMNS,
    assemble_predictions,
    build_pitcher_cards,
    run_refresh,
    write_outputs,
)
from src.predictions.tiering import LINE_PICKS_COLUMNS


# ---------------------------------------------------------------------------
# fixtures / stubs
# ---------------------------------------------------------------------------

SLATE_COLUMNS = [
    "pitcher", "pitcher_name", "pitcher_team", "opponent_team", "home_away",
    "game_pk", "game_date", "start_time", "pitcher_throws",
]


def _slate_row(**overrides):
    row = {
        "pitcher": 1,
        "pitcher_name": "Test Pitcher",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "game_pk": 99,
        "game_date": "2026-04-26",
        "start_time": "2026-04-26T23:05:00Z",
        "pitcher_throws": "R",
    }
    row.update(overrides)
    return row


def _pitch_row(pitcher, game_pk, game_date, home_team, away_team, top_bot, events, stand="R"):
    return {
        "pitcher": pitcher,
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": top_bot,
        "events": events,
        "description": "called_strike",
        "pitch_type": "FF",
        "release_speed": 95.0,
        "stand": stand,
        "p_throws": "R",
    }


def _prior_game_pitches(pitcher=1, game_pk=1, game_date="2026-04-21",
                         home_team="NYY", away_team="BOS", n_k=5, n_other=15):
    """
    One prior start's worth of pitch-level rows -- enough batters faced for
    a non-NaN k_rate_last5/whiff_rate_last5/velo_avg_last5/pitch_count_avg_
    last5 on a synthetic same-day row built on top of it (mirrors tests/
    test_predict_features.py's single-history-game convention, but at the
    pitch-level statcast_fetcher needs to return).
    """
    rows = [
        _pitch_row(pitcher, game_pk, game_date, home_team, away_team, "Top", "strikeout", "R" if i % 2 else "L")
        for i in range(n_k)
    ]
    rows += [
        _pitch_row(pitcher, game_pk, game_date, home_team, away_team, "Top", "field_out", "R" if i % 2 else "L")
        for i in range(n_other)
    ]
    return pd.DataFrame(rows)


class FakeModel:
    family = "poisson"
    alpha = None

    def __init__(self, mu=5.0):
        self._mu = mu
        self.preprocessor = {
            "impute_means": {col: 0.0 for col in bm.IMPUTE_COLUMNS},
            "scale_stats": {col: (0.0, 1.0) for col in bm.CONTINUOUS_REGRESSOR_COLUMNS},
        }

    def predict_mean(self, X):
        return np.full(len(X), self._mu)

    def predict_mean_with_se(self, X):
        """Stub: returns (mu, zeros) — no fitted result to query."""
        return np.full(len(X), self._mu), np.zeros(len(X))


def _fake_model_loader(trained_at="2026-06-25T00:00:00+00:00"):
    def loader(path):
        return FakeModel(), {"trained_at": trained_at}
    return loader


# ---------------------------------------------------------------------------
# EmptySlateError -- the one fatal case (no scipy needed: raises before the
# model is even loaded)
# ---------------------------------------------------------------------------

def test_empty_slate_raises_empty_slate_error():
    empty_slate = pd.DataFrame(columns=SLATE_COLUMNS)
    with pytest.raises(EmptySlateError):
        run_refresh(
            "2026-04-26",
            schedule_fetcher=lambda d: empty_slate,
            model_loader=_fake_model_loader(),
        )


def test_none_slate_raises_empty_slate_error():
    with pytest.raises(EmptySlateError):
        run_refresh(
            "2026-04-26",
            schedule_fetcher=lambda d: None,
            model_loader=_fake_model_loader(),
        )


# ---------------------------------------------------------------------------
# assemble_predictions() -- unit-level, scipy-free
# ---------------------------------------------------------------------------

def test_assemble_predictions_attaches_name_family_alpha_and_mu():
    slate = pd.DataFrame([_slate_row(pitcher=1, pitcher_name="Test Pitcher")])
    feature_rows = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5000, "pitcher_team": "NYY", "opponent_team": "BOS",
        "game_date": "2026-04-26",
        "k_rate_last5": 0.3, "whiff_rate_last5": 0.25,
        "velo_avg_last5": 95.0, "pitch_count_avg_last5": 90.0,
        "opponent_k_rate_last10": 0.22, "opponent_k_rate_vs_hand_season": 0.21,
        "park_k_factor": 100, "rest_days": 5.0, "home_away": "home",
    }])
    model = FakeModel(mu=4.2)

    predictions, dropped = assemble_predictions(model, feature_rows, slate)

    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert len(predictions) == 1
    row = predictions.iloc[0]
    assert row["pitcher"] == 1
    assert row["pitcher_name"] == "Test Pitcher"
    assert row["family"] == "poisson"
    assert row["alpha"] is None
    assert np.isclose(row["mu"], 4.2)
    assert np.isclose(row["mu_se"], 0.0)  # FakeModel stub returns zeros
    assert dropped == []


def test_build_pitcher_cards_carries_workload_and_context_stats():
    slate = pd.DataFrame([_slate_row(pitcher=1, pitcher_name="Test Pitcher")])
    feature_rows = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5000, "pitcher_team": "NYY", "opponent_team": "BOS",
        "game_date": "2026-04-26", "pitcher_throws": "R",
        "k_rate_last5": 0.3, "k_rate_season": 0.27, "k_rate_vs_LHB": 0.25, "k_rate_vs_RHB": 0.29,
        "k_rate_home": 0.28, "k_rate_away": 0.26, "k_rate_vs_opponent_career": 0.27,
        "whiff_rate_last5": 0.25, "velo_avg_last5": 95.0, "pitch_count_avg_last5": 90.0,
        "ip_avg_last5": 5.6, "bf_avg_last5": 23.1,
        "opponent_k_rate_last10": 0.22, "opponent_k_rate_vs_hand_season": 0.21,
        "opponent_k_rate_home": 0.22, "opponent_k_rate_away": 0.2,
        "park_k_factor": 1.02, "rest_days": 5.0, "home_away": "home",
    }])
    model = FakeModel(mu=4.2)
    predictions, _ = assemble_predictions(model, feature_rows, slate)

    cards = build_pitcher_cards(feature_rows, predictions)

    assert list(cards.columns) == PITCHER_CARD_COLUMNS
    assert len(cards) == 1
    row = cards.iloc[0]
    assert row["pitcher_name"] == "Test Pitcher"
    assert np.isclose(row["mu"], 4.2)
    assert np.isclose(row["ip_avg_last5"], 5.6)        # avg innings pitched
    assert np.isclose(row["bf_avg_last5"], 23.1)        # expected batters faced
    assert np.isclose(row["pitch_count_avg_last5"], 90.0)  # avg pitches thrown
    assert row["is_home"] == 1.0
    assert row["was_imputed"] == 0.0


def test_build_pitcher_cards_empty_when_no_predictions():
    cards = build_pitcher_cards(pd.DataFrame(), pd.DataFrame())
    assert list(cards.columns) == PITCHER_CARD_COLUMNS
    assert cards.empty


def test_assemble_predictions_drops_and_reports_pitcher_missing_core_features():
    slate = pd.DataFrame([
        _slate_row(pitcher=1, pitcher_name="Has History"),
        _slate_row(pitcher=999, pitcher_name="Rookie", game_pk=100),
    ])
    feature_rows = pd.DataFrame([
        {
            "pitcher": 1, "game_pk": 5000, "pitcher_team": "NYY", "opponent_team": "BOS",
            "game_date": "2026-04-26",
            "k_rate_last5": 0.3, "whiff_rate_last5": 0.25,
            "velo_avg_last5": 95.0, "pitch_count_avg_last5": 90.0,
            "opponent_k_rate_last10": 0.22, "opponent_k_rate_vs_hand_season": 0.21,
            "park_k_factor": 100, "rest_days": 5.0, "home_away": "home",
        },
        {
            "pitcher": 999, "game_pk": 100, "pitcher_team": "BOS", "opponent_team": "NYY",
            "game_date": "2026-04-26",
            "k_rate_last5": np.nan, "whiff_rate_last5": np.nan,
            "velo_avg_last5": np.nan, "pitch_count_avg_last5": np.nan,
            "opponent_k_rate_last10": np.nan, "opponent_k_rate_vs_hand_season": np.nan,
            "park_k_factor": np.nan, "rest_days": np.nan, "home_away": "away",
        },
    ])
    model = FakeModel()

    predictions, dropped = assemble_predictions(model, feature_rows, slate)

    assert len(predictions) == 1
    assert predictions.iloc[0]["pitcher"] == 1
    assert len(dropped) == 1
    assert dropped[0]["pitcher"] == 999
    assert dropped[0]["pitcher_name"] == "Rookie"
    assert "no usable pre-game features" in dropped[0]["reason"]


def test_assemble_predictions_empty_feature_rows_returns_empty_well_formed():
    slate = pd.DataFrame([_slate_row()])
    empty_features = pd.DataFrame(columns=[
        "pitcher", "pitcher_team", "opponent_team", "game_date",
        "k_rate_last5", "whiff_rate_last5", "velo_avg_last5",
        "pitch_count_avg_last5", "opponent_k_rate_last10",
        "opponent_k_rate_vs_hand_season", "park_k_factor", "rest_days", "home_away",
    ])
    model = FakeModel()

    predictions, dropped = assemble_predictions(model, empty_features, slate)

    assert predictions.empty
    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert dropped == []


# ---------------------------------------------------------------------------
# run_refresh() end-to-end -- needs scipy (build_threshold_table is always
# called inside run_refresh)
# ---------------------------------------------------------------------------

def test_run_refresh_happy_path_predictions_and_threshold_contract():
    pytest.importorskip("scipy")

    slate = [_slate_row(pitcher=1, pitcher_name="Test Pitcher")]
    history_pitches = _prior_game_pitches(pitcher=1)

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=lambda pid, name, season: history_pitches,
        lines_fetcher=lambda: pd.DataFrame(),
        register_fetcher=lambda: pd.DataFrame(),
        model_loader=_fake_model_loader(),
    )

    assert results["game_date"] == "2026-04-26"
    predictions = results["predictions"]
    assert len(predictions) == 1
    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert predictions.iloc[0]["pitcher"] == 1
    assert not predictions.iloc[0][["mu"]].isna().any()

    threshold_table = results["threshold_table"]
    assert len(threshold_table) > 0

    diagnostics = results["diagnostics"]
    assert list(diagnostics["skipped_pitchers"].columns) == SKIPPED_PITCHERS_COLUMNS
    assert len(diagnostics["skipped_pitchers"]) == 0
    assert diagnostics["model_age_days"] > 0
    assert isinstance(diagnostics["model_stale"], bool)


def test_run_refresh_statcast_failure_for_one_pitcher_skips_only_that_pitcher():
    pytest.importorskip("scipy")

    slate = [
        _slate_row(pitcher=1, pitcher_name="Has History", game_pk=99),
        _slate_row(pitcher=2, pitcher_name="Bad Pull", pitcher_team="BOS",
                    opponent_team="NYY", home_away="away", game_pk=100),
    ]
    history_pitches = _prior_game_pitches(pitcher=1)

    def statcast_fetcher(pid, name, season):
        if pid == 2:
            raise RuntimeError("pybaseball lookup failed")
        return history_pitches

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=statcast_fetcher,
        lines_fetcher=lambda: pd.DataFrame(),
        register_fetcher=lambda: pd.DataFrame(),
        model_loader=_fake_model_loader(),
    )

    predictions = results["predictions"]
    assert len(predictions) == 1
    assert predictions.iloc[0]["pitcher"] == 1

    skipped = results["diagnostics"]["skipped_pitchers"]
    assert len(skipped) == 1
    assert skipped.iloc[0]["pitcher"] == 2
    assert "Statcast pull failed" in skipped.iloc[0]["reason"]


def test_run_refresh_statcast_empty_result_is_skipped_not_fatal():
    pytest.importorskip("scipy")

    slate = [_slate_row(pitcher=1, pitcher_name="Debutant")]

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=lambda pid, name, season: pd.DataFrame(),
        lines_fetcher=lambda: pd.DataFrame(),
        register_fetcher=lambda: pd.DataFrame(),
        model_loader=_fake_model_loader(),
    )

    assert results["predictions"].empty
    skipped = results["diagnostics"]["skipped_pitchers"]
    assert len(skipped) == 1
    assert skipped.iloc[0]["pitcher"] == 1
    assert "no Statcast rows returned" in skipped.iloc[0]["reason"]


def test_run_refresh_line_source_failure_degrades_to_empty_line_picks():
    pytest.importorskip("scipy")

    slate = [_slate_row(pitcher=1, pitcher_name="Test Pitcher")]
    history_pitches = _prior_game_pitches(pitcher=1)

    def failing_lines_fetcher():
        raise RuntimeError("Underdog endpoint 503")

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=lambda pid, name, season: history_pitches,
        lines_fetcher=failing_lines_fetcher,
        register_fetcher=lambda: pd.DataFrame(),
        model_loader=_fake_model_loader(),
    )

    # predictions/threshold_table still fully produced
    assert len(results["predictions"]) == 1
    assert len(results["threshold_table"]) > 0

    # line_picks forced empty but well-formed
    assert results["line_picks"].empty
    assert list(results["line_picks"].columns) == LINE_PICKS_COLUMNS

    diagnostics = results["diagnostics"]
    assert "Underdog endpoint 503" in diagnostics["line_source_error"]
    assert diagnostics["register_error"] is None
    # nothing could be matched -> predicted_no_line is a full copy of predictions
    assert len(diagnostics["predicted_no_line"]) == len(results["predictions"])


def test_run_refresh_register_failure_also_degrades_line_picks():
    pytest.importorskip("scipy")

    slate = [_slate_row(pitcher=1, pitcher_name="Test Pitcher")]
    history_pitches = _prior_game_pitches(pitcher=1)

    def failing_register_fetcher():
        raise RuntimeError("chadwick_register network error")

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=lambda pid, name, season: history_pitches,
        lines_fetcher=lambda: pd.DataFrame(),
        register_fetcher=failing_register_fetcher,
        model_loader=_fake_model_loader(),
    )

    assert results["line_picks"].empty
    diagnostics = results["diagnostics"]
    assert diagnostics["line_source_error"] is None
    assert "chadwick_register network error" in diagnostics["register_error"]


def test_run_refresh_model_staleness_surfaced_not_hidden():
    pytest.importorskip("scipy")

    slate = [_slate_row(pitcher=1, pitcher_name="Test Pitcher")]
    history_pitches = _prior_game_pitches(pitcher=1)

    stale_loader = _fake_model_loader(trained_at="2026-01-01T00:00:00+00:00")

    results = run_refresh(
        "2026-04-26",
        schedule_fetcher=lambda d: pd.DataFrame(slate),
        statcast_fetcher=lambda pid, name, season: history_pitches,
        lines_fetcher=lambda: pd.DataFrame(),
        register_fetcher=lambda: pd.DataFrame(),
        model_loader=stale_loader,
        stale_warning_days=7,
    )

    diagnostics = results["diagnostics"]
    assert diagnostics["model_age_days"] > 7
    assert diagnostics["model_stale"] is True


# ---------------------------------------------------------------------------
# write_outputs() -- file-writing / idempotency, scipy-free (built from
# hand-made results dicts, no run_refresh() call needed)
# ---------------------------------------------------------------------------

def _hand_built_results(game_date="2026-04-26"):
    predictions = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5000, "pitcher_name": "Test Pitcher", "pitcher_team": "NYY",
        "opponent_team": "BOS", "game_date": game_date,
        "family": "poisson", "mu": 5.0, "alpha": None,
    }])
    threshold_table = pd.DataFrame([{"pitcher": 1, "game_pk": 5000, "threshold": 5, "p_over": 0.5, "tier": "Low"}])
    line_picks = pd.DataFrame(columns=LINE_PICKS_COLUMNS)
    diagnostics = {
        "skipped_pitchers": pd.DataFrame(columns=SKIPPED_PITCHERS_COLUMNS),
        "unmatched_lines": pd.DataFrame(),
        "predicted_no_line": predictions.copy(),
        "line_source_error": None,
        "register_error": None,
        "model_age_days": 1.5,
        "model_stale": False,
    }
    return {
        "game_date": game_date,
        "predictions": predictions,
        "threshold_table": threshold_table,
        "line_picks": line_picks,
        "diagnostics": diagnostics,
    }


def test_write_outputs_creates_expected_files_and_manifest(tmp_path):
    results = _hand_built_results()
    out_dir = write_outputs(results, processed_dir=str(tmp_path))

    assert os.path.exists(os.path.join(out_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(out_dir, "threshold_table.csv"))
    assert os.path.exists(os.path.join(out_dir, "line_picks.csv"))
    assert os.path.exists(os.path.join(out_dir, "run_manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "diagnostics", "skipped_pitchers.csv"))

    with open(os.path.join(out_dir, "run_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["game_date"] == "2026-04-26"
    assert "run_at" in manifest
