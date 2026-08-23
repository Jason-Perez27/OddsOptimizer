"""
Tests for Spec ④ (2026-06-30): weather + Vegas context + boosted ensemble.

Covers:
  - src/data/weather.py      : fetch_weather, get_game_weather
  - src/data/vegas.py        : get_game_odds, fetch_espn_scoreboard
  - src/models/boosted_model.py : fit_boosted_model, BoostedModel, compute_agreement
  - src/models/baseline_model.py : CONTEXT_CANDIDATE_COLUMNS present
  - Compile sweep: all five new/modified modules import cleanly

All tests are network-free: fetchers are lambda injections.
"""

import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers shared across weather / vegas / booster tests
# ---------------------------------------------------------------------------

def _make_hourly_weather_payload(
    date: str = "2025-06-15",
    first_pitch_hour: int = 19,
    temp_f: float = 72.5,
    humidity: float = 55.0,
    wind_mph: float = 8.5,
) -> dict:
    """Return a minimal Open-Meteo-shaped hourly payload with 24 hours."""
    times = [f"{date}T{h:02d}:00" for h in range(24)]
    temps = [60.0 + h for h in range(24)]
    temps[first_pitch_hour] = temp_f
    hums = [50.0 + h * 0.5 for h in range(24)]
    hums[first_pitch_hour] = humidity
    winds = [5.0 + h * 0.25 for h in range(24)]
    winds[first_pitch_hour] = wind_mph
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "windspeed_10m": winds,
        }
    }


def _make_espn_payload(
    home_abbr: str = "NYY",
    away_abbr: str = "BOS",
    over_under: float = 8.5,
    home_ml: int = -150,
    away_ml: int = 130,
) -> dict:
    """Return a minimal ESPN scoreboard-shaped payload."""
    return {
        "events": [
            {
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": home_abbr}},
                            {"homeAway": "away", "team": {"abbreviation": away_abbr}},
                        ],
                        "odds": [
                            {
                                "overUnder": over_under,
                                "homeTeamOdds": {"moneyLine": home_ml},
                                "awayTeamOdds": {"moneyLine": away_ml},
                            }
                        ],
                    }
                ]
            }
        ]
    }


# ---------------------------------------------------------------------------
# 1. weather.py — fetch_weather extracts the correct first-pitch hour
# ---------------------------------------------------------------------------

def test_weather_correct_hour_extracted():
    from src.data.weather import fetch_weather

    date = "2025-06-15"
    hour = 19
    payload = _make_hourly_weather_payload(
        date=date, first_pitch_hour=hour,
        temp_f=72.5, humidity=55.0, wind_mph=8.5,
    )
    result = fetch_weather(
        lat=40.8296, lon=-73.9262,
        game_date=date, first_pitch_hour=hour,
        fetcher=lambda url: payload,
    )

    assert result["temp_f"] == pytest.approx(72.5, abs=0.1)
    assert result["humidity"] == pytest.approx(55.0, abs=0.1)
    assert result["wind_mph"] == pytest.approx(8.5, abs=0.1)
    assert result["is_dome"] == 0.0
    assert result["weather_was_imputed"] is False


def test_weather_fallback_nearest_hour_when_exact_missing():
    """If the exact hour isn't in the time series, nearest available hour is used."""
    from src.data.weather import fetch_weather

    date = "2025-06-15"
    # Payload has hours 17, 18, 20 — no exact 19. Nearest to 19 is 18 or 20 (distance 1).
    payload = {
        "hourly": {
            "time": [f"{date}T17:00", f"{date}T18:00", f"{date}T20:00"],
            "temperature_2m": [60.0, 65.0, 75.0],
            "relative_humidity_2m": [50.0, 52.0, 56.0],
            "windspeed_10m": [5.0, 6.0, 9.0],
        }
    }
    result = fetch_weather(
        lat=40.0, lon=-74.0, game_date=date, first_pitch_hour=19,
        fetcher=lambda url: payload,
    )
    # Should not be imputed — fell back to a valid nearest hour
    assert result["weather_was_imputed"] is False
    # Temperature must be one of the known values (not made up)
    assert result["temp_f"] in {60.0, 65.0, 75.0}


# ---------------------------------------------------------------------------
# 2. weather.py — dome park → neutral result, no fetcher call
# ---------------------------------------------------------------------------

def test_dome_returns_neutral_no_fetcher_call():
    """Dome teams (e.g. HOU) return is_dome=1 without calling the fetcher."""
    from src.data.weather import get_game_weather

    calls = []

    def _fetcher(url):
        calls.append(url)
        return {}

    result = get_game_weather("HOU", "2025-06-15", fetcher=_fetcher)

    assert len(calls) == 0, "Dome park should not call the network fetcher"
    assert result["is_dome"] == 1.0
    assert result["weather_was_imputed"] is True
    assert math.isnan(result["temp_f"])
    assert math.isnan(result["wind_mph"])
    assert math.isnan(result["humidity"])


def test_dome_result_for_all_known_dome_teams():
    from src.data.weather import get_game_weather, BALLPARK_TABLE
    dome_teams = [abbr for abbr, info in BALLPARK_TABLE.items() if info["is_dome"]]
    for team in dome_teams:
        result = get_game_weather(team, "2025-06-15", fetcher=lambda url: {})
        assert result["is_dome"] == 1.0, f"{team} should be dome"


# ---------------------------------------------------------------------------
# 3. weather.py — unknown team → imputed; fetch failure → imputed
# ---------------------------------------------------------------------------

def test_unknown_team_returns_imputed():
    from src.data.weather import get_game_weather

    result = get_game_weather("XYZ", "2025-06-15")
    assert result["weather_was_imputed"] is True
    assert result["is_dome"] == 0.0
    assert math.isnan(result["temp_f"])


def test_fetch_failure_returns_imputed():
    from src.data.weather import fetch_weather

    def _failing_fetcher(url):
        raise RuntimeError("Network down")

    result = fetch_weather(
        lat=40.0, lon=-74.0, game_date="2025-06-15",
        fetcher=_failing_fetcher,
    )
    assert result["weather_was_imputed"] is True
    assert result["is_dome"] == 0.0
    assert math.isnan(result["temp_f"])


def test_empty_api_response_returns_imputed():
    from src.data.weather import fetch_weather

    result = fetch_weather(
        lat=40.0, lon=-74.0, game_date="2025-06-15",
        fetcher=lambda url: {"hourly": {"time": []}},
    )
    assert result["weather_was_imputed"] is True


# ---------------------------------------------------------------------------
# 4. vegas.py — ESPN fixture → game_total and is_favorite
# ---------------------------------------------------------------------------

def test_espn_fixture_home_favorite():
    from src.data.vegas import get_game_odds

    payload = _make_espn_payload(
        home_abbr="NYY", away_abbr="BOS",
        over_under=8.5, home_ml=-150, away_ml=130,
    )
    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        pitcher_home_away="home",
        fetcher=lambda url: payload,
    )

    assert result["game_total"] == pytest.approx(8.5)
    assert result["is_favorite"] == 1.0   # home ML is -150 → favourite
    assert result["vegas_was_imputed"] is False


def test_espn_fixture_away_underdog():
    from src.data.vegas import get_game_odds

    payload = _make_espn_payload(
        home_abbr="NYY", away_abbr="BOS",
        over_under=8.5, home_ml=-150, away_ml=130,
    )
    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        pitcher_home_away="away",
        fetcher=lambda url: payload,
    )

    assert result["game_total"] == pytest.approx(8.5)
    assert result["is_favorite"] == 0.0   # away ML is +130 → underdog
    assert result["vegas_was_imputed"] is False


def test_espn_no_matching_event_is_imputed():
    """If no event matches the home_team, result is imputed."""
    from src.data.vegas import get_game_odds

    payload = _make_espn_payload(home_abbr="LAD", away_abbr="SF")
    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        fetcher=lambda url: payload,
    )

    assert result["vegas_was_imputed"] is True
    assert math.isnan(result["game_total"])


def test_espn_fetch_failure_is_imputed():
    from src.data.vegas import get_game_odds

    def _err(url):
        raise ConnectionError("timeout")

    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        fetcher=_err,
    )
    assert result["vegas_was_imputed"] is True


def test_espn_no_odds_in_competition_is_imputed():
    """Competition with empty odds list → imputed."""
    from src.data.vegas import get_game_odds

    payload = {
        "events": [
            {
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "NYY"}},
                            {"homeAway": "away", "team": {"abbreviation": "BOS"}},
                        ],
                        "odds": [],
                    }
                ]
            }
        ]
    }
    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        fetcher=lambda url: payload,
    )
    assert result["vegas_was_imputed"] is True


def test_espn_team_total_columns_always_nan():
    """team_total_for / team_total_against are always NaN per spec limitation."""
    from src.data.vegas import get_game_odds

    payload = _make_espn_payload()
    result = get_game_odds(
        home_team="NYY", game_date="2025-06-15",
        fetcher=lambda url: payload,
    )
    assert math.isnan(result["team_total_for"])
    assert math.isnan(result["team_total_against"])


# ---------------------------------------------------------------------------
# 5. boosted_model.py — BoostedModel trains; predict_mean ≥ 0
# ---------------------------------------------------------------------------

sklearn = pytest.importorskip("sklearn", reason="scikit-learn required for BoostedModel tests")


def _make_train_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Minimal game-log-schema DataFrame sufficient for fit_boosted_model."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "pitcher": f"P{i % 5}",
            "game_pk": 1000 + i,
            "game_date": f"2025-{(i // 30) + 4:02d}-{(i % 30) + 1:02d}",
            "pitcher_team": "NYY",
            "opponent_team": "BOS",
            "home_away": "home",
            "strikeouts": int(rng.integers(2, 12)),
            "k_rate_last5": rng.uniform(0.18, 0.35),
            "whiff_rate_last5": rng.uniform(0.20, 0.38),
            "velo_avg_last5": rng.uniform(90.0, 98.0),
            "pitch_count_avg_last5": rng.uniform(85.0, 110.0),
            "opponent_k_rate_last10": rng.uniform(0.18, 0.28),
            "opponent_k_rate_vs_hand_season": rng.uniform(0.18, 0.28),
            "park_k_factor": rng.uniform(95.0, 105.0),
            "rest_days": rng.uniform(4.0, 6.0),
            "batters_faced": int(rng.integers(18, 28)),
            "pitch_count": int(rng.integers(80, 110)),
            "whiff_rate": rng.uniform(0.20, 0.38),
            "fastball_velo_avg": rng.uniform(90.0, 98.0),
            "innings_pitched": rng.uniform(4.5, 7.0),
            "strikeouts_vs_LHB": int(rng.integers(1, 6)),
            "batters_faced_vs_LHB": int(rng.integers(8, 14)),
            "strikeouts_vs_RHB": int(rng.integers(1, 6)),
            "batters_faced_vs_RHB": int(rng.integers(8, 14)),
        })
    return pd.DataFrame(rows)


def test_boosted_model_trains_and_predict_mean_nonnegative():
    from src.models.boosted_model import fit_boosted_model

    train_df = _make_train_df(n=80)
    model = fit_boosted_model(
        train_df,
        hgb_params={"max_iter": 10},  # fast for tests
    )
    assert model is not None
    assert model.gbr is not None
    assert len(model.feature_columns) > 0

    from src.models.baseline_model import build_design_matrix
    X, y, _, _, _ = build_design_matrix(train_df[:5])
    # Drop const for HGB
    X_feat = X[[c for c in X.columns if c != "const"]]
    mu = model.predict_mean(X_feat)
    assert (mu >= 0).all(), "predict_mean must return non-negative values"


def test_boosted_model_predict_mean_via_transform():
    """Verify BoostedModel.predict_mean works via transform_design_matrix path."""
    from src.models.boosted_model import fit_boosted_model
    from src.models.baseline_model import build_design_matrix, transform_design_matrix

    train_df = _make_train_df(n=60)
    model = fit_boosted_model(train_df, hgb_params={"max_iter": 5})

    test_df = _make_train_df(n=5, seed=99)
    X_test = transform_design_matrix(test_df, model.preprocessor)
    mu = model.predict_mean(X_test)
    assert len(mu) == len(X_test)
    assert (mu >= 0).all()


# ---------------------------------------------------------------------------
# 6. boosted_model.py — isotonic calibration is monotonic
# ---------------------------------------------------------------------------

def test_isotonic_calibration_monotonic():
    """
    Raw P(K >= t) that increases monotonically should produce calibrated
    probabilities that are also monotonically non-decreasing.
    """
    from src.models.boosted_model import fit_boosted_model, THRESHOLDS
    from src.models.baseline_model import build_design_matrix, transform_design_matrix

    train_df = _make_train_df(n=120, seed=7)
    model = fit_boosted_model(train_df, hgb_params={"max_iter": 20})

    if not model.calibrators:
        pytest.skip("No calibrators fitted (not enough calibration data)")

    test_df = _make_train_df(n=20, seed=55)
    X_test = transform_design_matrix(test_df, model.preprocessor)
    prob_df = model.predict_over_prob_sweep(X_test, thresholds=sorted(THRESHOLDS))

    # For each row: P(K >= t) should be non-increasing as t increases
    # (higher threshold → less likely to exceed it).
    sorted_thresholds = sorted(THRESHOLDS)
    for i in range(len(X_test)):
        probs = [prob_df[t].iloc[i] for t in sorted_thresholds]
        for a, b in zip(probs, probs[1:]):
            assert a >= b - 1e-8, (
                f"Row {i}: P(K>={sorted_thresholds[probs.index(a)]}) = {a:.4f} "
                f"< P(K>={sorted_thresholds[probs.index(b)]}) = {b:.4f} "
                "— calibration broke monotonicity"
            )


# ---------------------------------------------------------------------------
# 7. boosted_model.py — compute_agreement cases
# ---------------------------------------------------------------------------

def test_compute_agreement_both_above_line():
    from src.models.boosted_model import compute_agreement
    # GLM predicts 6.5 Ks, booster predicts 6.2 Ks, line is 5.5 → both above
    assert compute_agreement(6.5, 6.2, 5.5) == pytest.approx(1.0)


def test_compute_agreement_both_below_line():
    from src.models.boosted_model import compute_agreement
    # Both predict below the line → bearish agreement
    assert compute_agreement(4.5, 4.8, 5.5) == pytest.approx(-1.0)


def test_compute_agreement_disagree():
    from src.models.boosted_model import compute_agreement
    # GLM above line, booster below → disagree
    assert compute_agreement(6.0, 4.5, 5.5) == pytest.approx(0.0)
    # Booster above, GLM below → disagree
    assert compute_agreement(4.5, 6.0, 5.5) == pytest.approx(0.0)


def test_compute_agreement_glm_equals_line():
    from src.models.boosted_model import compute_agreement
    # GLM exactly on the line → np.sign = 0 → return 0.0
    assert compute_agreement(5.5, 6.0, 5.5) == pytest.approx(0.0)


def test_compute_agreement_nan_inputs():
    from src.models.boosted_model import compute_agreement
    assert math.isnan(compute_agreement(np.nan, 6.0, 5.5))
    assert math.isnan(compute_agreement(6.0, np.nan, 5.5))
    assert math.isnan(compute_agreement(6.0, 6.5, np.nan))


# ---------------------------------------------------------------------------
# 8. baseline_model.py — CONTEXT_CANDIDATE_COLUMNS present
# ---------------------------------------------------------------------------

def test_context_candidate_columns_present():
    from src.models.baseline_model import CONTEXT_CANDIDATE_COLUMNS

    assert isinstance(CONTEXT_CANDIDATE_COLUMNS, list)
    assert len(CONTEXT_CANDIDATE_COLUMNS) > 0

    expected = {"temp_f", "wind_mph", "humidity", "is_dome", "game_total", "is_favorite"}
    missing = expected - set(CONTEXT_CANDIDATE_COLUMNS)
    assert not missing, f"Missing from CONTEXT_CANDIDATE_COLUMNS: {missing}"


def test_context_candidate_columns_distinct_from_matchup():
    from src.models.baseline_model import CONTEXT_CANDIDATE_COLUMNS, MATCHUP_CANDIDATE_COLUMNS

    overlap = set(CONTEXT_CANDIDATE_COLUMNS) & set(MATCHUP_CANDIDATE_COLUMNS)
    assert not overlap, f"Unexpected column overlap: {overlap}"


# ---------------------------------------------------------------------------
# 9. Compile sweep — all five new / modified modules import without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "src.data.weather",
    "src.data.vegas",
    "src.models.boosted_model",
    "src.models.baseline_model",
    "src.pipeline.refresh",
])
def test_module_imports_cleanly(module):
    """Each modified file must be syntactically valid and import without error."""
    # Run from project root so "src.*" package paths resolve (python -c adds "" to sys.path)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True,
        cwd=project_root,
    )
    assert result.returncode == 0, (
        f"Import of {module} failed:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 10. Weather column constants match PITCHER_CARD_COLUMNS in refresh
# ---------------------------------------------------------------------------

def test_weather_candidate_columns_in_pitcher_card():
    from src.data.weather import WEATHER_CANDIDATE_COLUMNS
    from src.pipeline.refresh import PITCHER_CARD_COLUMNS

    for col in WEATHER_CANDIDATE_COLUMNS:
        assert col in PITCHER_CARD_COLUMNS, f"{col} missing from PITCHER_CARD_COLUMNS"

    assert "weather_was_imputed" in PITCHER_CARD_COLUMNS


def test_vegas_candidate_columns_in_pitcher_card():
    from src.data.vegas import VEGAS_CANDIDATE_COLUMNS
    from src.pipeline.refresh import PITCHER_CARD_COLUMNS

    for col in VEGAS_CANDIDATE_COLUMNS:
        assert col in PITCHER_CARD_COLUMNS, f"{col} missing from PITCHER_CARD_COLUMNS"

    assert "vegas_was_imputed" in PITCHER_CARD_COLUMNS


def test_booster_columns_in_pitcher_card():
    from src.pipeline.refresh import PITCHER_CARD_COLUMNS

    assert "booster_mu" in PITCHER_CARD_COLUMNS
    assert "glm_booster_agreement" in PITCHER_CARD_COLUMNS


# ---------------------------------------------------------------------------
# 11. Integration: refresh run calls weather/vegas fetchers when injected
# ---------------------------------------------------------------------------

def _minimal_game_log(pitcher_id: int = 1001, game_date: str = "2026-04-01"):
    """
    Game-log-schema row following test_matchup_umpire.py's pattern:
    start with all OUTPUT_COLUMNS as NaN, then populate the minimum set
    needed for add_rolling_features (strikeouts, batters_faced, etc.).
    """
    from src.features.game_logs import OUTPUT_COLUMNS
    row = {col: np.nan for col in OUTPUT_COLUMNS}
    row.update({
        "pitcher": pitcher_id,
        "game_pk": 1001,
        "game_date": pd.Timestamp(game_date),
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "pitcher_throws": "R",
        "strikeouts": 7,
        "batters_faced": 24,
        "rest_days": 5.0,
        "innings_pitched": 6.0,
        "pitch_count": 95,
        "whiff_rate": 0.28,
        "fastball_velo_avg": 95.0,
        "strikeouts_vs_LHB": 3,
        "batters_faced_vs_LHB": 12,
        "strikeouts_vs_RHB": 4,
        "batters_faced_vs_RHB": 12,
    })
    return pd.DataFrame([row])


def test_refresh_wiring_weather_vegas_fetchers_are_called():
    """
    Smoke test: run_refresh with injected weather/vegas fetchers calls them
    (verifies the wiring loop runs without error).
    """
    import src.pipeline.refresh as rmod
    from src.pipeline.refresh import run_refresh
    from unittest.mock import patch, MagicMock

    pitcher_id = 1001
    game_pk = 777
    game_log = _minimal_game_log(pitcher_id=pitcher_id)

    slate = pd.DataFrame([{
        "pitcher": pitcher_id,
        "pitcher_name": "Ace Pitcher",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "game_pk": game_pk,
        "game_date": "2026-04-01",
        "pitcher_throws": "R",
    }])

    model_mock = MagicMock()
    model_mock.preprocessor = {"impute_means": {}, "scale_stats": {}, "extra_columns": []}
    model_mock.family = "poisson"
    model_mock.alpha = None
    model_mock.predict_mean_with_se.return_value = (np.array([5.5]), np.array([0.3]))

    weather_result = {
        "temp_f": 72.0, "wind_mph": 10.0, "humidity": 60.0,
        "is_dome": 0.0, "weather_was_imputed": False,
    }
    vegas_result = {
        "game_total": 8.5, "is_favorite": 1.0,
        "team_total_for": np.nan, "team_total_against": np.nan,
        "vegas_was_imputed": False,
    }

    weather_calls = []
    vegas_calls = []

    def _wx(team, date, *a, **kw):
        weather_calls.append(team)
        return weather_result

    def _vx(team, date, ha, *a, **kw):
        vegas_calls.append(team)
        return vegas_result

    def fake_build_pitcher_cards(feature_rows, predictions):
        from src.pipeline.refresh import PITCHER_CARD_COLUMNS
        return pd.DataFrame(columns=PITCHER_CARD_COLUMNS)

    with (
        patch.object(rmod, "aggregate_pitcher_games", side_effect=lambda df: game_log),
        patch.object(rmod, "load_ump_tendency", return_value=pd.DataFrame()),
        patch.object(rmod, "build_pitcher_cards", side_effect=fake_build_pitcher_cards),
    ):
        results = run_refresh(
            game_date="2026-04-01",
            schedule_fetcher=lambda d: slate,
            statcast_fetcher=lambda pid, pname, season: game_log,
            lines_fetcher=lambda: pd.DataFrame(),
            register_fetcher=lambda: pd.DataFrame(),
            model_loader=lambda path: (model_mock, {"trained_at": "2026-04-01",
                                                     "train_through_date": None}),
            weather_fetcher=_wx,
            vegas_fetcher=_vx,
        )

    assert len(weather_calls) > 0, "weather_fetcher was never called"
    assert len(vegas_calls) > 0, "vegas_fetcher was never called"
    assert "pitcher_cards" in results
    assert "predictions" in results
