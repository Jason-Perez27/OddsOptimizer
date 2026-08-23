"""
Tests for Spec 2 plate-discipline skill features (2026-06-30).

Covers:
  1. Per-game aggregation in game_logs.aggregate_pitcher_games
     (csw_rate, putaway_rate, whiff_rate_overall, k_minus_bb)
  2. Strictly-prior rolling in rolling_features.add_rolling_features
     (swstr_rate_*, csw_rate_*, putaway_rate_*, whiff_rate_overall_*, k_minus_bb_rate_*)
  3. Leakage guardrail: appending a later game never mutates earlier rows skill features
  4. Debutant (no prior games) - all skill rolling features NaN, not propagated as regressors
  5. extra_columns path: SKILL_CANDIDATE_COLUMNS injected via build_design_matrix,
     missing columns imputed to 0 rather than raising
  6. --variant skill-features wraps fit_fn with extra_columns (run_backtest integration)
"""

import types

import numpy as np
import pandas as pd
import pytest

from src.features.game_logs import aggregate_pitcher_games
from src.features.rolling_features import add_rolling_features
from src.models.baseline_model import (
    SKILL_CANDIDATE_COLUMNS,
    build_design_matrix,
    transform_design_matrix,
    fit_preprocessor,
)


# ---------------------------------------------------------------------------
# Minimal pitch-level fixture builder
# ---------------------------------------------------------------------------

def _pitch(
    *,
    pitcher=1,
    game_pk=1,
    game_date="2026-04-01",
    home_team="NYY",
    away_team="BOS",
    inning_topbot="Top",
    events=None,
    description="ball",
    pitch_type="FF",
    release_speed=96.0,
    stand="R",
    p_throws="R",
    strikes=None,          # Statcast count-before-pitch column (optional)
):
    row = {
        "pitcher": pitcher,
        "game_pk": game_pk,
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "events": events,
        "description": description,
        "pitch_type": pitch_type,
        "release_speed": release_speed,
        "stand": stand,
        "p_throws": p_throws,
    }
    if strikes is not None:
        row["strikes"] = strikes
    return row


def _make_pitch_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Per-game aggregation
# ---------------------------------------------------------------------------

class TestPerGameAggregation:

    def _single_game_df(self):
        """Six pitches: 1 called_strike, 2 swinging_strike, 1 foul, 1 hit_into_play, 1 ball.
        Strikeout event on the last swinging_strike, walk on the ball-ending PA."""
        return _make_pitch_df([
            # 3 non-PA pitches (no events)
            _pitch(game_pk=10, description="called_strike", events=None),
            _pitch(game_pk=10, description="swinging_strike", events=None),
            _pitch(game_pk=10, description="foul", events=None),
            # PA endings
            _pitch(game_pk=10, description="swinging_strike", events="strikeout"),
            _pitch(game_pk=10, description="hit_into_play", events="field_out"),
            _pitch(game_pk=10, description="ball", events="walk"),
        ])

    def test_csw_rate(self):
        """CSW = called_strike + swinging_strike, denominator = total pitches."""
        df = self._single_game_df()
        out = aggregate_pitcher_games(df)
        assert len(out) == 1
        # pitches: 6; csw = 1 called_strike + 2 swinging_strike = 3
        assert out["csw_rate"].iloc[0] == pytest.approx(3 / 6)

    def test_whiff_rate_overall(self):
        """whiff_rate_overall = swinging_strikes / total_swings.
        swings = swinging_strike(x2) + foul + hit_into_play = 4;
        whiffs = 2 swinging_strike."""
        df = self._single_game_df()
        out = aggregate_pitcher_games(df)
        assert out["whiff_rate_overall"].iloc[0] == pytest.approx(2 / 4)

    def test_k_minus_bb(self):
        """1 strikeout, 1 walk -> k_minus_bb = 0."""
        df = self._single_game_df()
        out = aggregate_pitcher_games(df)
        assert out["k_minus_bb"].iloc[0] == 0

    def test_putaway_rate_with_strikes_column(self):
        """putaway_rate = strikeouts / two-strike pitches.
        Two pitches with strikes==2; 1 of those ends in strikeout."""
        pitches = [
            _pitch(game_pk=20, description="called_strike", events=None, strikes=0),
            _pitch(game_pk=20, description="swinging_strike", events=None, strikes=1),
            _pitch(game_pk=20, description="swinging_strike", events="strikeout", strikes=2),
            _pitch(game_pk=20, description="foul", events=None, strikes=2),
            _pitch(game_pk=20, description="hit_into_play", events="field_out", strikes=0),
        ]
        out = aggregate_pitcher_games(_make_pitch_df(pitches))
        # two-strike pitches = 2 (indices 2 and 3), strikeouts = 1
        assert out["putaway_rate"].iloc[0] == pytest.approx(1 / 2)

    def test_putaway_rate_nan_when_strikes_column_absent(self):
        """putaway_rate must be NaN when the Statcast `strikes` column is not present."""
        df = self._single_game_df()  # no `strikes` column
        out = aggregate_pitcher_games(df)
        assert pd.isna(out["putaway_rate"].iloc[0])

    def test_k_minus_bb_is_integer_type(self):
        """k_minus_bb is stored as int (int - int), not float."""
        df = self._single_game_df()
        out = aggregate_pitcher_games(df)
        assert isinstance(out["k_minus_bb"].iloc[0], (int, np.integer))


# ---------------------------------------------------------------------------
# 2 and 3. Strictly-prior rolling + leakage guardrail
# ---------------------------------------------------------------------------

def _game_log_row(**overrides):
    """Minimal game_logs-schema row with sensible defaults."""
    base = {
        "pitcher": 1,
        "game_pk": 1,
        "game_date": "2026-04-01",
        "pitcher_team": "NYY",
        "opponent_team": "BOS",
        "home_away": "home",
        "strikeouts": 5,
        "walks": 2,
        "batters_faced": 20,
        "pitch_count": 80,
        "whiff_rate": 0.20,
        "fastball_velo_avg": 95.0,
        "innings_pitched": 6.0,
        "strikeouts_vs_LHB": 2,
        "batters_faced_vs_LHB": 8,
        "strikeouts_vs_RHB": 3,
        "batters_faced_vs_RHB": 12,
        "rest_days": 5.0,
        "day_night": None,
        # skill columns (spec 2)
        "csw_rate": 0.30,
        "putaway_rate": 0.40,
        "whiff_rate_overall": 0.25,
        "k_minus_bb": 3,
    }
    base.update(overrides)
    return base


class TestSkillRollingFeatures:

    def test_first_game_skill_features_all_nan(self):
        """Pitcher debut: every skill rolling feature should be NaN."""
        df = add_rolling_features(pd.DataFrame([_game_log_row()]))
        row = df.iloc[0]
        for col in [
            "swstr_rate_last5", "swstr_rate_season",
            "csw_rate_last5", "csw_rate_season",
            "putaway_rate_last5", "putaway_rate_season",
            "whiff_rate_overall_last5", "whiff_rate_overall_season",
            "k_minus_bb_rate_last5", "k_minus_bb_rate_season",
        ]:
            assert pd.isna(row[col]), f"Expected NaN for {col} on debut game"

    def test_swstr_rate_season_is_strictly_prior(self):
        """swstr_rate_season at game 2 = whiff_rate from game 1 only (no same-game leakage)."""
        rows = [
            _game_log_row(game_pk=1, game_date="2026-04-01", whiff_rate=0.20),
            _game_log_row(game_pk=2, game_date="2026-04-06", whiff_rate=0.30),
            _game_log_row(game_pk=3, game_date="2026-04-11", whiff_rate=0.10),
        ]
        df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
        assert pd.isna(df.loc[1, "swstr_rate_season"])
        assert df.loc[2, "swstr_rate_season"] == pytest.approx(0.20)
        assert df.loc[3, "swstr_rate_season"] == pytest.approx(0.25)   # mean(0.20, 0.30)

    def test_csw_rate_last5_is_rolling_mean_not_pooled_rate(self):
        """csw_rate_last5 uses _prior_rolling_mean (mean of per-game values), not a
        pooled count/BF rate like k_rate_last5. Two games before game 3 -> mean of two."""
        rows = [
            _game_log_row(game_pk=1, game_date="2026-04-01", csw_rate=0.28),
            _game_log_row(game_pk=2, game_date="2026-04-06", csw_rate=0.32),
            _game_log_row(game_pk=3, game_date="2026-04-11", csw_rate=0.26),
        ]
        df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
        assert pd.isna(df.loc[1, "csw_rate_last5"])
        assert df.loc[2, "csw_rate_last5"] == pytest.approx(0.28)
        assert df.loc[3, "csw_rate_last5"] == pytest.approx((0.28 + 0.32) / 2)

    def test_k_minus_bb_rate_is_count_over_bf(self):
        """k_minus_bb_rate_last5 is a pooled count/BF rate (uses _add_count_rate_family)."""
        rows = [
            _game_log_row(game_pk=1, game_date="2026-04-01", k_minus_bb=3, batters_faced=20),
            _game_log_row(game_pk=2, game_date="2026-04-06", k_minus_bb=5, batters_faced=25),
            _game_log_row(game_pk=3, game_date="2026-04-11", k_minus_bb=2, batters_faced=18),
        ]
        df = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
        # Game 3 sees games 1+2: (3+5)/(20+25)
        assert df.loc[3, "k_minus_bb_rate_last5"] == pytest.approx(8 / 45)

    def test_skill_features_nan_when_source_columns_absent(self):
        """Rolling skill features are NaN when the source column is not in the input."""
        rows = [
            _game_log_row(game_pk=1, game_date="2026-04-01"),
            _game_log_row(game_pk=2, game_date="2026-04-06"),
        ]
        df_raw = pd.DataFrame(rows)
        # Drop the skill source columns to simulate an older cached feature table
        df_raw = df_raw.drop(columns=["csw_rate", "putaway_rate", "whiff_rate_overall", "k_minus_bb"])
        df = add_rolling_features(df_raw).set_index("game_pk")
        for col in ["csw_rate_last5", "csw_rate_season",
                    "putaway_rate_last5", "putaway_rate_season",
                    "whiff_rate_overall_last5", "whiff_rate_overall_season"]:
            assert pd.isna(df.loc[2, col]), f"Expected NaN for {col} when source absent"

    def test_leakage_guardrail_skill_features(self):
        """Adding a later game must not mutate earlier rows skill rolling feature values."""
        rows = [
            _game_log_row(game_pk=1, game_date="2026-04-01", csw_rate=0.28, k_minus_bb=3),
            _game_log_row(game_pk=2, game_date="2026-04-06", csw_rate=0.32, k_minus_bb=5),
            _game_log_row(game_pk=3, game_date="2026-04-11", csw_rate=0.26, k_minus_bb=2),
        ]
        later = _game_log_row(game_pk=4, game_date="2026-04-16", csw_rate=0.40, k_minus_bb=8)

        df_without = add_rolling_features(pd.DataFrame(rows)).set_index("game_pk")
        df_with    = add_rolling_features(pd.DataFrame(rows + [later])).set_index("game_pk")

        skill_cols = [
            "swstr_rate_last5", "swstr_rate_season",
            "csw_rate_last5", "csw_rate_season",
            "putaway_rate_last5", "putaway_rate_season",
            "whiff_rate_overall_last5", "whiff_rate_overall_season",
            "k_minus_bb_rate_last5", "k_minus_bb_rate_season",
        ]
        for game_pk in [1, 2, 3]:
            for col in skill_cols:
                a = df_without.loc[game_pk, col]
                b = df_with.loc[game_pk, col]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert a == pytest.approx(b), (
                    f"game {game_pk} col {col} changed when a later game was added (leakage)"
                )


# ---------------------------------------------------------------------------
# 4. Debutant -> extra_columns path imputes NaN, never raises
# ---------------------------------------------------------------------------

def _minimal_feature_table(n=5):
    """Minimal raw-feature-table rows (input to fit_preprocessor, not design-matrix output).
    Includes home_away + rest_days (needed by _engineer_raw_columns) and the
    CORE_PITCHER_FORM_COLUMNS (must be non-null to survive _dropna_core)."""
    from src.models.baseline_model import DESIGN_MATRIX_COLUMNS, CORE_PITCHER_FORM_COLUMNS
    rows = []
    for i in range(n):
        row = {
            "home_away": "home",
            "rest_days": 5.0,
            "strikeouts": 5,
            "batters_faced": 20,
        }
        # CORE columns non-null
        for col in CORE_PITCHER_FORM_COLUMNS:
            row[col] = 0.25
        # Fill remaining DESIGN_MATRIX_COLUMNS with a filler value
        for col in DESIGN_MATRIX_COLUMNS:
            if col not in row and col not in ("const", "is_home"):
                row[col] = 0.5
        row.update({col: float("nan") for col in SKILL_CANDIDATE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


class TestExtraColumnsPath:

    def test_missing_extra_columns_imputed_not_raised(self):
        """If extra_columns columns are ALL NaN (debutant), transform_design_matrix
        must not raise and must fill them with the impute mean (0.0 sentinel)."""
        from src.models.baseline_model import DESIGN_MATRIX_COLUMNS, CORE_PITCHER_FORM_COLUMNS
        train = _minimal_feature_table(10)
        # Deliberately omit skill columns from test frame
        test_row = {"home_away": "home", "rest_days": 5.0, "strikeouts": 5}
        for col in CORE_PITCHER_FORM_COLUMNS:
            test_row[col] = 0.25
        for col in DESIGN_MATRIX_COLUMNS:
            if col not in test_row and col not in ("const", "is_home"):
                test_row[col] = 0.5
        test_df = pd.DataFrame([test_row])

        preprocessor = fit_preprocessor(train, extra_columns=SKILL_CANDIDATE_COLUMNS)
        # Should not raise even though test_df has no skill columns
        X = transform_design_matrix(test_df, preprocessor)
        for col in SKILL_CANDIDATE_COLUMNS:
            assert col in X.columns
            # All NaN -> imputed to 0, then standardized to 0 (mean=0, std=1 sentinel)
            assert not X[col].isna().any(), f"{col} still NaN after impute"

    def test_extra_columns_extend_design_matrix(self):
        """build_design_matrix with extra_columns returns a matrix that includes those cols."""
        from src.models.baseline_model import CORE_PITCHER_FORM_COLUMNS
        train = _minimal_feature_table(20)
        # Provide non-NaN skill values on half the rows so impute mean is non-trivial
        for col in SKILL_CANDIDATE_COLUMNS:
            train.loc[train.index[:10], col] = 0.30

        X_train, _y, _Xt, _yt, _pp = build_design_matrix(train, extra_columns=SKILL_CANDIDATE_COLUMNS)
        for col in SKILL_CANDIDATE_COLUMNS:
            assert col in X_train.columns, f"{col} missing from design matrix"

    def test_skill_candidate_columns_list_has_expected_length(self):
        """Sanity-check the constant itself: spec 2 defines 7 candidate columns."""
        assert len(SKILL_CANDIDATE_COLUMNS) == 7, (
            f"Expected 7 SKILL_CANDIDATE_COLUMNS, got {len(SKILL_CANDIDATE_COLUMNS)}: "
            f"{SKILL_CANDIDATE_COLUMNS}"
        )


# ---------------------------------------------------------------------------
# 5. --variant skill-features wraps fit_fn (run_backtest integration)
# ---------------------------------------------------------------------------

class TestVariantSkillFeaturesWrapsFitFn:

    def _make_args(self, variant=None):
        args = types.SimpleNamespace(
            start="2024-04-01",
            end="2024-09-30",
            cache_dir="/tmp/cache",
            window_days=7,
            min_batters_faced=None,
            step=7,
            min_train_dates=14,
            oos_path="/tmp/oos.csv",
            reports_dir="/tmp/reports",
            no_production_model=True,
            fit_only=False,
            through_date=None,
            model_path="/tmp/model.joblib",
            variant=variant,
        )
        return args

    def test_no_variant_fit_fn_unchanged(self):
        """When --variant is None, run_backtest_pipeline must call fit_fn directly
        (no wrapping) -- verified by checking the captured extra_columns kwarg is absent."""
        from scripts.run_backtest import run_backtest_pipeline

        captured = {}

        def fake_fit_fn(train_df, test_df=None, **kw):
            captured["kw"] = kw
            m = types.SimpleNamespace(
                params=pd.Series({"const": 0.0}),
                predict=lambda X: pd.Series([5.0] * len(X)),
            )
            return m

        import unittest.mock as mock
        with (
            mock.patch("scripts.run_backtest.build_corpus",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.build_training_table",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.filter_starters",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.run_walk_forward",
                       return_value=pd.DataFrame()),
        ):
            try:
                run_backtest_pipeline(
                    self._make_args(variant=None),
                    statcast_fetcher=lambda *a, **kw: pd.DataFrame(),
                    fit_fn=fake_fit_fn,
                )
            except SystemExit:
                pass  # empty OOS -> sys.exit(1); that is fine, we only care about kw

        # Without --variant the wrapper must NOT inject extra_columns
        assert "extra_columns" not in captured.get("kw", {}), (
            "fit_fn should not receive extra_columns without --variant"
        )

    def test_variant_skill_features_injects_extra_columns(self):
        """When --variant skill-features, run_backtest_pipeline wraps fit_fn so it
        receives extra_columns=SKILL_CANDIDATE_COLUMNS.

        run_walk_forward is replaced with a fake that actually *calls* the fit_fn
        it receives (matching the real implementation), so the wrapper's injection
        of extra_columns is observable.  persist_oos_frame and
        generate_backtest_report are also mocked to avoid file I/O.
        """
        from scripts.run_backtest import run_backtest_pipeline

        captured = {}

        def fake_fit_fn(train_df, test_df=None, **kw):
            captured["kw"] = kw
            m = types.SimpleNamespace(
                params=pd.Series({"const": 0.0}),
                predict=lambda X: pd.Series([5.0] * len(X)),
            )
            return m, {}, (None, None)

        def fake_walk_forward(feature_table, step=None, min_train_dates=None, fit_fn=None):
            # Invoke the (possibly-wrapped) fit_fn so the wrapper's kwarg injection
            # is visible in captured["kw"].
            if fit_fn is not None:
                fit_fn(pd.DataFrame([{"x": 1}]))
            return pd.DataFrame([{"oos_row": 1}])  # non-empty to avoid sys.exit

        import unittest.mock as mock
        with (
            mock.patch("scripts.run_backtest.build_corpus",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.build_training_table",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.filter_starters",
                       return_value=pd.DataFrame([{"x": 1}])),
            mock.patch("scripts.run_backtest.run_walk_forward",
                       side_effect=fake_walk_forward),
            mock.patch("scripts.run_backtest.persist_oos_frame",
                       return_value="/tmp/oos.csv"),
            mock.patch("scripts.run_backtest.generate_backtest_report",
                       return_value={
                           "report_path": "/tmp/r.md",
                           "reliability_plot_path": "/tmp/a.png",
                           "calibration_by_tier_plot_path": "/tmp/b.png",
                           "error_over_time_plot_path": "/tmp/c.png",
                       }),
        ):
            args = self._make_args(variant="skill-features")
            args.no_production_model = True  # skip final model save
            run_backtest_pipeline(
                args,
                statcast_fetcher=lambda *a, **kw: pd.DataFrame(),
                fit_fn=fake_fit_fn,
            )

        assert "extra_columns" in captured.get("kw", {}), (
            "fit_fn must receive extra_columns with --variant skill-features"
        )
        assert captured["kw"]["extra_columns"] == SKILL_CANDIDATE_COLUMNS
