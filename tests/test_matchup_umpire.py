"""
Tests for spec ③ (2026-06-30) matchup + umpire features:
  - src/data/lineups.py
  - src/features/batter_logs.py
  - src/data/umpires.py
  - src/features/opponent_features.build_lineup_weighted_opp_k
  - refresh.py lineup/ump wiring (injected-fetcher smoke test)

All tests are network-free: fetchers are lambda injections returning
pre-built dicts.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.lineups import (
    parse_lineup,
    get_lineup,
    LINEUP_COLUMNS,
    MIN_LINEUP_SIZE,
)
from src.features.batter_logs import (
    aggregate_batter_games,
    add_batter_rolling_features,
    BATTER_GAME_COLUMNS,
    ROLLING_BATTER_COLUMNS,
)
from src.data.umpires import (
    load_ump_tendency,
    fetch_hp_umpire,
    get_ump_k_factor,
    NEUTRAL_K_FACTOR,
    MIN_GAMES,
)
from src.features.opponent_features import (
    build_lineup_weighted_opp_k,
    MATCHUP_FEATURE_COLUMNS,
)
from src.models.baseline_model import MATCHUP_CANDIDATE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nine_batters(side: str, bat_side: str = "R", k_rate: float = 0.25):
    """Return a DataFrame with 9 batters for one team side."""
    return pd.DataFrame([
        {
            "game_pk": 1,
            "team_side": side,
            "batter_id": 100 + i,
            "batter_name": f"Batter{i}",
            "bat_side": bat_side,
            "lineup_slot": i,
            "lineup_source": "confirmed",
        }
        for i in range(1, 10)
    ])


def _raw_lineup_payload(
    game_pk: int = 1,
    n_home: int = 9,
    n_away: int = 9,
    bat_side: str = "R",
) -> dict:
    """Build a minimal StatsAPI schedule+lineups payload."""
    def _players(side_key, n, id_offset):
        return [
            {
                "id": id_offset + i,
                "fullName": f"Player{id_offset + i}",
                "batSide": {"code": bat_side},
                "lineupPosition": i,
            }
            for i in range(1, n + 1)
        ]

    return {
        "dates": [{
            "games": [{
                "gamePk": game_pk,
                "lineups": {
                    "homePlayers": _players("home", n_home, 100),
                    "awayPlayers": _players("away", n_away, 200),
                },
            }]
        }]
    }


# ---------------------------------------------------------------------------
# TestLineupParse
# ---------------------------------------------------------------------------

class TestLineupParse:
    """parse_lineup pure-function tests."""

    def test_full_confirmed_lineup_both_sides(self):
        raw = _raw_lineup_payload(game_pk=42, n_home=9, n_away=9)
        df = parse_lineup(raw, game_pk=42)
        assert not df.empty
        assert set(df.columns) == set(LINEUP_COLUMNS)
        assert len(df) == 18
        assert (df["lineup_source"] == "confirmed").all()
        assert set(df["team_side"]) == {"home", "away"}

    def test_partial_lineup_marked_projected(self):
        # Only home side has batters
        raw = _raw_lineup_payload(game_pk=1, n_home=9, n_away=0)
        df = parse_lineup(raw, game_pk=1)
        assert not df.empty
        assert (df["lineup_source"] == "projected").all()

    def test_empty_payload_returns_empty_df(self):
        df = parse_lineup({}, game_pk=1)
        assert df.empty
        assert list(df.columns) == LINEUP_COLUMNS

    def test_missing_dates_returns_empty(self):
        df = parse_lineup({"dates": []}, game_pk=1)
        assert df.empty

    def test_battingorder_encoded_as_hundreds(self):
        """StatsAPI sometimes encodes slot as 100, 200, ..."""
        raw = {
            "dates": [{
                "games": [{
                    "gamePk": 1,
                    "lineups": {
                        "homePlayers": [
                            {"id": 101, "fullName": "A", "batSide": {"code": "L"},
                             "battingOrder": 100},
                            {"id": 102, "fullName": "B", "batSide": {"code": "R"},
                             "battingOrder": 900},
                        ],
                        "awayPlayers": [],
                    },
                }]
            }]
        }
        df = parse_lineup(raw, game_pk=1)
        slots = sorted(df["lineup_slot"].tolist())
        assert slots == [1, 9]

    def test_lineup_source_below_min_size_is_projected(self):
        raw = _raw_lineup_payload(game_pk=1, n_home=MIN_LINEUP_SIZE - 1, n_away=MIN_LINEUP_SIZE - 1)
        df = parse_lineup(raw, game_pk=1)
        assert (df["lineup_source"] == "projected").all()

    def test_get_lineup_injected_fetcher(self):
        raw = _raw_lineup_payload(game_pk=7, n_home=9, n_away=9)
        df = get_lineup(7, fetcher=lambda gp: raw)
        assert len(df) == 18
        assert (df["lineup_source"] == "confirmed").all()

    def test_get_lineup_fetch_error_returns_empty(self):
        df = get_lineup(99, fetcher=lambda gp: (_ for _ in ()).throw(RuntimeError("down")))
        assert df.empty
        assert list(df.columns) == LINEUP_COLUMNS


# ---------------------------------------------------------------------------
# TestBatterLogs
# ---------------------------------------------------------------------------

def _pitch_df():
    """Minimal pitch-level Statcast frame for 2 batters over 3 games each."""
    rows = []
    for batter_id in [10, 20]:
        for game_num in range(3):
            game_pk = 1000 + game_num
            game_date = pd.Timestamp(f"2025-04-{10 + game_num}")
            # 3 PA vs RHP: 1 K, 2 outs
            for _ in range(3):
                rows.append({
                    "batter": batter_id,
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "events": "strikeout",
                    "stand": "L",
                    "p_throws": "R",
                })
            # 2 PA vs LHP: 0 K
            for _ in range(2):
                rows.append({
                    "batter": batter_id,
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "events": "field_out",
                    "stand": "L",
                    "p_throws": "L",
                })
    return pd.DataFrame(rows)


class TestBatterLogs:
    def test_aggregate_batter_games_shape(self):
        pitches = _pitch_df()
        bg = aggregate_batter_games(pitches)
        assert list(bg.columns) == BATTER_GAME_COLUMNS
        # 2 batters × 3 games = 6 rows
        assert len(bg) == 6

    def test_aggregate_counts_correct(self):
        pitches = _pitch_df()
        bg = aggregate_batter_games(pitches)
        row = bg[(bg["batter"] == 10) & (bg["game_pk"] == 1000)].iloc[0]
        assert row["pa_vs_rhp"] == 3
        assert row["strikeouts_vs_rhp"] == 3
        assert row["pa_vs_lhp"] == 2
        assert row["strikeouts_vs_lhp"] == 0

    def test_empty_returns_empty(self):
        bg = aggregate_batter_games(pd.DataFrame())
        assert bg.empty
        assert list(bg.columns) == BATTER_GAME_COLUMNS

    def test_rolling_features_columns_added(self):
        pitches = _pitch_df()
        bg = aggregate_batter_games(pitches)
        rf = add_batter_rolling_features(bg)
        for col in ROLLING_BATTER_COLUMNS:
            assert col in rf.columns

    def test_shift1_leakage_guard_first_game_nan(self):
        """First game of each batter's season must have NaN rolling stats."""
        pitches = _pitch_df()
        bg = aggregate_batter_games(pitches)
        rf = add_batter_rolling_features(bg)
        first_rows = rf.groupby("batter").head(1)
        # First game has no prior data → season rate should be NaN
        assert first_rows["k_rate_vs_rhp_season"].isna().all()
        assert first_rows["k_rate_vs_lhp_season"].isna().all()

    def test_shift1_second_game_sees_first(self):
        """Second game should have the first game's stats as its rolling rate."""
        pitches = _pitch_df()
        bg = aggregate_batter_games(pitches)
        rf = add_batter_rolling_features(bg)
        batter10 = rf[rf["batter"] == 10].sort_values("game_date").reset_index(drop=True)
        # Second game sees first game: 3 K / 3 PA vs RHP = 1.0
        second = batter10.iloc[1]
        assert abs(second["k_rate_vs_rhp_season"] - 1.0) < 1e-9

    def test_rolling_empty_returns_nan_columns(self):
        result = add_batter_rolling_features(pd.DataFrame(columns=BATTER_GAME_COLUMNS))
        for col in ROLLING_BATTER_COLUMNS:
            assert col in result.columns


# ---------------------------------------------------------------------------
# TestUmpires
# ---------------------------------------------------------------------------

def _tendency_df(ump_id=50, k_factor=1.08, games_sampled=100):
    return pd.DataFrame([{
        "ump_id": ump_id, "ump_name": "Angel H.", "k_factor": k_factor,
        "games_sampled": games_sampled,
    }])


def _officials_payload(ump_id=50, name="Angel H.", office_type="Home Plate"):
    return {
        "dates": [{
            "games": [{
                "officials": [
                    {"official": {"id": ump_id, "fullName": name},
                     "officialType": office_type}
                ]
            }]
        }]
    }


class TestUmpires:
    def test_load_tendency_missing_file_returns_empty(self, tmp_path):
        df = load_ump_tendency(str(tmp_path / "nonexistent.csv"))
        assert df.empty

    def test_load_tendency_reads_csv(self, tmp_path):
        p = tmp_path / "ump_tendency.csv"
        p.write_text("ump_id,ump_name,k_factor,games_sampled\n50,Hernandez,1.08,120\n")
        df = load_ump_tendency(str(p))
        assert len(df) == 1
        assert df.iloc[0]["k_factor"] == pytest.approx(1.08)

    def test_fetch_hp_umpire_returns_id(self):
        raw = _officials_payload(ump_id=99)
        result = fetch_hp_umpire(1, fetcher=lambda gp: raw)
        assert result == 99

    def test_fetch_hp_umpire_wrong_type_returns_none(self):
        raw = _officials_payload(ump_id=99, office_type="First Base")
        result = fetch_hp_umpire(1, fetcher=lambda gp: raw)
        assert result is None

    def test_fetch_hp_umpire_network_error_returns_none(self):
        result = fetch_hp_umpire(1, fetcher=lambda gp: (_ for _ in ()).throw(OSError()))
        assert result is None

    def test_get_ump_k_factor_known_ump(self):
        tendency = _tendency_df(ump_id=50, k_factor=1.08, games_sampled=100)
        raw = _officials_payload(ump_id=50)
        kf, imputed = get_ump_k_factor(1, tendency, fetcher=lambda gp: raw)
        assert abs(kf - 1.08) < 1e-9
        assert imputed is False

    def test_get_ump_k_factor_unknown_ump_neutral(self):
        tendency = _tendency_df(ump_id=50)
        raw = _officials_payload(ump_id=99)  # ump 99 not in tendency
        kf, imputed = get_ump_k_factor(1, tendency, fetcher=lambda gp: raw)
        assert kf == NEUTRAL_K_FACTOR
        assert imputed is True

    def test_get_ump_k_factor_thin_sample_neutral(self):
        tendency = _tendency_df(ump_id=50, k_factor=1.15, games_sampled=MIN_GAMES - 1)
        raw = _officials_payload(ump_id=50)
        kf, imputed = get_ump_k_factor(1, tendency, fetcher=lambda gp: raw)
        assert kf == NEUTRAL_K_FACTOR
        assert imputed is True

    def test_get_ump_k_factor_empty_tendency_neutral(self):
        kf, imputed = get_ump_k_factor(1, pd.DataFrame(), fetcher=lambda gp: _officials_payload())
        assert kf == NEUTRAL_K_FACTOR
        assert imputed is True

    def test_get_ump_k_factor_fetch_fails_neutral(self):
        tendency = _tendency_df(ump_id=50)
        kf, imputed = get_ump_k_factor(1, tendency, fetcher=lambda gp: (_ for _ in ()).throw(OSError()))
        assert kf == NEUTRAL_K_FACTOR
        assert imputed is True


# ---------------------------------------------------------------------------
# TestLineupWeightedOppK
# ---------------------------------------------------------------------------

def _batter_rolling(batter_ids, k_rate_lhp=0.2, k_rate_rhp=0.3):
    """Simple rolling frame: one 'final' row per batter with pre-game stats."""
    rows = []
    for i, bid in enumerate(batter_ids):
        rows.append({
            "batter": bid,
            "game_date": pd.Timestamp("2025-05-01"),
            "k_rate_vs_lhp_season": k_rate_lhp,
            "k_rate_vs_rhp_season": k_rate_rhp,
            "k_rate_vs_lhp_last10": k_rate_lhp,
            "k_rate_vs_rhp_last10": k_rate_rhp,
        })
    return pd.DataFrame(rows)


class TestLineupWeightedOppK:
    def test_empty_lineup_returns_nan(self):
        result = build_lineup_weighted_opp_k(
            pd.DataFrame(columns=LINEUP_COLUMNS), None, "R"
        )
        assert np.isnan(result["opponent_lineup_k_rate_vs_hand"])
        assert np.isnan(result["opp_share_opposite_hand"])

    def test_none_lineup_returns_nan(self):
        result = build_lineup_weighted_opp_k(None, None, "R")
        assert np.isnan(result["opponent_lineup_k_rate_vs_hand"])

    def test_rhp_uses_rhp_k_rate(self):
        lineup = _nine_batters("away", bat_side="R")
        batter_ids = lineup["batter_id"].tolist()
        rolling = _batter_rolling(batter_ids, k_rate_lhp=0.10, k_rate_rhp=0.35)
        result = build_lineup_weighted_opp_k(lineup, rolling, pitcher_throws="R")
        assert abs(result["opponent_lineup_k_rate_vs_hand"] - 0.35) < 1e-6

    def test_lhp_uses_lhp_k_rate(self):
        lineup = _nine_batters("away", bat_side="L")
        batter_ids = lineup["batter_id"].tolist()
        rolling = _batter_rolling(batter_ids, k_rate_lhp=0.20, k_rate_rhp=0.35)
        result = build_lineup_weighted_opp_k(lineup, rolling, pitcher_throws="L")
        assert abs(result["opponent_lineup_k_rate_vs_hand"] - 0.20) < 1e-6

    def test_platoon_share_all_opposite(self):
        """All LHB vs RHP → opp_share = 1.0."""
        lineup = _nine_batters("away", bat_side="L")
        result = build_lineup_weighted_opp_k(lineup, None, pitcher_throws="R")
        assert abs(result["opp_share_opposite_hand"] - 1.0) < 1e-9

    def test_platoon_share_all_same(self):
        """All RHB vs RHP → opp_share = 0.0."""
        lineup = _nine_batters("away", bat_side="R")
        result = build_lineup_weighted_opp_k(lineup, None, pitcher_throws="R")
        assert abs(result["opp_share_opposite_hand"] - 0.0) < 1e-9

    def test_switch_hitters_count_as_opposite(self):
        """Switch hitters (S) always count as opposite hand."""
        lineup = _nine_batters("away", bat_side="S")
        result = build_lineup_weighted_opp_k(lineup, None, pitcher_throws="R")
        assert abs(result["opp_share_opposite_hand"] - 1.0) < 1e-9

    def test_no_batter_rolling_gives_nan_k_rate_but_valid_platoon(self):
        lineup = _nine_batters("away", bat_side="L")
        result = build_lineup_weighted_opp_k(lineup, None, pitcher_throws="R")
        # No rolling data → k_rate NaN, but platoon share still computable
        assert np.isnan(result["opponent_lineup_k_rate_vs_hand"])
        assert abs(result["opp_share_opposite_hand"] - 1.0) < 1e-9

    def test_slot_weights_higher_slots_contribute_more(self):
        """Slot 1 batter has higher K rate; should pull weighted mean up vs equal weight."""
        lineup = pd.DataFrame([
            {"game_pk": 1, "team_side": "away", "batter_id": 1, "batter_name": "A",
             "bat_side": "R", "lineup_slot": 1, "lineup_source": "confirmed"},
            {"game_pk": 1, "team_side": "away", "batter_id": 2, "batter_name": "B",
             "bat_side": "R", "lineup_slot": 9, "lineup_source": "confirmed"},
        ])
        rolling = pd.DataFrame([
            {"batter": 1, "game_date": pd.Timestamp("2025-05-01"),
             "k_rate_vs_rhp_season": 0.40, "k_rate_vs_lhp_season": 0.0,
             "k_rate_vs_rhp_last10": 0.0, "k_rate_vs_lhp_last10": 0.0},
            {"batter": 2, "game_date": pd.Timestamp("2025-05-01"),
             "k_rate_vs_rhp_season": 0.10, "k_rate_vs_lhp_season": 0.0,
             "k_rate_vs_rhp_last10": 0.0, "k_rate_vs_lhp_last10": 0.0},
        ])
        result = build_lineup_weighted_opp_k(lineup, rolling, pitcher_throws="R")
        # slot 1 weight=9, slot 9 weight=1 → weighted = (0.40*9 + 0.10*1)/10 = 0.37
        expected = (0.40 * 9 + 0.10 * 1) / 10
        assert abs(result["opponent_lineup_k_rate_vs_hand"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestMatchupCandidateColumns
# ---------------------------------------------------------------------------

class TestMatchupCandidateColumns:
    def test_columns_defined(self):
        assert "opponent_lineup_k_rate_vs_hand" in MATCHUP_CANDIDATE_COLUMNS
        assert "opp_share_opposite_hand" in MATCHUP_CANDIDATE_COLUMNS
        assert "ump_k_factor" in MATCHUP_CANDIDATE_COLUMNS

    def test_matchup_feature_columns_subset_of_candidate(self):
        for col in MATCHUP_FEATURE_COLUMNS:
            assert col in MATCHUP_CANDIDATE_COLUMNS or col == "opp_share_opposite_hand"


# ---------------------------------------------------------------------------
# TestRefreshWiring (smoke test only -- no network, no model disk I/O)
# ---------------------------------------------------------------------------

class TestRefreshWiring:
    """
    Verify refresh.run_refresh enriches feature_rows with the new
    lineup/ump columns when injected fetchers return valid data.

    We reuse the existing refresh test pattern: all external calls are
    injected lambdas; the model is a minimal stub.
    """

    def _make_stub_model(self):
        """Minimal model stub that passes assemble_predictions."""
        from unittest.mock import MagicMock
        import numpy as np
        model = MagicMock()
        model.family = "poisson"
        model.alpha = None
        model.preprocessor = {
            "impute_means": {}, "scale_stats": {}, "extra_columns": []
        }
        model.predict_mean_with_se.return_value = (np.array([6.0]), np.array([0.5]))
        return model

    def _minimal_game_log(self, pitcher_id=999, game_date="2025-04-10"):
        """Return a game_logs-schema DataFrame with enough columns for rolling."""
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
            "walk": 2,
            "is_home": 1,
            "rest_days": 5,
        })
        return pd.DataFrame([row])

    def test_refresh_injects_lineup_and_ump_columns(self):
        """feature_rows after run_refresh should have all new matchup columns."""
        from unittest.mock import patch, MagicMock
        from src.pipeline import refresh as rmod

        pitcher_id = 555
        game_pk = 7777

        slate_row = {
            "pitcher": pitcher_id,
            "pitcher_name": "Test Pitcher",
            "pitcher_team": "NYY",
            "opponent_team": "BOS",
            "game_date": "2025-06-15",
            "game_pk": game_pk,
            "home_away": "home",
            "pitcher_throws": "R",
        }
        slate_df = pd.DataFrame([slate_row])

        # Statcast history for the pitcher
        game_log = self._minimal_game_log(pitcher_id=pitcher_id)

        # Raw lineup payload (9+9 full lineup)
        lineup_raw = _raw_lineup_payload(game_pk=game_pk, n_home=9, n_away=9)
        from src.data.lineups import parse_lineup
        lineup_df_fixture = parse_lineup(lineup_raw, game_pk)

        def fake_lineup_fetcher(gp):
            return lineup_df_fixture

        officials_raw = _officials_payload(ump_id=77)

        def fake_officials_fetcher(gp):
            return officials_raw

        tendency_df = _tendency_df(ump_id=77, k_factor=1.05, games_sampled=120)

        stub_model = self._make_stub_model()

        captured = {}

        original_build_pitcher_cards = rmod.build_pitcher_cards

        def fake_build_pitcher_cards(feature_rows, predictions):
            captured["feature_rows"] = feature_rows.copy()
            return original_build_pitcher_cards(feature_rows, predictions)

        with (
            patch.object(rmod, "aggregate_pitcher_games", side_effect=lambda df: game_log),
            patch.object(rmod, "load_ump_tendency", return_value=tendency_df),
            patch.object(rmod, "build_pitcher_cards", side_effect=fake_build_pitcher_cards),
        ):
            results = rmod.run_refresh(
                game_date="2025-06-15",
                schedule_fetcher=lambda gd: slate_df,
                statcast_fetcher=lambda pid, pname, season: game_log,
                lines_fetcher=lambda: pd.DataFrame(columns=["projection_id", "player_name",
                    "stat_type", "line_score", "odds_type", "start_time", "game_pk"]),
                register_fetcher=lambda: pd.DataFrame(columns=["key_mlbam", "name_first", "name_last"]),
                model_loader=lambda path: (stub_model, {"trained_at": "2025-06-14T12:00:00+00:00"}),
                lineup_fetcher=fake_lineup_fetcher,
                officials_fetcher=fake_officials_fetcher,
            )

        fr = captured.get("feature_rows")
        assert fr is not None, "build_pitcher_cards was not called"
        assert "lineup_source" in fr.columns
        assert "ump_k_factor" in fr.columns
        assert "ump_was_imputed" in fr.columns
        assert "opponent_lineup_k_rate_vs_hand" in fr.columns
        assert "opp_share_opposite_hand" in fr.columns

        row = fr.iloc[0]
        assert row["lineup_source"] == "confirmed"
        # ump k_factor comes from tendency_df stub (ump 77 is unknown to get_ump_k_factor
        # unless tendency is patched in — it was patched via load_ump_tendency)
        # ump_was_imputed=1.0 is fine since officials_fetcher returns id 77 but
        # load_ump_tendency was monkeypatched to return tendency_df with ump_id=77
        # NOTE: get_ump_k_factor uses the tendency_df loaded inside run_refresh,
        # which we patched → kf=1.05, imputed=False
        assert float(row["ump_k_factor"]) == pytest.approx(1.05)
        assert float(row["ump_was_imputed"]) == pytest.approx(0.0)

    def test_refresh_team_fallback_when_no_lineup(self):
        """When lineup_fetcher returns empty, lineup_source = 'team_fallback'."""
        from unittest.mock import patch
        from src.pipeline import refresh as rmod

        pitcher_id = 556
        game_pk = 7778
        slate_df = pd.DataFrame([{
            "pitcher": pitcher_id, "pitcher_name": "P2",
            "pitcher_team": "NYM", "opponent_team": "PHI",
            "game_date": "2025-06-15", "game_pk": game_pk,
            "home_away": "away", "pitcher_throws": "L",
        }])
        game_log = self._minimal_game_log(pitcher_id=pitcher_id)
        stub_model = self._make_stub_model()
        captured = {}
        original_bpc = rmod.build_pitcher_cards
        def fake_bpc(fr, p):
            captured["fr"] = fr.copy()
            return original_bpc(fr, p)

        with (
            patch.object(rmod, "aggregate_pitcher_games", side_effect=lambda df: game_log),
            patch.object(rmod, "load_ump_tendency", return_value=pd.DataFrame()),
            patch.object(rmod, "build_pitcher_cards", side_effect=fake_bpc),
        ):
            rmod.run_refresh(
                game_date="2025-06-15",
                schedule_fetcher=lambda gd: slate_df,
                statcast_fetcher=lambda pid, pname, season: game_log,
                lines_fetcher=lambda: pd.DataFrame(columns=["projection_id", "player_name",
                    "stat_type", "line_score", "odds_type", "start_time", "game_pk"]),
                register_fetcher=lambda: pd.DataFrame(columns=["key_mlbam", "name_first", "name_last"]),
                model_loader=lambda path: (stub_model, {"trained_at": "2025-06-14T12:00:00+00:00"}),
                # lineup_fetcher returns empty → team_fallback
                lineup_fetcher=lambda gp: pd.DataFrame(columns=rmod.lineup_mod.LINEUP_COLUMNS),
                officials_fetcher=lambda gp: {},
            )

        fr = captured["fr"]
        assert fr.iloc[0]["lineup_source"] == "team_fallback"
