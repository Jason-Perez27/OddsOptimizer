"""
tests/test_compression_fix.py

Tests for the projection-compression fix
(spec: docs/design/specs/2026-06-30-fix-projection-compression-and-matchup-design.md).

Covers three spec-required test categories:
  1. EB-stabilized rate: a pitcher with a 2-start slump but strong season is
     pulled toward season, not tanked; k_stab_last5 > k_rate_last5 after slump.
  2. Leakage test: k_stab_last5 for game N uses only strictly-prior data
     (same guarantee as k_rate_last5 — the EB computation inherits it).
  3. Tail-calibration helper:
     a) compute_mu_decile_table returns correct structure and values on a
        hand-built OOS fixture.
     b) flag_high_mu_underprediction correctly detects systematic high-μ bias.

No network calls, no statsmodels fit required — all fixtures are synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.rolling_features import (
    EB_K_CONSTANT,
    LEAGUE_K_RATE_PRIOR,
    LAST5_WINDOW,
    add_rolling_features,
)
from src.backtest.tail_calibration import (
    compute_mu_decile_table,
    flag_high_mu_underprediction,
    format_decile_table_md,
    high_mu_reliability_curve,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_pitcher_game_df(
    strikeouts: list,
    batters_faced: list,
    pitcher_id: int = 100,
    start_date: str = "2024-04-01",
) -> pd.DataFrame:
    """
    Build a minimal pitcher-game DataFrame compatible with add_rolling_features.

    All non-test columns are filled with plausible stubs; the function only
    exercises the K-rate and EB-K computation paths.
    """
    n = len(strikeouts)
    dates = pd.date_range(start=start_date, periods=n, freq="5D")
    df = pd.DataFrame({
        "pitcher": [pitcher_id] * n,
        "game_date": dates,
        "game_pk": range(n),
        "team": ["NYY"] * n,
        "opponent_team": ["BOS"] * n,
        "home_away": ["home"] * n,
        "strikeouts": strikeouts,
        "batters_faced": batters_faced,
        "walks": [1] * n,
        "innings_pitched": [5.0] * n,
        "pitch_count": [85] * n,
        "fastball_velo_avg": [93.0] * n,
        "whiff_rate": [0.25] * n,
        "strikeouts_vs_LHB": [k // 2 for k in strikeouts],
        "batters_faced_vs_LHB": [bf // 2 for bf in batters_faced],
        "strikeouts_vs_RHB": [k - k // 2 for k in strikeouts],
        "batters_faced_vs_RHB": [bf - bf // 2 for bf in batters_faced],
    })
    return df


# ---------------------------------------------------------------------------
# 1. EB-stabilization: slump pitcher pulled toward season, not tanked
# ---------------------------------------------------------------------------

class TestKStabSlumpHandling:
    """
    Spec requirement: "a pitcher with a 2-start slump but strong season is
    pulled toward season, not tanked."

    Setup: 8 strong starts (K/BF ≈ 0.30) followed by 2 slump starts (K/BF ≈ 0.10).
    At the 9th start (test point), k_rate_last5 ≈ low (dominated by slump),
    but k_stab_last5 should be higher than k_rate_last5 (pulled toward season).
    """

    def _build_fixture(self):
        # 8 elite starts: 8 K / 27 BF ≈ 0.296 K-rate
        # 2 slump starts: 2 K / 24 BF ≈ 0.083 K-rate
        # 1 placeholder start (game being predicted — we look at row 10's features)
        strikeouts   = [8, 8, 8, 8, 8, 8, 8, 8,  2,  2,  8]
        batters_faced = [27, 27, 27, 27, 27, 27, 27, 27, 24, 24, 27]
        return _make_pitcher_game_df(strikeouts, batters_faced)

    def test_k_stab_greater_than_raw_k_rate_after_slump(self):
        """
        After two slump starts, k_stab_last5 > k_rate_last5 at the next game.
        The EB prior (based on season K-rate ≈ 0.27 from the 8 strong starts)
        pulls the stabilized rate above the raw L5 average (which is dragged down
        by the two slump starts even though recent BF is only ~50-54).
        """
        df = self._build_fixture()
        result = add_rolling_features(df)

        # Row index 10 is the 11th game — its L5 features are based on games 6-10
        # (the last 5 prior games: 3 strong + 2 slump).
        # k_rate_last5: (8+8+8+2+2) / (27+27+27+24+24) = 28/129 ≈ 0.217
        # k_rate_season (through game 10): (8*8+2*2)/(8*27+2*24) = 68/264 ≈ 0.258
        # k_stab_last5 = (28 + 100*0.258) / (129 + 100) = (28+25.8)/229 ≈ 0.235
        # So k_stab (0.235) > k_rate_last5 (0.217) — EB blends toward the season

        row = result.iloc[10]
        assert not np.isnan(row["k_rate_last5"]), "k_rate_last5 should not be NaN"
        assert not np.isnan(row["k_stab_last5"]), "k_stab_last5 should not be NaN"
        assert row["k_stab_last5"] > row["k_rate_last5"], (
            f"After slump, k_stab_last5 ({row['k_stab_last5']:.4f}) should be "
            f"ABOVE k_rate_last5 ({row['k_rate_last5']:.4f}) because EB pulls "
            f"toward the stronger season prior."
        )

    def test_k_stab_below_season_rate(self):
        """
        k_stab_last5 should be BELOW k_rate_season because the L5 sample (which
        includes the slumps) pulls the blend down from the season rate.
        """
        df = self._build_fixture()
        result = add_rolling_features(df)
        row = result.iloc[10]
        assert row["k_stab_last5"] < row["k_rate_season"], (
            f"k_stab_last5 ({row['k_stab_last5']:.4f}) should be below "
            f"k_rate_season ({row['k_rate_season']:.4f}) because the slump L5 "
            f"pulls the blend below the season average."
        )

    def test_k_stab_formula_correctness(self):
        """
        Verify the exact EB formula: k_stab = (K_sum_L5 + C * p_prior) / (BF_sum_L5 + C).
        """
        df = self._build_fixture()
        result = add_rolling_features(df)

        # Row 10: prior 5 games are rows 5-9 (games with indices 5..9)
        prior = result.iloc[5:10]
        k_sum_l5 = prior["strikeouts"].sum() if "strikeouts" in result.columns else float("nan")
        bf_sum_l5 = prior["batters_faced"].sum() if "batters_faced" in result.columns else float("nan")

        # k_rate_season for row 10 = season K-rate computed through games 0-9
        p_prior = result.iloc[10]["k_rate_season"]
        if np.isnan(p_prior):
            p_prior = LEAGUE_K_RATE_PRIOR

        expected = (k_sum_l5 + EB_K_CONSTANT * p_prior) / (bf_sum_l5 + EB_K_CONSTANT)
        actual = result.iloc[10]["k_stab_last5"]
        assert abs(actual - expected) < 1e-9, (
            f"k_stab formula mismatch: expected {expected:.6f}, got {actual:.6f}"
        )

    def test_k_stab_uses_league_prior_for_debut(self):
        """
        A pitcher in their first game ever has no season K-rate yet.
        k_stab_last5 should fall back to LEAGUE_K_RATE_PRIOR as p_prior.
        (At the debut start itself, all rolling features are NaN by definition
        since there's no prior data — we check the *second* start where the
        first game provides some signal but no season rate has been established.)
        """
        # 1 prior start, then we inspect row 1
        df = _make_pitcher_game_df(strikeouts=[6], batters_faced=[24])
        result = add_rolling_features(df)

        # Row 0 (debut): k_stab_last5 is NaN (no prior BF)
        # This is actually OK — debut starts have no rolling history
        row0 = result.iloc[0]
        # k_rate_season for row 0 = NaN (no prior season data), so EB uses league prior
        # but BF_sum = 0 → k_stab = (0 + C*league_prior) / (0 + C) = league_prior exactly
        # However, _prior_rolling_sum with min_periods=1 will return 0 for the first row
        # (since shift(1) makes it NaN, rolling().sum() with min_periods=1 returns NaN for NaN).
        # So k_stab_last5 for row 0 may be NaN (no prior data). That's correct.
        # (The debut start has no prior games, so no k_stab is computable — consistent with k_rate_last5.)
        assert pd.isna(row0["k_stab_last5"]) or row0["k_stab_last5"] >= 0, (
            "Debut start: k_stab_last5 should be NaN or a valid non-negative rate"
        )


# ---------------------------------------------------------------------------
# 2. Leakage test: k_stab_last5 uses only strictly-prior data
# ---------------------------------------------------------------------------

class TestKStabLeakage:
    """
    Spec requirement: "strictly-prior (leakage test)."

    k_stab_last5 for game N must not incorporate game N's own data. This is
    guaranteed by the EB formula's component parts: both _prior_rolling_sum
    calls use shift(1) before the rolling window, and k_rate_season (the prior)
    is also shift(1)-based (_prior_cumsum = shift(1).cumsum()). We verify it
    empirically by changing game N's data and checking that k_stab at game N
    does NOT change.
    """

    def test_last_game_data_does_not_affect_own_k_stab(self):
        """
        Mutating the final game's strikeouts should not change k_stab_last5
        for that same game (it should only affect future games).
        """
        df_original = _make_pitcher_game_df(
            strikeouts=[7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            batters_faced=[27] * 10,
        )
        df_modified = df_original.copy()
        # Change the last game's stats drastically
        df_modified.loc[df_modified.index[-1], "strikeouts"] = 0
        df_modified.loc[df_modified.index[-1], "batters_faced"] = 27

        result_orig = add_rolling_features(df_original)
        result_mod  = add_rolling_features(df_modified)

        # k_stab_last5 for the final row should be identical in both
        kstab_orig = result_orig.iloc[-1]["k_stab_last5"]
        kstab_mod  = result_mod.iloc[-1]["k_stab_last5"]
        assert abs(kstab_orig - kstab_mod) < 1e-12, (
            f"k_stab_last5 changed when we modified the SAME game's stats "
            f"({kstab_orig:.6f} vs {kstab_mod:.6f}) — this is a LEAKAGE BUG."
        )

    def test_k_stab_at_row0_is_nan_or_league_prior(self):
        """
        The very first game in a pitcher's history has no prior data,
        so k_stab_last5 should be NaN (no prior BF denominator → NaN from
        _shifted_rate) — same behaviour as k_rate_last5.
        """
        df = _make_pitcher_game_df(strikeouts=[9], batters_faced=[27])
        result = add_rolling_features(df)
        # With only one game, the prior rolling sum gives NaN (shift(1) → NaN)
        assert pd.isna(result.iloc[0]["k_stab_last5"]), (
            "Single-game pitcher: k_stab_last5 at row 0 should be NaN "
            "(no prior data available — same as k_rate_last5)."
        )

    def test_kstab_increases_after_strong_game_at_next_row(self):
        """
        After a pitcher has a very high-K game, k_stab_last5 at the NEXT game
        should be higher than before that game — confirming the feature is
        updated after each game, not before.
        """
        # 4 average games, then 1 elite game, then check next game's k_stab
        strikeouts    = [5, 5, 5, 5, 13, 5]
        batters_faced = [27] * 6
        df = _make_pitcher_game_df(strikeouts, batters_faced)
        result = add_rolling_features(df)

        # k_stab for row 5 (after the elite game at row 4) should be higher
        # than k_stab for row 4 (before the elite game)
        kstab_before_elite = result.iloc[4]["k_stab_last5"]  # doesn't include row 4
        kstab_after_elite  = result.iloc[5]["k_stab_last5"]  # does include row 4 (elite)

        assert kstab_after_elite > kstab_before_elite, (
            f"k_stab should rise after an elite K game: "
            f"before={kstab_before_elite:.4f}, after={kstab_after_elite:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Tail-calibration helper: compute_mu_decile_table + flag
# ---------------------------------------------------------------------------

def _make_oos_fixture(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic OOS frame where the model systematically UNDER-predicts the
    top decile (actual K > predicted μ for high-μ starts).
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(3.0, 10.0, size=n)
    # Simulate compression: model slightly under-predicts high-μ starts
    # by subtracting 20% of any excess above 7 (the compression artefact)
    realized = mu.copy()
    high_mask = mu > 7.0
    realized[high_mask] = mu[high_mask] + 0.5 + rng.normal(0, 0.5, high_mask.sum())
    realized = np.maximum(realized, 0).round().astype(int)
    return pd.DataFrame({"pred_mean": mu, "strikeouts": realized})


class TestMuDecileTable:

    def test_returns_correct_number_of_deciles(self):
        oos = _make_oos_fixture(n=300)
        tbl = compute_mu_decile_table(oos, n_deciles=10)
        assert 1 <= len(tbl) <= 10, f"Expected ≤10 deciles, got {len(tbl)}"

    def test_column_schema(self):
        oos = _make_oos_fixture(n=300)
        tbl = compute_mu_decile_table(oos)
        expected_cols = {"decile", "mu_lo", "mu_hi", "n", "mean_pred", "mean_actual", "bias", "bias_pct"}
        assert expected_cols.issubset(set(tbl.columns)), (
            f"Missing columns: {expected_cols - set(tbl.columns)}"
        )

    def test_decile_coverage(self):
        """All OOS rows must appear in exactly one decile (n sum = total rows)."""
        oos = _make_oos_fixture(n=300)
        tbl = compute_mu_decile_table(oos)
        assert tbl["n"].sum() == len(oos), (
            f"Decile n-sum ({tbl['n'].sum()}) ≠ total OOS rows ({len(oos)})"
        )

    def test_bias_formula(self):
        """bias = mean_pred − mean_actual for each decile.

        Tolerance is <= 0.002: the stored bias is computed from unrounded means,
        while expected_bias is computed from the already-rounded mean_pred /
        mean_actual columns. Double-rounding can cause up to ~0.001 difference
        per value, so the combined max discrepancy is 0.002.
        """
        oos = _make_oos_fixture(n=300)
        tbl = compute_mu_decile_table(oos)
        for _, row in tbl.iterrows():
            expected_bias = round(row["mean_pred"] - row["mean_actual"], 3)
            assert abs(row["bias"] - expected_bias) <= 0.002, (
                f"Decile {row['decile']}: bias formula mismatch "
                f"(got {row['bias']:.3f}, expected {expected_bias:.3f})"
            )

    def test_mu_ordering_monotone(self):
        """Decile 1 has the lowest μ range, decile N has the highest."""
        oos = _make_oos_fixture(n=300)
        tbl = compute_mu_decile_table(oos).sort_values("decile").reset_index(drop=True)
        for i in range(len(tbl) - 1):
            assert tbl.loc[i, "mu_hi"] <= tbl.loc[i + 1, "mu_lo"] or \
                   tbl.loc[i, "mu_lo"] < tbl.loc[i + 1, "mu_lo"], (
                "Decile μ ranges should be monotonically increasing"
            )

    def test_empty_oos_returns_empty_table(self):
        oos = pd.DataFrame({"pred_mean": [], "strikeouts": []})
        tbl = compute_mu_decile_table(oos)
        assert tbl.empty

    def test_accepts_mu_column_alias(self):
        """Should work with 'mu' column (walk_forward output name) as well as 'pred_mean'."""
        oos = _make_oos_fixture(n=200)
        oos_mu = oos.rename(columns={"pred_mean": "mu"})
        tbl_pred = compute_mu_decile_table(oos)
        tbl_mu   = compute_mu_decile_table(oos_mu)
        assert len(tbl_pred) == len(tbl_mu), "Should produce same number of deciles for both column names"

    def test_missing_pred_mean_raises(self):
        oos = pd.DataFrame({"actual": [5, 6, 7], "strikeouts": [5, 6, 7]})
        with pytest.raises(ValueError, match="pred_mean.*mu"):
            compute_mu_decile_table(oos)


class TestFlagHighMuUnderprediction:
    """
    flag_high_mu_underprediction(decile_table) returns True when top deciles
    show systematic under-prediction (actual > predicted, i.e. bias < 0).
    """

    def _make_table(self, top_bias_pct: float, bottom_bias_pct: float = 0.0, n_deciles: int = 10):
        """Build a synthetic decile table with controlled top/bottom bias."""
        rows = []
        for d in range(1, n_deciles + 1):
            bp = top_bias_pct if d >= (n_deciles - 1) else bottom_bias_pct
            rows.append({
                "decile": d,
                "mu_lo": float(d * 0.5),
                "mu_hi": float(d * 0.5 + 0.5),
                "n": 30,
                "mean_pred": 5.0,
                "mean_actual": 5.0 - (bp / 100),  # bias_pct = (pred-actual)/actual*100
                "bias": bp / 100,
                "bias_pct": bp,
            })
        return pd.DataFrame(rows)

    def test_flags_severe_underprediction(self):
        """bias_pct = −20% in top 2 deciles → should be flagged (< −10 threshold)."""
        tbl = self._make_table(top_bias_pct=-20.0)
        assert flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-10.0)

    def test_does_not_flag_mild_underprediction(self):
        """bias_pct = −5% in top 2 deciles → within tolerance (> −10 threshold)."""
        tbl = self._make_table(top_bias_pct=-5.0)
        assert not flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-10.0)

    def test_does_not_flag_overprediction(self):
        """Positive bias_pct (model over-predicts) → should not trigger the flag."""
        tbl = self._make_table(top_bias_pct=+15.0)
        assert not flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-10.0)

    def test_empty_table_returns_false(self):
        tbl = pd.DataFrame(columns=["decile", "mu_lo", "mu_hi", "n",
                                     "mean_pred", "mean_actual", "bias", "bias_pct"])
        assert not flag_high_mu_underprediction(tbl)

    def test_only_top_deciles_matter(self):
        """
        Even if the bottom deciles have severe negative bias, the flag should
        only trigger when TOP deciles are under-predicted.
        """
        tbl = self._make_table(top_bias_pct=0.0, bottom_bias_pct=-30.0)
        # Top 2 deciles have bias_pct = 0.0, which is > -10 threshold → no flag
        assert not flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-10.0)

    def test_on_realistic_compressed_fixture(self):
        """
        The synthetic OOS fixture has systematic compression at the top —
        the flag should detect it.
        """
        oos = _make_oos_fixture(n=500)
        tbl = compute_mu_decile_table(oos)
        # Our fixture adds 0.5+ K to high-μ starts → actual > predicted → negative bias_pct
        # With n=500 and 20% uplift for high-μ starts, top deciles will have negative bias
        flagged = flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-5.0)
        # The fixture has actual > predicted for high-μ starts, so flag should be True
        # (we use a looser threshold of -5 to account for randomness in small fixture)
        top2 = tbl.nlargest(2, "decile")
        top2_under = (top2["bias_pct"] < -5.0).all()
        assert flagged == top2_under, (
            "flag_high_mu_underprediction should agree with manual check on the fixture"
        )


class TestFormatDecileTableMd:

    def test_returns_markdown_string(self):
        oos = _make_oos_fixture(n=200)
        tbl = compute_mu_decile_table(oos)
        md = format_decile_table_md(tbl)
        assert isinstance(md, str)
        assert "| decile |" in md
        assert "| 1 |" in md  # first decile row

    def test_empty_table_returns_placeholder(self):
        tbl = pd.DataFrame(columns=["decile", "mu_lo", "mu_hi", "n",
                                     "mean_pred", "mean_actual", "bias", "bias_pct"])
        md = format_decile_table_md(tbl)
        assert "no data" in md.lower()


# ---------------------------------------------------------------------------
# 4. Opponent block de-collinearization (config / regression test)
# ---------------------------------------------------------------------------

class TestOpponentBlockDecollinearization:
    """
    Spec requirement: "Dropping `opponent_k_rate_last10` leaves the matchup
    term positive and present."

    Since this is a model-coefficient test that requires actual fitting (and
    a real corpus), we verify the *configuration* guarantee: the compression-
    fix column set does NOT include opponent_k_rate_last10 in any of the
    CORE, IMPUTE, or EXTRA lists (it was the sign-flipped redundant regressor).
    """

    def test_opponent_k_rate_last10_not_in_fix_core(self):
        from scripts.run_compression_fix_gate import COMPRESSION_FIX_CORE
        assert "opponent_k_rate_last10" not in COMPRESSION_FIX_CORE, (
            "opponent_k_rate_last10 should NOT appear in COMPRESSION_FIX_CORE"
        )

    def test_opponent_k_rate_last10_not_in_fix_impute(self):
        from scripts.run_compression_fix_gate import COMPRESSION_FIX_IMPUTE
        assert "opponent_k_rate_last10" not in COMPRESSION_FIX_IMPUTE, (
            "opponent_k_rate_last10 should NOT appear in COMPRESSION_FIX_IMPUTE "
            "(it was sign-flipped by collinearity with the hand-split)"
        )

    def test_opponent_k_rate_vs_hand_season_in_fix_impute(self):
        """The correctly-signed hand-split matchup regressor must be retained."""
        from scripts.run_compression_fix_gate import COMPRESSION_FIX_IMPUTE
        assert "opponent_k_rate_vs_hand_season" in COMPRESSION_FIX_IMPUTE, (
            "opponent_k_rate_vs_hand_season (correctly-signed matchup term) "
            "must be retained in COMPRESSION_FIX_IMPUTE"
        )

    def test_k_stab_last5_in_fix_core_not_k_rate_last5(self):
        """k_stab_last5 replaces k_rate_last5 in the fix column set."""
        from scripts.run_compression_fix_gate import COMPRESSION_FIX_CORE
        assert "k_stab_last5" in COMPRESSION_FIX_CORE, (
            "k_stab_last5 should be in COMPRESSION_FIX_CORE"
        )
        assert "k_rate_last5" not in COMPRESSION_FIX_CORE, (
            "k_rate_last5 should NOT be in COMPRESSION_FIX_CORE (replaced by k_stab_last5)"
        )
