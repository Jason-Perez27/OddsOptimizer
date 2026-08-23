"""
tests/test_compression_fix_v2.py

Tests for the compression-fix v2 spec (2026-06-30):
  1. Tail-calibration subset_mask: decile bias computed only on filtered subset
  2. EB tiered prior in rolling_features.add_rolling_features:
       (a) ace with season data → shrinks toward own season rate, not league
       (b) pitcher with no own-history → falls back to league prior on debut game
       (c) strictly-prior leakage guard (k_stab_last5 never sees same-game K)
  3. Iterative VIF prune terminates at max_vif < 10 on a collinear fixture
  4. flag_high_mu_underprediction: oos_df + subset_mask overload matches pre-built table
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oos_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic OOS frame with pred_mean, strikeouts, game_date."""
    rng = np.random.default_rng(seed)
    pred_mean = rng.uniform(3.0, 9.0, n)
    noise = rng.normal(0, 1.5, n)
    strikeouts = np.clip(pred_mean + noise, 0, 20).round().astype(float)
    years = np.where(rng.random(n) < 0.5, 2025, 2026)
    months = rng.integers(4, 10, n)
    days = rng.integers(1, 28, n)
    dates = pd.to_datetime({"year": years, "month": months, "day": days})
    return pd.DataFrame({"pred_mean": pred_mean, "strikeouts": strikeouts, "game_date": dates})


def _make_pitcher_game_df(
    pitchers: list,
    n_games_per: int,
    k_rates: dict,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Build a minimal game_df for rolling-features tests.
    Each pitcher gets n_games_per starts with K drawn from Binomial(20 BF, rate).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pitcher in pitchers:
        rate = k_rates.get(pitcher, 0.22)
        for g in range(n_games_per):
            bf = 20
            ks = int(rng.binomial(bf, min(rate, 0.99)))
            rows.append({
                "pitcher": pitcher,
                "game_date": pd.Timestamp(f"2025-04-{(g % 28) + 1:02d}"),
                "batters_faced": bf,
                "strikeouts": ks,
                "walks": int(rng.binomial(bf, 0.08)),
                "innings_pitched": 6.0,
                "pitch_count": 90,
                "whiff_rate": 0.12,
                "fastball_velo_avg": 93.0,
                "home_away": "home" if g % 2 == 0 else "away",
                "opponent_team": "OPP",
                "strikeouts_vs_LHB": ks // 2,
                "strikeouts_vs_RHB": ks - ks // 2,
                "batters_faced_vs_LHB": bf // 2,
                "batters_faced_vs_RHB": bf - bf // 2,
            })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)


# ===========================================================================
# 1. subset_mask: decile table restricted to 2026 rows
# ===========================================================================

class TestSubsetMask(unittest.TestCase):
    def test_mask_restricts_rows(self):
        """compute_mu_decile_table with subset_mask only uses masked rows."""
        from src.backtest.tail_calibration import compute_mu_decile_table

        oos = _make_oos_df(n=400)
        mask_2026 = pd.to_datetime(oos["game_date"]).dt.year == 2026
        n_2026 = int(mask_2026.sum())

        tbl_all = compute_mu_decile_table(oos)
        tbl_masked = compute_mu_decile_table(oos, subset_mask=mask_2026)

        self.assertEqual(
            int(tbl_masked["n"].sum()), n_2026,
            f"Masked decile table has {tbl_masked['n'].sum()} rows; expected {n_2026}",
        )
        self.assertGreater(int(tbl_all["n"].sum()), n_2026)

    def test_mask_mu_range_within_subset(self):
        """μ range in masked table stays within the μ range of 2026 rows."""
        from src.backtest.tail_calibration import compute_mu_decile_table

        oos = _make_oos_df(n=300)
        mask = pd.to_datetime(oos["game_date"]).dt.year == 2026
        tbl = compute_mu_decile_table(oos, subset_mask=mask)
        subset_mu = oos.loc[mask, "pred_mean"]

        if tbl.empty or subset_mu.empty:
            return
        self.assertGreaterEqual(tbl["mu_lo"].min(), subset_mu.min() - 1e-6)
        self.assertLessEqual(tbl["mu_hi"].max(), subset_mu.max() + 1e-6)

    def test_none_mask_equals_no_mask(self):
        """subset_mask=None behaves identically to omitting it."""
        from src.backtest.tail_calibration import compute_mu_decile_table

        oos = _make_oos_df(n=200)
        tbl_default = compute_mu_decile_table(oos)
        tbl_none = compute_mu_decile_table(oos, subset_mask=None)
        pd.testing.assert_frame_equal(tbl_default, tbl_none)

    def test_empty_mask_returns_empty_table(self):
        """All-False mask returns an empty DataFrame, not an error."""
        from src.backtest.tail_calibration import compute_mu_decile_table

        oos = _make_oos_df(n=100)
        empty_mask = pd.Series([False] * len(oos), index=oos.index)
        tbl = compute_mu_decile_table(oos, subset_mask=empty_mask)
        self.assertTrue(tbl.empty)


# ===========================================================================
# 2. EB tiered prior in rolling_features
# ===========================================================================

class TestEBTieredPrior(unittest.TestCase):

    def _build(self, game_df: pd.DataFrame) -> pd.DataFrame:
        from src.features.rolling_features import add_rolling_features
        return add_rolling_features(game_df)

    def test_ace_stays_above_league_prior(self):
        """
        An ace (K-rate 0.38) in mid-season: k_stab_last5 must stay above
        LEAGUE_K_RATE_PRIOR (0.22) once own-season history is established.
        """
        from src.features.rolling_features import LEAGUE_K_RATE_PRIOR

        df = _make_pitcher_game_df(["Ace"], n_games_per=12, k_rates={"Ace": 0.38})
        feat = self._build(df)
        ace = feat[feat["pitcher"] == "Ace"].iloc[4:]   # skip first 4 (thin sample)
        if ace.empty:
            return

        below_league = ace[ace["k_stab_last5"] <= LEAGUE_K_RATE_PRIOR]
        self.assertEqual(
            len(below_league), 0,
            f"Ace k_stab_last5 fell to/below league ({LEAGUE_K_RATE_PRIOR}) on "
            f"{len(below_league)} rows — tiered prior is capping the ace:\n"
            f"{below_league[['game_date','k_stab_last5','k_rate_season']].to_string()}",
        )

    def test_ace_closer_to_own_season_than_league(self):
        """
        Mid-season ace: mean k_stab_last5 must be closer to own season rate
        than to the league prior (i.e. tiered prior is working).
        """
        from src.features.rolling_features import LEAGUE_K_RATE_PRIOR

        df = _make_pitcher_game_df(["Ace"], n_games_per=10, k_rates={"Ace": 0.38})
        feat = self._build(df)
        mid = feat[feat["pitcher"] == "Ace"].iloc[5:]
        stab = mid["k_stab_last5"].dropna()
        season = mid["k_rate_season"].dropna()

        if stab.empty or season.empty:
            return  # not enough data for mid-season test

        stab_mean = float(stab.mean())
        season_mean = float(season.mean())
        dist_season = abs(stab_mean - season_mean)
        dist_league = abs(stab_mean - LEAGUE_K_RATE_PRIOR)
        self.assertLess(
            dist_season, dist_league,
            f"k_stab={stab_mean:.3f} should be closer to own season ({season_mean:.3f}) "
            f"than to league ({LEAGUE_K_RATE_PRIOR:.3f}). "
            f"dist_season={dist_season:.4f} vs dist_league={dist_league:.4f}",
        )

    def test_debut_game_equals_league_prior(self):
        """
        A pitcher's first game has no prior history. k_stab_last5 must equal
        LEAGUE_K_RATE_PRIOR  (=(0 + C*prior)/(0 + C) = prior).
        """
        from src.features.rolling_features import LEAGUE_K_RATE_PRIOR

        df = _make_pitcher_game_df(["Rookie"], n_games_per=5, k_rates={"Rookie": 0.22})
        feat = self._build(df)
        debut = feat[feat["pitcher"] == "Rookie"].iloc[0]
        actual = float(debut["k_stab_last5"])
        self.assertAlmostEqual(
            actual, LEAGUE_K_RATE_PRIOR, places=2,
            msg=f"Debut k_stab_last5={actual:.4f} should equal LEAGUE_K_RATE_PRIOR={LEAGUE_K_RATE_PRIOR:.4f}",
        )

    def test_no_same_game_leakage(self):
        """
        k_stab_last5 for game 0 must not change when game 0's strikeouts are mutated.
        Downstream games should change (confirming the lookback actually reads prior data).
        """
        from src.features.rolling_features import add_rolling_features

        df = _make_pitcher_game_df(["P1"], n_games_per=8, k_rates={"P1": 0.30})

        feat_orig = add_rolling_features(df.copy())
        stab_orig = feat_orig[feat_orig["pitcher"] == "P1"]["k_stab_last5"].values.copy()

        mutated = df.copy()
        mutated["strikeouts"] = mutated["strikeouts"] * 2
        feat_mut = add_rolling_features(mutated)
        stab_mut = feat_mut[feat_mut["pitcher"] == "P1"]["k_stab_last5"].values

        # Game 0: zero prior data → same result regardless of game-0 K
        # Both should be equal (and finite since we fillna(0) on rolling sums).
        s0_orig = float(stab_orig[0])
        s0_mut  = float(stab_mut[0])
        import math
        if math.isnan(s0_orig) and math.isnan(s0_mut):
            pass  # both NaN: leakage-free
        else:
            self.assertAlmostEqual(
                s0_orig, s0_mut, places=9,
                msg="First-game k_stab must be independent of same-game strikeout count",
            )
        # Games 2+: doubling K should shift k_stab (the rolling lookback is working)
        if len(stab_orig) > 2:
            max_diff = float(np.abs(stab_orig[2:] - stab_mut[2:]).max())
            self.assertGreater(
                max_diff, 1e-6,
                "Mutating strikeouts had no downstream effect — "
                "shift-based lookback may be broken",
            )


# ===========================================================================
# 3. Iterative VIF prune
# ===========================================================================

class TestIterativeVIFPrune(unittest.TestCase):

    def test_collinear_pair_pruned_below_10(self):
        """Near-perfectly-correlated pair gets pruned until max VIF < 10."""
        from src.backtest.vif_prune import compute_vif, iterative_vif_prune

        rng = np.random.default_rng(7)
        x1 = rng.normal(0, 1, 300)
        x2 = x1 + rng.normal(0, 0.01, 300)  # near-perfect collinearity
        x3 = rng.normal(0, 1, 300)

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        vif_before = compute_vif(df, ["x1", "x2", "x3"])
        self.assertGreater(
            float(vif_before["vif"].max()), 10.0,
            "Test fixture must start with max VIF > 10",
        )

        survivors, prune_log = iterative_vif_prune(df, ["x1", "x2", "x3"], max_vif=10.0)
        vif_after = compute_vif(df, survivors)
        max_after = float(vif_after["vif"].max())

        self.assertLess(
            max_after, 10.0,
            f"After prune, max VIF={max_after:.2f} should be < 10. Survivors: {survivors}",
        )
        self.assertIn("x3", survivors, "Independent column x3 must survive VIF pruning")
        self.assertGreaterEqual(len(survivors), 1)

    def test_prune_log_records_each_drop(self):
        """Each prune step records the dropped column and before-VIF."""
        from src.backtest.vif_prune import iterative_vif_prune

        rng = np.random.default_rng(11)
        x1 = rng.normal(0, 1, 200)
        x2 = x1 * 1.5 + rng.normal(0, 0.05, 200)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        _, prune_log = iterative_vif_prune(df, ["x1", "x2"], max_vif=10.0)

        self.assertGreaterEqual(len(prune_log), 1)
        for entry in prune_log:
            self.assertIn("dropped", entry, f"prune_log entry missing 'dropped': {entry}")
            self.assertIn("dropped_vif", entry)
            self.assertGreater(entry["dropped_vif"], 10.0)

    def test_no_prune_when_already_below_threshold(self):
        """If all VIF < 10 from the start, no columns are dropped."""
        from src.backtest.vif_prune import iterative_vif_prune

        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
            "c": rng.normal(0, 1, 100),
        })
        survivors, prune_log = iterative_vif_prune(df, ["a", "b", "c"], max_vif=10.0)
        self.assertEqual(set(survivors), {"a", "b", "c"})
        self.assertEqual(len(prune_log), 0)

    def test_near_duplicate_vif_is_very_high(self):
        """compute_vif returns extreme VIF for a near-duplicate column pair."""
        from src.backtest.vif_prune import compute_vif

        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 500)
        df = pd.DataFrame({"x": x, "x_dup": x + rng.normal(0, 1e-6, 500)})
        vif = compute_vif(df, ["x", "x_dup"])
        self.assertGreater(float(vif["vif"].max()), 1000.0)

    def test_single_column_is_noop(self):
        """Single-column input returns unchanged; no prune steps run."""
        from src.backtest.vif_prune import iterative_vif_prune

        rng = np.random.default_rng(5)
        df = pd.DataFrame({"only": rng.normal(0, 1, 100)})
        survivors, prune_log = iterative_vif_prune(df, ["only"], max_vif=10.0, min_features=1)
        self.assertEqual(survivors, ["only"])
        self.assertEqual(len(prune_log), 0)


# ===========================================================================
# 4. flag_high_mu_underprediction: oos_df + subset_mask overload
# ===========================================================================

class TestFlagHighMuOverload(unittest.TestCase):

    def test_overload_matches_prebuilt_table(self):
        """
        flag_high_mu_underprediction(oos_df=..., subset_mask=...) must give
        the same result as computing the decile table externally and passing it.
        """
        from src.backtest.tail_calibration import (
            compute_mu_decile_table,
            flag_high_mu_underprediction,
        )

        oos = _make_oos_df(n=400)
        mask = pd.to_datetime(oos["game_date"]).dt.year == 2026
        tbl = compute_mu_decile_table(oos, subset_mask=mask)

        result_prebuilt = flag_high_mu_underprediction(tbl, top_n_deciles=2, min_bias_pct=-10.0)
        result_overload = flag_high_mu_underprediction(
            decile_table=pd.DataFrame(),  # ignored when oos_df is present
            top_n_deciles=2,
            min_bias_pct=-10.0,
            oos_df=oos,
            subset_mask=mask,
        )
        self.assertEqual(result_prebuilt, result_overload)


if __name__ == "__main__":
    unittest.main()
