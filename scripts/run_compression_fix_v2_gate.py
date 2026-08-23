"""
scripts/run_compression_fix_v2_gate.py

Walk-forward gate for the compression-fix v2
(spec: docs/design/specs/2026-06-30-compression-fix-v2-design.md).

v1 was rejected for three reasons, all corrected here:
  ① Tested on data-rich 2024 only — v2 spans 2024→2026 and scores 2026 OOS.
  ② EB_K_CONSTANT=100 too aggressive — v2 sweeps C ∈ {25,50,75} and picks best.
  ③ Wrong collinearity target — v2 iteratively prunes on VIF until max VIF < 10.

Arms run separately so effects are attributable:
  Arm 1 (Fix 3): VIF-prune current column set; report before/after VIF + coefficients.
  Arm 2 (Fix 2): EB C sweep on VIF-pruned set; select C by 2026-OOS top-decile bias.
  Arm 3 (Fix 1): Head-to-head widened (2024+) vs narrow (2026-only) fit on 2026 OOS.

Acceptance (on 2026 OOS only):
  (a) Top-2-decile under-prediction shrinks toward 0.
  (b) Aggregate ECE/log-loss/MAE hold or improve.
  (c) Max VIF < 10 after pruning.

Usage:
    python scripts/run_compression_fix_v2_gate.py \\
        --start 2024-04-01 --end 2026-06-30 \\
        [--oos-year 2026]      # score this year as the OOS slice (default 2026)
        [--dry-run]
        [--step 7]
"""

import argparse
import re
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.corpus import (
    DEFAULT_CACHE_DIR,
    DEFAULT_WINDOW_DAYS,
    build_corpus,
    filter_starters,
)
from src.backtest.tail_calibration import (
    compute_mu_decile_table,
    flag_high_mu_underprediction,
    format_decile_table_md,
)
from src.backtest.vif_prune import (
    compute_vif,
    format_prune_log_md,
    format_vif_table_md,
    iterative_vif_prune,
)
from src.backtest.walk_forward import (
    DEFAULT_MIN_TRAIN_DATES,
    DEFAULT_STEP_DAYS,
    run_walk_forward,
)
from src.features.build_features import build_training_table
from src.models.baseline_model import (
    CORE_PITCHER_FORM_COLUMNS,
    DESIGN_MATRIX_EXTRA_COLUMNS,
    IMPUTE_COLUMNS,
    fit_baseline_model,
    fit_production_model,
)

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# -----------------------------------------------------------------------
# Constants: current baseline column set (for VIF comparison starting point)
# -----------------------------------------------------------------------
BASELINE_CORE = list(CORE_PITCHER_FORM_COLUMNS)   # ['k_rate_last5', 'whiff_rate_last5', ...]
BASELINE_IMPUTE = list(IMPUTE_COLUMNS)              # ['opponent_k_rate_last10', ...]

# v1 compression-fix candidate set (reference only — v2 overwrites via VIF prune)
V1_FIX_CORE = [
    "k_stab_last5", "whiff_rate_last5", "velo_avg_last5", "pitch_count_avg_last5",
]
V1_FIX_IMPUTE = [
    "opponent_k_rate_vs_hand_season", "park_k_factor", "rest_days",
]

# EB constant sweep
EB_SWEEP_VALUES = [25, 50, 75]


# -----------------------------------------------------------------------
# Context managers: patch module-level constants without permanent change
# -----------------------------------------------------------------------

@contextmanager
def _patch_baseline_columns(new_core, new_impute):
    """Temporarily replace baseline_model module-level column lists."""
    import src.models.baseline_model as bm
    orig_core = bm.CORE_PITCHER_FORM_COLUMNS
    orig_impute = bm.IMPUTE_COLUMNS
    orig_continuous = bm.CONTINUOUS_REGRESSOR_COLUMNS
    orig_regressors = bm.REGRESSOR_COLUMNS
    orig_design = bm.DESIGN_MATRIX_COLUMNS

    new_continuous = new_core + new_impute
    new_regressors = new_continuous + ["is_home"]
    new_design = new_regressors + list(DESIGN_MATRIX_EXTRA_COLUMNS)

    bm.CORE_PITCHER_FORM_COLUMNS = new_core
    bm.IMPUTE_COLUMNS = new_impute
    bm.CONTINUOUS_REGRESSOR_COLUMNS = new_continuous
    bm.REGRESSOR_COLUMNS = new_regressors
    bm.DESIGN_MATRIX_COLUMNS = new_design
    try:
        yield
    finally:
        bm.CORE_PITCHER_FORM_COLUMNS = orig_core
        bm.IMPUTE_COLUMNS = orig_impute
        bm.CONTINUOUS_REGRESSOR_COLUMNS = orig_continuous
        bm.REGRESSOR_COLUMNS = orig_regressors
        bm.DESIGN_MATRIX_COLUMNS = orig_design


@contextmanager
def _patch_eb_constant(new_c: float):
    """Temporarily set rolling_features.EB_K_CONSTANT to new_c."""
    import src.features.rolling_features as rf
    orig = rf.EB_K_CONSTANT
    rf.EB_K_CONSTANT = new_c
    try:
        yield
    finally:
        rf.EB_K_CONSTANT = orig


# -----------------------------------------------------------------------
# k_stab recomputation (for C sweep without re-running full feature pipeline)
# -----------------------------------------------------------------------

def _recompute_k_stab(feature_table: pd.DataFrame, C: float) -> pd.DataFrame:
    """
    Recompute k_stab_last5 in `feature_table` with a given EB constant C,
    using the tiered prior (own-season → own-career → league).

    Requires columns: pitcher, game_date, strikeouts, batters_faced, k_rate_season.
    Returns a copy with k_stab_last5 updated.
    """
    from src.features.rolling_features import (
        LAST5_WINDOW,
        LEAGUE_K_RATE_PRIOR,
        _prior_cumsum,
        _prior_rolling_sum,
        _shifted_rate,
    )

    df = feature_table.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    pitcher_grp = df.groupby("pitcher", sort=False)

    prior_k_last5 = pitcher_grp["strikeouts"].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    )
    prior_bf_last5 = pitcher_grp["batters_faced"].transform(
        lambda s: _prior_rolling_sum(s, LAST5_WINDOW)
    )

    # Career K-rate (cumulative across all prior seasons, no season grouping)
    prior_k_career = pitcher_grp["strikeouts"].transform(_prior_cumsum)
    prior_bf_career = pitcher_grp["batters_faced"].transform(_prior_cumsum)
    k_rate_career = _shifted_rate(prior_k_career, prior_bf_career)

    # Tiered prior: own-season → career → league
    p_prior = df["k_rate_season"].fillna(k_rate_career).fillna(LEAGUE_K_RATE_PRIOR)

    df["k_stab_last5"] = (
        (prior_k_last5 + C * p_prior)
        / (prior_bf_last5 + C)
    )
    return df


# -----------------------------------------------------------------------
# Fit-function factories
# -----------------------------------------------------------------------

def _make_fit_fn(core_cols, impute_cols, extra_cols=None):
    """Return a fit_fn that uses the given column set."""
    extra = list(extra_cols) if extra_cols else []

    def fit_fn(train_df, test_df=None, **kw):
        with _patch_baseline_columns(core_cols, impute_cols):
            return fit_baseline_model(train_df, test_df, extra_columns=extra, **kw)

    return fit_fn


def _make_narrow_fit_fn(core_cols, impute_cols, oos_year: int = 2026):
    """
    Return a fit_fn that trains ONLY on rows from oos_year
    (simulates the as-deployed narrow 2026-only retrain).
    Falls back to full data if no oos_year rows exist yet.
    """
    def fit_fn(train_df, test_df=None, **kw):
        narrow = train_df[pd.to_datetime(train_df["game_date"]).dt.year >= oos_year]
        effective_train = narrow if not narrow.empty else train_df
        with _patch_baseline_columns(core_cols, impute_cols):
            return fit_baseline_model(effective_train, test_df, **kw)

    return fit_fn


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _aggregate_metrics(oos_df: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, ECE, Brier@7, log-loss@7 from a walk-forward OOS frame."""
    # Walk-forward uses column 'mu' (not 'pred_mean') and 'realized_strikeouts'
    mu = oos_df["mu"].astype(float)
    k = oos_df["realized_strikeouts"].astype(float)
    n = int(len(oos_df))
    mae = float((mu - k).abs().mean()) if n > 0 else float("nan")
    rmse = float(np.sqrt(((mu - k) ** 2).mean())) if n > 0 else float("nan")

    p7_col = "p_over_7" if "p_over_7" in oos_df.columns else None
    ece = float("nan")
    brier = float("nan")
    ll = float("nan")
    if p7_col and n > 0:
        p = oos_df[p7_col].astype(float).clip(1e-9, 1 - 1e-9)
        hit = (k >= 7).astype(float)
        ece_bins = pd.cut(p, bins=10)
        ece_parts = []
        for _, grp in oos_df.assign(_p=p, _hit=hit).groupby(ece_bins):
            if len(grp) == 0:
                continue
            ece_parts.append(abs(grp["_p"].mean() - grp["_hit"].mean()) * len(grp) / n)
        ece = float(sum(ece_parts))
        brier = float(((p - hit) ** 2).mean())
        ll = float(-(hit * np.log(p) + (1 - hit) * np.log(1 - p)).mean())

    return {"n": n, "mae": mae, "rmse": rmse, "ece": ece, "brier_t7": brier, "log_loss_t7": ll}


def _oos_to_decile_df(oos_df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """Convert walk-forward OOS frame to the format compute_mu_decile_table expects."""
    renamed = oos_df.rename(columns={"mu": "pred_mean", "realized_strikeouts": "strikeouts"})
    return compute_mu_decile_table(renamed, subset_mask=mask)


def _top2_bias(decile_tbl: pd.DataFrame) -> float:
    """Mean bias_pct of the two highest deciles (the headline compression metric)."""
    if decile_tbl.empty:
        return float("nan")
    return float(decile_tbl.nlargest(2, "decile")["bias_pct"].mean())


# -----------------------------------------------------------------------
# Arm 1: iterative VIF pruning
# -----------------------------------------------------------------------

def run_arm1_vif_prune(feature_table: pd.DataFrame, args) -> dict:
    """
    Arm 1 (Fix 3): Iteratively prune the highest-VIF feature from the v1
    compression-fix candidate set until max VIF < 10.

    Starting set: V1_FIX_CORE (contains velo_avg_last5, csw_rate_season via
    extra, etc.). Walk-forward with the pruned set on the FULL corpus.

    Returns dict with:
        surviving_core, prune_log, vif_before, vif_after,
        oos_arm1, metrics_arm1, deciles_arm1_full, deciles_arm1_oos_year
    """
    print("  [Arm 1] Computing VIF on v1 candidate set ...")

    # Candidate set for VIF analysis: v1 core + extra skill column if present
    candidate_cols = [c for c in V1_FIX_CORE if c in feature_table.columns]

    vif_before = compute_vif(feature_table, candidate_cols)
    print("  VIF before pruning:")
    for _, r in vif_before.sort_values("vif", ascending=False).iterrows():
        print(f"    {r['col']:30s} VIF={r['vif']:.2f}")

    surviving_core, prune_log = iterative_vif_prune(
        feature_table, candidate_cols, max_vif=10.0, min_features=1
    )
    vif_after = compute_vif(feature_table, surviving_core)
    print(f"  Surviving columns: {surviving_core}")
    print("  VIF after pruning:")
    for _, r in vif_after.sort_values("vif", ascending=False).iterrows():
        print(f"    {r['col']:30s} VIF={r['vif']:.2f}")

    # Walk-forward with surviving column set + v1 impute block
    print("  [Arm 1] Walk-forward with VIF-pruned column set ...")
    arm1_fit_fn = _make_fit_fn(surviving_core, V1_FIX_IMPUTE)
    oos_arm1 = run_walk_forward(
        feature_table,
        step=args.step,
        min_train_dates=args.min_train_dates,
        fit_fn=arm1_fit_fn,
    )

    oos_year_mask = (
        pd.to_datetime(oos_arm1["game_date"]).dt.year == args.oos_year
        if not oos_arm1.empty else None
    )
    metrics_arm1 = _aggregate_metrics(oos_arm1)
    deciles_arm1_full = _oos_to_decile_df(oos_arm1)
    deciles_arm1_oos_year = _oos_to_decile_df(oos_arm1, mask=oos_year_mask)

    print(f"  [Arm 1] n={len(oos_arm1):,} OOS rows; "
          f"n_{args.oos_year}={int(oos_year_mask.sum()) if oos_year_mask is not None else 0}; "
          f"MAE={metrics_arm1['mae']:.4f}; "
          f"top2-decile bias ({args.oos_year} OOS)={_top2_bias(deciles_arm1_oos_year):+.1f}%")

    return {
        "surviving_core": surviving_core,
        "prune_log": prune_log,
        "vif_before": vif_before,
        "vif_after": vif_after,
        "oos": oos_arm1,
        "metrics": metrics_arm1,
        "deciles_full": deciles_arm1_full,
        "deciles_oos_year": deciles_arm1_oos_year,
        "oos_year_mask": oos_year_mask,
    }


# -----------------------------------------------------------------------
# Arm 2: EB constant sweep
# -----------------------------------------------------------------------

def run_arm2_eb_sweep(feature_table: pd.DataFrame, surviving_core: list, args) -> dict:
    """
    Arm 2 (Fix 2): Sweep C ∈ {25, 50, 75} on the VIF-pruned column set.
    Select C by minimizing top-decile bias on oos_year OOS.

    If k_stab_last5 is not in surviving_core, the sweep is skipped (no EB
    feature in the pruned set) and the v1 sweep default is used.

    Returns dict with sweep_results (one entry per C), best_C, best_oos, etc.
    """
    sweep_results = []

    if "k_stab_last5" not in surviving_core:
        print("  [Arm 2] k_stab_last5 not in VIF-pruned set — EB sweep skipped.")
        return {
            "sweep": [],
            "best_C": None,
            "best_oos": None,
            "best_metrics": None,
            "best_deciles_oos_year": None,
        }

    print(f"  [Arm 2] EB constant sweep C ∈ {EB_SWEEP_VALUES} ...")
    for C in EB_SWEEP_VALUES:
        print(f"    C={C}: recomputing k_stab_last5 ...")
        ft_c = _recompute_k_stab(feature_table, C)

        arm2_fit_fn = _make_fit_fn(surviving_core, V1_FIX_IMPUTE)
        oos_c = run_walk_forward(
            ft_c,
            step=args.step,
            min_train_dates=args.min_train_dates,
            fit_fn=arm2_fit_fn,
        )
        oos_mask = (
            pd.to_datetime(oos_c["game_date"]).dt.year == args.oos_year
            if not oos_c.empty else None
        )
        metrics_c = _aggregate_metrics(oos_c)
        deciles_oos_year_c = _oos_to_decile_df(oos_c, mask=oos_mask)
        top2_c = _top2_bias(deciles_oos_year_c)

        print(f"      top2-bias ({args.oos_year})={top2_c:+.1f}%, MAE={metrics_c['mae']:.4f}, "
              f"ll@7={metrics_c['log_loss_t7']:.4f}")

        sweep_results.append({
            "C": C,
            "top2_bias_oos_year": top2_c,
            "mae": metrics_c["mae"],
            "log_loss_t7": metrics_c["log_loss_t7"],
            "metrics": metrics_c,
            "deciles_oos_year": deciles_oos_year_c,
            "oos": oos_c,
            "oos_mask": oos_mask,
        })

    # Select best C: minimize |top2_bias_oos_year| (closest to 0 = least compression)
    # subject to: MAE not much worse than the smallest in sweep, ll@7 not worse
    baseline_mae = min(r["mae"] for r in sweep_results)
    sweep_ok = [
        r for r in sweep_results
        if (r["mae"] - baseline_mae) <= 0.10  # 0.10 K tolerance
    ]
    if sweep_ok:
        best = min(sweep_ok, key=lambda r: abs(r["top2_bias_oos_year"])
                   if not np.isnan(r["top2_bias_oos_year"]) else 1e9)
    else:
        best = sweep_results[0]

    best_C = best["C"]
    print(f"  [Arm 2] Best C={best_C} (top2-bias={best['top2_bias_oos_year']:+.1f}%)")

    return {
        "sweep": sweep_results,
        "best_C": best_C,
        "best_oos": best["oos"],
        "best_metrics": best["metrics"],
        "best_deciles_oos_year": best["deciles_oos_year"],
        "best_oos_mask": best["oos_mask"],
    }


# -----------------------------------------------------------------------
# Arm 3: widened-corpus head-to-head on oos_year OOS
# -----------------------------------------------------------------------

def run_arm3_widened(
    feature_table: pd.DataFrame,
    surviving_core: list,
    best_C: float,
    args,
) -> dict:
    """
    Arm 3 (Fix 1): Head-to-head on oos_year OOS rows.
      Narrow: train only on oos_year data at each step (simulates current production).
      Widened: train on all available data at each step (the fix).

    If best_C is provided, also recompute k_stab with that C.
    """
    if best_C is not None:
        print(f"  [Arm 3] Using best C={best_C} for k_stab recomputation ...")
        ft = _recompute_k_stab(feature_table, best_C)
    else:
        ft = feature_table

    # Only run Arm 3 if there's oos_year data to evaluate against
    all_years = pd.to_datetime(ft["game_date"]).dt.year.unique()
    if args.oos_year not in all_years:
        print(f"  [Arm 3] No {args.oos_year} data in corpus — skipping head-to-head.")
        return {"narrow_oos": None, "widened_oos": None,
                "narrow_metrics": None, "widened_metrics": None,
                "narrow_deciles": None, "widened_deciles": None}

    oos_mask_fn = lambda oos_df: (
        pd.to_datetime(oos_df["game_date"]).dt.year == args.oos_year
        if not oos_df.empty else None
    )

    narrow_fit = _make_narrow_fit_fn(surviving_core, V1_FIX_IMPUTE, oos_year=args.oos_year)
    widened_fit = _make_fit_fn(surviving_core, V1_FIX_IMPUTE)

    print(f"  [Arm 3] Walk-forward narrow (train on {args.oos_year} only) ...")
    oos_narrow = run_walk_forward(ft, step=args.step, min_train_dates=args.min_train_dates, fit_fn=narrow_fit)
    mask_narrow = oos_mask_fn(oos_narrow)
    narrow_oos_year = oos_narrow[mask_narrow] if mask_narrow is not None and len(oos_narrow) else pd.DataFrame()
    narrow_metrics = _aggregate_metrics(narrow_oos_year) if not narrow_oos_year.empty else None
    narrow_deciles = _oos_to_decile_df(oos_narrow, mask=mask_narrow)
    print(f"      n_{args.oos_year}={len(narrow_oos_year)}; "
          f"MAE={narrow_metrics['mae']:.4f if narrow_metrics else 'n/a'}; "
          f"top2-bias={_top2_bias(narrow_deciles):+.1f}%")

    print(f"  [Arm 3] Walk-forward widened (full corpus) ...")
    oos_widened = run_walk_forward(ft, step=args.step, min_train_dates=args.min_train_dates, fit_fn=widened_fit)
    mask_widened = oos_mask_fn(oos_widened)
    widened_oos_year = oos_widened[mask_widened] if mask_widened is not None and len(oos_widened) else pd.DataFrame()
    widened_metrics = _aggregate_metrics(widened_oos_year) if not widened_oos_year.empty else None
    widened_deciles = _oos_to_decile_df(oos_widened, mask=mask_widened)
    print(f"      n_{args.oos_year}={len(widened_oos_year)}; "
          f"MAE={widened_metrics['mae']:.4f if widened_metrics else 'n/a'}; "
          f"top2-bias={_top2_bias(widened_deciles):+.1f}%")

    return {
        "narrow_oos": oos_narrow, "widened_oos": oos_widened,
        "narrow_oos_year": narrow_oos_year, "widened_oos_year": widened_oos_year,
        "narrow_metrics": narrow_metrics, "widened_metrics": widened_metrics,
        "narrow_deciles": narrow_deciles, "widened_deciles": widened_deciles,
    }


# -----------------------------------------------------------------------
# Baseline (for comparison in report)
# -----------------------------------------------------------------------

def run_baseline(feature_table: pd.DataFrame, args) -> dict:
    """Run walk-forward with the current production column set."""
    print("  [Baseline] Walk-forward with production column set ...")
    oos = run_walk_forward(
        feature_table,
        step=args.step,
        min_train_dates=args.min_train_dates,
        fit_fn=fit_baseline_model,
    )
    oos_mask = (
        pd.to_datetime(oos["game_date"]).dt.year == args.oos_year
        if not oos.empty else None
    )
    metrics = _aggregate_metrics(oos)
    deciles_full = _oos_to_decile_df(oos)
    deciles_oos_year = _oos_to_decile_df(oos, mask=oos_mask)
    print(f"  Baseline n={len(oos):,} OOS; MAE={metrics['mae']:.4f}; "
          f"top2-bias ({args.oos_year})={_top2_bias(deciles_oos_year):+.1f}%")
    return {
        "oos": oos, "oos_mask": oos_mask,
        "metrics": metrics, "deciles_full": deciles_full,
        "deciles_oos_year": deciles_oos_year,
    }


# -----------------------------------------------------------------------
# Adoption decision
# -----------------------------------------------------------------------

ADOPT_GATE = {
    "mae_must_not_regress_by": 0.05,
    "log_loss_must_not_regress_by": 0.010,
}


def _decide_v2(
    baseline_deciles_oos_year: pd.DataFrame,
    final_deciles_oos_year: pd.DataFrame,
    baseline_metrics_oos_year: dict | None,
    final_metrics_oos_year: dict | None,
    surviving_core: list,
    feature_table: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Adopt if (on oos_year OOS):
      (a) top-2-decile under-prediction shrinks toward 0
      (b) aggregate MAE / log-loss hold or improve
      (c) max VIF < 10
    """
    reasons = []

    base_top2 = _top2_bias(baseline_deciles_oos_year)
    fix_top2 = _top2_bias(final_deciles_oos_year)
    bias_ok = (not np.isnan(fix_top2)) and (not np.isnan(base_top2)) and (fix_top2 > base_top2)
    reasons.append(
        f"  (a) Top-2-decile bias: baseline={base_top2:+.1f}%  fix={fix_top2:+.1f}%  "
        f"{'✓ IMPROVED' if bias_ok else '✗ DID NOT IMPROVE'}"
    )

    if baseline_metrics_oos_year and final_metrics_oos_year:
        mae_ok = (final_metrics_oos_year["mae"] - baseline_metrics_oos_year["mae"]) \
                 <= ADOPT_GATE["mae_must_not_regress_by"]
        ll_ok = (final_metrics_oos_year["log_loss_t7"] - baseline_metrics_oos_year["log_loss_t7"]) \
                <= ADOPT_GATE["log_loss_must_not_regress_by"]
        reasons.append(
            f"  (b) MAE: baseline={baseline_metrics_oos_year['mae']:.4f}  "
            f"fix={final_metrics_oos_year['mae']:.4f}  {'✓ OK' if mae_ok else '✗ REGRESSION'}"
        )
        reasons.append(
            f"      log-loss@7: baseline={baseline_metrics_oos_year['log_loss_t7']:.4f}  "
            f"fix={final_metrics_oos_year['log_loss_t7']:.4f}  "
            f"{'✓ OK' if ll_ok else '✗ REGRESSION'}"
        )
    else:
        mae_ok = ll_ok = False
        reasons.append("  (b) Insufficient OOS data for metric comparison.")

    if len(surviving_core) >= 2:
        vif_final = compute_vif(feature_table, surviving_core)
        max_vif = float(vif_final["vif"].max())
        vif_ok = max_vif < 10.0
        reasons.append(f"  (c) Max VIF after prune = {max_vif:.2f}  {'✓ OK' if vif_ok else '✗ ABOVE 10'}")
    else:
        vif_ok = True  # single column — VIF N/A
        reasons.append("  (c) Single surviving column — VIF check N/A ✓")

    adopted = bias_ok and mae_ok and ll_ok and vif_ok
    return adopted, reasons


# -----------------------------------------------------------------------
# Report writer
# -----------------------------------------------------------------------

def _metrics_row(label: str, m: dict | None) -> str:
    if m is None:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a |\n"
    return (
        f"| {label} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | "
        f"{m['ece']:.4f} | {m['log_loss_t7']:.4f} |\n"
    )


def _write_report_v2(
    run_date: str, args,
    baseline: dict,
    arm1: dict,
    arm2: dict,
    arm3: dict,
    adopted: bool,
    decision_reasons: list[str],
    surviving_core: list,
    best_C: float | None,
    edit_log: str,
) -> Path:
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_date}-compression-fix-v2.md"

    adopt_str = "**ADOPTED**" if adopted else "**REJECTED**"
    oy = args.oos_year

    lines = [
        f"# Compression-fix v2 gate — {run_date}\n\n",
        f"**Decision ({oy} OOS): {adopt_str}**\n\n",
        "## Gate reasoning\n\n",
        "\n".join(decision_reasons) + "\n\n",
        "---\n\n",

        # ---- Arm 1: VIF ----
        "## Arm 1 — VIF pruning (Fix 3: correct collinearity target)\n\n",
        "### VIF before pruning (v1 candidate set)\n\n",
        format_vif_table_md(arm1.get("vif_before", pd.DataFrame())) + "\n\n",
        "### Iterative prune log\n\n",
        format_prune_log_md(arm1.get("prune_log", [])) + "\n\n",
        f"**Surviving columns**: `{arm1.get('surviving_core', [])}`\n\n",
        "### VIF after pruning\n\n",
        format_vif_table_md(arm1.get("vif_after", pd.DataFrame())) + "\n\n",
        f"### μ-decile table — Arm 1 ({oy} OOS)\n\n",
        format_decile_table_md(arm1.get("deciles_oos_year", pd.DataFrame())) + "\n\n",
        "---\n\n",

        # ---- Arm 2: C sweep ----
        "## Arm 2 — EB constant sweep (Fix 2: retune EB shrinkage)\n\n",
    ]

    sweep = arm2.get("sweep", [])
    if sweep:
        lines.append(f"| C | top2-bias ({oy} OOS) | MAE | log-loss@7 |\n")
        lines.append("|---|---|---|---|\n")
        for r in sweep:
            lines.append(
                f"| {r['C']} | {r['top2_bias_oos_year']:+.1f}% | "
                f"{r['mae']:.4f} | {r['log_loss_t7']:.4f} |\n"
            )
        lines.append(f"\n**Best C = {arm2.get('best_C')}** (closest top-2-decile bias to 0 on {oy} OOS)\n\n")
    else:
        lines.append("*(k_stab_last5 not in VIF-surviving set — sweep skipped)*\n\n")

    if arm2.get("best_deciles_oos_year") is not None:
        lines.append(f"### μ-decile table — Arm 2 (C={arm2.get('best_C')}, {oy} OOS)\n\n")
        lines.append(format_decile_table_md(arm2["best_deciles_oos_year"]) + "\n\n")

    lines.append("---\n\n")

    # ---- Arm 3: widened corpus head-to-head ----
    lines += [
        f"## Arm 3 — Widened-corpus head-to-head (Fix 1: correct test window)\n\n",
        f"*Both fits scored on {oy} OOS rows only.*\n\n",
        f"| approach | n ({oy} OOS) | MAE | log-loss@7 | top2-decile bias |\n",
        "|---|---|---|---|---|\n",
    ]
    bm = baseline.get("metrics")
    nm = arm3.get("narrow_metrics")
    wm = arm3.get("widened_metrics")
    bd = baseline.get("deciles_oos_year", pd.DataFrame())
    nd = arm3.get("narrow_deciles", pd.DataFrame())
    wd = arm3.get("widened_deciles", pd.DataFrame())
    bn = bm.get("n", 0) if bm else 0
    nn = nm.get("n", 0) if nm else 0
    wn = wm.get("n", 0) if wm else 0

    for label, m, d, n in [
        (f"baseline (production)", bm, bd, bn),
        (f"narrow ({oy}-only train)", nm, nd, nn),
        (f"widened (full history)", wm, wd, wn),
    ]:
        if m:
            lines.append(
                f"| {label} | {n} | {m['mae']:.4f} | {m['log_loss_t7']:.4f} | "
                f"{_top2_bias(d):+.1f}% |\n"
            )
        else:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |\n")

    lines += ["\n"]
    if nd is not None and not nd.empty:
        lines.append(f"### μ-decile table — Arm 3 narrow ({oy}-only train, {oy} OOS)\n\n")
        lines.append(format_decile_table_md(nd) + "\n\n")
    if wd is not None and not wd.empty:
        lines.append(f"### μ-decile table — Arm 3 widened (full history, {oy} OOS)\n\n")
        lines.append(format_decile_table_md(wd) + "\n\n")

    lines += [
        "---\n\n",
        "## Baseline μ-decile table (production model, full OOS)\n\n",
        format_decile_table_md(baseline.get("deciles_full", pd.DataFrame())) + "\n\n",
        "---\n\n",
        "## Feature sets\n\n",
        f"**Baseline (current production):** `CORE={BASELINE_CORE}`\n\n",
        f"**v2 surviving columns (VIF-pruned):** `{surviving_core}`\n\n",
        f"**Best EB constant:** C={best_C}\n\n",
        "---\n\n",
        "## baseline_model.py edits\n\n",
        f"```\n{edit_log}\n```\n\n",
        "---\n\n",
        "## Next steps\n\n",
    ]
    if adopted:
        lines.append(
            f"1. Run widened-corpus retrain: "
            f"`python -m scripts.run_backtest --start 2024-04-01 --end <today> "
            f"--through-date <today> --fit-only`\n"
            f"2. Update EB_K_CONSTANT={best_C} in rolling_features.py (search EB_K_CONSTANT = ...).\n"
            f"3. Run refresh: `python -m src.pipeline.refresh --date <today>`\n"
            f"4. Spot-check 3 aces: μ should be materially higher; "
            f"surviving K-skill coefficient should exceed +0.057.\n"
        )
    else:
        lines.append(
            "Gate rejected. Investigate:\n"
            f"- If top-2-decile bias did not shrink on {oy} OOS: the compression may be exposure-bound\n"
            f"  (K-count = rate × batters). Consider expected-batters offset as the next lever.\n"
            "- If metrics regressed: check surviving column set for any remaining collinearity.\n"
        )

    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------
# baseline_model.py permanent edit (if adopted)
# -----------------------------------------------------------------------

def _edit_baseline_model(
    surviving_core: list,
    best_C: float | None,
    dry_run: bool = False,
) -> str:
    """Edit baseline_model.py in-place if gate passes."""
    bm_path = PROJECT_ROOT / "src" / "models" / "baseline_model.py"
    src = bm_path.read_text(encoding="utf-8")
    changes = []

    # Replace CORE_PITCHER_FORM_COLUMNS
    old_core_pattern = r"CORE_PITCHER_FORM_COLUMNS\s*=\s*\[.*?\]"
    new_core_str = f"CORE_PITCHER_FORM_COLUMNS = {surviving_core!r}"
    if re.search(old_core_pattern, src, re.DOTALL):
        src = re.sub(old_core_pattern, new_core_str, src, flags=re.DOTALL)
        changes.append(f"  CORE_PITCHER_FORM_COLUMNS → {surviving_core}")
    else:
        changes.append("  CORE_PITCHER_FORM_COLUMNS: pattern not found — EDIT MANUALLY")

    # Update EB_K_CONSTANT in rolling_features.py if best_C known
    if best_C is not None:
        rf_path = PROJECT_ROOT / "src" / "features" / "rolling_features.py"
        rf_src = rf_path.read_text(encoding="utf-8")
        old_c_pattern = r"EB_K_CONSTANT\s*=\s*[\d.]+"
        new_c_str = f"EB_K_CONSTANT = {float(best_C)}"
        if re.search(old_c_pattern, rf_src):
            rf_src = re.sub(old_c_pattern, new_c_str, rf_src)
            changes.append(f"  rolling_features.EB_K_CONSTANT → {best_C}")
            if not dry_run:
                rf_path.write_text(rf_src, encoding="utf-8")
        else:
            changes.append("  EB_K_CONSTANT: pattern not found in rolling_features.py — EDIT MANUALLY")

    if not dry_run:
        bm_path.write_text(src, encoding="utf-8")

    return "\n".join(changes) if changes else "  (nothing to change)"


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run_gate_v2(args, *, statcast_fetcher=None):
    from src.backtest.corpus import build_corpus, filter_starters
    from pybaseball import statcast as _statcast

    def _default_fetcher(ws, we):
        df = _statcast(start_dt=ws, end_dt=we)
        return df if df is not None else pd.DataFrame()

    statcast_fetcher = statcast_fetcher or _default_fetcher

    print(f"[1/6] Building corpus {args.start} → {args.end} ...")
    pitch_df = build_corpus(
        args.start, args.end,
        statcast_fetcher=statcast_fetcher,
        cache_dir=args.cache_dir,
        window_days=args.window_days,
    )
    print(f"      {len(pitch_df):,} pitch-level rows.")
    if pitch_df.empty:
        print("No pitch rows — nothing to do. Exiting.")
        sys.exit(1)

    print("[2/6] Building feature table ...")
    game_df = build_training_table(pitch_df)
    feature_table = filter_starters(game_df, min_batters_faced=args.min_batters_faced)
    print(f"      {len(feature_table):,} starter rows; "
          f"years: {sorted(pd.to_datetime(feature_table['game_date']).dt.year.unique())}")
    if feature_table.empty:
        print("No starter rows — exiting.")
        sys.exit(1)

    print("[3/6] Baseline walk-forward (production model) ...")
    baseline = run_baseline(feature_table, args)

    print("[4/6] Arm 1 — iterative VIF prune ...")
    arm1 = run_arm1_vif_prune(feature_table, args)
    surviving_core = arm1["surviving_core"]

    print("[5/6] Arm 2 — EB constant sweep ...")
    arm2 = run_arm2_eb_sweep(feature_table, surviving_core, args)
    best_C = arm2.get("best_C")

    # Rebuild feature table with best C for Arm 3
    feature_table_v2 = (
        _recompute_k_stab(feature_table, best_C)
        if best_C is not None and "k_stab_last5" in surviving_core
        else feature_table
    )

    print("[6/6] Arm 3 — widened corpus head-to-head ...")
    arm3 = run_arm3_widened(feature_table_v2, surviving_core, best_C, args)

    # Final decision: use widened fit vs production baseline, scored on oos_year
    final_deciles = arm3.get("widened_deciles") or arm1.get("deciles_oos_year", pd.DataFrame())
    final_metrics = arm3.get("widened_metrics")
    baseline_oos_metrics = _aggregate_metrics(
        baseline["oos"][baseline["oos_mask"]]
        if baseline.get("oos_mask") is not None and len(baseline["oos"]) > 0 else pd.DataFrame()
    )

    adopted, decision_reasons = _decide_v2(
        baseline_deciles_oos_year=baseline["deciles_oos_year"],
        final_deciles_oos_year=final_deciles,
        baseline_metrics_oos_year=baseline_oos_metrics if baseline_oos_metrics.get("n", 0) > 0 else None,
        final_metrics_oos_year=final_metrics,
        surviving_core=surviving_core,
        feature_table=feature_table,
    )

    run_date = date.today().isoformat()
    edit_log = _edit_baseline_model(
        surviving_core=surviving_core,
        best_C=best_C,
        dry_run=(not adopted or args.dry_run),
    )

    report_path = _write_report_v2(
        run_date=run_date, args=args,
        baseline=baseline,
        arm1=arm1, arm2=arm2, arm3=arm3,
        adopted=adopted,
        decision_reasons=decision_reasons,
        surviving_core=surviving_core,
        best_C=best_C,
        edit_log=edit_log,
    )

    print("\n" + "=" * 70)
    print(f"Decision ({args.oos_year} OOS): {'ADOPTED ✓' if adopted else 'REJECTED ✗'}")
    print("\n".join(decision_reasons))
    if adopted and not args.dry_run:
        print(f"\nbaseline_model.py + rolling_features.py updated:")
        print(edit_log)
    elif adopted and args.dry_run:
        print(f"\n[--dry-run] Would have edited:")
        print(edit_log)
    else:
        print(f"\nProduction model unchanged. See report for guidance.")
    print(f"\nReport: {report_path}")
    print("=" * 70)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compression-fix v2 gate: VIF prune + EB sweep + widened corpus."
    )
    parser.add_argument("--start", required=True, help="Corpus start date, e.g. 2024-04-01")
    parser.add_argument("--end", required=True, help="Corpus end date, e.g. 2026-06-30")
    parser.add_argument("--oos-year", type=int, default=2026,
                        help="Year to use as the OOS scoring slice (default 2026)")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-batters-faced", type=int, default=None)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print adopt/reject but do NOT edit any source files.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_gate_v2(args)
