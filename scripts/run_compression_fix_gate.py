"""
scripts/run_compression_fix_gate.py

Walk-forward gate for the projection-compression fix
(spec: docs/design/specs/2026-06-30-fix-projection-compression-and-matchup-design.md).

Runs two walk-forwards in-process against the same cached corpus:
  1. Baseline  — current production column set (k_rate_last5, opponent_k_rate_last10)
  2. Candidate — compression-fix set (k_stab_last5, no opponent_k_rate_last10,
                  + csw_rate_season trial candidate)

Computes:
  • Standard aggregate metrics (MAE, RMSE, ECE, Brier, log-loss) for both
  • μ-decile table (predicted vs realized K per decile) for both — THIS IS THE KEY CHECK
  • VIF on the candidate K-skill cluster to catch remaining collinearity
  • Adopt/reject decision (spec gate: high-μ bias must shrink; aggregate metrics must hold)

Writes:
  • reports/<date>-compression-matchup-fix.md  — full comparison report
  • Edits src/models/baseline_model.py in-place if gate passes (unless --dry-run)

Usage:
    python scripts/run_compression_fix_gate.py \\
        --start 2024-04-01 --end 2024-09-30 \\
        [--dry-run]                      # print adopt/reject but don't edit baseline_model.py
        [--no-production-model]          # skip the --fit-only retrain step when adopted
        [--step 7]                       # walk-forward step in days (default 7)
        [--min-train-dates 14]           # minimum distinct training dates per step

Note: uses the same Statcast cache that run_backtest.py writes, so if you've already
run run_backtest.py --start ... --end ..., the corpus is already cached and this script
runs cheaply (no network calls).
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
# Compression-fix column set
# -----------------------------------------------------------------------
COMPRESSION_FIX_CORE = [
    "k_stab_last5",          # EB-stabilized K rate replaces raw k_rate_last5
    "whiff_rate_last5",
    "velo_avg_last5",
    "pitch_count_avg_last5",
]
COMPRESSION_FIX_IMPUTE = [
    # opponent_k_rate_last10 intentionally DROPPED (sign-flipped by collinearity
    # with the hand-split; keeping it dilutes the correctly-signed matchup term)
    "opponent_k_rate_vs_hand_season",
    "park_k_factor",
    "rest_days",
]
COMPRESSION_FIX_EXTRA = ["csw_rate_season"]   # trial candidate for VIF check


@contextmanager
def _patch_baseline_columns(new_core, new_impute):
    """
    Temporarily replace baseline_model's module-level column lists so that
    _dropna_core, fit_preprocessor, and transform_design_matrix all pick up the
    candidate column set without a permanent code change.

    Restores the originals on exit (even on exception).
    """
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


def _fit_fn_compression(train_df, test_df=None, **kw):
    """fit_fn wrapper that applies the compression-fix column patch + csw_rate_season extra."""
    with _patch_baseline_columns(COMPRESSION_FIX_CORE, COMPRESSION_FIX_IMPUTE):
        return fit_baseline_model(
            train_df, test_df, extra_columns=COMPRESSION_FIX_EXTRA, **kw
        )


# -----------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------

def _aggregate_metrics(oos_df: pd.DataFrame) -> dict:
    """Compute MAE, RMSE, ECE from a walk-forward OOS frame."""
    mu = oos_df["mu"].astype(float)
    k = oos_df["realized_strikeouts"].astype(float)
    n = len(oos_df)
    mae = float((mu - k).abs().mean()) if n > 0 else float("nan")
    rmse = float(np.sqrt(((mu - k) ** 2).mean())) if n > 0 else float("nan")

    # ECE: computed at threshold=7 (most common line; Brier at a representative threshold)
    p7_col = "p_over_7" if "p_over_7" in oos_df.columns else None
    ece = float("nan")
    brier = float("nan")
    ll = float("nan")
    if p7_col:
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


def _vif_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each column in cols from df."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return pd.DataFrame({"col": cols, "vif": [float("nan")] * len(cols)})

    sub = df[cols].dropna()
    if len(sub) < len(cols) + 2:
        return pd.DataFrame({"col": cols, "vif": [float("nan")] * len(cols)})

    vifs = []
    sub_arr = sub.values
    for i in range(sub_arr.shape[1]):
        try:
            v = variance_inflation_factor(sub_arr, i)
        except Exception:
            v = float("nan")
        vifs.append(round(v, 2))

    return pd.DataFrame({"col": cols, "vif": vifs})


# -----------------------------------------------------------------------
# Adoption decision
# -----------------------------------------------------------------------

ADOPT_GATE = {
    # High-μ top-2-decile bias must be less negative (closer to 0) after the fix
    # OR the baseline already had no systematic under-prediction (still adopt if metrics improve)
    "mae_must_not_regress_by": 0.05,       # MAE can worsen by at most 0.05 K
    "log_loss_must_not_regress_by": 0.010, # log-loss at t=7 can worsen by at most 0.010
    "top_decile_bias_improvement_needed": 0.0,  # baseline top-decile bias_pct must shrink
}


def _decide(baseline_metrics, fix_metrics, baseline_deciles, fix_deciles):
    """
    Apply the spec adoption rule:
    Adopt if:
      (a) High-μ-decile bias shrinks (aces less systematically under-predicted)
      (b) Aggregate calibration doesn't regress and OOS log-loss/MAE improve or hold

    Returns (adopted: bool, reasons: list[str])
    """
    reasons = []

    # Gate (a): top-2 decile bias_pct should move toward 0 (less negative = better)
    base_top_bias = float(baseline_deciles.nlargest(2, "decile")["bias_pct"].mean())
    fix_top_bias = float(fix_deciles.nlargest(2, "decile")["bias_pct"].mean())
    bias_improved = fix_top_bias > base_top_bias
    reasons.append(
        f"  (a) Top-decile bias_pct: baseline={base_top_bias:+.1f}%  fix={fix_top_bias:+.1f}%  "
        f"{'✓ IMPROVED' if bias_improved else '✗ DID NOT IMPROVE'}"
    )

    # Gate (b): MAE and log-loss must not regress beyond tolerance
    mae_ok = (fix_metrics["mae"] - baseline_metrics["mae"]) <= ADOPT_GATE["mae_must_not_regress_by"]
    ll_ok  = (fix_metrics["log_loss_t7"] - baseline_metrics["log_loss_t7"]) <= ADOPT_GATE["log_loss_must_not_regress_by"]
    reasons.append(
        f"  (b) MAE: baseline={baseline_metrics['mae']:.4f}  fix={fix_metrics['mae']:.4f}  "
        f"{'✓ OK' if mae_ok else '✗ REGRESSION'}"
    )
    reasons.append(
        f"      log-loss@7: baseline={baseline_metrics['log_loss_t7']:.4f}  "
        f"fix={fix_metrics['log_loss_t7']:.4f}  {'✓ OK' if ll_ok else '✗ REGRESSION'}"
    )

    adopted = bias_improved and mae_ok and ll_ok
    return adopted, reasons


# -----------------------------------------------------------------------
# baseline_model.py edit (permanent adoption)
# -----------------------------------------------------------------------

def _edit_baseline_model(dry_run: bool = False) -> str:
    """
    If gate passes: edit src/models/baseline_model.py in-place to:
      1. Replace k_rate_last5 with k_stab_last5 in CORE_PITCHER_FORM_COLUMNS
      2. Remove opponent_k_rate_last10 from IMPUTE_COLUMNS
      3. Add k_stab_last5 to LEAKAGE_COLUMNS (k_stab is still a derived column,
         not a same-game raw stat, but we record it to keep the allowlist accurate)

    Returns a string describing what changed (or what WOULD change in --dry-run).
    """
    bm_path = PROJECT_ROOT / "src" / "models" / "baseline_model.py"
    src = bm_path.read_text(encoding="utf-8")

    changes = []

    # 1. CORE_PITCHER_FORM_COLUMNS: swap k_rate_last5 → k_stab_last5
    old_core = (
        'CORE_PITCHER_FORM_COLUMNS = [\n'
        '    "k_rate_last5", "whiff_rate_last5", "velo_avg_last5", "pitch_count_avg_last5",\n'
        ']'
    )
    new_core = (
        'CORE_PITCHER_FORM_COLUMNS = [\n'
        '    "k_stab_last5", "whiff_rate_last5", "velo_avg_last5", "pitch_count_avg_last5",\n'
        ']'
    )
    if old_core in src:
        src = src.replace(old_core, new_core)
        changes.append("  CORE_PITCHER_FORM_COLUMNS: k_rate_last5 → k_stab_last5")
    elif "k_stab_last5" in src and "k_rate_last5" not in src:
        changes.append("  CORE_PITCHER_FORM_COLUMNS: already uses k_stab_last5 (no change needed)")
    else:
        changes.append("  CORE_PITCHER_FORM_COLUMNS: COULD NOT FIND exact string to replace — EDIT MANUALLY")

    # 2. IMPUTE_COLUMNS: drop opponent_k_rate_last10
    old_impute = (
        'IMPUTE_COLUMNS = [\n'
        '    "opponent_k_rate_last10", "opponent_k_rate_vs_hand_season", "park_k_factor",\n'
        '    "rest_days",\n'
        ']'
    )
    new_impute = (
        'IMPUTE_COLUMNS = [\n'
        '    # opponent_k_rate_last10 dropped: sign-flipped by collinearity with the\n'
        '    # hand-split (see spec fix-projection-compression, 2026-06-30).\n'
        '    "opponent_k_rate_vs_hand_season", "park_k_factor",\n'
        '    "rest_days",\n'
        ']'
    )
    if old_impute in src:
        src = src.replace(old_impute, new_impute)
        changes.append("  IMPUTE_COLUMNS: opponent_k_rate_last10 removed")
    elif "opponent_k_rate_last10" not in src:
        changes.append("  IMPUTE_COLUMNS: opponent_k_rate_last10 already absent (no change needed)")
    else:
        changes.append("  IMPUTE_COLUMNS: COULD NOT FIND exact string to replace — EDIT MANUALLY")

    if not dry_run:
        bm_path.write_text(src, encoding="utf-8")

    return "\n".join(changes) if changes else "  (nothing to change)"


# -----------------------------------------------------------------------
# Report writer
# -----------------------------------------------------------------------

def _write_report(
    run_date: str,
    args,
    baseline_metrics: dict,
    fix_metrics: dict,
    baseline_deciles: pd.DataFrame,
    fix_deciles: pd.DataFrame,
    vif_df: pd.DataFrame,
    adopted: bool,
    decision_reasons: list,
    edit_log: str,
) -> Path:
    """Write reports/<date>-compression-matchup-fix.md."""
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{run_date}-compression-matchup-fix.md"

    adopt_str = "**ADOPTED**" if adopted else "**REJECTED**"

    lines = [
        f"# Compression-fix gate — {run_date}\n\n",
        f"**Decision: {adopt_str}**\n\n",
        "## Gate reasoning\n\n",
        "\n".join(decision_reasons) + "\n\n",
        "---\n\n",
        "## Aggregate metrics (walk-forward OOS)\n\n",
        "| metric | baseline | compression-fix | Δ |\n",
        "|--------|----------|-----------------|---|\n",
        f"| n starts | {baseline_metrics['n']} | {fix_metrics['n']} | — |\n",
        f"| MAE | {baseline_metrics['mae']:.4f} | {fix_metrics['mae']:.4f} | "
        f"{fix_metrics['mae'] - baseline_metrics['mae']:+.4f} |\n",
        f"| RMSE | {baseline_metrics['rmse']:.4f} | {fix_metrics['rmse']:.4f} | "
        f"{fix_metrics['rmse'] - baseline_metrics['rmse']:+.4f} |\n",
        f"| ECE | {baseline_metrics['ece']:.4f} | {fix_metrics['ece']:.4f} | "
        f"{fix_metrics['ece'] - baseline_metrics['ece']:+.4f} |\n",
        f"| Brier@7 | {baseline_metrics['brier_t7']:.4f} | {fix_metrics['brier_t7']:.4f} | "
        f"{fix_metrics['brier_t7'] - baseline_metrics['brier_t7']:+.4f} |\n",
        f"| log-loss@7 | {baseline_metrics['log_loss_t7']:.4f} | {fix_metrics['log_loss_t7']:.4f} | "
        f"{fix_metrics['log_loss_t7'] - baseline_metrics['log_loss_t7']:+.4f} |\n\n",
        "---\n\n",
        "## μ-decile table — BASELINE\n\n",
        "*bias = mean_pred − mean_actual; negative = model under-predicted (actual > predicted)*\n\n",
        format_decile_table_md(baseline_deciles),
        "\n\n",
        "## μ-decile table — COMPRESSION FIX\n\n",
        format_decile_table_md(fix_deciles),
        "\n\n",
        "---\n\n",
        "## VIF on candidate K-skill cluster (compression-fix variant)\n\n",
        "*VIF > 10 = problematic collinearity; drop the highest-VIF redundant feature*\n\n",
        "| feature | VIF |\n|---------|-----|\n",
    ]
    for _, row in vif_df.iterrows():
        lines.append(f"| {row['col']} | {row['vif']} |\n")

    lines += [
        "\n---\n\n",
        "## baseline_model.py edits\n\n",
        f"```\n{edit_log}\n```\n\n",
        "---\n\n",
        "## Feature sets compared\n\n",
        "**Baseline (current production):**\n",
        f"  CORE: {CORE_PITCHER_FORM_COLUMNS}\n",
        f"  IMPUTE: {IMPUTE_COLUMNS}\n\n",
        "**Compression-fix candidate:**\n",
        f"  CORE: {COMPRESSION_FIX_CORE}\n",
        f"  IMPUTE: {COMPRESSION_FIX_IMPUTE}\n",
        f"  EXTRA: {COMPRESSION_FIX_EXTRA}\n\n",
        "---\n\n",
        "## Next steps\n\n",
    ]
    if adopted:
        lines.append(
            "1. Run widened-corpus retrain: "
            "`python -m scripts.run_backtest --start 2024-04-01 --end <today> "
            "--through-date <today> --fit-only`\n"
            "2. Run refresh: "
            "`python -m src.pipeline.refresh --date <today>`\n"
            "3. Spot-check 3 aces: μ should be materially higher than before; "
            "coefficient on k_stab_last5 should exceed the old +0.057 on k_rate_last5.\n"
        )
    else:
        lines.append(
            "Gate rejected — production model unchanged. Investigate:\n"
            "- If top-decile bias didn't shrink: k_stab_last5 may need a different C constant "
            "(tune EB_K_CONSTANT in rolling_features.py).\n"
            "- If aggregate metrics regressed: widened-corpus retrain (the PRIMARY lever from "
            "the spec) may be needed before the feature changes show their full benefit.\n"
        )

    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run_gate(args, *, statcast_fetcher=None):
    from src.backtest.corpus import build_corpus, filter_starters
    from pybaseball import statcast as _statcast  # imported lazily

    def _default_fetcher(ws, we):
        df = _statcast(start_dt=ws, end_dt=we)
        return df if df is not None else pd.DataFrame()

    statcast_fetcher = statcast_fetcher or _default_fetcher

    print(f"[1/5] Building corpus {args.start} → {args.end} ...")
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

    print("[2/5] Building feature table ...")
    game_df = build_training_table(pitch_df)
    feature_table = filter_starters(game_df, min_batters_faced=args.min_batters_faced)
    print(f"      {len(feature_table):,} starter rows.")
    if feature_table.empty:
        print("No starter rows — exiting.")
        sys.exit(1)

    print(f"[3/5] Walk-forward baseline (step={args.step}d, min_train={args.min_train_dates}) ...")
    oos_baseline = run_walk_forward(
        feature_table,
        step=args.step,
        min_train_dates=args.min_train_dates,
        fit_fn=fit_baseline_model,
    )
    print(f"      {len(oos_baseline):,} OOS rows.")

    print("[4/5] Walk-forward compression-fix candidate ...")
    oos_fix = run_walk_forward(
        feature_table,
        step=args.step,
        min_train_dates=args.min_train_dates,
        fit_fn=_fit_fn_compression,
    )
    print(f"      {len(oos_fix):,} OOS rows.")

    print("[5/5] Computing metrics, deciles, VIF ...")
    baseline_metrics = _aggregate_metrics(oos_baseline.rename(columns={"mu": "mu", "realized_strikeouts": "realized_strikeouts"}))
    fix_metrics = _aggregate_metrics(oos_fix)

    baseline_deciles = compute_mu_decile_table(oos_baseline.rename(columns={"mu": "pred_mean", "realized_strikeouts": "strikeouts"}))
    fix_deciles      = compute_mu_decile_table(oos_fix.rename(columns={"mu": "pred_mean", "realized_strikeouts": "strikeouts"}))

    vif_cols = COMPRESSION_FIX_CORE + COMPRESSION_FIX_EXTRA
    vif_df = _vif_table(feature_table, [c for c in vif_cols if c in feature_table.columns])

    adopted, decision_reasons = _decide(baseline_metrics, fix_metrics, baseline_deciles, fix_deciles)

    run_date = date.today().isoformat()
    edit_log = _edit_baseline_model(dry_run=(not adopted or args.dry_run))

    report_path = _write_report(
        run_date=run_date,
        args=args,
        baseline_metrics=baseline_metrics,
        fix_metrics=fix_metrics,
        baseline_deciles=baseline_deciles,
        fix_deciles=fix_deciles,
        vif_df=vif_df,
        adopted=adopted,
        decision_reasons=decision_reasons,
        edit_log=edit_log,
    )

    print("\n" + "=" * 70)
    print(f"Decision: {'ADOPTED ✓' if adopted else 'REJECTED ✗'}")
    print("\n".join(decision_reasons))
    if adopted and not args.dry_run:
        print("\nbaseline_model.py updated:")
        print(edit_log)
        print("\nNext: retrain with widened corpus (see report for command).")
    elif adopted and args.dry_run:
        print("\n[--dry-run] Would have updated baseline_model.py:")
        print(edit_log)
    else:
        print("\nProduction model unchanged. See report for investigation guidance.")
    print(f"\nReport: {report_path}")
    print("=" * 70)

    return {
        "adopted": adopted,
        "baseline_metrics": baseline_metrics,
        "fix_metrics": fix_metrics,
        "baseline_deciles": baseline_deciles,
        "fix_deciles": fix_deciles,
        "vif_df": vif_df,
        "report_path": str(report_path),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Walk-forward gate for the projection-compression fix."
    )
    parser.add_argument("--start", required=True, help="Corpus start date, e.g. 2024-04-01")
    parser.add_argument("--end", required=True, help="Corpus end date, e.g. 2024-09-30")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--min-batters-faced", type=int, default=None)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--dry-run", action="store_true",
                         help="Print adopt/reject but do NOT edit baseline_model.py.")
    parser.add_argument("--no-production-model", action="store_true",
                         help="If adopted: skip the retrain step (just update baseline_model.py).")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_gate(args)
