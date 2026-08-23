"""
scripts/compare_skill_features.py
Gate comparison: baseline vs skill-features variant.

Loads the already-cached Statcast corpus (no network needed if corpus
cache files already exist under data/raw/statcast/), runs both walk-forward
evaluations in-process, computes VIF across skill candidates, decides which
subset to promote, and writes reports/<date>-skill-features.md.

Usage:
    python scripts/compare_skill_features.py --start 2024-04-01 --end 2024-09-30
    python scripts/compare_skill_features.py --start 2024-04-01 --end 2024-09-30 --step 7
"""

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.backtest.corpus import build_corpus, filter_starters, DEFAULT_CACHE_DIR
from src.backtest.walk_forward import run_walk_forward, DEFAULT_STEP_DAYS, DEFAULT_MIN_TRAIN_DATES
from src.features.build_features import build_training_table
from src.models.baseline_model import fit_baseline_model, SKILL_CANDIDATE_COLUMNS

DEFAULT_REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _oos_metrics(oos_df: pd.DataFrame, thresholds=(6, 7)) -> dict:
    """Compute MAE, RMSE, ECE, and per-threshold Brier/log-loss."""
    mu = oos_df["mu"].values
    y = oos_df["realized_strikeouts"].values
    mae = float(np.mean(np.abs(mu - y)))
    rmse = float(np.sqrt(np.mean((mu - y) ** 2)))

    # ECE: mean absolute calibration error across 10 probability bins
    ece_vals = []
    for t in thresholds:
        p_col = f"p_over_{t}"
        hit_col = f"over_hit_{t}"
        if p_col not in oos_df.columns or hit_col not in oos_df.columns:
            continue
        p = oos_df[p_col].values
        h = oos_df[hit_col].values
        mask = ~(np.isnan(p) | np.isnan(h))
        if mask.sum() < 20:
            continue
        p, h = p[mask], h[mask]
        bins = np.linspace(0, 1, 11)
        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = (p >= lo) & (p < hi)
            if idx.sum() < 5:
                continue
            ece_vals.append(abs(p[idx].mean() - h[idx].mean()))
    ece = float(np.mean(ece_vals)) if ece_vals else np.nan

    per_t = {}
    for t in thresholds:
        p_col = f"p_over_{t}"
        hit_col = f"over_hit_{t}"
        if p_col not in oos_df.columns or hit_col not in oos_df.columns:
            continue
        p = np.clip(oos_df[p_col].values, 1e-7, 1 - 1e-7)
        h = oos_df[hit_col].values
        mask = ~(np.isnan(p) | np.isnan(h))
        p, h = p[mask], h[mask]
        brier = float(np.mean((p - h) ** 2))
        logloss = float(-np.mean(h * np.log(p) + (1 - h) * np.log(1 - p)))
        per_t[t] = {"brier": brier, "log_loss": logloss, "n": int(mask.sum())}

    return {"mae": mae, "rmse": rmse, "ece": ece, "n": len(oos_df), "per_threshold": per_t}


# ──────────────────────────────────────────────────────────────────────────────
# VIF
# ──────────────────────────────────────────────────────────────────────────────

def _compute_vif(feature_table: pd.DataFrame, candidates: list) -> pd.DataFrame:
    """Return VIF for each skill candidate using the full training feature table."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  [VIF] statsmodels not available — skipping VIF computation")
        return pd.DataFrame({"feature": candidates, "vif": [np.nan] * len(candidates)})

    # Use only rows where ALL candidates are non-null
    sub = feature_table[candidates].dropna()
    if len(sub) < 50:
        print(f"  [VIF] Only {len(sub)} complete rows — skipping VIF")
        return pd.DataFrame({"feature": candidates, "vif": [np.nan] * len(candidates)})

    X = sub.values.astype(float)
    rows = []
    for i, col in enumerate(candidates):
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "vif": round(vif, 1) if not np.isnan(vif) else np.nan})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Promotion decision
# ──────────────────────────────────────────────────────────────────────────────

VIF_THRESHOLD = 10.0
IMPROVEMENT_MIN_DELTA = 0.001  # MAE or log-loss must improve by at least this


def _decide_promotion(baseline_m: dict, skill_m: dict, vif_df: pd.DataFrame) -> dict:
    """
    Return a promotion decision dict:
      promote_cols   : list of columns to add to CORE_PITCHER_FORM_COLUMNS
      reject_cols    : list to keep as candidates
      reason         : human-readable rationale
    """
    mae_delta = baseline_m["mae"] - skill_m["mae"]  # positive = skill better
    ece_delta = baseline_m["ece"] - skill_m["ece"]

    # Log-loss at t=6 (most informative threshold)
    base_ll6 = baseline_m["per_threshold"].get(6, {}).get("log_loss", np.nan)
    skill_ll6 = skill_m["per_threshold"].get(6, {}).get("log_loss", np.nan)
    ll6_delta = base_ll6 - skill_ll6  # positive = skill better

    high_vif = set(vif_df[vif_df["vif"] > VIF_THRESHOLD]["feature"].tolist())

    if mae_delta < IMPROVEMENT_MIN_DELTA and ll6_delta < IMPROVEMENT_MIN_DELTA:
        return {
            "promote_cols": [],
            "reject_cols": SKILL_CANDIDATE_COLUMNS,
            "verdict": "REJECT — no OOS improvement",
            "reason": (
                f"Skill-features variant did not beat baseline OOS "
                f"(ΔMAE={mae_delta:+.4f}, Δlog-loss@6={ll6_delta:+.4f}). "
                f"All skill candidates remain in SKILL_CANDIDATE_COLUMNS."
            ),
        }

    # Something improved — pick non-collinear subset
    # Priority: k_minus_bb_rate_last5 (compound), then swstr_rate_last5 (best single-pitch),
    # then putaway_rate_last5.  csw_rate and whiff_rate_overall are
    # typically collinear with swstr.
    candidate_priority = [
        "k_minus_bb_rate_last5",
        "swstr_rate_last5",
        "putaway_rate_last5",
        "csw_rate_last5",
        "whiff_rate_overall_last5",
    ]
    promote = []
    reject = []
    for col in candidate_priority:
        if col not in SKILL_CANDIDATE_COLUMNS:
            continue
        vif_row = vif_df[vif_df["feature"] == col]
        col_vif = float(vif_row["vif"].iloc[0]) if len(vif_row) and not np.isnan(vif_row["vif"].iloc[0]) else 1.0
        if col_vif > VIF_THRESHOLD and promote:
            reject.append(col)
        else:
            promote.append(col)

    return {
        "promote_cols": promote,
        "reject_cols": reject,
        "verdict": f"PROMOTE {promote}",
        "reason": (
            f"Skill-features improved OOS (ΔMAE={mae_delta:+.4f}, "
            f"Δlog-loss@6={ll6_delta:+.4f}, ΔECE={ece_delta:+.4f}). "
            f"High-VIF columns (VIF>{VIF_THRESHOLD}): {sorted(high_vif) or 'none'}. "
            f"Promoting non-redundant subset: {promote}. "
            f"Remaining candidates: {reject}."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_report(
    path: str,
    as_of: str,
    args,
    baseline_m: dict,
    skill_m: dict,
    vif_df: pd.DataFrame,
    decision: dict,
) -> None:
    lines = [
        f"# Skill-features gate comparison — {as_of}",
        "",
        f"Window: {args.start} → {args.end} | step={args.step}d | n_starts={baseline_m['n']}",
        "",
        "## OOS metrics",
        "",
        "| metric | baseline | skill-features | delta | direction |",
        "|--------|----------|----------------|-------|-----------|",
    ]
    for label, bv, sv in [
        ("MAE", baseline_m["mae"], skill_m["mae"]),
        ("RMSE", baseline_m["rmse"], skill_m["rmse"]),
        ("ECE", baseline_m["ece"], skill_m["ece"]),
    ]:
        delta = sv - bv
        direction = "✓ better" if delta < -IMPROVEMENT_MIN_DELTA else ("✗ worse" if delta > IMPROVEMENT_MIN_DELTA else "≈ flat")
        lines.append(f"| {label} | {bv:.4f} | {sv:.4f} | {delta:+.4f} | {direction} |")

    lines += ["", "## Per-threshold (Brier / log-loss)", "",
              "| threshold | base_brier | skill_brier | Δbrier | base_ll | skill_ll | Δll |",
              "|-----------|-----------|------------|--------|---------|---------|-----|"]
    for t in sorted(set(list(baseline_m["per_threshold"]) + list(skill_m["per_threshold"]))):
        bpt = baseline_m["per_threshold"].get(t, {})
        spt = skill_m["per_threshold"].get(t, {})
        db = (spt.get("brier", np.nan) - bpt.get("brier", np.nan))
        dl = (spt.get("log_loss", np.nan) - bpt.get("log_loss", np.nan))
        lines.append(
            f"| {t} | {bpt.get('brier', float('nan')):.4f} | {spt.get('brier', float('nan')):.4f} | {db:+.4f} | "
            f"{bpt.get('log_loss', float('nan')):.4f} | {spt.get('log_loss', float('nan')):.4f} | {dl:+.4f} |"
        )

    lines += ["", "## VIF — skill candidates", "",
              "| feature | VIF | collinear? |",
              "|---------|-----|------------|"]
    for _, row in vif_df.iterrows():
        v = row["vif"]
        collinear = "⚠ YES" if (not np.isnan(v) and v > VIF_THRESHOLD) else "no"
        lines.append(f"| {row['feature']} | {v if not np.isnan(v) else 'n/a'} | {collinear} |")

    lines += [
        "", "## Decision", "",
        f"**{decision['verdict']}**", "",
        decision["reason"], "",
        f"Columns promoted to `CORE_PITCHER_FORM_COLUMNS`: `{decision['promote_cols']}`",
        f"Columns remaining as `SKILL_CANDIDATE_COLUMNS`: `{decision['reject_cols']}`",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report written → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Promote: edit baseline_model.py in place
# ──────────────────────────────────────────────────────────────────────────────

def _promote_columns(promote_cols: list) -> None:
    """
    Move `promote_cols` from SKILL_CANDIDATE_COLUMNS to CORE_PITCHER_FORM_COLUMNS
    in src/models/baseline_model.py by direct string manipulation.
    Idempotent: skips columns already in CORE or not in SKILL_CANDIDATE.
    """
    if not promote_cols:
        print("  [promote] Nothing to promote.")
        return

    path = os.path.join(PROJECT_ROOT, "src", "models", "baseline_model.py")
    with open(path) as f:
        src = f.read()

    import ast, re

    def _extract_list(src, varname):
        m = re.search(rf'{varname}\s*=\s*(\[[\s\S]*?\])', src)
        if not m:
            raise ValueError(f"Could not find {varname} in baseline_model.py")
        return ast.literal_eval(m.group(1)), m.start(1), m.end(1)

    core_list, core_start, core_end = _extract_list(src, "CORE_PITCHER_FORM_COLUMNS")
    skill_list, skill_start, skill_end = _extract_list(src, "SKILL_CANDIDATE_COLUMNS")

    to_promote = [c for c in promote_cols if c in skill_list and c not in core_list]
    if not to_promote:
        print(f"  [promote] All columns already promoted or not in candidates: {promote_cols}")
        return

    new_core = core_list + to_promote
    new_skill = [c for c in skill_list if c not in to_promote]

    def _fmt_list(lst):
        items = ",\n    ".join(repr(c) for c in lst)
        return f"[\n    {items},\n]"

    # Replace in reverse order (higher offset first) to keep offsets valid
    if skill_start < core_start:
        src = src[:skill_start] + _fmt_list(new_skill) + src[skill_end:]
        # Recalculate core offsets after skill replacement
        core_list2, core_start2, core_end2 = _extract_list(src, "CORE_PITCHER_FORM_COLUMNS")
        src = src[:core_start2] + _fmt_list(new_core) + src[core_end2:]
    else:
        src = src[:core_start] + _fmt_list(new_core) + src[core_end:]
        skill_list2, skill_start2, skill_end2 = _extract_list(src, "SKILL_CANDIDATE_COLUMNS")
        src = src[:skill_start2] + _fmt_list(new_skill) + src[skill_end2:]

    with open(path, "w") as f:
        f.write(src)
    print(f"  [promote] Moved to CORE_PITCHER_FORM_COLUMNS: {to_promote}")
    print(f"  [promote] Remaining in SKILL_CANDIDATE_COLUMNS: {new_skill}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gate: baseline vs skill-features OOS comparison")
    parser.add_argument("--start", default="2024-04-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compare and report but do NOT edit baseline_model.py")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[5, 6, 7, 8])
    args = parser.parse_args()

    as_of = date.today().isoformat()

    # ── 1. Load corpus (uses cache, no network needed if already built) ────────
    print("[1/5] Loading corpus from cache (no network needed if already cached)...")
    from src.backtest.corpus import build_corpus
    import pybaseball  # will raise ImportError if not installed

    def _statcast_fetcher(start, end):
        from pybaseball import statcast
        return statcast(start, end, verbose=False)

    pitch_df = build_corpus(
        args.start, args.end,
        statcast_fetcher=_statcast_fetcher,
        cache_dir=args.cache_dir,
    )
    print(f"  Corpus: {len(pitch_df):,} pitch rows")

    # ── 2. Build training table ──────────────────────────────────────────────
    print("[2/5] Building training table...")
    game_df = build_training_table(pitch_df)
    feature_table = filter_starters(game_df)
    print(f"  Feature table: {len(feature_table):,} starter-games")

    # ── 3. VIF on skill candidates ───────────────────────────────────────────
    print("[3/5] Computing VIF on skill candidates...")
    vif_df = _compute_vif(feature_table, SKILL_CANDIDATE_COLUMNS)
    print(vif_df.to_string(index=False))

    # ── 4. Walk-forward: baseline then skill-features ────────────────────────
    print("[4/5] Running baseline walk-forward...")
    oos_base = run_walk_forward(
        feature_table,
        fit_fn=lambda train, test, **kw: fit_baseline_model(train, test, **kw),
        step=args.step,
        min_train_dates=args.min_train_dates,
    )
    baseline_m = _oos_metrics(oos_base, thresholds=args.thresholds)
    print(f"  Baseline  — MAE={baseline_m['mae']:.4f} RMSE={baseline_m['rmse']:.4f} ECE={baseline_m['ece']:.4f} n={baseline_m['n']}")

    print("  Running skill-features walk-forward...")
    def _skill_fit_fn(train, test, **kw):
        return fit_baseline_model(train, test, extra_columns=SKILL_CANDIDATE_COLUMNS, **kw)

    oos_skill = run_walk_forward(
        feature_table,
        fit_fn=_skill_fit_fn,
        step=args.step,
        min_train_dates=args.min_train_dates,
    )
    skill_m = _oos_metrics(oos_skill, thresholds=args.thresholds)
    print(f"  Skill-feat — MAE={skill_m['mae']:.4f} RMSE={skill_m['rmse']:.4f} ECE={skill_m['ece']:.4f} n={skill_m['n']}")

    # ── 5. Decision + report ─────────────────────────────────────────────────
    print("[5/5] Making promotion decision...")
    decision = _decide_promotion(baseline_m, skill_m, vif_df)
    print(f"\n  {decision['verdict']}")
    print(f"  {decision['reason']}")

    report_path = os.path.join(args.reports_dir, f"{as_of}-skill-features.md")
    _write_report(report_path, as_of, args, baseline_m, skill_m, vif_df, decision)

    if not args.dry_run and decision["promote_cols"]:
        print("\n  Editing baseline_model.py to promote columns...")
        _promote_columns(decision["promote_cols"])
        print("\n  Done. Next steps:")
        print("    python -m scripts.run_backtest --start 2026-03-26 --end 2026-06-29 --through-date 2026-06-29 --fit-only")
        print("    python -m src.pipeline.refresh --date $(date +%Y-%m-%d)")
    elif args.dry_run:
        print("\n  --dry-run: baseline_model.py NOT modified.")
    else:
        print("\n  No columns promoted. Model unchanged.")


if __name__ == "__main__":
    main()
