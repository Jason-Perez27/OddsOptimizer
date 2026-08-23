"""
End-to-end historical backtest runner (task #11): the script that actually
executes the walk-forward backtest pipeline against REAL multi-season
Statcast data on a real machine with network access and scipy installed --
this sandbox has neither, so every piece this script wires together
(corpus.py, build_features.py, walk_forward.py, report.py, baseline_model.py)
has only ever been unit-tested against synthetic/hand-built fixtures until
this script is actually run.

Pipeline wired here, in order:
  1. build_corpus(start, end, statcast_fetcher=...)
       -> league-wide pitch-level Statcast rows, windowed + cached + resumable
          under data/raw/statcast/ (a network failure mid-run is safe to
          re-run with the same --start/--end: cached windows are skipped).
  2. build_training_table(pitch_df)
       -> one row per pitcher-game, every rolling/opponent/park feature
          attached, `strikeouts` as the label (src/features/build_features.py).
  3. filter_starters(game_df, min_batters_faced=...)
       -> the evaluation set: one row per (pitcher_team, game_pk), the
          starter-proxy pitcher in that team's half of that game. THIS is the
          `feature_table` both run_walk_forward and fit_production_model
          expect -- neither re-derives features or re-filters starters.
  4. run_walk_forward(feature_table, step=..., min_train_dates=...)
       -> wide OOS frame, one row per evaluated pitcher-start, persisted via
          persist_oos_frame() and rendered via generate_backtest_report().
       SKIPPED when --fit-only is passed (task #12 go-live: a cheap weekly
       retrain that reuses corpus/feature/filter but skips the walk-forward
       evaluation, which is the expensive part).
  5. (optional, default on) fit_production_model(feature_table, through_date=...,
     save_path=...)
       -> the go-live model artifact src/pipeline/refresh.py's default model
          loader expects. Trains on ALL starts through --through-date (default:
          --end), no holdout -- this is NOT the walk-forward evaluation, it's
          the final full fit meant to be loaded daily.

statcast_fetcher: build_corpus's contract is `statcast_fetcher(window_start,
window_end) -> pd.DataFrame` for the FULL LEAGUE, not one pitcher --
src/data/pitcher_logs.py only wraps pybaseball's per-pitcher
`statcast_pitcher`, so this script defines its own thin wrapper here over
pybaseball's league-wide `statcast(start_dt, end_dt)` instead of reusing
pitcher_logs.py (there is nothing there to reuse for this call).

--fit-only (task #12 go-live): skips steps 4-5's walk-forward + report and
runs only corpus -> training table -> starter-filter -> fit_production_model,
saving to --model-path. Mutually exclusive with --no-production-model
(fitting is the entire point of --fit-only). The core orchestration lives in
run_backtest_pipeline() (injectable statcast_fetcher/fit_fn), kept separate
from main() so it's unit-testable with hand-built fixtures and no network --
matching the rest of this repo's run_refresh()/run_settlement() convention.

Usage:
    python -m scripts.run_backtest --start 2024-04-01 --end 2024-09-30
    python -m scripts.run_backtest --start 2024-04-01 --end 2024-09-30 \
        --min-batters-faced 5 --step 7 --min-train-dates 14 \
        --no-production-model
    python -m scripts.run_backtest --start 2026-03-26 --end 2026-06-28 \
        --through-date 2026-06-28 --fit-only
"""

import argparse
import sys

import pandas as pd

from src.backtest.corpus import (
    DEFAULT_CACHE_DIR,
    DEFAULT_WINDOW_DAYS,
    build_corpus,
    filter_starters,
)
from src.backtest.walk_forward import (
    DEFAULT_MIN_TRAIN_DATES,
    DEFAULT_STEP_DAYS,
    run_walk_forward,
)
from src.backtest.report import (
    DEFAULT_OOS_PATH,
    DEFAULT_REPORTS_DIR,
    generate_backtest_report,
    persist_oos_frame,
)
from src.features.build_features import build_training_table
from src.models.baseline_model import fit_baseline_model, fit_production_model
from src.pipeline.refresh import DEFAULT_MODEL_PATH as REFRESH_DEFAULT_MODEL_PATH

# Must match src.pipeline.refresh.DEFAULT_MODEL_PATH exactly -- this script's
# whole point on --model-path is to hand off an artifact that refresh.py's
# default loader (no --model-path passed) will pick up. Re-deriving the same
# constant here previously drifted ("models/..." vs "data/models/..."),
# silently defeating that handoff. Importing it directly instead of
# redefining it makes that drift impossible going forward.
DEFAULT_MODEL_PATH = REFRESH_DEFAULT_MODEL_PATH


def pybaseball_statcast_fetcher(window_start: str, window_end: str) -> pd.DataFrame:
    """
    Thin wrapper over pybaseball's LEAGUE-WIDE statcast() pull -- the real
    network call this script makes. Imported lazily inside the function (not
    at module top) so this module stays importable in environments without
    pybaseball installed (e.g. anything that only wants to import the CLI
    parser for inspection/testing).
    """
    from pybaseball import statcast

    df = statcast(start_dt=window_start, end_dt=window_end)
    return df if df is not None else pd.DataFrame()


def run_backtest_pipeline(args, *, statcast_fetcher=None, fit_fn=None) -> dict:
    """
    Core orchestration, separated from main() so it's unit-testable with an
    injected statcast_fetcher (no network) and an injected fit_fn (no real
    statsmodels fit) -- matches src/pipeline/refresh.py's run_refresh() /
    src/pipeline/settle.py's run_settlement() convention of keeping the CLI
    a thin parser+printer over a pure-ish core.

    Returns a dict describing what ran:
    {"pitch_rows", "game_rows", "feature_rows", "fit_only", "oos_rows",
     "oos_path", "report_result", "model_path", "model"} -- later keys are
     None when the corresponding step was skipped or not reached (e.g. an
     empty corpus exits early; --fit-only never populates oos_*/report_result).

    Raises SystemExit(1) (via sys.exit, printing a message first) on the
    same "nothing to do" conditions main() always has: an empty corpus, an
    empty starter-filtered feature table, or (non-fit-only) zero walk-forward
    OOS rows -- callers that want to assert on these in tests should expect
    SystemExit, not a falsy return.
    """
    statcast_fetcher = statcast_fetcher or pybaseball_statcast_fetcher
    fit_fn = fit_fn or fit_baseline_model

    # --variant: walk-forward trial with a modified feature set.
    # 'skill-features': adds SKILL_CANDIDATE_COLUMNS on top of the v1 allowlist.
    # 'compression-fix': patches CORE/IMPUTE_COLUMNS to replace k_rate_last5 with
    #   k_stab_last5 and drop opponent_k_rate_last10; adds csw_rate_season candidate.
    #   Uses a context-manager patch of baseline_model module constants so that
    #   _dropna_core, fit_preprocessor, and transform_design_matrix all pick up the
    #   new column set without permanent code changes (the gate script promotes to
    #   permanent if the walk-forward passes).
    variant_extra = None
    if getattr(args, "variant", None) == "skill-features":
        from src.models.baseline_model import SKILL_CANDIDATE_COLUMNS
        variant_extra = SKILL_CANDIDATE_COLUMNS
        print(
            f"[--variant skill-features] Adding {len(variant_extra)} skill candidate "
            f"columns as extra regressors: {variant_extra}"
        )
        _base_fit_fn = fit_fn
        def fit_fn(train_df, test_df=None, **kw):  # noqa: E306
            return _base_fit_fn(train_df, test_df, extra_columns=variant_extra, **kw)

    elif getattr(args, "variant", None) == "compression-fix":
        from contextlib import contextmanager
        import src.models.baseline_model as _bm

        @contextmanager
        def _patch_compression_fix():
            """Temporarily swap module-level column lists for the compression-fix trial."""
            orig_core = _bm.CORE_PITCHER_FORM_COLUMNS
            orig_impute = _bm.IMPUTE_COLUMNS
            orig_continuous = _bm.CONTINUOUS_REGRESSOR_COLUMNS
            orig_regressors = _bm.REGRESSOR_COLUMNS
            orig_design = _bm.DESIGN_MATRIX_COLUMNS

            new_core = [
                "k_stab_last5", "whiff_rate_last5", "velo_avg_last5", "pitch_count_avg_last5",
            ]
            new_impute = [
                # opponent_k_rate_last10 intentionally dropped (sign-flipped by collinearity)
                "opponent_k_rate_vs_hand_season",
                "park_k_factor",
                "rest_days",
            ]
            new_continuous = new_core + new_impute
            new_regressors = new_continuous + ["is_home"]
            new_design = new_regressors + _bm.DESIGN_MATRIX_EXTRA_COLUMNS

            _bm.CORE_PITCHER_FORM_COLUMNS = new_core
            _bm.IMPUTE_COLUMNS = new_impute
            _bm.CONTINUOUS_REGRESSOR_COLUMNS = new_continuous
            _bm.REGRESSOR_COLUMNS = new_regressors
            _bm.DESIGN_MATRIX_COLUMNS = new_design
            try:
                yield
            finally:
                _bm.CORE_PITCHER_FORM_COLUMNS = orig_core
                _bm.IMPUTE_COLUMNS = orig_impute
                _bm.CONTINUOUS_REGRESSOR_COLUMNS = orig_continuous
                _bm.REGRESSOR_COLUMNS = orig_regressors
                _bm.DESIGN_MATRIX_COLUMNS = orig_design

        # Adds csw_rate_season as extra candidate alongside the patched core/impute
        _compression_fix_extras = ["csw_rate_season"]
        _base_fit_fn = fit_fn
        print(
            "[--variant compression-fix] Patching CORE_PITCHER_FORM_COLUMNS "
            "(k_stab_last5 replaces k_rate_last5) and IMPUTE_COLUMNS "
            "(opponent_k_rate_last10 dropped). Extra candidate: csw_rate_season."
        )

        def fit_fn(train_df, test_df=None, **kw):  # noqa: E306
            with _patch_compression_fix():
                return _base_fit_fn(train_df, test_df, extra_columns=_compression_fix_extras, **kw)

    print(f"[1/{'3' if args.fit_only else '5'}] Building corpus {args.start} -> {args.end} "
          f"(window={args.window_days}d, cache={args.cache_dir}) ...")
    pitch_df = build_corpus(
        args.start, args.end,
        statcast_fetcher=statcast_fetcher,
        cache_dir=args.cache_dir,
        window_days=args.window_days,
    )
    print(f"      {len(pitch_df)} pitch-level rows pulled/loaded.")
    if pitch_df.empty:
        print("No pitch-level rows for this date range -- nothing to build. Exiting.")
        sys.exit(1)

    print(f"[2/{'3' if args.fit_only else '5'}] Building per-pitcher-game training table (features + label) ...")
    game_df = build_training_table(pitch_df)
    print(f"      {len(game_df)} pitcher-game rows built.")

    print(f"[3/{'3' if args.fit_only else '5'}] Filtering to starters (min_batters_faced={args.min_batters_faced}) ...")
    feature_table = filter_starters(game_df, min_batters_faced=args.min_batters_faced)
    print(f"      {len(feature_table)} starter rows -- this is the walk-forward evaluation set.")
    if feature_table.empty:
        print("No starter rows survived filtering -- nothing to walk-forward. Exiting.")
        sys.exit(1)

    result = {
        "pitch_rows": len(pitch_df),
        "game_rows": len(game_df),
        "feature_rows": len(feature_table),
        "fit_only": args.fit_only,
        "oos_rows": None,
        "oos_path": None,
        "report_result": None,
        "model_path": None,
        "model": None,
    }

    if args.fit_only:
        through_date = args.through_date or args.end
        print(f"Fitting go-live production model (through_date={through_date}, --fit-only) ...")
        model = fit_production_model(
            feature_table,
            through_date=through_date,
            fit_fn=fit_fn,
            save_path=args.model_path,
        )
        print(f"      Production model saved -> {args.model_path}")
        print("Done. src/pipeline/refresh.py's default model loader will pick this artifact up.")
        result["model_path"] = args.model_path
        result["model"] = model
        return result

    print(f"[4/5] Running walk-forward (step={args.step}d, min_train_dates={args.min_train_dates}) ...")
    oos_df = run_walk_forward(
        feature_table,
        step=args.step,
        min_train_dates=args.min_train_dates,
        fit_fn=fit_fn,
    )
    print(f"      {len(oos_df)} out-of-sample pitcher-starts evaluated.")
    if oos_df.empty:
        print(
            "Walk-forward produced zero OOS rows -- likely the date range is too short to "
            "clear --min-train-dates distinct training dates. Try a longer range or a lower "
            "--min-train-dates. Exiting without writing a report."
        )
        sys.exit(1)

    oos_path = persist_oos_frame(oos_df, path=args.oos_path)
    print(f"      Persisted OOS frame -> {oos_path}")
    result["oos_rows"] = len(oos_df)
    result["oos_path"] = oos_path

    print(f"[5/5] Generating backtest report (reports_dir={args.reports_dir}) ...")
    report_result = generate_backtest_report(oos_df=oos_df, reports_dir=args.reports_dir)
    print(f"      Report   -> {report_result['report_path']}")
    print(f"      Plots    -> {report_result['reliability_plot_path']}")
    print(f"                  {report_result['calibration_by_tier_plot_path']}")
    print(f"                  {report_result['error_over_time_plot_path']}")
    result["report_result"] = report_result

    if args.no_production_model:
        print("Skipping production-model fit (--no-production-model). Done.")
        return result

    through_date = args.through_date or args.end
    print(f"Fitting go-live production model (through_date={through_date}) ...")
    model = fit_production_model(
        feature_table,
        through_date=through_date,
        fit_fn=fit_fn,
        save_path=args.model_path,
    )
    print(f"      Production model saved -> {args.model_path}")
    print("Done. src/pipeline/refresh.py's default model loader will pick this artifact up.")
    result["model_path"] = args.model_path
    result["model"] = model
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end historical walk-forward backtest against real Statcast data."
    )
    parser.add_argument("--start", required=True, help="Corpus start date, e.g. 2024-04-01")
    parser.add_argument("--end", required=True, help="Corpus end date, e.g. 2024-09-30")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                         help=f"Statcast window cache dir (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                         help=f"Statcast pull window size in days (default: {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--min-batters-faced", type=int, default=None,
                         help="Optional floor to guard filter_starters' max-BF proxy against opener/short-relief misclassification.")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_DAYS,
                         help=f"Walk-forward expanding-window step, in days (default: {DEFAULT_STEP_DAYS})")
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_MIN_TRAIN_DATES,
                         help=f"Minimum distinct training dates before a walk-forward step is attempted (default: {DEFAULT_MIN_TRAIN_DATES})")
    parser.add_argument("--oos-path", default=DEFAULT_OOS_PATH,
                         help=f"Where to persist the wide walk-forward OOS frame (default: {DEFAULT_OOS_PATH})")
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR,
                         help=f"Where to write the backtest report + plots (default: {DEFAULT_REPORTS_DIR})")
    parser.add_argument("--no-production-model", action="store_true",
                         help="Skip the final go-live production-model fit/save step.")
    parser.add_argument("--fit-only", action="store_true",
                         help="Skip the walk-forward evaluation + report (steps 4-5) and only "
                              "refit + save the go-live production model (task #12 weekly retrain). "
                              "Mutually exclusive with --no-production-model."
                         )
    parser.add_argument("--through-date", default=None,
                         help="Production model train-through date (default: --end).")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                         help=f"Where to save the production model artifact (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--variant", default=None,
                         choices=["skill-features", "compression-fix"],
                         help="Feature variant for walk-forward trial. "
                              "'skill-features': adds SKILL_CANDIDATE_COLUMNS as extra regressors "
                              "(spec \u2461, 2026-06-30). "
                              "'compression-fix': replaces k_rate_last5 with k_stab_last5, drops "
                              "opponent_k_rate_last10, adds csw_rate_season candidate "
                              "(spec fix-projection-compression, 2026-06-30).")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.fit_only and args.no_production_model:
        parser.error(
            "--fit-only and --no-production-model are mutually exclusive -- "
            "fitting the production model is the entire point of --fit-only."
        )

    run_backtest_pipeline(args)


if __name__ == "__main__":
    main()
