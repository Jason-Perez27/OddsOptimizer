"""
Daily settlement pipeline (task #10, module 5): pulls realized outcomes for a
date's predicted starters and writes graded results.

Design: docs/design/specs/2026-06-27-outcome-tracking-design.md
(sections 1 "Outcome ingestion & timing" and 3 "Storage"). Mirrors
src/pipeline/refresh.py's shape on purpose -- dependency-injected fetcher,
pure orchestration logic, a write_* step, and a thin CLI -- so the two
pipelines read the same way.

Pure-function boundary (matches refresh.py): `run_settlement()` takes every
tunable (`now`, `scratch_void_max_wait_days`, `settlement_lag_hours`) as an
explicit keyword argument and never reads configs/config.yaml itself, so it
stays unit-testable with hand-built fixtures and no filesystem/network beyond
the injected `outcome_fetcher`. Only the CLI (`main()`) reads config.yaml,
for defaults -- the same split refresh.py uses (module-level constants there,
a config read here only because these particular knobs are genuinely
operator-tunable, per the spec).

Status-merge responsibility named in src/evaluation/grading.py's own
docstring: grading.grade_line_picks / grade_threshold_sweep deliberately
don't assign the 3-way `settlement_status` (line_picks_df has no game_date
to time from). This module is where that gets resolved -- one
attach_outcomes() call against predictions.csv (which does have game_date)
produces the authoritative status per (pitcher, game_pk), then that status
(and game_date, for line_picks which lacks it) is merged onto both graded
frames before they're written, to match the graded_line_picks.csv /
graded_threshold_sweep.csv schemas in the spec exactly.
"""

import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from src.data.pitcher_logs import get_pitcher_logs_by_id
from src.evaluation import grading
from src.features.game_logs import aggregate_pitcher_games, OUTPUT_COLUMNS as GAME_LOG_COLUMNS
from src.predictions.tiering import LINE_PICKS_COLUMNS, THRESHOLD_TABLE_COLUMNS
from src.props import DEFAULT_PROP

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
DEFAULT_SCRATCH_VOID_MAX_WAIT_DAYS = 3
DEFAULT_SETTLEMENT_LAG_HOURS = 4

PREDICTIONS_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "pitcher_team", "opponent_team",
    "game_date", "family", "mu", "alpha",
]

GRADED_LINE_PICKS_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "team", "game_date", "line",
    "line_threshold", "lean", "tier", "p_over", "p_under", "edge",
    "edge_vs_coinflip", "p_market", "over_payout_multiplier", "under_payout_multiplier",
    "push_mass", "pulled_at", "realized_strikeouts", "settlement_status",
    "over_hit", "push", "pick_correct", "pnl_units",
]

GRADED_THRESHOLD_SWEEP_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "game_date", "threshold", "p_over",
    "tier", "realized_strikeouts", "settlement_status", "over_hit",
]


# ---------------------------------------------------------------------------
# Default (live) outcome fetcher -- network-touching, never exercised by tests
# ---------------------------------------------------------------------------

def default_outcome_fetcher(pitcher_id, game_date: str) -> pd.DataFrame:
    """
    Pull pitch-level Statcast rows for one pitcher on one calendar date, by
    MLBAM id (see src/data/pitcher_logs.get_pitcher_logs_by_id's own
    docstring -- written specifically for this call site). A single-day
    [game_date, game_date] window naturally covers a doubleheader (both
    game_pk rows for that pitcher on that date come back together;
    aggregate_pitcher_games splits them apart again by game_pk).
    """
    return get_pitcher_logs_by_id(pitcher_id, game_date, game_date)


# ---------------------------------------------------------------------------
# Reading a day's refresh output (the input side of settlement)
# ---------------------------------------------------------------------------

def _predictions_dir(processed_dir: str, game_date: str, prop: str = DEFAULT_PROP) -> str:
    """Return the predictions partition directory for a given date and prop.

    For the default prop ("strikeouts") the flat game_date=*/ path is used for
    backward compatibility with existing partitions and tests.  Non-default
    props live under game_date=*/prop={key}/.
    """
    base = os.path.join(processed_dir, "predictions", f"game_date={game_date}")
    if prop == DEFAULT_PROP:
        return base
    return os.path.join(base, f"prop={prop}")


def read_predictions_partition(processed_dir: str, game_date: str, prop: str = DEFAULT_PROP) -> dict | None:
    """
    Read predictions.csv / threshold_table.csv / line_picks.csv written by
    src/pipeline/refresh.py for this date. Returns None if the partition
    doesn't exist at all -- the "predictions partition missing" edge case
    (spec): a clean no-op, not an exception.
    """
    pred_dir = _predictions_dir(processed_dir, game_date, prop)
    if not os.path.isdir(pred_dir):
        return None

    def _read(name, columns):
        path = os.path.join(pred_dir, name)
        if not os.path.exists(path):
            return pd.DataFrame(columns=columns)
        return pd.read_csv(path)

    return {
        "predictions": _read("predictions.csv", PREDICTIONS_COLUMNS),
        "threshold_table": _read("threshold_table.csv", THRESHOLD_TABLE_COLUMNS),
        "line_picks": _read("line_picks.csv", LINE_PICKS_COLUMNS),
    }


# ---------------------------------------------------------------------------
# Realized-outcome assembly
# ---------------------------------------------------------------------------

def fetch_realized_outcomes(predictions_df: pd.DataFrame, game_date: str, outcome_fetcher) -> tuple:
    """
    Call `outcome_fetcher(pitcher_id, game_date)` once per unique predicted
    pitcher and aggregate the results into the (pitcher, game_pk,
    strikeouts) shape grading.py expects.

    A single pitcher's fetch failing (network error, etc.) does not abort
    the run -- that pitcher is simply treated as unsettled for this pass
    (resolves on a later settle re-run) and the error is recorded for the
    manifest, the same partial-failure spirit as refresh.py's
    skipped_pitchers.

    Returns (realized_df, fetch_errors).
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=GAME_LOG_COLUMNS), []

    fetch_errors = []
    pitch_level_frames = []
    for pitcher_id in predictions_df["pitcher"].unique():
        try:
            pitches = outcome_fetcher(pitcher_id, game_date)
        except Exception as exc:
            fetch_errors.append({"pitcher": pitcher_id, "reason": f"outcome fetch failed: {exc}"})
            continue
        if pitches is not None and not pitches.empty:
            pitch_level_frames.append(pitches)

    realized_df = (
        aggregate_pitcher_games(pd.concat(pitch_level_frames, ignore_index=True))
        if pitch_level_frames else pd.DataFrame(columns=GAME_LOG_COLUMNS)
    )
    return realized_df, fetch_errors


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_settlement(
    game_date: str,
    *,
    prop: str = DEFAULT_PROP,
    outcome_fetcher=None,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    now=None,
    scratch_void_max_wait_days: float = DEFAULT_SCRATCH_VOID_MAX_WAIT_DAYS,
    settlement_lag_hours: float = DEFAULT_SETTLEMENT_LAG_HOURS,
) -> dict:
    """
    Settle one date's predictions against realized outcomes. Returns a
    results dict: {"game_date", "prop", "graded_line_picks",
    "graded_threshold_sweep", "manifest"}.

    Edge cases handled (spec's numbered list):
    - Missing predictions partition for `game_date` -> clean no-op: empty
      graded frames, manifest notes "no predictions found", no exception.
    - Missing `game_pk` on a predictions-shaped table -> fails fast (the
      ValueError raised inside grading.py's _require_game_pk propagates
      unchanged -- this indicates an upstream task #9 regression, not
      something settle.py should paper over).
    - Realized-but-unpredicted pitcher -> never enters the pipeline at all
      (outcomes are only fetched for predicted pitchers).
    - Doubleheader -> graded independently by game_pk (grading.py's join key;
      default_outcome_fetcher's single-day window returns both games for
      that pitcher/date together, split apart again by aggregate_pitcher_games).
    """
    outcome_fetcher = outcome_fetcher or default_outcome_fetcher
    now = now if now is not None else datetime.now(timezone.utc)

    partition = read_predictions_partition(processed_dir, game_date, prop)
    if partition is None:
        manifest = {
            "game_date": game_date,
            "prop": prop,
            "settled_at": pd.Timestamp(now).isoformat(),
            "note": f"no predictions partition found for {game_date} -- nothing to settle",
            "n_predicted": 0, "n_settled": 0, "n_pending": 0,
            "n_void_scratched": 0, "n_picks_graded": 0,
            "fetch_errors": [],
        }
        return {
            "game_date": game_date,
            "prop": prop,
            "graded_line_picks": pd.DataFrame(columns=GRADED_LINE_PICKS_COLUMNS),
            "graded_threshold_sweep": pd.DataFrame(columns=GRADED_THRESHOLD_SWEEP_COLUMNS),
            "manifest": manifest,
        }

    predictions_df = partition["predictions"]
    threshold_table_df = partition["threshold_table"]
    line_picks_df = partition["line_picks"]

    realized_df, fetch_errors = fetch_realized_outcomes(predictions_df, game_date, outcome_fetcher)

    attached = grading.attach_outcomes(
        predictions_df, realized_df,
        now=now, scratch_void_max_wait_days=scratch_void_max_wait_days,
        settlement_lag_hours=settlement_lag_hours,
    )
    status_map = attached[["pitcher", "game_pk", "settlement_status"]].drop_duplicates()
    date_map = attached[["pitcher", "game_pk", "game_date"]].drop_duplicates()

    graded_sweep = grading.grade_threshold_sweep(threshold_table_df, realized_df)
    graded_sweep = graded_sweep.merge(status_map, on=["pitcher", "game_pk"], how="left")
    if not graded_sweep.empty:
        for col in GRADED_THRESHOLD_SWEEP_COLUMNS:
            if col not in graded_sweep.columns:
                graded_sweep[col] = pd.NA
        graded_sweep = graded_sweep[GRADED_THRESHOLD_SWEEP_COLUMNS]
    else:
        graded_sweep = pd.DataFrame(columns=GRADED_THRESHOLD_SWEEP_COLUMNS)

    graded_picks = grading.grade_line_picks(line_picks_df, realized_df)
    graded_picks = graded_picks.merge(status_map, on=["pitcher", "game_pk"], how="left")
    graded_picks = graded_picks.merge(date_map, on=["pitcher", "game_pk"], how="left")
    if not graded_picks.empty:
        for col in GRADED_LINE_PICKS_COLUMNS:
            if col not in graded_picks.columns:
                graded_picks[col] = pd.NA
        graded_picks = graded_picks[GRADED_LINE_PICKS_COLUMNS]
    else:
        graded_picks = pd.DataFrame(columns=GRADED_LINE_PICKS_COLUMNS)

    status_counts = attached["settlement_status"].value_counts() if not attached.empty else pd.Series(dtype=int)
    manifest = {
        "game_date": game_date,
        "prop": prop,
        "settled_at": pd.Timestamp(now).isoformat(),
        "note": None,
        "n_predicted": int(len(predictions_df)),
        "n_settled": int(status_counts.get(grading.SETTLED, 0)),
        "n_pending": int(status_counts.get(grading.PENDING, 0)),
        "n_void_scratched": int(status_counts.get(grading.VOID_SCRATCHED, 0)),
        "n_picks_graded": int(len(graded_picks)),
        "fetch_errors": fetch_errors,
    }

    return {
        "game_date": game_date,
        "prop": prop,
        "graded_line_picks": graded_picks,
        "graded_threshold_sweep": graded_sweep,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_graded(results: dict, processed_dir: str = DEFAULT_PROCESSED_DIR, overwrite: bool = True) -> str:
    """
    Write graded_line_picks.csv / graded_threshold_sweep.csv /
    settle_manifest.json to {processed_dir}/outcomes/game_date=YYYY-MM-DD/
    (or .../prop={key}/ for non-default props) -- mirrors refresh.py's
    write_outputs partition/overwrite conventions exactly.

    Re-settling a date OVERWRITES by default (idempotent -- re-running
    settlement for the same date with newer Statcast data should replace,
    not duplicate). `overwrite=False` raises FileExistsError if this date's
    manifest already exists.
    """
    game_date = results["game_date"]
    prop = results.get("prop", DEFAULT_PROP)
    base_dir = os.path.join(processed_dir, "outcomes", f"game_date={game_date}")
    out_dir = base_dir if prop == DEFAULT_PROP else os.path.join(base_dir, f"prop={prop}")
    manifest_path = os.path.join(out_dir, "settle_manifest.json")

    if not overwrite and os.path.exists(manifest_path):
        raise FileExistsError(
            f"Settlement output already exists for {game_date} at {out_dir} -- "
            f"pass overwrite=True to replace it."
        )

    os.makedirs(out_dir, exist_ok=True)

    results["graded_line_picks"].to_csv(os.path.join(out_dir, "graded_line_picks.csv"), index=False)
    results["graded_threshold_sweep"].to_csv(os.path.join(out_dir, "graded_threshold_sweep.csv"), index=False)

    with open(manifest_path, "w") as f:
        json.dump(results["manifest"], f, indent=2, default=str)

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def window_days_to_dates(window_days: int, now=None) -> list:
    """
    Expand `--window-days N` into the trailing N-day window ending
    yesterday: [today - N, today - 1] inclusive, ascending. Takes the same
    injectable `now` clock run_settlement() already accepts, so the
    expansion is unit-testable without depending on the real wall clock
    (task #12 go-live spec, "settle.py -- add --window-days N").
    """
    now = now if now is not None else datetime.now(timezone.utc)
    today = pd.Timestamp(now).date()
    return [
        (today - timedelta(days=offset)).isoformat()
        for offset in range(window_days, 0, -1)
    ]


def _load_evaluation_config(config_path: str = os.path.join("configs", "config.yaml")) -> dict:
    """
    Read the `evaluation:` block from config.yaml for CLI defaults only --
    run_settlement() itself never calls this (see module docstring on the
    pure-function boundary). Missing file or missing keys fall back to this
    module's own DEFAULT_* constants rather than raising -- a config file is
    a convenience here, not a hard dependency.
    """
    defaults = {
        "settlement_lag_hours": DEFAULT_SETTLEMENT_LAG_HOURS,
        "scratch_void_max_wait_days": DEFAULT_SCRATCH_VOID_MAX_WAIT_DAYS,
    }
    if not os.path.exists(config_path):
        return defaults
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        evaluation = cfg.get("evaluation", {}) or {}
        defaults.update({k: evaluation[k] for k in defaults if k in evaluation})
    except Exception:
        pass
    return defaults


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Settle predictions against realized outcomes.")
    parser.add_argument("--date", default=None, help="Single game date YYYY-MM-DD to settle")
    parser.add_argument("--from", dest="from_date", default=None, help="Start of a date range (inclusive)")
    parser.add_argument("--to", dest="to_date", default=None, help="End of a date range (inclusive)")
    parser.add_argument(
        "--window-days", type=int, default=None,
        help="Settle the trailing N days ending yesterday: [today-N, today-1] inclusive. "
             "Mutually exclusive with --date and --from/--to.",
    )
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Abort instead of overwriting an existing settlement for a date",
    )
    args = parser.parse_args()

    if args.window_days is not None and (args.date or args.from_date or args.to_date):
        parser.error(
            "--window-days is mutually exclusive with --date and --from/--to -- "
            "pick one way to select dates."
        )

    if args.window_days is not None:
        dates = window_days_to_dates(args.window_days)
    elif args.from_date and args.to_date:
        start = pd.Timestamp(args.from_date)
        end = pd.Timestamp(args.to_date)
        dates = [
            (start + timedelta(days=i)).date().isoformat()
            for i in range((end - start).days + 1)
        ]
    else:
        dates = [args.date or date.today().isoformat()]

    config = _load_evaluation_config()

    for game_date in dates:
        results = run_settlement(
            game_date,
            processed_dir=args.processed_dir,
            scratch_void_max_wait_days=config["scratch_void_max_wait_days"],
            settlement_lag_hours=config["settlement_lag_hours"],
        )
        manifest = results["manifest"]
        if manifest.get("note"):
            print(f"{game_date}: {manifest['note']}")
            continue

        out_dir = write_graded(results, processed_dir=args.processed_dir, overwrite=not args.no_overwrite)
        print(
            f"Settled {game_date}: {manifest['n_settled']} settled, "
            f"{manifest['n_pending']} pending, {manifest['n_void_scratched']} void/scratched "
            f"-> {out_dir}"
        )
        if manifest["fetch_errors"]:
            print(f"  WARNING: {len(manifest['fetch_errors'])} outcome fetch error(s) -- see settle_manifest.json")


if __name__ == "__main__":
    main()
