"""
Backtest report generation for task #10 (spec section 6) and task #11 step 6
(historical-backtest report).

Design (live report): docs/design/specs/2026-06-27-outcome-tracking-design.md,
section 6 "src/backtest/report.py". A thin layer over src/evaluation/metrics.py and
src/backtest/roi.py -- those return numbers/tables and are independently
testable; this module's only job is to (a) glob settled outcome partitions
and concatenate them into the cumulative frames the spec calls for ("no
separate rolling ledger" -- section 3), (b) call the metrics/roi helpers on
that combined data, and (c) render the result as a markdown report plus two
PNGs (reliability diagram, cumulative-ROI curve).

Design (historical-backtest report): docs/design/specs/2026-06-27-
baseline-validation-design.md, section 3 "Metrics & reporting". A SIBLING to
the live results report above -- distinct filename
(`reports/YYYY-MM-DD-baseline-backtest.md`, never `results.md`), its own
load/build/render/plot functions, but sharing the same metrics.py +
matplotlib plumbing. The two tracks never visually merge: Track A (the
backtest report) is model calibration on historical completed games; Track B
(the live results report) is live betting performance. No line/edge/ROI
field ever appears in the backtest report, by construction
(src.backtest.walk_forward emits none).

On-demand cadence (spec): no scheduler here, just `main()` -- run it whenever
you want a fresh report.

Kept pure where it matters: `build_report()` / `build_backtest_report()` take
already-loaded frames and return a plain dict of numbers/tables -- no file
IO, fully unit-testable with hand-built fixtures. Only the `load_*` functions
(read CSVs) and `generate_*report()`/the plot functions (write files) touch
disk.
"""

import os
from datetime import date

import matplotlib

matplotlib.use("Agg")  # no display in this pipeline -- write PNGs, never show()
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import roi
from src.backtest.walk_forward import _oos_columns, melt_oos_sweep
from src.evaluation import metrics
from src.models.baseline_model import THRESHOLDS
from src.pipeline.settle import GRADED_LINE_PICKS_COLUMNS, GRADED_THRESHOLD_SWEEP_COLUMNS

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_ROLLING_WINDOW = 50
DEFAULT_N_BUCKETS = 10


# ---------------------------------------------------------------------------
# Loading: glob settled partitions, concatenate (spec section 3: cumulative
# metrics computed by globbing partitions at report time)
# ---------------------------------------------------------------------------

def discover_outcome_dates(processed_dir: str = DEFAULT_PROCESSED_DIR) -> list:
    """Sorted list of game_date strings with a written outcomes partition."""
    outcomes_root = os.path.join(processed_dir, "outcomes")
    if not os.path.isdir(outcomes_root):
        return []
    dates = []
    for name in os.listdir(outcomes_root):
        if name.startswith("game_date="):
            dates.append(name[len("game_date="):])
    return sorted(dates)


def load_graded_frames(processed_dir: str = DEFAULT_PROCESSED_DIR, game_dates: list = None) -> tuple:
    """
    Concatenate graded_line_picks.csv / graded_threshold_sweep.csv across
    every settled partition (or just `game_dates`, if given). A partition
    missing one of the two files contributes an empty frame for that file
    rather than aborting the whole load.
    """
    dates = game_dates if game_dates is not None else discover_outcome_dates(processed_dir)

    picks_frames = []
    sweep_frames = []
    for game_date in dates:
        part_dir = os.path.join(processed_dir, "outcomes", f"game_date={game_date}")
        picks_path = os.path.join(part_dir, "graded_line_picks.csv")
        sweep_path = os.path.join(part_dir, "graded_threshold_sweep.csv")
        if os.path.exists(picks_path):
            picks_frames.append(pd.read_csv(picks_path))
        if os.path.exists(sweep_path):
            sweep_frames.append(pd.read_csv(sweep_path))

    picks = pd.concat(picks_frames, ignore_index=True) if picks_frames else pd.DataFrame(columns=GRADED_LINE_PICKS_COLUMNS)
    sweep = pd.concat(sweep_frames, ignore_index=True) if sweep_frames else pd.DataFrame(columns=GRADED_THRESHOLD_SWEEP_COLUMNS)
    return picks, sweep


# ---------------------------------------------------------------------------
# Pure metric/roi aggregation -- no file IO, fully unit-testable
# ---------------------------------------------------------------------------

def build_report(
    graded_line_picks_df: pd.DataFrame,
    graded_sweep_df: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    n_buckets: int = DEFAULT_N_BUCKETS,
) -> dict:
    """
    Run every model-honesty + pick-profitability helper over the combined
    (cumulative) graded frames and return a single dict of numbers/tables.
    Sample-size visibility by tier (spec section 5's stated revisit-trigger
    floor: >=100 settled per tier) comes for free from `by_tier`'s `n_settled`
    column -- no separate logic needed here, just don't drop it from the
    report.

    Defensively reindexed onto the canonical graded-frame columns first:
    metrics.py's helpers index columns like `over_hit` directly, which
    raises a KeyError on a bare, columnless `pd.DataFrame()` (as opposed to
    one with 0 rows but the right columns, which is what every real
    settle.py partition produces). Reindexing a frame that already has all
    the columns is a no-op; reindexing a truly empty one just gives back an
    empty frame with the right (0-row) columns.
    """
    graded_line_picks_df = graded_line_picks_df.reindex(columns=GRADED_LINE_PICKS_COLUMNS) \
        if not set(GRADED_LINE_PICKS_COLUMNS).issubset(graded_line_picks_df.columns) else graded_line_picks_df
    graded_sweep_df = graded_sweep_df.reindex(columns=GRADED_THRESHOLD_SWEEP_COLUMNS) \
        if not set(GRADED_THRESHOLD_SWEEP_COLUMNS).issubset(graded_sweep_df.columns) else graded_sweep_df

    reliability = metrics.reliability_table(graded_sweep_df, n_buckets=n_buckets)
    return {
        "n_picks_total": int(len(graded_line_picks_df)),
        "n_sweep_total": int(len(graded_sweep_df)),
        "hit_rate": roi.hit_rate(graded_line_picks_df),
        "flat_bet_roi": roi.flat_bet_roi(graded_line_picks_df),
        "by_tier": roi.by_tier(graded_line_picks_df),
        "by_line": roi.by_line(graded_line_picks_df),
        "time_series": roi.time_series(graded_line_picks_df, rolling_window=rolling_window),
        "reliability_table": reliability,
        "ece": metrics.expected_calibration_error(reliability),
        "brier_log_loss_at_line": metrics.brier_and_log_loss_at_line(graded_line_picks_df),
        "brier_log_loss_at_thresholds": metrics.brier_and_log_loss_at_thresholds(graded_sweep_df),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(x, pct=False, decimals=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if pct:
        return f"{x * 100:.1f}%"
    return f"{x:.{decimals}f}"


def render_markdown(report: dict, as_of_date: str) -> str:
    """Render `build_report()`'s output as a markdown document."""
    hr = report["hit_rate"]
    rr = report["flat_bet_roi"]

    lines = [
        f"# Backtest results -- as of {as_of_date}",
        "",
        f"Settled picks: {hr['n']} (of {report['n_picks_total']} graded rows)",
        f"Hit rate: {_fmt(hr['hit_rate'], pct=True)} ({hr['wins']}W-{hr['losses']}L)",
        f"Flat-bet ROI (proxy, even-money units -- see deferred section): "
        f"{_fmt(rr['roi'], pct=True)} ({_fmt(rr['total_pnl'])} units / {rr['n_settled_non_push']} picks)",
        "",
        "## By tier",
        "",
        "| tier | n_settled | hit_rate | roi |",
        "|---|---|---|---|",
    ]
    for _, row in report["by_tier"].iterrows():
        lines.append(
            f"| {row['tier']} | {row['n_settled']} | {_fmt(row['hit_rate'], pct=True)} | {_fmt(row['roi'], pct=True)} |"
        )

    lines += [
        "",
        "## By line",
        "",
        "| line | n_settled | hit_rate | roi |",
        "|---|---|---|---|",
    ]
    for _, row in report["by_line"].iterrows():
        lines.append(
            f"| {row['line']} | {row['n_settled']} | {_fmt(row['hit_rate'], pct=True)} | {_fmt(row['roi'], pct=True)} |"
        )

    bll = report["brier_log_loss_at_line"]
    lines += [
        "",
        "## Model honesty",
        "",
        f"Expected calibration error (ECE): {_fmt(report['ece'])}",
        f"Brier score at line: {_fmt(bll['brier_score'])} (n={bll['n']})",
        f"Log loss at line: {_fmt(bll['log_loss'])} (n={bll['n']})",
        "",
        "Tiers with fewer than 100 settled picks are not yet reliable enough "
        "to judge -- see decision log / spec section 5 before redefining "
        "confidence-tier cutoffs based on a small sample.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability_diagram(reliability_df: pd.DataFrame, out_path: str) -> str:
    """Predicted vs. empirical hit frequency, one point per bucket, plus the y=x reference line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    non_empty = reliability_df[reliability_df["n"] > 0]
    ax.scatter(non_empty["mean_predicted"], non_empty["empirical_rate"], label="observed buckets")
    ax.set_xlabel("Mean predicted P(over)")
    ax.set_ylabel("Empirical hit rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram")
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cumulative_roi(time_series_df: pd.DataFrame, out_path: str) -> str:
    """Cumulative flat-bet ROI over settled, ordered picks."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if not time_series_df.empty:
        ax.plot(range(len(time_series_df)), time_series_df["cum_roi"], label="cumulative ROI")
        if "rolling_roi" in time_series_df.columns:
            ax.plot(range(len(time_series_df)), time_series_df["rolling_roi"], label="rolling ROI", alpha=0.6)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Settled pick #, ordered by date")
    ax.set_ylabel("ROI (units / pick)")
    ax.set_title("Cumulative ROI")
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# End-to-end generation (live results report -- task #10)
# ---------------------------------------------------------------------------

def generate_report(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    reports_dir: str = DEFAULT_REPORTS_DIR,
    game_dates: list = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    n_buckets: int = DEFAULT_N_BUCKETS,
    as_of: str = None,
) -> dict:
    """
    Load every settled partition (or just `game_dates`), build the report,
    and write `{reports_dir}/{as_of}-results.md` plus
    `{as_of}-reliability.png` / `{as_of}-cumulative-roi.png`.
    """
    as_of = as_of or date.today().isoformat()
    picks, sweep = load_graded_frames(processed_dir, game_dates)
    report = build_report(picks, sweep, rolling_window=rolling_window, n_buckets=n_buckets)

    os.makedirs(reports_dir, exist_ok=True)
    reliability_path = os.path.join(reports_dir, f"{as_of}-reliability.png")
    roi_path = os.path.join(reports_dir, f"{as_of}-cumulative-roi.png")
    plot_reliability_diagram(report["reliability_table"], reliability_path)
    plot_cumulative_roi(report["time_series"], roi_path)

    md = render_markdown(report, as_of)
    report_path = os.path.join(reports_dir, f"{as_of}-results.md")
    with open(report_path, "w") as f:
        f.write(md)

    return {
        "report_path": report_path,
        "reliability_plot_path": reliability_path,
        "roi_plot_path": roi_path,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Historical backtest report -- task #11 step 6 (spec section 3 / "Metrics &
# reporting"). A SIBLING to the live results report above: distinct
# filename (`baseline-backtest.md`, never `results.md`), its own load/build/
# render/plot functions, but sharing the same metrics.py + matplotlib
# plumbing. The two tracks never visually merge -- Track A (this report) is
# model calibration on historical completed games; Track B (generate_report
# above) is live betting performance. No line/edge/ROI field ever appears
# here, by construction (src.backtest.walk_forward emits none).
# ---------------------------------------------------------------------------

DEFAULT_OOS_DIR = os.path.join("data", "processed", "backtest")
DEFAULT_OOS_PATH = os.path.join(DEFAULT_OOS_DIR, "walk_forward_oos.csv")


def persist_oos_frame(oos_df: pd.DataFrame, path: str = DEFAULT_OOS_PATH) -> str:
    """Write the wide walk-forward OOS frame to CSV (spec section 3 / output schema)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    oos_df.to_csv(path, index=False)
    return path


def load_oos_frame(path: str = DEFAULT_OOS_PATH, thresholds=THRESHOLDS) -> pd.DataFrame:
    """
    Read the persisted OOS frame back; a missing file returns an empty,
    correctly-columned frame (matching run_walk_forward's own empty-input
    convention) rather than raising.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=_oos_columns(thresholds))
    return pd.read_csv(path)


def build_backtest_report(
    oos_df: pd.DataFrame,
    n_buckets: int = DEFAULT_N_BUCKETS,
    thresholds=THRESHOLDS,
) -> dict:
    """
    Run every model-honesty helper over the wide walk-forward OOS frame and
    return a single dict of numbers/tables -- no file IO, fully
    unit-testable with a hand-built fixture (testing item #13: "the
    backtest report's metric assembly returns the dict/tables tests assert
    on; markdown/plots are a thin layer").

    `oos_df` is the wide schema `run_walk_forward`/`load_oos_frame` produce
    (one row per pitcher-start). Internally melted to the long,
    one-row-per-(pitcher-game, threshold) shape via `melt_oos_sweep` for the
    threshold/tier/time-indexed helpers (reliability, ECE,
    brier_log_loss_at_thresholds, by_sweep_tier, over_time); `point_accuracy`
    and `pit_histogram` use the wide frame directly since they need
    `mu`/`family`/`alpha`, which the long melt drops.

    `pit_histogram` is PIT-for-counts and needs scipy; if scipy is
    unavailable, "pit_histogram" is None in the result rather than raising
    -- the rest of the report still renders.
    """
    long_df = melt_oos_sweep(oos_df, thresholds=thresholds)
    reliability = metrics.reliability_table(long_df, n_buckets=n_buckets)

    try:
        pit = metrics.pit_histogram(oos_df, n_buckets=n_buckets)
    except ImportError:
        pit = None

    return {
        "n_oos_total": int(len(oos_df)),
        "reliability_table": reliability,
        "ece": metrics.expected_calibration_error(reliability),
        "brier_log_loss_at_thresholds": metrics.brier_and_log_loss_at_thresholds(
            long_df, thresholds=metrics.REPRESENTATIVE_THRESHOLDS
        ),
        "by_sweep_tier": metrics.by_sweep_tier(long_df),
        "over_time": metrics.over_time(long_df),
        "point_accuracy": metrics.point_accuracy(oos_df),
        "pit_histogram": pit,
    }


def render_backtest_markdown(report: dict, as_of_date: str) -> str:
    """
    Render `build_backtest_report()`'s output as markdown. The header states
    plainly what this is NOT, per spec section 3: model calibration on
    historical completed games, not betting performance, no lines involved.
    """
    pa = report["point_accuracy"]

    lines = [
        f"# Baseline backtest -- as of {as_of_date}",
        "",
        "**Model calibration on historical completed games -- not betting "
        "performance, no lines involved.** This is Track A (model honesty); "
        "see the live results report (`YYYY-MM-DD-results.md`) for Track B "
        "(pick profitability), a separate, forward-only track.",
        "",
        f"Out-of-sample starts evaluated: {report['n_oos_total']}",
        f"Point accuracy: MAE {_fmt(pa['mae'])}, RMSE {_fmt(pa['rmse'])} (n={pa['n']})",
        f"Expected calibration error (ECE): {_fmt(report['ece'])}",
        "",
        "## Brier / log loss at representative thresholds",
        "",
        "| threshold | n | brier_score | log_loss |",
        "|---|---|---|---|",
    ]
    for t, vals in report["brier_log_loss_at_thresholds"].items():
        lines.append(f"| {t} | {vals['n']} | {_fmt(vals['brier_score'])} | {_fmt(vals['log_loss'])} |")

    lines += [
        "",
        "## By sweep tier",
        "",
        "| tier | n | mean_predicted | empirical_rate | brier_score | log_loss |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in report["by_sweep_tier"].iterrows():
        lines.append(
            f"| {row['tier']} | {row['n']} | {_fmt(row['mean_predicted'])} | "
            f"{_fmt(row['empirical_rate'])} | {_fmt(row['brier_score'])} | {_fmt(row['log_loss'])} |"
        )

    lines += [
        "",
        "## Over time (by walk-forward step)",
        "",
        "| wf_step | n | mean_predicted | empirical_rate | brier_score | log_loss |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in report["over_time"].iterrows():
        lines.append(
            f"| {row['wf_step']} | {row['n']} | {_fmt(row['mean_predicted'])} | "
            f"{_fmt(row['empirical_rate'])} | {_fmt(row['brier_score'])} | {_fmt(row['log_loss'])} |"
        )

    if report["pit_histogram"] is not None:
        lines += [
            "",
            "## PIT histogram (probability integral transform, for counts)",
            "",
            "A well-calibrated model's PIT histogram is approximately flat. "
            "Systematic skew signals mis-calibration.",
            "",
            "| bucket_lo | bucket_hi | n |",
            "|---|---|---|",
        ]
        for _, row in report["pit_histogram"].iterrows():
            lines.append(f"| {_fmt(row['bucket_lo'])} | {_fmt(row['bucket_hi'])} | {row['n']} |")
    else:
        lines += ["", "_PIT histogram unavailable (scipy not installed)._"]

    lines.append("")
    return "\n".join(lines)


def plot_calibration_by_tier(by_tier_df: pd.DataFrame, out_path: str) -> str:
    """Mean predicted vs. empirical hit rate, one bar pair per sweep tier."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if not by_tier_df.empty:
        x = list(range(len(by_tier_df)))
        width = 0.35
        ax.bar([i - width / 2 for i in x], by_tier_df["mean_predicted"], width, label="mean predicted")
        ax.bar([i + width / 2 for i in x], by_tier_df["empirical_rate"], width, label="empirical rate")
        ax.set_xticks(x)
        ax.set_xticklabels(by_tier_df["tier"])
    ax.set_ylabel("P(over)")
    ax.set_title("Calibration by sweep tier")
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_error_over_time(over_time_df: pd.DataFrame, out_path: str) -> str:
    """
    Brier score by walk-forward step -- shows whether calibration holds or
    drifts as the season progresses (spec section 3).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    if not over_time_df.empty:
        ax.plot(over_time_df["wf_step"], over_time_df["brier_score"], marker="o")
    ax.set_xlabel("Walk-forward step")
    ax.set_ylabel("Brier score")
    ax.set_title("Calibration error over time")
    ax.tick_params(axis="x", rotation=45)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_backtest_report(
    oos_df: pd.DataFrame = None,
    oos_path: str = DEFAULT_OOS_PATH,
    reports_dir: str = DEFAULT_REPORTS_DIR,
    n_buckets: int = DEFAULT_N_BUCKETS,
    thresholds=THRESHOLDS,
    as_of: str = None,
) -> dict:
    """
    Load the OOS frame (from `oos_df` if given, else `load_oos_frame(oos_path)`),
    build the report, and write `{reports_dir}/{as_of}-baseline-backtest.md`
    plus `{as_of}-backtest-reliability.png`,
    `{as_of}-backtest-calibration-by-tier.png`, and
    `{as_of}-backtest-error-over-time.png`. Distinct filenames from
    `generate_report()`'s live-report outputs by construction -- the two
    tracks never visually merge.
    """
    as_of = as_of or date.today().isoformat()
    if oos_df is None:
        oos_df = load_oos_frame(oos_path, thresholds=thresholds)

    report = build_backtest_report(oos_df, n_buckets=n_buckets, thresholds=thresholds)

    os.makedirs(reports_dir, exist_ok=True)
    reliability_path = os.path.join(reports_dir, f"{as_of}-backtest-reliability.png")
    tier_path = os.path.join(reports_dir, f"{as_of}-backtest-calibration-by-tier.png")
    time_path = os.path.join(reports_dir, f"{as_of}-backtest-error-over-time.png")
    plot_reliability_diagram(report["reliability_table"], reliability_path)
    plot_calibration_by_tier(report["by_sweep_tier"], tier_path)
    plot_error_over_time(report["over_time"], time_path)

    md = render_backtest_markdown(report, as_of)
    report_path = os.path.join(reports_dir, f"{as_of}-baseline-backtest.md")
    with open(report_path, "w") as f:
        f.write(md)

    return {
        "report_path": report_path,
        "reliability_plot_path": reliability_path,
        "calibration_by_tier_plot_path": tier_path,
        "error_over_time_plot_path": time_path,
        "report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a results or baseline-backtest report.")
    parser.add_argument("--backtest", action="store_true", help="Generate the historical Track-A backtest report instead of the live Track-B results report.")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--oos-path", default=DEFAULT_OOS_PATH, help="(--backtest only) path to the walk-forward OOS CSV.")
    parser.add_argument("--from", dest="from_date", default=None)
    parser.add_argument("--to", dest="to_date", default=None)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--n-buckets", type=int, default=DEFAULT_N_BUCKETS)
    args = parser.parse_args()

    if args.backtest:
        result = generate_backtest_report(
            oos_path=args.oos_path,
            reports_dir=args.reports_dir,
            n_buckets=args.n_buckets,
        )
        print(f"Wrote report: {result['report_path']}")
        print(
            f"Wrote plots: {result['reliability_plot_path']}, "
            f"{result['calibration_by_tier_plot_path']}, {result['error_over_time_plot_path']}"
        )
        return

    game_dates = None
    if args.from_date and args.to_date:
        all_dates = discover_outcome_dates(args.processed_dir)
        game_dates = [d for d in all_dates if args.from_date <= d <= args.to_date]

    result = generate_report(
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        game_dates=game_dates,
        rolling_window=args.rolling_window,
        n_buckets=args.n_buckets,
    )
    print(f"Wrote report: {result['report_path']}")
    print(f"Wrote plots: {result['reliability_plot_path']}, {result['roi_plot_path']}")


if __name__ == "__main__":
    main()
