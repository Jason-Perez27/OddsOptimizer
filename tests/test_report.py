"""
Unit tests for src/backtest/report.py (task #10, spec section 6 + the
"Testing approach" list, item 13: cumulative aggregation across multiple
settled partitions is correct, and the rendered report contains the right
top-line figures).

Covers:
  13. Multi-partition cumulative aggregation: hand-built graded_line_picks /
      graded_threshold_sweep CSVs written across two separate
      game_date=YYYY-MM-DD outcome partitions, loaded and concatenated by
      load_graded_frames(), produce a build_report() whose hit_rate/roi/
      reliability numbers match what you'd get from hand-computing the
      combined (not per-partition) data -- this is the behavior that's
      easy to get wrong (e.g. only reading the latest partition, or
      double-counting).

Plus: load_graded_frames handles a missing outcomes dir / missing CSV inside
a partition without crashing; render_markdown's output contains the expected
top-line numbers as a sanity check (not an exact-string match, which would be
brittle); the plot functions actually write PNG files (smoke test, no pixel
assertions); generate_report's end-to-end file-writing wiring.

No network, no real settle.py/grading.py calls -- every fixture here is a
hand-built graded CSV, matching the rest of this repo's test style
(test_backtest_roi.py).

Run with: pytest tests/test_report.py -v
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import report


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _picks_row(pitcher, game_pk, game_date, line, tier, pick_correct, pnl_units, p_over=0.55):
    over_hit = pick_correct  # lean=over fixtures throughout, kept simple
    return {
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "P", "team": "NYY",
        "game_date": game_date, "line": line, "line_threshold": int(line) + 1,
        "lean": "over", "tier": tier, "p_over": p_over, "p_under": 1 - p_over,
        "edge": 0.05, "push_mass": 0.0, "pulled_at": f"{game_date}T12:00:00Z",
        "realized_strikeouts": 8 if pick_correct else 4,
        "settlement_status": "settled", "over_hit": over_hit, "push": False,
        "pick_correct": pick_correct, "pnl_units": pnl_units,
    }


def _sweep_row(pitcher, game_pk, game_date, threshold, p_over, over_hit):
    return {
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "P",
        "game_date": game_date, "threshold": threshold, "p_over": p_over,
        "tier": "low", "realized_strikeouts": 8 if over_hit else 4,
        "settlement_status": "settled", "over_hit": over_hit,
    }


def _write_partition(processed_dir, game_date, picks_rows, sweep_rows):
    part_dir = os.path.join(processed_dir, "outcomes", f"game_date={game_date}")
    os.makedirs(part_dir, exist_ok=True)
    pd.DataFrame(picks_rows).to_csv(os.path.join(part_dir, "graded_line_picks.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(part_dir, "graded_threshold_sweep.csv"), index=False)
    return part_dir


# ---------------------------------------------------------------------------
# discover_outcome_dates / load_graded_frames
# ---------------------------------------------------------------------------

def test_discover_outcome_dates_missing_dir_returns_empty(tmp_path):
    assert report.discover_outcome_dates(str(tmp_path)) == []


def test_discover_outcome_dates_sorted(tmp_path):
    _write_partition(str(tmp_path), "2026-06-21", [], [])
    _write_partition(str(tmp_path), "2026-06-19", [], [])
    _write_partition(str(tmp_path), "2026-06-20", [], [])
    assert report.discover_outcome_dates(str(tmp_path)) == ["2026-06-19", "2026-06-20", "2026-06-21"]


def test_load_graded_frames_missing_outcomes_dir_returns_empty_frames(tmp_path):
    picks, sweep = report.load_graded_frames(str(tmp_path))
    assert picks.empty
    assert sweep.empty


def test_load_graded_frames_partition_missing_one_csv_does_not_crash(tmp_path):
    # Write only graded_line_picks.csv for this partition, no sweep file.
    part_dir = os.path.join(str(tmp_path), "outcomes", "game_date=2026-06-20")
    os.makedirs(part_dir, exist_ok=True)
    pd.DataFrame([_picks_row(1, 100, "2026-06-20", 6.5, "low", True, 1.0)]).to_csv(
        os.path.join(part_dir, "graded_line_picks.csv"), index=False
    )

    picks, sweep = report.load_graded_frames(str(tmp_path))
    assert len(picks) == 1
    assert sweep.empty


# ---------------------------------------------------------------------------
# 13. Multi-partition cumulative aggregation is correct
# ---------------------------------------------------------------------------

def test_load_graded_frames_concatenates_across_partitions(tmp_path):
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    _write_partition(
        str(tmp_path), "2026-06-20",
        [_picks_row(2, 200, "2026-06-20", 5.5, "high", False, -1.0)],
        [_sweep_row(2, 200, "2026-06-20", 6, 0.75, False)],
    )

    picks, sweep = report.load_graded_frames(str(tmp_path))
    assert len(picks) == 2
    assert len(sweep) == 2
    assert set(picks["game_date"]) == {"2026-06-19", "2026-06-20"}


def test_build_report_hit_rate_and_roi_are_combined_not_per_partition(tmp_path):
    # Partition 1: 1 win. Partition 2: 1 win, 1 loss. Combined: 2W-1L,
    # hit_rate = 2/3, total_pnl = 1 + 1 - 1 = 1, roi = 1/3.
    # Reading only the latest partition would wrongly give hit_rate=0.5.
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    _write_partition(
        str(tmp_path), "2026-06-20",
        [
            _picks_row(2, 200, "2026-06-20", 5.5, "high", True, 1.0),
            _picks_row(3, 300, "2026-06-20", 5.5, "high", False, -1.0),
        ],
        [
            _sweep_row(2, 200, "2026-06-20", 6, 0.75, True),
            _sweep_row(3, 300, "2026-06-20", 6, 0.75, False),
        ],
    )

    picks, sweep = report.load_graded_frames(str(tmp_path))
    built = report.build_report(picks, sweep)

    assert built["hit_rate"]["wins"] == 2
    assert built["hit_rate"]["losses"] == 1
    assert built["hit_rate"]["n"] == 3
    assert built["hit_rate"]["hit_rate"] == pytest.approx(2 / 3)
    assert built["flat_bet_roi"]["total_pnl"] == pytest.approx(1.0)
    assert built["flat_bet_roi"]["roi"] == pytest.approx(1 / 3)
    assert built["n_picks_total"] == 3
    assert built["n_sweep_total"] == 3


def test_build_report_by_tier_reflects_combined_partitions(tmp_path):
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    _write_partition(
        str(tmp_path), "2026-06-20",
        [_picks_row(2, 200, "2026-06-20", 5.5, "low", False, -1.0)],
        [_sweep_row(2, 200, "2026-06-20", 6, 0.75, False)],
    )

    picks, sweep = report.load_graded_frames(str(tmp_path))
    built = report.build_report(picks, sweep)

    by_tier = built["by_tier"].set_index("tier")
    assert by_tier.loc["low", "n_settled"] == 2
    assert by_tier.loc["low", "hit_rate"] == pytest.approx(0.5)


def test_build_report_time_series_length_spans_all_partitions(tmp_path):
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    _write_partition(
        str(tmp_path), "2026-06-20",
        [_picks_row(2, 200, "2026-06-20", 5.5, "low", True, 1.0)],
        [_sweep_row(2, 200, "2026-06-20", 6, 0.75, True)],
    )

    picks, sweep = report.load_graded_frames(str(tmp_path))
    built = report.build_report(picks, sweep, rolling_window=5)

    ts = built["time_series"]
    assert len(ts) == 2
    assert list(ts["cum_n"]) == [1, 2]
    assert "rolling_roi" in ts.columns


def test_build_report_game_dates_filter_excludes_other_partitions(tmp_path):
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    _write_partition(
        str(tmp_path), "2026-06-20",
        [_picks_row(2, 200, "2026-06-20", 5.5, "low", False, -1.0)],
        [_sweep_row(2, 200, "2026-06-20", 6, 0.75, False)],
    )

    picks, sweep = report.load_graded_frames(str(tmp_path), game_dates=["2026-06-19"])
    assert len(picks) == 1
    assert picks.iloc[0]["game_date"] == "2026-06-19"


# ---------------------------------------------------------------------------
# build_report on fully empty input -- never crashes
# ---------------------------------------------------------------------------

def test_build_report_empty_input_does_not_crash():
    built = report.build_report(pd.DataFrame(), pd.DataFrame())
    assert built["n_picks_total"] == 0
    assert built["hit_rate"]["n"] == 0
    assert pd.isna(built["hit_rate"]["hit_rate"])
    assert pd.isna(built["flat_bet_roi"]["roi"])
    assert pd.isna(built["ece"])


# ---------------------------------------------------------------------------
# render_markdown -- sanity check on content, not exact string match
# ---------------------------------------------------------------------------

def test_render_markdown_contains_top_line_figures(tmp_path):
    _write_partition(
        str(tmp_path), "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )
    picks, sweep = report.load_graded_frames(str(tmp_path))
    built = report.build_report(picks, sweep)
    md = report.render_markdown(built, "2026-06-21")

    assert "2026-06-21" in md
    assert "Settled picks: 1" in md
    assert "100.0%" in md  # 1-for-1 hit rate
    assert "## By tier" in md
    assert "## By line" in md
    assert "low" in md


def test_render_markdown_handles_empty_report_without_crashing():
    built = report.build_report(pd.DataFrame(), pd.DataFrame())
    md = report.render_markdown(built, "2026-06-21")
    assert "n/a" in md
    assert "Settled picks: 0" in md


# ---------------------------------------------------------------------------
# Plot functions -- smoke tests, file gets written, no pixel assertions
# ---------------------------------------------------------------------------

def test_plot_reliability_diagram_writes_png(tmp_path):
    sweep = pd.DataFrame([_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)])
    from src.evaluation import metrics
    table = metrics.reliability_table(sweep, n_buckets=5)

    out_path = os.path.join(str(tmp_path), "reliability.png")
    report.plot_reliability_diagram(table, out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_cumulative_roi_writes_png_even_for_empty_series(tmp_path):
    empty_ts = pd.DataFrame(columns=report.roi.TIME_SERIES_COLUMNS)
    out_path = os.path.join(str(tmp_path), "roi.png")
    report.plot_cumulative_roi(empty_ts, out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


# ---------------------------------------------------------------------------
# generate_report -- end-to-end file-writing wiring
# ---------------------------------------------------------------------------

def test_generate_report_writes_md_and_pngs(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    reports_dir = os.path.join(str(tmp_path), "reports")
    _write_partition(
        processed_dir, "2026-06-19",
        [_picks_row(1, 100, "2026-06-19", 6.5, "low", True, 1.0)],
        [_sweep_row(1, 100, "2026-06-19", 7, 0.55, True)],
    )

    result = report.generate_report(processed_dir=processed_dir, reports_dir=reports_dir, as_of="2026-06-21")

    assert os.path.exists(result["report_path"])
    assert os.path.exists(result["reliability_plot_path"])
    assert os.path.exists(result["roi_plot_path"])
    assert result["report_path"].endswith("2026-06-21-results.md")
    with open(result["report_path"]) as f:
        content = f.read()
    assert "Settled picks: 1" in content


def test_generate_report_with_no_partitions_still_writes_files(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    reports_dir = os.path.join(str(tmp_path), "reports")

    result = report.generate_report(processed_dir=processed_dir, reports_dir=reports_dir, as_of="2026-06-21")

    assert os.path.exists(result["report_path"])
    assert os.path.exists(result["reliability_plot_path"])
    assert os.path.exists(result["roi_plot_path"])


# ---------------------------------------------------------------------------
# CLV section (2026-08 CLV feature) -- read-only over data/raw/underdog_ticks/
# and each date's already-written line_picks.csv; never writes either.
# ---------------------------------------------------------------------------

def _write_ticks_csv(ticks_root, game_date, rows):
    part_dir = os.path.join(ticks_root, f"game_date={game_date}")
    os.makedirs(part_dir, exist_ok=True)
    pd.DataFrame(rows, columns=report.TICK_COLUMNS).to_csv(
        os.path.join(part_dir, "ticks.csv"), index=False,
    )


def _write_line_picks_csv(processed_dir, game_date, rows):
    part_dir = os.path.join(processed_dir, "predictions", f"game_date={game_date}")
    os.makedirs(part_dir, exist_ok=True)
    pd.DataFrame(rows, columns=report.LINE_PICKS_COLUMNS).to_csv(
        os.path.join(part_dir, "line_picks.csv"), index=False,
    )


def _tick_row(poll_at, start_time, over_american=-148, under_american=124,
              live_event=False, game_status="scheduled"):
    return {
        "poll_at": poll_at, "over_under_id": "ou-1", "projection_id": "line-1",
        "player_id": "p-1", "pitcher": "Gerrit Cole", "stat_type": "strikeouts",
        "line": 6.5, "over_american": over_american, "under_american": under_american,
        "over_payout_multiplier": 0.68, "under_payout_multiplier": 1.24,
        "p_over_implied": None, "p_under_implied": None, "p_market": None,
        "over_updated_at": poll_at, "under_updated_at": poll_at,
        "game_id": "m-1", "game_title": "NYY @ BOS", "start_time": start_time,
        "game_status": game_status, "live_event": live_event, "status": "active",
    }


def _line_pick_row(game_date):
    return {
        "pitcher": 543037, "game_pk": 1001, "pitcher_name": "Gerrit Cole", "team": "NYY",
        "start_time": "2026-08-23T23:05:00+00:00", "line": 6.5, "line_threshold": 7,
        "p_over": 0.55, "p_under": 0.45, "tier": "medium", "lean": "over",
        "edge": 0.03, "edge_vs_coinflip": 0.05, "push_mass": 0.0,
        "projection_id": "line-1", "pulled_at": f"{game_date}T14:00:00+00:00",
        "over_american": -148, "under_american": 124,
        "over_payout_multiplier": 0.68, "under_payout_multiplier": 1.24,
        "p_over_implied": 0.6, "p_under_implied": 0.45, "vig": 0.05, "p_market": 0.57,
        "p_over_lo": 0.50, "p_over_hi": 0.60, "conviction": 1.5, "actionability": "lean_over",
    }


def test_build_clv_section_empty_when_no_tick_log(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    ticks_root = os.path.join(str(tmp_path), "ticks")
    result = report.build_clv_section(processed_dir=processed_dir, ticks_root=ticks_root)
    assert result["n_total"] == 0
    assert result["by_tier"] == {}


def test_build_clv_section_combines_across_dates(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    ticks_root = os.path.join(str(tmp_path), "ticks")

    _write_ticks_csv(ticks_root, "2026-08-23", [
        _tick_row("2026-08-23T14:00:00+00:00", "2026-08-23T23:05:00+00:00", -148, 124),
        _tick_row("2026-08-23T22:30:00+00:00", "2026-08-23T23:05:00+00:00", -160, 138),
    ])
    _write_line_picks_csv(processed_dir, "2026-08-23", [_line_pick_row("2026-08-23")])

    result = report.build_clv_section(processed_dir=processed_dir, ticks_root=ticks_root)

    assert result["n_total"] == 1
    assert result["overall"]["n"] == 1


def test_render_clv_markdown_includes_honest_caveat_and_no_close_message():
    empty = report.clv.clv_summary(pd.DataFrame())
    lines = report.render_clv_markdown(empty)
    text = "\n".join(lines)
    assert "Closing line value (CLV)" in text
    assert "DFS pick'em operator" in text
    assert "No resolved closes yet" in text


def test_generate_report_appends_clv_section_when_tick_log_present(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    reports_dir = os.path.join(str(tmp_path), "reports")
    ticks_root = os.path.join(str(tmp_path), "ticks")

    _write_ticks_csv(ticks_root, "2026-08-23", [
        _tick_row("2026-08-23T14:00:00+00:00", "2026-08-23T23:05:00+00:00", -148, 124),
        _tick_row("2026-08-23T22:30:00+00:00", "2026-08-23T23:05:00+00:00", -160, 138),
    ])
    _write_line_picks_csv(processed_dir, "2026-08-23", [_line_pick_row("2026-08-23")])

    result = report.generate_report(
        processed_dir=processed_dir, reports_dir=reports_dir, ticks_root=ticks_root, as_of="2026-08-24",
    )

    assert result["clv_report"]["n_total"] == 1
    with open(result["report_path"]) as f:
        content = f.read()
    assert "Closing line value (CLV)" in content


def test_generate_report_include_clv_false_skips_the_section(tmp_path):
    processed_dir = os.path.join(str(tmp_path), "processed")
    reports_dir = os.path.join(str(tmp_path), "reports")

    result = report.generate_report(
        processed_dir=processed_dir, reports_dir=reports_dir, as_of="2026-06-21", include_clv=False,
    )
    assert result["clv_report"] is None
    with open(result["report_path"]) as f:
        content = f.read()
    assert "Closing line value (CLV)" not in content


# ---------------------------------------------------------------------------
# Historical backtest report (task #11 step 6, testing item #13)
# ---------------------------------------------------------------------------
#
# Hand-built wide OOS fixtures matching src.backtest.walk_forward's exact
# output schema (_oos_columns): pitcher, game_pk, game_date, wf_step, family,
# mu, alpha, p_over_<t>..., realized_strikeouts, over_hit_<t>..., tier_<t>...
# Only thresholds 6 and 7 are populated here (a tiny, hand-computable
# fixture) -- build_backtest_report's helpers all tolerate a partial sweep.

def _oos_row(pitcher, game_pk, game_date, wf_step, mu, p_over_6, p_over_7, realized):
    from src.predictions.tiering import tier as tier_fn

    over_hit_6 = realized >= 6
    over_hit_7 = realized >= 7
    return {
        "pitcher": pitcher, "game_pk": game_pk, "game_date": game_date,
        "wf_step": wf_step, "family": "poisson", "mu": mu, "alpha": float("nan"),
        "p_over_6": p_over_6, "p_over_7": p_over_7,
        "realized_strikeouts": realized,
        "over_hit_6": over_hit_6, "over_hit_7": over_hit_7,
        "tier_6": tier_fn(p_over_6), "tier_7": tier_fn(p_over_7),
    }


def _oos_fixture():
    return pd.DataFrame([
        _oos_row(1, 100, "2026-04-05", "2026-04-07", 6.0, 0.55, 0.35, 7),
        _oos_row(2, 200, "2026-04-12", "2026-04-14", 5.0, 0.40, 0.20, 4),
    ])


def test_load_oos_frame_missing_file_returns_empty_correctly_columned_frame(tmp_path):
    missing_path = os.path.join(str(tmp_path), "walk_forward_oos.csv")
    oos = report.load_oos_frame(missing_path, thresholds=(6, 7))
    assert oos.empty
    assert "pitcher" in oos.columns
    assert "p_over_6" in oos.columns


def test_persist_and_load_oos_frame_round_trips(tmp_path):
    path = os.path.join(str(tmp_path), "backtest", "walk_forward_oos.csv")
    oos = _oos_fixture()

    written_path = report.persist_oos_frame(oos, path)
    assert written_path == path
    assert os.path.exists(path)

    loaded = report.load_oos_frame(path, thresholds=(6, 7))
    assert len(loaded) == 2
    assert set(loaded["pitcher"]) == {1, 2}


def test_build_backtest_report_returns_numbers_not_crashing():
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, n_buckets=5, thresholds=(6, 7))

    assert built["n_oos_total"] == 2
    assert built["point_accuracy"]["n"] == 2
    # row1: |6-7|=1, row2: |5-4|=1 -> MAE = 1.0
    assert built["point_accuracy"]["mae"] == pytest.approx(1.0)
    assert not built["reliability_table"].empty
    assert not built["by_sweep_tier"].empty
    assert not built["over_time"].empty
    assert 6 in built["brier_log_loss_at_thresholds"]
    assert 7 in built["brier_log_loss_at_thresholds"]


def test_build_backtest_report_by_sweep_tier_hand_computed():
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))

    # Long-melted rows: (1,100,t6,p=0.55,hit=True,tier=?), (1,100,t7,p=0.35,hit=False,tier=?)
    #                   (2,200,t6,p=0.40,hit=False,tier=?), (2,200,t7,p=0.20,hit=False,tier=?)
    by_tier = built["by_sweep_tier"]
    assert by_tier["n"].sum() == 4


def test_build_backtest_report_over_time_sorted_chronologically():
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))

    over_time_table = built["over_time"]
    assert list(over_time_table["wf_step"]) == sorted(over_time_table["wf_step"])
    assert over_time_table["n"].sum() == 4


def test_build_backtest_report_no_roi_fields_present():
    # Testing item #7's no-ROI guard, re-asserted at the report layer: the
    # backtest report's dict must never carry a line/edge/roi key, matching
    # walk_forward.py's own output-schema guarantee.
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))

    forbidden = {"roi", "line", "edge", "flat_bet_roi", "by_line", "hit_rate"}
    assert forbidden.isdisjoint(built.keys())


def test_build_backtest_report_empty_input_does_not_crash():
    from src.backtest.walk_forward import _oos_columns

    empty_oos = pd.DataFrame(columns=_oos_columns((6, 7)))
    built = report.build_backtest_report(empty_oos, thresholds=(6, 7))

    assert built["n_oos_total"] == 0
    assert built["by_sweep_tier"].empty
    assert built["over_time"].empty
    assert pd.isna(built["point_accuracy"]["mae"])


def test_render_backtest_markdown_states_not_betting_performance():
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))
    md = report.render_backtest_markdown(built, "2026-06-21")

    assert "2026-06-21" in md
    assert "not betting performance" in md.lower()
    assert "no lines involved" in md.lower()
    assert "## By sweep tier" in md
    assert "## Over time" in md
    assert "Out-of-sample starts evaluated: 2" in md


def test_render_backtest_markdown_handles_empty_report_without_crashing():
    from src.backtest.walk_forward import _oos_columns

    empty_oos = pd.DataFrame(columns=_oos_columns((6, 7)))
    built = report.build_backtest_report(empty_oos, thresholds=(6, 7))
    md = report.render_backtest_markdown(built, "2026-06-21")

    assert "Out-of-sample starts evaluated: 0" in md


def test_plot_calibration_by_tier_writes_png(tmp_path):
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))
    out_path = os.path.join(str(tmp_path), "by_tier.png")

    report.plot_calibration_by_tier(built["by_sweep_tier"], out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_calibration_by_tier_writes_png_even_when_empty(tmp_path):
    out_path = os.path.join(str(tmp_path), "by_tier_empty.png")
    empty = pd.DataFrame(columns=["tier"] + list(report.metrics.SLICE_COLUMNS))
    report.plot_calibration_by_tier(empty, out_path)
    assert os.path.exists(out_path)


def test_plot_error_over_time_writes_png(tmp_path):
    oos = _oos_fixture()
    built = report.build_backtest_report(oos, thresholds=(6, 7))
    out_path = os.path.join(str(tmp_path), "over_time.png")

    report.plot_error_over_time(built["over_time"], out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_error_over_time_writes_png_even_when_empty(tmp_path):
    out_path = os.path.join(str(tmp_path), "over_time_empty.png")
    empty = pd.DataFrame(columns=["wf_step"] + list(report.metrics.SLICE_COLUMNS))
    report.plot_error_over_time(empty, out_path)
    assert os.path.exists(out_path)


def test_generate_backtest_report_writes_md_and_three_pngs(tmp_path):
    reports_dir = os.path.join(str(tmp_path), "reports")
    oos = _oos_fixture()

    result = report.generate_backtest_report(oos_df=oos, reports_dir=reports_dir, as_of="2026-06-21", thresholds=(6, 7))

    assert os.path.exists(result["report_path"])
    assert result["report_path"].endswith("2026-06-21-baseline-backtest.md")
    assert os.path.exists(result["reliability_plot_path"])
    assert os.path.exists(result["calibration_by_tier_plot_path"])
    assert os.path.exists(result["error_over_time_plot_path"])
    with open(result["report_path"]) as f:
        content = f.read()
    assert "not betting performance" in content.lower()


def test_generate_backtest_report_loads_from_disk_when_oos_df_not_given(tmp_path):
    reports_dir = os.path.join(str(tmp_path), "reports")
    oos_path = os.path.join(str(tmp_path), "backtest", "walk_forward_oos.csv")
    report.persist_oos_frame(_oos_fixture(), oos_path)

    result = report.generate_backtest_report(
        oos_path=oos_path, reports_dir=reports_dir, as_of="2026-06-21", thresholds=(6, 7),
    )

    assert os.path.exists(result["report_path"])
    assert result["report"]["n_oos_total"] == 2


def test_generate_backtest_report_with_no_oos_data_still_writes_files(tmp_path):
    reports_dir = os.path.join(str(tmp_path), "reports")
    missing_oos_path = os.path.join(str(tmp_path), "backtest", "walk_forward_oos.csv")

    result = report.generate_backtest_report(
        oos_path=missing_oos_path, reports_dir=reports_dir, as_of="2026-06-21", thresholds=(6, 7),
    )

    assert os.path.exists(result["report_path"])
    assert result["report"]["n_oos_total"] == 0
