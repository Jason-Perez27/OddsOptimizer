"""
Unit tests for src/pipeline/settle.py (task #10, spec section 1/3 +
the "Testing approach" list, items 10-12 plus settle-layer coverage of 14).

Covers:
  10. End-to-end settlement: a written predictions partition, run through
      run_settlement with an injected outcome_fetcher, produces graded
      frames with the correct authoritative settlement_status merged in
      (resolving the status-merge split documented in grading.py).
  11. Re-settling is idempotent (overwrite=True replaces); --no-overwrite
      (overwrite=False) aborts with FileExistsError instead of duplicating.
  12. A single pitcher's outcome fetch failing doesn't abort the run -- it's
      recorded in the manifest's fetch_errors and that pitcher stays
      unsettled for this pass, never crashing the whole settlement.
  14 (settle-layer): a short, low-K start still settles normally (not void),
      verified again here at the full pipeline level, not just grading.py's.

Plus the "predictions partition missing" and "missing game_pk fails fast"
edge cases from the spec's numbered list.

No network: outcome_fetcher is always injected with hand-built pitch-level
fixtures, matching the rest of this repo's pipeline test style
(test_refresh.py).

Run with: pytest tests/test_settle.py -v
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import grading
from src.pipeline import settle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pitch_rows(pitcher, game_pk, game_date, n_strikeouts, n_other_outs=0):
    """Minimal pitch-level rows aggregate_pitcher_games can collapse."""
    rows = []
    for _ in range(n_strikeouts):
        rows.append({
            "pitcher": pitcher, "game_pk": game_pk, "game_date": game_date,
            "home_team": "NYY", "away_team": "BOS", "inning_topbot": "Top",
            "events": "strikeout", "description": "swinging_strike",
            "pitch_type": "FF", "release_speed": 95.0, "stand": "R",
        })
    for _ in range(n_other_outs):
        rows.append({
            "pitcher": pitcher, "game_pk": game_pk, "game_date": game_date,
            "home_team": "NYY", "away_team": "BOS", "inning_topbot": "Top",
            "events": "field_out", "description": "hit_into_play",
            "pitch_type": "FF", "release_speed": 95.0, "stand": "R",
        })
    return pd.DataFrame(rows)


def _write_predictions_partition(
    processed_dir, game_date, pitcher=1, game_pk=100, line=6.5, lean="over",
):
    pred_dir = os.path.join(processed_dir, "predictions", f"game_date={game_date}")
    os.makedirs(pred_dir, exist_ok=True)

    predictions = pd.DataFrame([{
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "Test Pitcher",
        "pitcher_team": "NYY", "opponent_team": "BOS", "game_date": game_date,
        "family": "poisson", "mu": 6.2, "alpha": None,
    }])
    threshold_table = pd.DataFrame([{
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "Test Pitcher",
        "team": "NYY", "opponent_team": "BOS", "game_date": game_date,
        "threshold": 6, "p_over": 0.55, "tier": "low",
    }])
    line_picks = pd.DataFrame([{
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "Test Pitcher",
        "team": "NYY", "start_time": f"{game_date}T18:00:00Z",
        "line": line, "line_threshold": int(line) + 1, "p_over": 0.55,
        "p_under": 0.45, "tier": "low", "lean": lean, "edge": 0.05,
        "push_mass": 0.0, "projection_id": "proj-1",
        "pulled_at": f"{game_date}T12:00:00Z",
    }])

    predictions.to_csv(os.path.join(pred_dir, "predictions.csv"), index=False)
    threshold_table.to_csv(os.path.join(pred_dir, "threshold_table.csv"), index=False)
    line_picks.to_csv(os.path.join(pred_dir, "line_picks.csv"), index=False)
    return pred_dir


# ---------------------------------------------------------------------------
# Missing predictions partition -> clean no-op
# ---------------------------------------------------------------------------

def test_run_settlement_missing_partition_is_clean_noop(tmp_path):
    results = settle.run_settlement(
        "2026-06-26",
        outcome_fetcher=lambda pid, d: pd.DataFrame(),
        processed_dir=str(tmp_path),
        now="2026-06-27",
    )

    assert results["graded_line_picks"].empty
    assert results["graded_threshold_sweep"].empty
    assert "no predictions partition" in results["manifest"]["note"]
    assert results["manifest"]["n_predicted"] == 0


def test_write_graded_noop_manifest_still_writes_cleanly(tmp_path):
    # Even a no-op result should be writable without crashing, in case a
    # caller writes every date in a --from/--to range unconditionally.
    results = settle.run_settlement(
        "2026-06-26",
        outcome_fetcher=lambda pid, d: pd.DataFrame(),
        processed_dir=str(tmp_path),
        now="2026-06-27",
    )
    out_dir = settle.write_graded(results, processed_dir=str(tmp_path))
    assert os.path.exists(os.path.join(out_dir, "settle_manifest.json"))


# ---------------------------------------------------------------------------
# 10. End-to-end settlement -- authoritative status merged onto both frames
# ---------------------------------------------------------------------------

@pytest.fixture
def settled_partition(tmp_path):
    game_date = "2026-06-20"
    _write_predictions_partition(str(tmp_path), game_date, pitcher=1, game_pk=100, line=6.5, lean="over")

    def outcome_fetcher(pitcher_id, d):
        assert d == game_date
        return _pitch_rows(1, 100, game_date, n_strikeouts=8)

    return tmp_path, game_date, outcome_fetcher


def test_run_settlement_end_to_end_settled_and_status_merged(settled_partition):
    tmp_path, game_date, outcome_fetcher = settled_partition

    results = settle.run_settlement(
        game_date,
        outcome_fetcher=outcome_fetcher,
        processed_dir=str(tmp_path),
        now="2026-06-21",
        scratch_void_max_wait_days=3,
    )

    manifest = results["manifest"]
    assert manifest["note"] is None
    assert manifest["n_settled"] == 1
    assert manifest["n_pending"] == 0
    assert manifest["n_void_scratched"] == 0

    picks = results["graded_line_picks"]
    assert len(picks) == 1
    row = picks.iloc[0]
    # status_map merge: grade_line_picks itself doesn't know status -- this
    # is the thing settle.py exists to fix.
    assert row["settlement_status"] == grading.SETTLED
    assert row["game_date"] == game_date  # merged in too -- line_picks_df has no game_date of its own
    assert row["realized_strikeouts"] == 8
    assert row["over_hit"] == True
    assert row["pick_correct"] == True  # lean=over, line=6.5 (threshold 7), 8>=7

    sweep = results["graded_threshold_sweep"]
    assert len(sweep) == 1
    assert sweep.iloc[0]["settlement_status"] == grading.SETTLED
    assert sweep.iloc[0]["over_hit"] == True


def test_run_settlement_pending_within_wait_window(tmp_path):
    game_date = "2026-06-26"
    _write_predictions_partition(str(tmp_path), game_date, pitcher=1, game_pk=100)

    results = settle.run_settlement(
        game_date,
        outcome_fetcher=lambda pid, d: pd.DataFrame(),  # no Statcast row yet
        processed_dir=str(tmp_path),
        now="2026-06-27",  # 1 day elapsed, within the 3-day wait window
        scratch_void_max_wait_days=3,
    )

    assert results["manifest"]["n_pending"] == 1
    picks = results["graded_line_picks"]
    assert picks.iloc[0]["settlement_status"] == grading.PENDING
    assert pd.isna(picks.iloc[0]["over_hit"])


def test_run_settlement_void_scratched_past_max_wait(tmp_path):
    game_date = "2026-06-10"
    _write_predictions_partition(str(tmp_path), game_date, pitcher=1, game_pk=100)

    results = settle.run_settlement(
        game_date,
        outcome_fetcher=lambda pid, d: pd.DataFrame(),  # never threw -- scratched/postponed
        processed_dir=str(tmp_path),
        now="2026-06-27",  # well past the wait window
        scratch_void_max_wait_days=3,
    )

    assert results["manifest"]["n_void_scratched"] == 1
    picks = results["graded_line_picks"]
    assert picks.iloc[0]["settlement_status"] == grading.VOID_SCRATCHED


# ---------------------------------------------------------------------------
# 14 (settle-layer): a short, low-K start settles normally, never void
# ---------------------------------------------------------------------------

def test_run_settlement_short_low_k_start_settles_normally(tmp_path):
    game_date = "2026-06-20"
    _write_predictions_partition(str(tmp_path), game_date, pitcher=1, game_pk=100, line=6.5, lean="under")

    def outcome_fetcher(pitcher_id, d):
        return _pitch_rows(1, 100, game_date, n_strikeouts=1)  # pulled early, only 1 K -- a real start

    results = settle.run_settlement(
        game_date, outcome_fetcher=outcome_fetcher, processed_dir=str(tmp_path),
        now="2026-06-27", scratch_void_max_wait_days=3,
    )

    assert results["manifest"]["n_void_scratched"] == 0
    picks = results["graded_line_picks"]
    assert picks.iloc[0]["settlement_status"] == grading.SETTLED
    assert picks.iloc[0]["pick_correct"] == True  # lean=under, realized=1 < threshold 7


# ---------------------------------------------------------------------------
# 12. Per-pitcher fetch failure doesn't abort the run
# ---------------------------------------------------------------------------

def test_run_settlement_one_pitcher_fetch_failure_does_not_abort(tmp_path):
    game_date = "2026-06-20"
    _write_predictions_partition(str(tmp_path), game_date, pitcher=1, game_pk=100)

    def flaky_fetcher(pitcher_id, d):
        raise ConnectionError("simulated network failure")

    results = settle.run_settlement(
        game_date, outcome_fetcher=flaky_fetcher, processed_dir=str(tmp_path),
        now="2026-06-21", scratch_void_max_wait_days=3,
    )

    # No exception raised -- the failure is recorded, and the pitcher is
    # simply unsettled (pending, since 1 day elapsed is within wait window).
    assert len(results["manifest"]["fetch_errors"]) == 1
    assert results["manifest"]["fetch_errors"][0]["pitcher"] == 1
    assert results["manifest"]["n_pending"] == 1


# ---------------------------------------------------------------------------
# 11. Re-settling: overwrite is idempotent; --no-overwrite aborts
# ---------------------------------------------------------------------------

def test_write_graded_overwrite_is_idempotent(settled_partition):
    tmp_path, game_date, outcome_fetcher = settled_partition

    results = settle.run_settlement(
        game_date, outcome_fetcher=outcome_fetcher, processed_dir=str(tmp_path), now="2026-06-21",
    )
    out_dir1 = settle.write_graded(results, processed_dir=str(tmp_path), overwrite=True)
    out_dir2 = settle.write_graded(results, processed_dir=str(tmp_path), overwrite=True)

    assert out_dir1 == out_dir2
    assert os.path.exists(os.path.join(out_dir2, "graded_line_picks.csv"))


def test_write_graded_no_overwrite_aborts_on_second_call(settled_partition):
    tmp_path, game_date, outcome_fetcher = settled_partition

    results = settle.run_settlement(
        game_date, outcome_fetcher=outcome_fetcher, processed_dir=str(tmp_path), now="2026-06-21",
    )
    settle.write_graded(results, processed_dir=str(tmp_path), overwrite=True)

    with pytest.raises(FileExistsError):
        settle.write_graded(results, processed_dir=str(tmp_path), overwrite=False)


# ---------------------------------------------------------------------------
# Doubleheader, at the settle layer
# ---------------------------------------------------------------------------

def test_run_settlement_doubleheader_graded_independently(tmp_path):
    game_date = "2026-06-20"
    pred_dir = os.path.join(str(tmp_path), "predictions", f"game_date={game_date}")
    os.makedirs(pred_dir, exist_ok=True)

    predictions = pd.DataFrame([
        {"pitcher": 1, "game_pk": 100, "pitcher_name": "P", "pitcher_team": "NYY",
         "opponent_team": "BOS", "game_date": game_date, "family": "poisson", "mu": 6.0, "alpha": None},
        {"pitcher": 1, "game_pk": 200, "pitcher_name": "P", "pitcher_team": "NYY",
         "opponent_team": "BOS", "game_date": game_date, "family": "poisson", "mu": 6.0, "alpha": None},
    ])
    threshold_table = pd.DataFrame([
        {"pitcher": 1, "game_pk": 100, "pitcher_name": "P", "team": "NYY", "opponent_team": "BOS",
         "game_date": game_date, "threshold": 6, "p_over": 0.55, "tier": "low"},
        {"pitcher": 1, "game_pk": 200, "pitcher_name": "P", "team": "NYY", "opponent_team": "BOS",
         "game_date": game_date, "threshold": 6, "p_over": 0.55, "tier": "low"},
    ])
    line_picks = pd.DataFrame(columns=settle.LINE_PICKS_COLUMNS)

    predictions.to_csv(os.path.join(pred_dir, "predictions.csv"), index=False)
    threshold_table.to_csv(os.path.join(pred_dir, "threshold_table.csv"), index=False)
    line_picks.to_csv(os.path.join(pred_dir, "line_picks.csv"), index=False)

    def outcome_fetcher(pitcher_id, d):
        # One call returns BOTH games' pitches -- aggregate_pitcher_games
        # must split them back apart by game_pk.
        game1 = _pitch_rows(1, 100, game_date, n_strikeouts=8)
        game2 = _pitch_rows(1, 200, game_date, n_strikeouts=3)
        return pd.concat([game1, game2], ignore_index=True)

    results = settle.run_settlement(
        game_date, outcome_fetcher=outcome_fetcher, processed_dir=str(tmp_path),
        now="2026-06-21", scratch_void_max_wait_days=3,
    )

    sweep = results["graded_threshold_sweep"]
    g1 = sweep[sweep["game_pk"] == 100].iloc[0]
    g2 = sweep[sweep["game_pk"] == 200].iloc[0]
    assert g1["over_hit"] == True
    assert g2["over_hit"] == False
    assert g1["settlement_status"] == grading.SETTLED
    assert g2["settlement_status"] == grading.SETTLED


# ---------------------------------------------------------------------------
# Missing game_pk fails fast
# ---------------------------------------------------------------------------

def test_run_settlement_missing_game_pk_fails_fast(tmp_path):
    game_date = "2026-06-20"
    pred_dir = os.path.join(str(tmp_path), "predictions", f"game_date={game_date}")
    os.makedirs(pred_dir, exist_ok=True)

    # No game_pk column at all -- a task #9 regression.
    predictions = pd.DataFrame([{"pitcher": 1, "game_date": game_date, "mu": 6.0}])
    predictions.to_csv(os.path.join(pred_dir, "predictions.csv"), index=False)
    pd.DataFrame(columns=["pitcher", "threshold"]).to_csv(os.path.join(pred_dir, "threshold_table.csv"), index=False)
    pd.DataFrame(columns=["pitcher", "line"]).to_csv(os.path.join(pred_dir, "line_picks.csv"), index=False)

    with pytest.raises(ValueError, match="game_pk"):
        settle.run_settlement(
            game_date, outcome_fetcher=lambda pid, d: pd.DataFrame(),
            processed_dir=str(tmp_path), now="2026-06-21",
        )


# ---------------------------------------------------------------------------
# --window-days (task #12 go-live spec, testing checklist items #4-#5)
# ---------------------------------------------------------------------------

def test_window_days_to_dates_expands_to_trailing_window_ending_yesterday():
    from datetime import datetime, timezone

    now = datetime(2026, 6, 29, 15, 0, 0, tzinfo=timezone.utc)
    dates = settle.window_days_to_dates(4, now=now)
    assert dates == ["2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28"]


def test_window_days_to_dates_equals_explicit_from_to_equivalent():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 6, 29, 15, 0, 0, tzinfo=timezone.utc)
    window_days = 4

    via_window = settle.window_days_to_dates(window_days, now=now)

    today = pd.Timestamp(now).date()
    start = today - timedelta(days=window_days)
    end = today - timedelta(days=1)
    via_explicit = [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]

    assert via_window == via_explicit


def test_window_days_mutually_exclusive_with_date(monkeypatch, capsys):
    import sys as _sys
    from src.pipeline import settle as settle_mod

    monkeypatch.setattr(_sys, "argv", ["settle.py", "--window-days", "4", "--date", "2026-06-28"])
    with pytest.raises(SystemExit):
        settle_mod.main()
    err = capsys.readouterr().err
    assert "--window-days is mutually exclusive" in err


def test_window_days_mutually_exclusive_with_from_to(monkeypatch, capsys):
    import sys as _sys
    from src.pipeline import settle as settle_mod

    monkeypatch.setattr(
        _sys, "argv",
        ["settle.py", "--window-days", "4", "--from", "2026-06-01", "--to", "2026-06-05"],
    )
    with pytest.raises(SystemExit):
        settle_mod.main()
    err = capsys.readouterr().err
    assert "--window-days is mutually exclusive" in err
