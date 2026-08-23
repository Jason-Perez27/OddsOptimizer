"""
Unit tests for src/serve/data.py -- the dashboard's pure data-join layer.

No web server, no network: builds a tiny on-disk partition in tmp_path mirroring
what src/pipeline/refresh.write_outputs produces, then asserts load_slate assembles
the right slate dict (KPIs, nested 1..10 ladder, two-sided market prices on a
line, a sweep-only pitcher with no line, and graceful handling of a missing
pitcher_cards.csv).

Run with: pytest tests/test_serve_data.py -v
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serve import data as slate_data


def _write_partition(processed_dir, game_date="2026-06-29", with_cards=True):
    part = os.path.join(processed_dir, "predictions", f"game_date={game_date}")
    os.makedirs(os.path.join(part, "diagnostics"), exist_ok=True)

    # 3 pitchers: standard line, goblin fallback, and a sweep-only (no line)
    predictions = pd.DataFrame([
        {"pitcher": 1, "game_pk": 100, "pitcher_name": "Ace One", "pitcher_team": "NYY",
         "opponent_team": "BOS", "game_date": game_date, "family": "poisson", "mu": 6.5, "alpha": None},
        {"pitcher": 2, "game_pk": 200, "pitcher_name": "Lefty Two", "pitcher_team": "OAK",
         "opponent_team": "LAD", "game_date": game_date, "family": "poisson", "mu": 5.0, "alpha": None},
        {"pitcher": 3, "game_pk": 300, "pitcher_name": "Nolan Three", "pitcher_team": "SEA",
         "opponent_team": "LAA", "game_date": game_date, "family": "poisson", "mu": 4.0, "alpha": None},
    ])
    predictions.to_csv(os.path.join(part, "predictions.csv"), index=False)

    th_rows = []
    for pid, gpk in [(1, 100), (2, 200), (3, 300)]:
        for t in range(1, 11):
            th_rows.append({"pitcher": pid, "game_pk": gpk, "pitcher_name": "x", "team": "x",
                            "opponent_team": "x", "game_date": game_date,
                            "threshold": t, "p_over": max(0.0, 1.0 - 0.1 * t), "tier": "high"})
    pd.DataFrame(th_rows).to_csv(os.path.join(part, "threshold_table.csv"), index=False)

    # pitcher 1 -> priced line; pitcher 2 -> priced line (higher vig); pitcher 3 -> NO line
    line_picks = pd.DataFrame([
        {"pitcher": 1, "game_pk": 100, "pitcher_name": "Ace One", "team": "NYY",
         "start_time": "2026-06-29T18:35:00Z", "line": 5.5, "line_threshold": 6, "p_over": 0.64,
         "p_under": 0.36, "tier": "medium", "lean": "over", "edge": 0.09, "edge_vs_coinflip": 0.14,
         "push_mass": 0.0, "projection_id": "p1", "pulled_at": "2026-06-29T13:00:00Z",
         "over_american": -140.0, "under_american": 118.0,
         "over_payout_multiplier": 0.71, "under_payout_multiplier": 1.18,
         "p_over_implied": 0.583, "p_under_implied": 0.459, "vig": 0.042, "p_market": 0.559},
        {"pitcher": 2, "game_pk": 200, "pitcher_name": "Lefty Two", "team": "OAK",
         "start_time": "2026-06-29T20:10:00Z", "line": 4.5, "line_threshold": 5, "p_over": 0.55,
         "p_under": 0.45, "tier": "low", "lean": "over", "edge": 0.03, "edge_vs_coinflip": 0.05,
         "push_mass": 0.0, "projection_id": "p2", "pulled_at": "2026-06-29T13:00:00Z",
         "over_american": -110.0, "under_american": -110.0,
         "over_payout_multiplier": 0.91, "under_payout_multiplier": 0.91,
         "p_over_implied": 0.524, "p_under_implied": 0.524, "vig": 0.048, "p_market": 0.520},
    ])
    line_picks.to_csv(os.path.join(part, "line_picks.csv"), index=False)

    if with_cards:
        cards = pd.DataFrame([
            {"pitcher": 1, "game_pk": 100, "game_date": game_date, "pitcher_name": "Ace One",
             "pitcher_team": "NYY", "opponent_team": "BOS", "pitcher_throws": "R", "is_home": 1.0,
             "rest_days": 5, "mu": 6.5, "k_rate_last5": 0.274, "k_rate_season": 0.241,
             "k_rate_vs_LHB": 0.22, "k_rate_vs_RHB": 0.26, "k_rate_home": 0.25, "k_rate_away": 0.23,
             "k_rate_vs_opponent_career": 0.24, "ip_avg_last5": 5.8, "pitch_count_avg_last5": 92.0,
             "bf_avg_last5": 23.4, "whiff_rate_last5": 0.26, "velo_avg_last5": 93.8,
             "opponent_k_rate_vs_hand_season": 0.238, "opponent_k_rate_last10": 0.216,
             "opponent_k_rate_home": 0.22, "opponent_k_rate_away": 0.21, "park_k_factor": 1.03,
             "was_imputed": 0.0},
        ])
        cards.to_csv(os.path.join(part, "pitcher_cards.csv"), index=False)

    for name in ["skipped_pitchers.csv", "unmatched_lines.csv", "predicted_no_line.csv"]:
        pd.DataFrame().to_csv(os.path.join(part, "diagnostics", name), index=False)

    with open(os.path.join(part, "run_manifest.json"), "w") as f:
        json.dump({"game_date": game_date, "model_age_days": 0.5, "model_stale": False,
                   "line_source_error": None, "n_predictions": 3}, f)
    return processed_dir


def test_list_and_latest_game_date(tmp_path):
    _write_partition(str(tmp_path), "2026-06-28")
    _write_partition(str(tmp_path), "2026-06-29")
    assert slate_data.list_game_dates(str(tmp_path)) == ["2026-06-28", "2026-06-29"]
    assert slate_data.latest_game_date(str(tmp_path)) == "2026-06-29"


def test_load_slate_kpis_and_ladder(tmp_path):
    _write_partition(str(tmp_path))
    slate = slate_data.load_slate(str(tmp_path), "2026-06-29")

    kpis = slate["kpis"]
    assert kpis["n_pitchers"] == 3
    assert kpis["n_with_line"] == 2
    assert kpis["n_no_line"] == 1
    assert kpis["median_vig"] == pytest.approx((0.042 + 0.048) / 2)
    assert kpis["model_age_days"] == 0.5
    assert kpis["model_stale"] is False
    assert kpis["line_source_error"] is None

    ace = next(p for p in slate["pitchers"] if p["pitcher"] == 1)
    assert len(ace["ladder"]) == 10
    assert [r["threshold"] for r in ace["ladder"]] == list(range(1, 11))
    assert ace["line"]["p_market"] == pytest.approx(0.559)
    assert ace["line"]["over_american"] == -140.0
    assert ace["line"]["under_american"] == 118.0


def test_priced_lines_and_sweep_only(tmp_path):
    _write_partition(str(tmp_path))
    slate = slate_data.load_slate(str(tmp_path), "2026-06-29")

    lefty = next(p for p in slate["pitchers"] if p["pitcher"] == 2)
    assert lefty["line"]["p_market"] == pytest.approx(0.520)
    assert lefty["line"]["vig"] == pytest.approx(0.048)

    nolan = next(p for p in slate["pitchers"] if p["pitcher"] == 3)
    assert nolan["line"] is None  # sweep-only, never dropped
    assert len(nolan["ladder"]) == 10


def test_stats_present_and_expected_batters(tmp_path):
    _write_partition(str(tmp_path))
    slate = slate_data.load_slate(str(tmp_path), "2026-06-29")
    ace = next(p for p in slate["pitchers"] if p["pitcher"] == 1)
    assert ace["stats"]["ip_avg_last5"] == 5.8
    assert ace["stats"]["bf_avg_last5"] == 23.4
    assert ace["stats"]["pitch_count_avg_last5"] == 92.0
    assert slate["has_stats"] is True


def test_missing_pitcher_cards_degrades_gracefully(tmp_path):
    _write_partition(str(tmp_path), with_cards=False)
    slate = slate_data.load_slate(str(tmp_path), "2026-06-29")
    assert slate["has_stats"] is False
    assert all(p["stats"] is None for p in slate["pitchers"])
    # ladder + lines still render
    assert any(p["line"] and p["line"]["p_market"] is not None for p in slate["pitchers"])


def test_missing_partition_raises(tmp_path):
    _write_partition(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        slate_data.load_slate(str(tmp_path), "2099-01-01")


# ---------------------------------------------------------------------------
# prop routing -- non-default prop reads the prop={key}/ sub-partition
# ---------------------------------------------------------------------------

def _write_walks_partition(processed_dir, game_date="2026-06-29"):
    """Write a minimal walks partition under game_date=*/prop=walks/."""
    import json
    part = os.path.join(processed_dir, "predictions",
                        f"game_date={game_date}", "prop=walks")
    os.makedirs(os.path.join(part, "diagnostics"), exist_ok=True)

    predictions = pd.DataFrame([
        {"pitcher": 10, "game_pk": 1000, "pitcher_name": "Walker One",
         "pitcher_team": "BOS", "opponent_team": "NYY", "game_date": game_date,
         "family": "poisson", "mu": 2.1, "alpha": None},
    ])
    predictions.to_csv(os.path.join(part, "predictions.csv"), index=False)
    pd.DataFrame().to_csv(os.path.join(part, "threshold_table.csv"), index=False)
    pd.DataFrame().to_csv(os.path.join(part, "line_picks.csv"), index=False)

    for name in ["skipped_pitchers.csv", "unmatched_lines.csv", "predicted_no_line.csv"]:
        pd.DataFrame().to_csv(os.path.join(part, "diagnostics", name), index=False)

    with open(os.path.join(part, "run_manifest.json"), "w") as f:
        json.dump({"game_date": game_date, "model_age_days": 0.5, "model_stale": False,
                   "line_source_error": None, "n_predictions": 1}, f)


def test_load_slate_with_prop_walks_reads_sub_partition(tmp_path):
    """load_slate(prop='walks') must read the prop=walks/ sub-partition, not the flat one."""
    # Write both default (strikeouts flat) and walks sub-partition.
    _write_partition(str(tmp_path))
    _write_walks_partition(str(tmp_path))

    walks_slate = slate_data.load_slate(str(tmp_path), "2026-06-29", prop="walks")
    assert len(walks_slate["pitchers"]) == 1
    assert walks_slate["pitchers"][0]["pitcher"] == 10

    # Default prop still reads the flat partition (3 pitchers)
    default_slate = slate_data.load_slate(str(tmp_path), "2026-06-29")
    assert len(default_slate["pitchers"]) == 3


def test_load_slate_prop_walks_missing_partition_raises(tmp_path):
    """If the walks sub-partition doesn't exist, FileNotFoundError is raised."""
    _write_partition(str(tmp_path))  # only the strikeouts flat partition
    with pytest.raises(FileNotFoundError):
        slate_data.load_slate(str(tmp_path), "2026-06-29", prop="walks")
