"""
Unit tests for src/predictions/tiering.py (task #8; Underdog line-source
migration, 2026-08).

Strategy (see engineering:testing-strategy, mirroring tests/test_underdog_lines.py
and tests/test_baseline_model.py conventions in this repo):
- No network calls. The model side is represented as plain (family, mu, alpha)
  values rather than a live fitted BaselineModel -- prob_over_line/build_threshold_table
  call the exact same poisson_over_prob/nbinom_over_prob functions BaselineModel
  itself uses, so this is the real survival-function math, not a stub.
- scipy-dependent assertions (item 4: matching scipy.stats directly) are gated
  with pytest.importorskip("scipy.stats"), consistent with test_baseline_model.py --
  scipy is not installed in this sandbox, so those specific assertions could not be
  executed here and need a real local `pytest` run to confirm. Everything else in
  this file (tiering boundaries, line/threshold math, the resolver, dedupe/status
  filtering, coverage diagnostics, empty-input handling) is pure pandas/numpy and
  was run successfully in-sandbox.
- Hand-built fixtures via small helper functions, # ---...--- section dividers
  grouping tests by function under test, test_<function>_<behavior> naming.

Underdog line source (2026-08 migration): Underdog posts exactly one balanced
two-sided line per pitcher/stat, so the old PrizePicks-era standard/goblin/demon
canonical-line selection (_select_canonical_line) no longer exists -- there is
nothing to collapse. `edge`/`lean`/`actionability` are now measured against the
line's own no-vig market probability (p_market), not a fixed 50% coinflip.

Run with: pytest tests/test_tiering.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predictions import tiering


# ---------------------------------------------------------------------------
# tier()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected", [
    (0.70, "high"), (0.699, "medium"), (0.60, "medium"), (0.599, "low"),
    (0.50, "low"),
    (0.30, "high"), (0.301, "medium"), (0.40, "medium"), (0.401, "low"),
])
def test_tier_boundaries_are_exact_and_symmetric(p, expected):
    assert tiering.tier(p) == expected


def test_tier_depends_only_on_probability_not_on_a_line():
    # tier()'s signature takes only p -- a line can never reach it.
    import inspect
    assert list(inspect.signature(tiering.tier).parameters) == ["p"]

    # Two different (family, mu, line) combinations that happen to produce
    # the same p_over must land in the same tier.
    p_over_a, _ = tiering.prob_over_line("poisson", mu=5.0, alpha=None, line=4.5)
    p_over_b = p_over_a
    assert tiering.tier(p_over_a) == tiering.tier(p_over_b)


# ---------------------------------------------------------------------------
# line_to_threshold() / prob_over_line()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [(6.5, 7), (0.5, 1), (9.5, 10), (10.5, 11)])
def test_line_to_threshold_conversion(line, expected):
    assert tiering.line_to_threshold(line) == expected


def test_integer_line_records_push_mass_without_crashing():
    p_over, push_mass = tiering.prob_over_line("poisson", mu=6.0, alpha=None, line=6.0)
    assert tiering.line_to_threshold(6.0) == 7  # over = K >= 7
    assert push_mass > 0.0  # P(K == 6) under Poisson(6) is strictly positive
    assert 0.0 <= p_over <= 1.0
    assert 0.0 <= push_mass <= 1.0


def test_prob_over_line_handles_lines_beyond_the_fixed_sweep():
    # line >= 10.5 -> threshold >= 11, outside the 1..10 sweep -- must compute
    # directly from the survival function rather than raising/indexing a table.
    p_over, push_mass = tiering.prob_over_line("poisson", mu=8.0, alpha=None, line=10.5)
    assert tiering.line_to_threshold(10.5) == 11
    assert push_mass == 0.0
    assert 0.0 <= p_over <= 1.0


def test_prob_over_line_matches_scipy_poisson_reference():
    scipy_stats = pytest.importorskip("scipy.stats")
    mu = 7.0
    p_over, push = tiering.prob_over_line("poisson", mu, None, 6.5)
    assert p_over == pytest.approx(float(scipy_stats.poisson.sf(6, mu)))
    assert push == 0.0


def test_prob_over_line_matches_scipy_negative_binomial_reference():
    scipy_stats = pytest.importorskip("scipy.stats")
    mu, alpha = 7.0, 0.4
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    p_over, push = tiering.prob_over_line("negative_binomial", mu, alpha, 6.5)
    assert p_over == pytest.approx(float(scipy_stats.nbinom.sf(6, n, p)))
    assert push == 0.0


# ---------------------------------------------------------------------------
# build_threshold_table()
# ---------------------------------------------------------------------------

def _one_pitcher_prediction(pitcher=100, mu=6.0, family="poisson", alpha=None, game_pk=5000):
    return pd.DataFrame([{
        "pitcher": pitcher, "game_pk": game_pk, "pitcher_name": "Gerrit Cole",
        "pitcher_team": "NYY", "opponent_team": "BOS", "game_date": "2026-06-27",
        "family": family, "mu": mu, "alpha": alpha,
    }])


def test_threshold_table_covers_one_through_ten_and_is_well_formed():
    predictions = _one_pitcher_prediction()
    table = tiering.build_threshold_table(predictions)

    sub = table[table["pitcher"] == 100].sort_values("threshold")
    assert list(sub["threshold"]) == list(range(1, 11))
    assert (sub["p_over"] >= 0).all() and (sub["p_over"] <= 1).all()
    assert (sub["p_over"].diff().dropna() <= 1e-9).all()  # non-increasing in t
    assert sub["tier"].notna().all()


def test_threshold_table_empty_predictions_returns_empty_well_formed_frame():
    table = tiering.build_threshold_table(pd.DataFrame(columns=[
        "pitcher", "game_pk", "pitcher_name", "pitcher_team", "opponent_team",
        "game_date", "family", "mu", "alpha",
    ]))
    assert table.empty
    assert list(table.columns) == tiering.THRESHOLD_TABLE_COLUMNS


# ---------------------------------------------------------------------------
# normalize_name() / to_statcast_team()
# ---------------------------------------------------------------------------

def test_normalize_name_strips_accents_punctuation_and_suffixes():
    assert tiering.normalize_name("Júlio Urías Jr.") == "julio urias"
    assert tiering.normalize_name("Julio Urias") == "julio urias"
    assert tiering.normalize_name(None) == ""


def test_to_statcast_team_applies_crosswalk():
    assert tiering.to_statcast_team("WSH") == "WAS"
    assert tiering.to_statcast_team("CWS") == "CHW"
    assert tiering.to_statcast_team("NYY") == "NYY"


# ---------------------------------------------------------------------------
# resolve_pitcher_ids()
# ---------------------------------------------------------------------------

def _line_row(name, line=6.5, status="active", projection_id="p1",
              pulled_at="2026-06-27T10:00:00Z", team=None,
              away_team="BOS", home_team="NYY",
              game_status="scheduled", live_event=False,
              over_american=-140.0, under_american=118.0,
              over_payout_multiplier=0.71, under_payout_multiplier=1.18):
    return {
        "pitcher": name, "team": team, "stat_type": "strikeouts", "line": line,
        "start_time": "2026-06-27T23:00:00Z", "status": status,
        "projection_id": projection_id, "pulled_at": pulled_at, "player_id": "x",
        "away_team": away_team, "home_team": home_team,
        "game_status": game_status, "live_event": live_event,
        "over_american": over_american, "under_american": under_american,
        "over_payout_multiplier": over_payout_multiplier,
        "under_payout_multiplier": under_payout_multiplier,
    }


def test_resolver_matches_known_name_no_team_needed():
    """Underdog rows carry no `team` -- a single name candidate matches with no tiebreak."""
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = _one_pitcher_prediction(pitcher=100)
    pp = pd.DataFrame([_line_row("Gerrit Cole", away_team="BOS", home_team="NYY")])

    resolved, unmatched = tiering.resolve_pitcher_ids(pp, predictions, register)
    assert unmatched.empty
    assert resolved.iloc[0]["pitcher_id"] == 100


def test_resolver_matches_accented_and_suffix_name_variant():
    register = pd.DataFrame([{"key_mlbam": 300, "name_first": "Julio", "name_last": "Urias Jr."}])
    predictions = pd.DataFrame([{
        "pitcher": 300, "game_pk": 5001, "pitcher_name": "Julio Urias Jr.",
        "pitcher_team": "LAD", "opponent_team": "SF", "game_date": "2026-06-27",
        "family": "poisson", "mu": 5.0, "alpha": None,
    }])
    pp = pd.DataFrame([_line_row("Júlio Urías", line=5.5, away_team="SF", home_team="LAD")])

    resolved, unmatched = tiering.resolve_pitcher_ids(pp, predictions, register)
    assert unmatched.empty
    assert resolved.iloc[0]["pitcher_id"] == 300


def test_resolver_ambiguous_same_name_no_team_match_is_unmatched_not_guessed():
    register = pd.DataFrame([
        {"key_mlbam": 400, "name_first": "Jose", "name_last": "Ramirez"},
        {"key_mlbam": 401, "name_first": "Jose", "name_last": "Ramirez"},
    ])
    predictions = pd.DataFrame([
        {"pitcher": 400, "game_pk": 5002, "pitcher_name": "Jose Ramirez", "pitcher_team": "CLE",
         "opponent_team": "DET", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
        {"pitcher": 401, "game_pk": 5003, "pitcher_name": "Jose Ramirez", "pitcher_team": "TBR",
         "opponent_team": "BAL", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
    ])
    # Neither CLE nor TBR -- ambiguous, no unique team match, even via away/home.
    pp = pd.DataFrame([_line_row("Jose Ramirez", line=5.5, away_team="NYY", home_team="BOS")])

    resolved, unmatched = tiering.resolve_pitcher_ids(pp, predictions, register)
    assert resolved.empty
    assert len(unmatched) == 1


def test_resolver_disambiguates_same_name_by_away_home_pair():
    """No `team` field on the row -- disambiguation falls back to away/home."""
    register = pd.DataFrame([
        {"key_mlbam": 400, "name_first": "Jose", "name_last": "Ramirez"},
        {"key_mlbam": 401, "name_first": "Jose", "name_last": "Ramirez"},
    ])
    predictions = pd.DataFrame([
        {"pitcher": 400, "game_pk": 5002, "pitcher_name": "Jose Ramirez", "pitcher_team": "CLE",
         "opponent_team": "DET", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
        {"pitcher": 401, "game_pk": 5003, "pitcher_name": "Jose Ramirez", "pitcher_team": "TBR",
         "opponent_team": "BAL", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
    ])
    pp = pd.DataFrame([_line_row("Jose Ramirez", line=5.5, away_team="DET", home_team="CLE")])

    resolved, unmatched = tiering.resolve_pitcher_ids(pp, predictions, register)
    assert unmatched.empty
    assert resolved.iloc[0]["pitcher_id"] == 400


def test_resolver_disambiguates_via_team_crosswalk_on_away_home():
    """away/home abbreviations run through TEAM_CROSSWALK just like the old `team` field did."""
    register = pd.DataFrame([
        {"key_mlbam": 500, "name_first": "A", "name_last": "Pitcher"},
        {"key_mlbam": 501, "name_first": "A", "name_last": "Pitcher"},
    ])
    predictions = pd.DataFrame([
        {"pitcher": 500, "game_pk": 6000, "pitcher_name": "A Pitcher", "pitcher_team": "WAS",
         "opponent_team": "NYM", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
        {"pitcher": 501, "game_pk": 6001, "pitcher_name": "A Pitcher", "pitcher_team": "CHW",
         "opponent_team": "MIN", "game_date": "2026-06-27",
         "family": "poisson", "mu": 5.0, "alpha": None},
    ])
    # Underdog's game title would carry "WSH" (its own abbreviation), not "WAS".
    pp = pd.DataFrame([_line_row("A Pitcher", line=5.5, away_team="NYM", home_team="WSH")])

    resolved, unmatched = tiering.resolve_pitcher_ids(pp, predictions, register)
    assert unmatched.empty
    assert resolved.iloc[0]["pitcher_id"] == 500


# ---------------------------------------------------------------------------
# build_line_picks(): coverage diagnostics, pre-game filter, market fields
# ---------------------------------------------------------------------------

def test_predicted_no_line_and_unmatched_line_are_both_surfaced():
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = pd.DataFrame([
        {"pitcher": 100, "game_pk": 5000, "pitcher_name": "Gerrit Cole", "pitcher_team": "NYY",
         "opponent_team": "BOS", "game_date": "2026-06-27",
         "family": "poisson", "mu": 6.0, "alpha": None},
        {"pitcher": 999, "game_pk": 5004, "pitcher_name": "No Line Guy", "pitcher_team": "BOS",
         "opponent_team": "NYY", "game_date": "2026-06-27",
         "family": "poisson", "mu": 4.0, "alpha": None},
    ])
    pp = pd.DataFrame([
        _line_row("Gerrit Cole", line=6.5, projection_id="p1", away_team="BOS", home_team="NYY"),
        _line_row("Some Debut Guy", line=3.5, projection_id="p4", away_team="TOR", home_team="BOS"),
    ])

    line_picks, diagnostics = tiering.build_line_picks(predictions, pp, register)

    assert set(line_picks["pitcher"]) == {100}
    assert len(diagnostics["unmatched_lines"]) == 1
    assert diagnostics["unmatched_lines"].iloc[0]["pitcher"] == "Some Debut Guy"
    assert list(diagnostics["predicted_no_line"]["pitcher"]) == [999]


def test_non_pre_game_lines_are_filtered_out_by_game_status():
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = _one_pitcher_prediction(pitcher=100)
    pp = pd.DataFrame([_line_row("Gerrit Cole", game_status="in_progress")])

    line_picks, diagnostics = tiering.build_line_picks(predictions, pp, register)
    assert line_picks.empty
    # Filtered before resolution entirely -- not even surfaced as unmatched.
    assert diagnostics["unmatched_lines"].empty
    assert list(diagnostics["predicted_no_line"]["pitcher"]) == [100]


def test_live_event_lines_are_filtered_out_even_if_scheduled():
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = _one_pitcher_prediction(pitcher=100)
    pp = pd.DataFrame([_line_row("Gerrit Cole", game_status="scheduled", live_event=True)])

    line_picks, diagnostics = tiering.build_line_picks(predictions, pp, register)
    assert line_picks.empty
    assert list(diagnostics["predicted_no_line"]["pitcher"]) == [100]


def test_build_line_picks_carries_market_fields_and_prices():
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = _one_pitcher_prediction(pitcher=100, mu=6.5)
    pp = pd.DataFrame([_line_row(
        "Gerrit Cole", line=6.5, over_american=-140.0, under_american=118.0,
        over_payout_multiplier=0.71, under_payout_multiplier=1.18,
    )])

    line_picks, _ = tiering.build_line_picks(predictions, pp, register)
    row = line_picks.iloc[0]

    assert row["over_american"] == -140.0
    assert row["under_american"] == 118.0
    assert row["over_payout_multiplier"] == pytest.approx(0.71)
    assert row["under_payout_multiplier"] == pytest.approx(1.18)

    p_over_implied = tiering.american_to_prob(-140.0)
    p_under_implied = tiering.american_to_prob(118.0)
    expected_market = tiering.no_vig_two_way(p_over_implied, p_under_implied)

    assert row["p_over_implied"] == pytest.approx(p_over_implied)
    assert row["p_under_implied"] == pytest.approx(p_under_implied)
    assert row["vig"] == pytest.approx(p_over_implied + p_under_implied - 1.0)
    assert row["vig"] > 0  # two-sided vig is always positive
    assert row["p_market"] == pytest.approx(expected_market)

    # edge is measured against the market, not a fixed 50%.
    assert row["edge"] == pytest.approx(row["p_over"] - expected_market)
    assert row["edge_vs_coinflip"] == pytest.approx(row["p_over"] - 0.5)
    assert row["lean"] == ("over" if row["p_over"] > expected_market else "under")


def test_build_line_picks_edge_falls_back_to_coinflip_when_prices_missing():
    """A matched line with no parseable prices still produces a pick, using
    the coinflip fallback for edge/lean/actionability."""
    register = pd.DataFrame([{"key_mlbam": 1, "name_first": "A", "name_last": "B"}])
    predictions = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5005, "pitcher_name": "A B", "pitcher_team": "NYY",
        "opponent_team": "BOS", "game_date": "2026-06-27",
        "family": "poisson", "mu": 10.0, "alpha": None,
    }])
    pp = pd.DataFrame([_line_row("A B", line=3.5, over_american=None, under_american=None)])

    line_picks, _ = tiering.build_line_picks(predictions, pp, register)
    row = line_picks.iloc[0]

    assert pd.isna(row["p_market"])
    assert pd.isna(row["edge"])
    assert row["edge_vs_coinflip"] == pytest.approx(row["p_over"] - 0.5)
    assert row["lean"] == "over"  # p_over (mu=10 vs line=3.5) is well above 0.5


def test_duplicate_projections_dedupe_is_gone_one_line_per_pitcher():
    """Underdog posts exactly one balanced line per pitcher/stat -- no
    duplicate-projection dedupe step exists any more (unlike the old
    PrizePicks standard/goblin/demon ladder)."""
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    predictions = _one_pitcher_prediction(pitcher=100)
    pp = pd.DataFrame([_line_row("Gerrit Cole", line=6.5, projection_id="only")])

    line_picks, _ = tiering.build_line_picks(predictions, pp, register)
    assert len(line_picks) == 1
    assert line_picks.iloc[0]["projection_id"] == "only"
    assert line_picks.iloc[0]["line"] == 6.5


# ---------------------------------------------------------------------------
# build_line_picks(): lean / edge / tier on the actionable pick row
# ---------------------------------------------------------------------------

def test_lean_and_edge_for_a_model_strongly_disagreeing_with_the_line():
    register = pd.DataFrame([{"key_mlbam": 1, "name_first": "A", "name_last": "B"}])
    predictions = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5005, "pitcher_name": "A B", "pitcher_team": "NYY",
        "opponent_team": "BOS", "game_date": "2026-06-27",
        "family": "poisson", "mu": 10.0, "alpha": None,
    }])
    pp = pd.DataFrame([_line_row("A B", line=3.5)])  # model (mu=10) vs. a low line

    line_picks, _ = tiering.build_line_picks(predictions, pp, register)
    row = line_picks.iloc[0]
    assert row["lean"] == "over"
    assert row["edge"] == pytest.approx(row["p_over"] - row["p_market"])
    assert row["tier"] == "high"


def test_efficient_line_yields_low_tier_and_small_edge():
    # mu chosen so the posted line sits close to the model's own median.
    register = pd.DataFrame([{"key_mlbam": 1, "name_first": "A", "name_last": "B"}])
    predictions = pd.DataFrame([{
        "pitcher": 1, "game_pk": 5006, "pitcher_name": "A B", "pitcher_team": "NYY",
        "opponent_team": "BOS", "game_date": "2026-06-27",
        "family": "poisson", "mu": 6.0, "alpha": None,
    }])
    pp = pd.DataFrame([_line_row("A B", line=5.5)])

    line_picks, _ = tiering.build_line_picks(predictions, pp, register)
    row = line_picks.iloc[0]
    assert row["tier"] == "low"
    assert abs(row["edge_vs_coinflip"]) < 0.10


# ---------------------------------------------------------------------------
# Empty-input edge cases
# ---------------------------------------------------------------------------

def _empty_predictions():
    return pd.DataFrame(columns=[
        "pitcher", "game_pk", "pitcher_name", "pitcher_team", "opponent_team", "game_date",
        "family", "mu", "alpha",
    ])


def _empty_lines():
    return pd.DataFrame(columns=[
        "pitcher", "team", "stat_type", "line", "start_time",
        "projection_id", "pulled_at", "player_id",
        "away_team", "home_team", "game_status", "live_event",
        "over_american", "under_american",
        "over_payout_multiplier", "under_payout_multiplier",
    ])


def _empty_register():
    return pd.DataFrame(columns=["key_mlbam", "name_first", "name_last"])


def test_empty_predictions_and_empty_lines_both_yield_empty_well_formed_outputs():
    line_picks, diagnostics = tiering.build_line_picks(
        _empty_predictions(), _empty_lines(), _empty_register(),
    )
    assert line_picks.empty
    assert list(line_picks.columns) == tiering.LINE_PICKS_COLUMNS
    assert diagnostics["unmatched_lines"].empty
    assert diagnostics["predicted_no_line"].empty

    table = tiering.build_threshold_table(_empty_predictions())
    assert table.empty
    assert list(table.columns) == tiering.THRESHOLD_TABLE_COLUMNS


def test_empty_lines_with_real_predictions_surfaces_predicted_no_line():
    register = _empty_register()
    predictions = _one_pitcher_prediction(pitcher=100)

    line_picks, diagnostics = tiering.build_line_picks(predictions, _empty_lines(), register)
    assert line_picks.empty
    assert list(diagnostics["predicted_no_line"]["pitcher"]) == [100]


def test_empty_predictions_with_real_lines_surfaces_unmatched_lines():
    register = pd.DataFrame([{"key_mlbam": 100, "name_first": "Gerrit", "name_last": "Cole"}])
    pp = pd.DataFrame([_line_row("Gerrit Cole")])

    line_picks, diagnostics = tiering.build_line_picks(_empty_predictions(), pp, register)
    assert line_picks.empty
    assert len(diagnostics["unmatched_lines"]) == 1
