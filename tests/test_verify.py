"""
Tests for src.pipeline.verify (task #11 step 5, "Live-data verification
gate"). Testing approach item #11 (verbatim, the contract these tests pin):
"a well-formed live-payload fixture passes; a malformed one (wrong sport_id
result, missing stat key, missing probablePitcher.id) is flagged with the
specific failure, and --dry-run writes nothing."

All fixtures here are hand-built raw payloads in the exact shape documented
in src/data/underdog_lines.py (Underdog `over_under_lines`/`appearances`/
`players`/`games`) and src/data/probable_pitchers.py (StatsAPI
`dates[].games[].teams.{home,away}`) -- never live network calls, per the
project's no-network-in-tests rule.
"""

import pandas as pd
import pytest

from src.pipeline import verify


# ---------------------------------------------------------------------------
# Fixtures: Underdog raw payload shape
# ---------------------------------------------------------------------------

def _well_formed_underdog_payload():
    return {
        "over_under_lines": [
            {
                "id": "line-1",
                "over_under_id": "ou-1",
                "stat_value": "6.5",
                "line_type": "balanced",
                "live_event": False,
                "status": "active",
                "over_under": {
                    "appearance_stat": {"appearance_id": "app-1", "stat": "strikeouts",
                                        "display_stat": "Strikeouts"},
                    "has_alternates": True,
                },
                "options": [
                    {"choice": "higher", "american_price": "-148", "decimal_price": 1.68,
                     "payout_multiplier": "0.68", "status": "active"},
                    {"choice": "lower", "american_price": "+124", "decimal_price": 2.24,
                     "payout_multiplier": "1.24", "status": "active"},
                ],
            },
            {
                "id": "line-2",
                "over_under_id": "ou-2",
                "stat_value": "16.5",
                "line_type": "balanced",
                "live_event": False,
                "status": "active",
                "over_under": {
                    "appearance_stat": {"appearance_id": "app-1", "stat": "pitch_outs",
                                        "display_stat": "Pitching Outs"},
                    "has_alternates": False,
                },
                "options": [
                    {"choice": "higher", "american_price": "-120", "decimal_price": 1.83,
                     "payout_multiplier": "0.83", "status": "active"},
                    {"choice": "lower", "american_price": "+100", "decimal_price": 2.00,
                     "payout_multiplier": "1.00", "status": "active"},
                ],
            },
        ],
        "appearances": [
            {"id": "app-1", "player_id": "p-1", "match_id": "m-1", "team_id": "t-away"},
        ],
        "players": [
            {"id": "p-1", "first_name": "Gerrit", "last_name": "Cole"},
        ],
        "games": [
            {"id": "m-1", "sport_id": "MLB", "title": "NYY @ BOS",
             "scheduled_at": "2026-06-28T23:05:00Z", "status": "scheduled"},
        ],
        "solo_games": [],
    }


def _underdog_payload_missing_over_under_lines_key():
    # e.g. the wrong sport_id returns a different top-level shape, or the
    # endpoint itself changed -- "over_under_lines" missing entirely.
    return {"errors": [{"detail": "sport not found"}]}


def _underdog_payload_with_no_strikeouts():
    # well-formed shape, but no line actually has stat=="strikeouts" -- the
    # kind of mismatch a wrong sport_id or a stat-key rename produces.
    payload = _well_formed_underdog_payload()
    for line in payload["over_under_lines"]:
        line["over_under"]["appearance_stat"]["stat"] = "pitch_outs"
    return payload


# ---------------------------------------------------------------------------
# Fixtures: StatsAPI raw schedule payload shape
# ---------------------------------------------------------------------------

def _well_formed_schedule_payload():
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 12345,
                        "gameDate": "2026-06-28T23:05:00Z",
                        "teams": {
                            "away": {
                                "team": {"abbreviation": "NYY"},
                                "probablePitcher": {
                                    "id": 543037,
                                    "fullName": "Gerrit Cole",
                                    "pitchHand": {"code": "R"},
                                },
                            },
                            "home": {
                                "team": {"abbreviation": "BOS"},
                                "probablePitcher": {
                                    "id": 999999,
                                    "fullName": "Some Guy",
                                },
                            },
                        },
                    }
                ]
            }
        ]
    }


def _schedule_payload_missing_probable_pitcher_id():
    # hydration didn't come through -- "probablePitcher" absent on both sides,
    # the exact failure mode the spec names by name.
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 12345,
                        "gameDate": "2026-06-28T23:05:00Z",
                        "teams": {
                            "away": {"team": {"abbreviation": "NYY"}},
                            "home": {"team": {"abbreviation": "BOS"}},
                        },
                    }
                ]
            }
        ]
    }


def _empty_schedule_payload():
    return {"dates": []}


# ---------------------------------------------------------------------------
# verify_underdog_lines
# ---------------------------------------------------------------------------

def test_verify_underdog_lines_passes_on_well_formed_payload():
    check = verify.verify_underdog_lines(lambda: _well_formed_underdog_payload())
    assert check["passed"] is True
    assert check["name"] == "underdog_lines"


def test_verify_underdog_lines_fails_with_specific_detail_on_missing_key():
    check = verify.verify_underdog_lines(lambda: _underdog_payload_missing_over_under_lines_key())
    assert check["passed"] is False
    assert "over_under_lines" in check["detail"]


def test_verify_underdog_lines_fails_when_no_matching_stat():
    check = verify.verify_underdog_lines(lambda: _underdog_payload_with_no_strikeouts())
    assert check["passed"] is False
    assert "strikeouts" in check["detail"]


def test_verify_underdog_lines_fails_on_non_dict_payload():
    check = verify.verify_underdog_lines(lambda: ["not", "a", "dict"])
    assert check["passed"] is False
    assert "over_under_lines" in check["detail"]


def test_verify_underdog_lines_catches_fetch_exception_without_raising():
    def boom():
        raise ConnectionError("network unreachable")

    check = verify.verify_underdog_lines(boom)
    assert check["passed"] is False
    assert "fetch failed" in check["detail"]
    assert "network unreachable" in check["detail"]


def test_verify_underdog_lines_reports_priced_count():
    check = verify.verify_underdog_lines(lambda: _well_formed_underdog_payload())
    assert check["passed"] is True
    assert "1 with a parsed over price" in check["detail"] or "1 'strikeouts' line" in check["detail"]


# ---------------------------------------------------------------------------
# verify_schedule_hydration
# ---------------------------------------------------------------------------

def test_verify_schedule_hydration_passes_on_well_formed_payload():
    check = verify.verify_schedule_hydration(
        lambda game_date: _well_formed_schedule_payload(), "2026-06-28"
    )
    assert check["passed"] is True
    assert check["name"] == "statsapi_schedule"


def test_verify_schedule_hydration_fails_with_specific_detail_on_missing_probable_pitcher_id():
    check = verify.verify_schedule_hydration(
        lambda game_date: _schedule_payload_missing_probable_pitcher_id(), "2026-06-28"
    )
    assert check["passed"] is False
    assert "probablePitcher.id" in check["detail"]


def test_verify_schedule_hydration_fails_on_empty_schedule():
    check = verify.verify_schedule_hydration(
        lambda game_date: _empty_schedule_payload(), "2026-06-28"
    )
    assert check["passed"] is False


def test_verify_schedule_hydration_catches_fetch_exception_without_raising():
    def boom(game_date):
        raise TimeoutError("statsapi timed out")

    check = verify.verify_schedule_hydration(boom, "2026-06-28")
    assert check["passed"] is False
    assert "fetch failed" in check["detail"]
    assert "statsapi timed out" in check["detail"]


# ---------------------------------------------------------------------------
# verify_live_sources (combined report)
# ---------------------------------------------------------------------------

def test_verify_live_sources_passes_when_both_checks_pass():
    report = verify.verify_live_sources(
        lines_fetcher=lambda: _well_formed_underdog_payload(),
        schedule_fetcher=lambda game_date: _well_formed_schedule_payload(),
        game_date="2026-06-28",
    )
    assert report["passed"] is True
    assert len(report["checks"]) == 2
    assert all(c["passed"] for c in report["checks"])


def test_verify_live_sources_fails_when_lines_check_fails_even_if_schedule_passes():
    report = verify.verify_live_sources(
        lines_fetcher=lambda: _underdog_payload_missing_over_under_lines_key(),
        schedule_fetcher=lambda game_date: _well_formed_schedule_payload(),
        game_date="2026-06-28",
    )
    assert report["passed"] is False
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["underdog_lines"]["passed"] is False
    assert by_name["statsapi_schedule"]["passed"] is True


def test_verify_live_sources_fails_when_schedule_check_fails_even_if_lines_passes():
    report = verify.verify_live_sources(
        lines_fetcher=lambda: _well_formed_underdog_payload(),
        schedule_fetcher=lambda game_date: _schedule_payload_missing_probable_pitcher_id(),
        game_date="2026-06-28",
    )
    assert report["passed"] is False
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["underdog_lines"]["passed"] is True
    assert by_name["statsapi_schedule"]["passed"] is False


def test_verify_live_sources_defaults_game_date_to_today_when_not_given():
    seen = {}

    def schedule_fetcher(game_date):
        seen["game_date"] = game_date
        return _well_formed_schedule_payload()

    verify.verify_live_sources(
        lines_fetcher=lambda: _well_formed_underdog_payload(),
        schedule_fetcher=schedule_fetcher,
    )
    assert seen["game_date"] is not None


# ---------------------------------------------------------------------------
# refresh --dry-run integration: verify it writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_verification_writes_nothing(tmp_path, monkeypatch):
    from src.pipeline import refresh

    monkeypatch.chdir(tmp_path)

    report = refresh.run_dry_run(
        lines_fetcher=lambda: _well_formed_underdog_payload(),
        schedule_fetcher=lambda game_date: _well_formed_schedule_payload(),
        game_date="2026-06-28",
    )

    assert report["passed"] is True
    assert list(tmp_path.iterdir()) == []


def test_dry_run_verification_reports_failure_without_raising_or_writing(tmp_path, monkeypatch):
    from src.pipeline import refresh

    monkeypatch.chdir(tmp_path)

    report = refresh.run_dry_run(
        lines_fetcher=lambda: _underdog_payload_missing_over_under_lines_key(),
        schedule_fetcher=lambda game_date: _schedule_payload_missing_probable_pitcher_id(),
        game_date="2026-06-28",
    )

    assert report["passed"] is False
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# balanced single-line gate (2026-08 migration -- no more odds_type ladder)
# ---------------------------------------------------------------------------

def test_verify_underdog_lines_fails_on_missing_required_column(monkeypatch):
    """Simulate an Underdog shape change (e.g. dropping american_price) by
    making flatten_lines return a frame missing a required column."""
    import src.pipeline.verify as verify_mod

    def _broken_flatten(payload, stat=None, sport_id=None):
        return pd.DataFrame([{"pitcher": "Gerrit Cole", "stat_type": "strikeouts", "line": 6.5}])

    monkeypatch.setattr(verify_mod, "flatten_lines", _broken_flatten)
    check = verify.verify_underdog_lines(lambda: _well_formed_underdog_payload())
    assert check["passed"] is False
    assert "missing expected column" in check["detail"]
