"""Tests for src/backtest/corpus.py -- historical corpus assembly (task #11)."""

import os

import pandas as pd
import pytest

from src.backtest.corpus import build_corpus, filter_starters, _date_windows


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def test_date_windows_splits_range_into_consecutive_non_overlapping_chunks():
    windows = _date_windows("2026-04-01", "2026-04-20", window_days=7)
    assert windows == [
        ("2026-04-01", "2026-04-07"),
        ("2026-04-08", "2026-04-14"),
        ("2026-04-15", "2026-04-20"),  # final window shorter than 7 days
    ]


def test_date_windows_single_day_range():
    assert _date_windows("2026-04-01", "2026-04-01", window_days=7) == [("2026-04-01", "2026-04-01")]


def test_date_windows_empty_when_start_after_end():
    assert _date_windows("2026-04-10", "2026-04-01") == []


# ---------------------------------------------------------------------------
# build_corpus: fetch, cache, resume, concatenate (spec testing items 8-9)
# ---------------------------------------------------------------------------

def _fake_pitch_row(game_pk, pitcher=1):
    return {"pitcher": pitcher, "game_pk": game_pk, "game_date": "2026-04-01", "events": "strikeout"}


def test_build_corpus_fetches_each_window_once_and_concatenates(tmp_path):
    calls = []

    def fetcher(win_start, win_end):
        calls.append((win_start, win_end))
        return pd.DataFrame([_fake_pitch_row(game_pk=len(calls))])

    cache_dir = os.path.join(tmp_path, "statcast")
    corpus = build_corpus(
        "2026-04-01", "2026-04-20",
        statcast_fetcher=fetcher, cache_dir=cache_dir, window_days=7,
    )

    assert len(calls) == 3  # 3 weekly windows over a 20-day range
    assert len(corpus) == 3  # one fixture row per window, concatenated
    assert set(corpus["game_pk"]) == {1, 2, 3}


def test_build_corpus_does_not_refetch_a_cached_window(tmp_path):
    calls = []

    def fetcher(win_start, win_end):
        calls.append((win_start, win_end))
        return pd.DataFrame([_fake_pitch_row(game_pk=1)])

    cache_dir = os.path.join(tmp_path, "statcast")

    build_corpus("2026-04-01", "2026-04-07", statcast_fetcher=fetcher, cache_dir=cache_dir, window_days=7)
    assert len(calls) == 1

    # Second call, same range -- the window is already cached on disk, so
    # the fetcher must NOT be called again (this is the resume guarantee).
    build_corpus("2026-04-01", "2026-04-07", statcast_fetcher=fetcher, cache_dir=cache_dir, window_days=7)
    assert len(calls) == 1


def test_build_corpus_resumes_after_a_failed_window_leaving_earlier_windows_cached(tmp_path):
    calls = []

    def flaky_fetcher(win_start, win_end):
        calls.append((win_start, win_end))
        if win_start == "2026-04-08":
            raise RuntimeError("simulated rate limit")
        return pd.DataFrame([_fake_pitch_row(game_pk=len(calls))])

    cache_dir = os.path.join(tmp_path, "statcast")

    with pytest.raises(RuntimeError):
        build_corpus("2026-04-01", "2026-04-20", statcast_fetcher=flaky_fetcher, cache_dir=cache_dir, window_days=7)

    # The first window succeeded and should be cached on disk even though
    # the overall run raised on the second window.
    assert os.path.exists(os.path.join(cache_dir, "2026-04-01_2026-04-07.csv"))
    assert not os.path.exists(os.path.join(cache_dir, "2026-04-08_2026-04-14.csv"))

    calls_before_retry = len(calls)

    def fixed_fetcher(win_start, win_end):
        calls.append((win_start, win_end))
        return pd.DataFrame([_fake_pitch_row(game_pk=len(calls))])

    corpus = build_corpus("2026-04-01", "2026-04-20", statcast_fetcher=fixed_fetcher, cache_dir=cache_dir, window_days=7)

    # Only the two remaining (previously-failed-or-unattempted) windows were
    # fetched on retry -- the first window's cache was reused, not refetched.
    assert len(calls) - calls_before_retry == 2
    assert len(corpus) == 3


def test_build_corpus_empty_range_returns_empty_frame(tmp_path):
    def fetcher(win_start, win_end):
        raise AssertionError("fetcher should never be called for an empty range")

    cache_dir = os.path.join(tmp_path, "statcast")
    corpus = build_corpus("2026-04-10", "2026-04-01", statcast_fetcher=fetcher, cache_dir=cache_dir)
    assert corpus.empty


# ---------------------------------------------------------------------------
# Starter filter
# ---------------------------------------------------------------------------

def _game_row(pitcher, game_pk, pitcher_team, batters_faced, game_date="2026-04-01"):
    return {
        "pitcher": pitcher, "game_pk": game_pk, "game_date": game_date,
        "pitcher_team": pitcher_team, "opponent_team": "OPP",
        "batters_faced": batters_faced,
    }


def test_filter_starters_picks_max_batters_faced_per_team_per_game():
    df = pd.DataFrame([
        _game_row(pitcher=1, game_pk=100, pitcher_team="NYY", batters_faced=22),  # starter
        _game_row(pitcher=2, game_pk=100, pitcher_team="NYY", batters_faced=4),   # reliever
        _game_row(pitcher=3, game_pk=100, pitcher_team="NYY", batters_faced=3),   # reliever
        _game_row(pitcher=10, game_pk=100, pitcher_team="BOS", batters_faced=25),  # opposing starter
    ])
    starters = filter_starters(df)
    assert set(starters["pitcher"]) == {1, 10}
    assert len(starters) == 2  # one starter per (pitcher_team, game_pk)


def test_filter_starters_doubleheader_keeps_games_distinct():
    df = pd.DataFrame([
        _game_row(pitcher=1, game_pk=100, pitcher_team="NYY", batters_faced=22),
        _game_row(pitcher=2, game_pk=101, pitcher_team="NYY", batters_faced=20),  # game 2 of DH
    ])
    starters = filter_starters(df)
    assert len(starters) == 2
    assert set(starters["game_pk"]) == {100, 101}


def test_filter_starters_min_batters_faced_floor_drops_short_outings():
    df = pd.DataFrame([
        _game_row(pitcher=1, game_pk=100, pitcher_team="NYY", batters_faced=10),  # opener, only 10 BF
    ])
    starters = filter_starters(df, min_batters_faced=15)
    assert starters.empty


def test_filter_starters_empty_input():
    out = filter_starters(pd.DataFrame())
    assert out.empty
