# Strikeout Feature Engineering — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (2026-06-27 entry), task #6

## Goal

Build a per-pitcher-game feature table for the strikeout model (task #7), with no
data leakage: every feature for game N is computed using only information known
before game N starts (games 1..N-1, and pre-game opponent/context info).

## Inputs

- `src/data/pitcher_logs.py` output — pitch-level Statcast rows per pitcher
  (already built, task #4).
- MLB-StatsAPI — probable starting pitcher handedness, probable/confirmed
  opponent lineup, game date/time (day/night), home/away, ballpark.
- Derived team-level batting logs (aggregated the same way as pitcher logs, but
  per batter/team) for opponent K-rate features.

## Modules

1. **`src/features/game_logs.py`** — aggregate pitch-level Statcast rows into
   one row per pitcher-game: strikeouts, innings pitched, pitch count, whiff
   rate, average fastball velocity, game date, opponent team, home/away,
   day/night.
   - `innings_pitched` is *estimated*, not pulled directly — Statcast has no
     per-pitcher-game IP field. It's derived from an `OUT_EVENTS` mapping
     (event string -> outs produced, e.g. `strikeout` -> 1,
     `grounded_into_double_play` -> 2, `triple_play` -> 3) summed across the
     game's plate appearances and divided by 3.
   - `pitcher_throws` (L/R) is sourced from Statcast's `p_throws` column when
     present, and `None` otherwise (kept optional so older callers/tests that
     don't supply `p_throws` still work). This is what `opponent_features.py`
     uses to compute a team's K-rate against left- vs right-handed pitching —
     a different axis from `strikeouts_vs_LHB`/`RHB`, which split by the
     *batter's* stand, not the pitcher's throwing hand.

2. **`src/features/rolling_features.py`** — for each pitcher-game, compute
   pitcher-side features using only prior games (shift-before-aggregate, no
   leakage):
   - `k_rate_last5` — rolling K-rate, last 5 starts.
   - `k_rate_season` — season-to-date K-rate.
   - `k_rate_vs_LHB`, `k_rate_vs_RHB` — season-to-date K-rate split by batter
     handedness faced.
   - `k_rate_home`, `k_rate_away` — season-to-date K-rate split by game site.
   - `k_rate_vs_opponent_career` — career K-rate vs today's specific opponent;
     **falls back to `k_rate_season`** if fewer than 3 career starts exist
     against that team (avoids overfitting to a 1-2-game sample).
   - `ip_avg_last5`, `pitch_count_avg_last5`, `whiff_rate_last5`,
     `velo_avg_last5` — rolling averages, last 5 starts.

3. **`src/features/opponent_features.py`** — for each pitcher-game, compute
   opponent-side features using only the opposing team's games before this
   date:
   - `opponent_k_rate_last10` — team K-rate, last 10 team games (teams play
     near-daily, so a larger window than the pitcher's last-5-starts is
     needed for a comparable sample size).
   - `opponent_k_rate_vs_hand_season` — opponent's season-to-date K-rate
     specifically against the handedness of today's starting pitcher.
   - `opponent_k_rate_home`, `opponent_k_rate_away` — opponent's season-to-date
     K-rate split by whether they're home or away today.
   - Implementation note: there's no separate batter-level ingestion pipeline
     yet, so team-level history is derived from the existing per-pitcher-game
     table via `build_team_game_logs()`, which groups by
     `(opponent_team, game_pk)` and sums across every pitcher (starter +
     relievers) that team faced in that game. The team's site is the
     opposite of the pitcher's site. The team-game's representative pitcher
     hand (for `opponent_k_rate_vs_hand_season`) is taken from whichever
     pitcher faced the most batters in that game (a starter proxy) — a v1
     approximation when relievers of a different hand also pitched.
     `opponent_k_rate_vs_hand_season` is computed via a per-row lookup (not a
     column join), since the hand filter is today's pitcher's own hand,
     which varies row to row even for the same opponent.

4. **`src/features/park_factors.py`** — `park_k_factor`: a ballpark strikeout
   index (park's K-rate relative to league average across all pitchers).
   Computed from our own accumulated game logs once enough season data
   exists; falls back to a small static reference table of known public
   park factors for early-season cold starts. The static table is documented
   inline with its source.

5. **`src/features/build_features.py`** — joins all of the above into one
   training table, one row per pitcher-game, with `strikeouts` (actual game
   total) as the label.

## Final feature list (v1)

`k_rate_last5`, `k_rate_season`, `k_rate_vs_LHB`, `k_rate_vs_RHB`,
`k_rate_home`, `k_rate_away`, `k_rate_vs_opponent_career` (with fallback),
`ip_avg_last5`, `pitch_count_avg_last5`, `whiff_rate_last5`, `velo_avg_last5`,
`rest_days`, `home_away`, `day_night`, `opponent_k_rate_last10`,
`opponent_k_rate_vs_hand_season`, `opponent_k_rate_home`,
`opponent_k_rate_away`, `park_k_factor`.

## Deferred (conscious scope cut, not an oversight)

- **Umpire strikeout tendency** — real signal, but no clean free per-game
  umpire-assignment data source identified yet.
- **Bullpen/usage patterns, weather** — out of scope until the baseline model
  (#7) is working and we know what's worth the added complexity.

## Leakage guardrail (the thing tests must verify)

Every rolling/season/split feature for game N must be computed using only
games strictly before game N's date. This is the primary thing
`tests/test_rolling_features.py` and `tests/test_opponent_features.py` need to
assert directly (e.g. by checking that a feature value for game 3 doesn't
change when game 4's data is added to the input).

## Testing approach

Unit tests per module with small hand-built DataFrames (no network calls):
- `game_logs.py`: pitch-level rows aggregate correctly to per-game rows.
- `rolling_features.py`: rolling/season/split values are correct, leakage
  guardrail holds, and the opponent-career fallback triggers at the right
  sample-size threshold.
- `opponent_features.py`: same leakage guardrail, last-10 window is correct,
  handedness weighting is correct.
- `park_factors.py`: static fallback table is used when season data is too
  thin; computed value is used once enough games exist.
- `build_features.py`: join produces one row per pitcher-game with no
  unexpected nulls in required columns.
