"""
Pull current pitcher prop lines from Underdog Fantasy's public pick'em feed.

Line source migration (2026-08): PrizePicks' public projections endpoint now
returns HTTP 403 (DataDome bot protection) and is permanently unusable; it
also replaced its standard/goblin/demon ladder with flexible two-sided
alternates whose payouts are unpublished, so even a working endpoint no
longer supports the fixed-payout assumption the edge calculation was built
on. Underdog Fantasy's public over/under feed replaces it: free, no API key,
no auth, no User-Agent requirement, ~0.5s, ~1.4 MB.

    GET https://api.underdogfantasy.com/beta/v6/over_under_lines?sport_id=MLB

Payload contract (verified live -- see docs/decision_log.md):
Top-level keys: `over_under_lines`, `appearances`, `players`, `games`,
`solo_games`. There is NO `teams` key in this feed.

Join path:
  over_under_lines[].over_under.appearance_stat.appearance_id
    -> appearances[] (has player_id, match_id, team_id)
    -> players[]  (first_name, last_name)
    -> games[] via match_id (sport_id, title "AWAY @ HOME", scheduled_at, status)

Each over_under_line has exactly TWO entries in `options`, distinguished by
`choice` == "higher" / "lower". Each option carries: american_price (string,
e.g. "-148"), decimal_price, payout_multiplier (string, e.g. "0.89"),
status, selection_header (player name), selection_subheader, and its own
`updated_at` timestamp (2026-08 tick-log addition -- src/data/underdog_ticks.py
uses each side's updated_at for change detection/dedup; see that module's
docstring). A price move is visible as a NEW updated_at on one or both
sides, never as a full-snapshot diff.

Line-level fields: id, over_under_id, stat_value (the line, string),
line_type (always "balanced" in this feed -- no ladder to collapse), live_event
(bool), status. Nested: over_under.appearance_stat.stat / .display_stat,
over_under.has_alternates.

MLB pitcher stat keys (NOTE the third one -- key and display name disagree):
  strikeouts     -> "Strikeouts"
  pitch_outs     -> "Pitching Outs"
  runs_allowed   -> "Earned Runs Allowed"     <-- key is runs_allowed, NOT earned_runs_allowed
  walks_allowed  -> "Walks Allowed"
  hits_allowed   -> "Hits Allowed"

Facts established by live testing (trust these, do not re-verify):
- All lines are line_type "balanced"; exactly one line per over_under_id --
  there is no canonical-line selection to do downstream any more.
- `has_alternates: true` appears but alternates are NOT exposed in the public
  feed. The balanced two-sided line is the canonical object.
- Team abbreviations are not in this feed. Derive them from games[].title
  split on " @ " (away @ home); `team` is left None here, resolved downstream.
- Observed two-sided vig is ~7.5%, so no-vig normalization is required to get
  an honest market probability (see no_vig_two_way below).
- Underdog posts the slate progressively through the morning -- fewer games
  than expected is normal, not a bug.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

UNDERDOG_API_BASE = "https://api.underdogfantasy.com"
DEFAULT_SPORT_ID = "MLB"
DEFAULT_STAT = "strikeouts"

# Underdog stat key -> display name (for reference / CLI help only; flatten_lines
# filters on the key, never the display name -- see the runs_allowed/"Earned Runs
# Allowed" naming trap in the module docstring above).
MLB_PITCHER_STATS = {
    "strikeouts": "Strikeouts",
    "pitch_outs": "Pitching Outs",
    "runs_allowed": "Earned Runs Allowed",
    "walks_allowed": "Walks Allowed",
    "hits_allowed": "Hits Allowed",
}


def fetch_over_under_lines(sport_id: str = DEFAULT_SPORT_ID, timeout: int = 20) -> dict:
    url = f"{UNDERDOG_API_BASE}/beta/v6/over_under_lines"
    params = {"sport_id": sport_id}
    headers = {"User-Agent": "Mozilla/5.0 (OddsOptimizer research project)"}
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    if response.status_code == 403:
        raise RuntimeError(
            "Underdog rejected the request (403). Their public endpoint is "
            "unofficial and may have added bot protection or changed shape."
        )
    response.raise_for_status()
    return response.json()


FLATTEN_COLUMNS = [
    "pulled_at", "projection_id", "over_under_id", "player_id", "pitcher", "team",
    "stat_type", "line", "line_type",
    "over_american", "under_american", "over_decimal", "under_decimal",
    "over_payout_multiplier", "under_payout_multiplier", "over_status", "under_status",
    "over_updated_at", "under_updated_at",
    "game_id", "game_title", "away_team", "home_team", "start_time",
    "game_status", "live_event", "status",
]

_NUMERIC_COLUMNS = [
    "line", "over_american", "under_american", "over_decimal", "under_decimal",
    "over_payout_multiplier", "under_payout_multiplier",
]


def flatten_lines(payload: dict, stat: str = DEFAULT_STAT, sport_id: str = DEFAULT_SPORT_ID) -> pd.DataFrame:
    """
    Flatten Underdog's over_under_lines payload into one row per line,
    filtered to a single stat key (see MLB_PITCHER_STATS -- filter on the
    KEY, e.g. "runs_allowed", never the display name).

    Every line in this feed is already the single canonical balanced
    two-sided line for that pitcher/stat -- unlike PrizePicks' old
    standard/goblin/demon ladder, there is nothing to collapse here.
    """
    lines = payload.get("over_under_lines", []) if isinstance(payload, dict) else []
    appearances = {a.get("id"): a for a in (payload.get("appearances", []) if isinstance(payload, dict) else [])}
    players = {p.get("id"): p for p in (payload.get("players", []) if isinstance(payload, dict) else [])}
    games = {g.get("id"): g for g in (payload.get("games", []) if isinstance(payload, dict) else [])}

    pulled_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for line in lines:
        over_under = line.get("over_under", {}) or {}
        appearance_stat = over_under.get("appearance_stat", {}) or {}
        if appearance_stat.get("stat") != stat:
            continue

        appearance_id = appearance_stat.get("appearance_id")
        appearance = appearances.get(appearance_id, {}) or {}
        player_id = appearance.get("player_id")
        match_id = appearance.get("match_id")

        player = players.get(player_id, {}) or {}
        pitcher = None
        if player:
            full_name = f"{player.get('first_name', '') or ''} {player.get('last_name', '') or ''}".strip()
            pitcher = full_name or None

        game = games.get(match_id, {}) or {}
        if sport_id and game.get("sport_id") not in (sport_id, None):
            continue

        game_title = game.get("title")
        away_team = home_team = None
        if game_title and " @ " in game_title:
            away_team, home_team = game_title.split(" @ ", 1)

        options = line.get("options", []) or []
        by_choice = {opt.get("choice"): opt for opt in options}
        over_opt = by_choice.get("higher", {}) or {}
        under_opt = by_choice.get("lower", {}) or {}

        rows.append({
            "pulled_at":               pulled_at,
            "projection_id":           line.get("id"),
            "over_under_id":           line.get("over_under_id"),
            "player_id":               player_id,
            "pitcher":                 pitcher,
            "team":                    None,  # not in this feed -- caller resolves via away/home
            "stat_type":               stat,
            "line":                    line.get("stat_value"),
            "line_type":               line.get("line_type"),
            "over_american":           over_opt.get("american_price"),
            "under_american":          under_opt.get("american_price"),
            "over_decimal":            over_opt.get("decimal_price"),
            "under_decimal":           under_opt.get("decimal_price"),
            "over_payout_multiplier":  over_opt.get("payout_multiplier"),
            "under_payout_multiplier": under_opt.get("payout_multiplier"),
            "over_status":             over_opt.get("status"),
            "under_status":            under_opt.get("status"),
            "over_updated_at":         over_opt.get("updated_at"),
            "under_updated_at":        under_opt.get("updated_at"),
            "game_id":                 match_id,
            "game_title":              game_title,
            "away_team":               away_team,
            "home_team":               home_team,
            "start_time":              game.get("scheduled_at"),
            "game_status":             game.get("status"),
            "live_event":              line.get("live_event"),
            "status":                  line.get("status"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_raw(df: pd.DataFrame, stat: str = DEFAULT_STAT) -> str:
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(raw_dir, f"underdog_{stat}_{timestamp}.csv")
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Odds helpers -- shared by tiering.py's market-probability / edge calc.
# ---------------------------------------------------------------------------

def american_to_prob(american_price) -> float:
    """American odds -> implied probability (still carries the vig)."""
    a = float(american_price)
    if a < 0:
        return (-a) / (-a + 100.0)
    return 100.0 / (a + 100.0)


def no_vig_two_way(p_over_implied: float, p_under_implied: float) -> float:
    """
    Normalize a two-sided market's implied probabilities (which sum to
    > 1.0 because of the vig) down to a no-vig P(over): divide by the sum
    of both sides' implied probabilities.
    """
    total = float(p_over_implied) + float(p_under_implied)
    return float(p_over_implied) / total


def payout_to_decimal(multiplier) -> float:
    """
    Convention: Underdog's `payout_multiplier` (e.g. "0.89") is the PROFIT
    multiplier on a winning unit stake -- a 1-unit stake that wins returns
    1 + multiplier units total. This is the standard "decimal odds minus 1"
    (i.e. net profit) convention, distinct from decimal odds themselves
    (where 1.0 means "stake back only"). decimal_odds = 1 + multiplier.
    """
    return 1.0 + float(multiplier)


def main():
    parser = argparse.ArgumentParser(description="Pull current Underdog Fantasy pitcher prop lines.")
    parser.add_argument(
        "--stat", default=DEFAULT_STAT,
        choices=sorted(MLB_PITCHER_STATS),
        help="Underdog stat key, e.g. strikeouts, walks_allowed, runs_allowed",
    )
    parser.add_argument("--sport-id", default=DEFAULT_SPORT_ID)
    args = parser.parse_args()

    print(f"Pulling current Underdog {args.stat!r} lines for sport_id={args.sport_id}...")
    payload = fetch_over_under_lines(sport_id=args.sport_id)

    df = flatten_lines(payload, stat=args.stat, sport_id=args.sport_id)
    if df.empty:
        print(f"No {args.stat!r} lines found right now (Underdog posts the slate progressively).")
        sys.exit(0)

    n_pitchers = df["pitcher"].nunique()
    out_path = save_raw(df, stat=args.stat)
    print(f"Pulled {len(df)} lines for {n_pitchers} pitcher(s).")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
