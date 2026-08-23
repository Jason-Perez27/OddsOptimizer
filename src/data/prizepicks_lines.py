"""
DECOMMISSIONED (2026-08): PrizePicks' public projections endpoint now
returns HTTP 403 (DataDome bot protection) and is permanently unusable; it
also replaced this standard/goblin/demon ladder with unpublished-payout
two-sided alternates, so even a working endpoint would no longer fit the
fixed-payout edge calculation this module's downstream consumers assumed.
The live pipeline (src/pipeline/refresh.py) has been migrated to
src/data/underdog_lines.py -- see that module's docstring and
docs/data_sources.md for the replacement contract. This module is kept,
unmodified and untested-against-live-traffic beyond what's below, purely as
a historical record of the pre-migration line source; nothing in the
pipeline imports it any more.

Pull current pitcher prop lines from PrizePicks public projections endpoint.

odds_type and alternate lines (2026-06-29): PrizePicks posts multiple
projections per pitcher per stat, distinguished by the odds_type field:
  - standard: the real over/under line (one per pitcher/stat).
  - goblin:   alternate with a lower line and a discounted payout.
  - demon:    alternate with a higher line and a boosted payout.
flatten_projections emits all lines (standard + goblin + demon), each
carrying odds_type. Canonical single-line selection was previously done
downstream in tiering._select_canonical_line (removed in the migration --
Underdog posts exactly one balanced line per pitcher/stat).
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

PRIZEPICKS_API_BASE = "https://api.prizepicks.com"
MLB_LEAGUE_ID = 2
DEFAULT_STAT_TYPE = "Pitcher Strikeouts"


def fetch_projections(league_id: int = MLB_LEAGUE_ID) -> dict:
    url = f"{PRIZEPICKS_API_BASE}/projections"
    params = {"league_id": league_id, "per_page": 250}
    headers = {"User-Agent": "Mozilla/5.0 (OddsOptimizer research project)"}
    response = requests.get(url, params=params, headers=headers, timeout=15)
    if response.status_code == 403:
        raise RuntimeError(
            "PrizePicks rejected the request (403). Their public endpoint is "
            "unofficial and may have added bot protection or changed shape."
        )
    response.raise_for_status()
    return response.json()


def _build_player_lookup(included: list) -> dict:
    lookup = {}
    for item in included:
        if item.get("type") != "new_player":
            continue
        attrs = item.get("attributes", {})
        lookup[item.get("id")] = {
            "player_name": attrs.get("name"),
            "team": attrs.get("team"),
            "position": attrs.get("position"),
        }
    return lookup


def flatten_projections(payload: dict, stat_type: str = DEFAULT_STAT_TYPE) -> pd.DataFrame:
    """
    Flatten PrizePicks JSON:API projections payload into one row per
    projection line, filtered to a single stat_type. All lines are emitted
    (standard, goblin, and demon alternates), each carrying odds_type so
    downstream consumers can select a canonical line per pitcher.
    """
    data = payload.get("data", [])
    included = payload.get("included", [])
    player_lookup = _build_player_lookup(included)

    pulled_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for projection in data:
        attrs = projection.get("attributes", {})
        if attrs.get("stat_type") != stat_type:
            continue

        player_rel = projection.get("relationships", {}).get("new_player", {}).get("data")
        player_id = player_rel.get("id") if player_rel else None
        player_info = player_lookup.get(player_id, {})

        rows.append({
            "pulled_at":     pulled_at,
            "projection_id": projection.get("id"),
            "player_id":     player_id,
            "pitcher":       player_info.get("player_name"),
            "team":          player_info.get("team"),
            "stat_type":     attrs.get("stat_type"),
            "odds_type":     attrs.get("odds_type"),
            "line":          attrs.get("line_score"),
            "start_time":    attrs.get("start_time"),
            "status":        attrs.get("status"),
        })

    return pd.DataFrame(rows)


def save_raw(df: pd.DataFrame, stat_type: str = DEFAULT_STAT_TYPE) -> str:
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    safe_stat = stat_type.replace(" ", "_").lower()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(raw_dir, f"prizepicks_{safe_stat}_{timestamp}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pull current PrizePicks pitcher prop lines.")
    parser.add_argument(
        "--stat-type",
        default=DEFAULT_STAT_TYPE,
        help="e.g. Pitcher Strikeouts, Pitching Outs, Earned Runs Allowed",
    )
    parser.add_argument("--league-id", type=int, default=MLB_LEAGUE_ID)
    args = parser.parse_args()

    print(f"Pulling current PrizePicks {args.stat_type!r} lines for league_id={args.league_id}...")
    payload = fetch_projections(league_id=args.league_id)

    df = flatten_projections(payload, stat_type=args.stat_type)
    if df.empty:
        print(f"No {args.stat_type!r} projections found right now.")
        sys.exit(0)

    n_pitchers = df["pitcher"].nunique()
    out_path = save_raw(df, stat_type=args.stat_type)
    print(f"Pulled {len(df)} lines for {n_pitchers} pitcher(s).")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
