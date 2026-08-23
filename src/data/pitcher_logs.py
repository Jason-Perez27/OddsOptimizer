"""
Pull a pitcher's season-to-date Statcast logs using pybaseball.

This is the first ingestion step in the OddsOptimizer pipeline: prove that we
can reliably pull pitch-level Statcast data for a single pitcher, for a given
season, with no API key. Output is saved locally under data/raw/ (gitignored)
as a CSV, never committed to GitHub.

Usage:
    python -m src.data.pitcher_logs --name "Gerrit Cole" --season 2026
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher


def lookup_pitcher_id(full_name: str) -> int:
    """
    Resolve a pitcher's full name (e.g. 'Gerrit Cole') to their MLBAM player ID
    using pybaseball's playerid_lookup, which matches on last/first name.
    """
    parts = full_name.strip().split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected 'First Last' name format, got: {full_name!r}")
    first, last = parts[0], parts[1]

    matches = playerid_lookup(last, first)
    if matches.empty:
        raise ValueError(f"No player found matching name: {full_name!r}")

    # If multiple matches (common names), take the most recently active player.
    matches = matches.sort_values("mlb_played_last", ascending=False)
    player_id = int(matches.iloc[0]["key_mlbam"])
    return player_id


def get_pitcher_season_logs(
    full_name: str,
    season: int,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Pull all Statcast pitch-level rows for a pitcher across a given season,
    from opening day through end_date (defaults to today).
    """
    player_id = lookup_pitcher_id(full_name)

    start_dt = f"{season}-03-01"  # safely before any season's opening day
    end_dt = end_date or date.today().isoformat()

    df = statcast_pitcher(start_dt, end_dt, player_id)
    return df


def get_pitcher_logs_by_id(player_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull Statcast pitch-level rows for a pitcher directly by MLBAM id, over an
    explicit [start_date, end_date] window -- no name lookup round-trip.

    Task #10 (outcome tracking, spec section 1): predictions already carry
    the MLBAM id, so settlement pulls realized outcomes by id, not by name.
    This is a thin wrapper over statcast_pitcher -- the only network call on
    the settlement path (src/pipeline/settle.py), injected there as
    `outcome_fetcher` and never unit-tested directly (no network in tests).
    """
    return statcast_pitcher(start_date, end_date, player_id)


def save_raw(df: pd.DataFrame, full_name: str, season: int) -> str:
    """Save the pulled data to data/raw/, creating the directory if needed."""
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    safe_name = full_name.replace(" ", "_").lower()
    out_path = os.path.join(raw_dir, f"{safe_name}_{season}_statcast.csv")
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Pull a pitcher's season Statcast logs.")
    parser.add_argument("--name", required=True, help="Pitcher full name, e.g. 'Gerrit Cole'")
    parser.add_argument("--season", required=True, type=int, help="Season year, e.g. 2026")
    args = parser.parse_args()

    print(f"Looking up {args.name}...")
    df = get_pitcher_season_logs(args.name, args.season)

    if df.empty:
        print(f"No Statcast rows found for {args.name} in {args.season}. "
              f"Pipeline ran successfully but returned no data (player may not "
              f"have pitched yet, or name lookup matched the wrong record).")
        sys.exit(0)

    out_path = save_raw(df, args.name, args.season)
    print(f"Pulled {len(df)} pitch-level rows for {args.name} ({args.season}).")
    print(f"Saved to {out_path}")
    print(f"Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")


if __name__ == "__main__":
    main()
