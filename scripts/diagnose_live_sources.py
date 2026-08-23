"""
One-off diagnostic for the go-live verification gate (task #12) FAILing.

Not part of the daily/weekly cadence -- a throwaway triage tool to inspect the
*real* PrizePicks and StatsAPI payload shapes when `refresh --dry-run` fails,
so the parser fix in src/data/prizepicks_lines.py / src/data/probable_pitchers.py
is based on evidence, not a second guess.

Usage:
    python -m scripts.diagnose_live_sources --date 2026-06-29
"""

import argparse
import json
import sys

import requests


def diagnose_prizepicks():
    print("=" * 70)
    print("PRIZEPICKS")
    print("=" * 70)

    base = "https://api.prizepicks.com"
    headers = {"User-Agent": "Mozilla/5.0 (OddsOptimizer research project)"}

    print("\n--- GET /leagues (find MLB's real league id) ---")
    try:
        resp = requests.get(f"{base}/leagues", headers=headers, timeout=15)
        print(f"HTTP {resp.status_code}")
        leagues = resp.json().get("data", [])
        for lg in leagues:
            name = lg.get("attributes", {}).get("name", "")
            if "mlb" in name.lower() or "baseball" in name.lower():
                print(f"  id={lg.get('id')}  name={name!r}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\n--- GET /projections?league_id=2 (current assumed MLB id) ---")
    try:
        resp = requests.get(
            f"{base}/projections", params={"league_id": 2, "per_page": 250},
            headers=headers, timeout=15,
        )
        print(f"HTTP {resp.status_code}")
        payload = resp.json()
        data = payload.get("data", [])
        print(f"  {len(data)} projection(s) total for league_id=2")
        stat_types = sorted({d.get("attributes", {}).get("stat_type") for d in data})
        print(f"  distinct stat_type values present: {stat_types}")
        if data:
            print("\n  Sample raw projection 'attributes' (first row):")
            print(json.dumps(data[0].get("attributes", {}), indent=2, default=str))
    except Exception as exc:
        print(f"  ERROR: {exc}")


def diagnose_statsapi(game_date: str):
    print("\n" + "=" * 70)
    print("STATSAPI SCHEDULE")
    print("=" * 70)

    try:
        import statsapi
    except ImportError:
        print("  MLB-StatsAPI not installed (pip install MLB-StatsAPI) -- "
              "falling back to a raw requests.get against the schedule endpoint.")
        try:
            resp = requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"sportId": 1, "date": game_date, "hydrate": "probablePitcher"},
                timeout=15,
            )
            print(f"  HTTP {resp.status_code}")
            payload = resp.json()
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return
    else:
        try:
            payload = statsapi.get("schedule", {"sportId": 1, "date": game_date, "hydrate": "probablePitcher"})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return

    dates = payload.get("dates", [])
    print(f"  {len(dates)} date entry(ies) for {game_date}")
    if not dates:
        print("  No games found for this date at all (genuine off-day, or wrong date param).")
        return

    games = dates[0].get("games", [])
    print(f"  {len(games)} game(s)")
    if games:
        print("\n  Sample raw 'teams' block (first game):")
        print(json.dumps(games[0].get("teams", {}), indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="Diagnose live-source shape for the verification gate.")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    diagnose_prizepicks()
    diagnose_statsapi(args.date)


if __name__ == "__main__":
    main()
