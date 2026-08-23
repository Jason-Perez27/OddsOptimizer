"""
One-off diagnostic for the go-live verification gate (task #12) FAILing.

Not part of the daily/weekly cadence -- a throwaway triage tool to inspect the
*real* Underdog and StatsAPI payload shapes when `refresh --dry-run` fails,
so the parser fix in src/data/underdog_lines.py / src/data/probable_pitchers.py
is based on evidence, not a second guess.

Usage:
    python -m scripts.diagnose_live_sources --date 2026-06-29
"""

import argparse
import json
import sys

import requests

UNDERDOG_SPORT_ID = "MLB"


def diagnose_underdog():
    print("=" * 70)
    print("UNDERDOG")
    print("=" * 70)

    base = "https://api.underdogfantasy.com"
    headers = {"User-Agent": "Mozilla/5.0 (OddsOptimizer research project)"}

    print(f"\n--- GET /beta/v6/over_under_lines?sport_id={UNDERDOG_SPORT_ID} ---")
    try:
        resp = requests.get(
            f"{base}/beta/v6/over_under_lines", params={"sport_id": UNDERDOG_SPORT_ID},
            headers=headers, timeout=15,
        )
        print(f"HTTP {resp.status_code}")
        payload = resp.json()
        lines = payload.get("over_under_lines", []) or []
        games = payload.get("games", []) or []
        mlb_games = [g for g in games if g.get("sport_id") == UNDERDOG_SPORT_ID]
        print(f"  {len(mlb_games)} MLB game(s) (of {len(games)} game(s) in the payload)")

        stat_values = sorted({
            (line.get("over_under", {}) or {}).get("appearance_stat", {}).get("stat")
            for line in lines
        } - {None})
        print(f"  distinct appearance_stat.stat values present: {stat_values}")

        if lines:
            sample = lines[0]
            options = sample.get("options", []) or []
            print("\n  Sample raw line (first row) -- both options entries:")
            print(json.dumps(sample, indent=2, default=str))
            print("\n  Sample line's two options, american_price / payout_multiplier only:")
            for opt in options:
                print(
                    f"    choice={opt.get('choice')!r}  "
                    f"american_price={opt.get('american_price')!r}  "
                    f"payout_multiplier={opt.get('payout_multiplier')!r}"
                )
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

    diagnose_underdog()
    diagnose_statsapi(args.date)


if __name__ == "__main__":
    main()
