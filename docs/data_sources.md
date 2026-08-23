# Data Sources

All sources below are free for this project's scale. None require payment. Anything optional/paid is labeled explicitly.

## Pitcher and batter performance data

### pybaseball
- **What:** Pulls Statcast pitch-level data, pitcher game logs, and batter stats from Baseball Savant / Baseball Reference / FanGraphs.
- **Cost:** Free, MIT-licensed, no API key.
- **Coverage:** Every MLB game from 2015–present, updated same-day.
- **Use in this project:** Pitcher rolling strikeout rates, pitch mix, opponent batter strikeout vulnerability.
- **Link:** https://github.com/jldbc/pybaseball

### MLB-StatsAPI
- **What:** Python wrapper around the official, public MLB Stats API.
- **Cost:** Free, no API key.
- **Coverage:** Probable starting pitchers, live boxscores, rosters, schedules.
- **Use in this project:** Pre-game pipeline — confirming today's probable pitcher and opposing lineup.
- **Link:** https://github.com/toddrob99/MLB-StatsAPI

## Prop lines

### Underdog Fantasy (primary, as of 2026-08)
- **What:** Pick'em-style DFS app. Posts a genuine two-sided over/under line per prop (e.g. "Pitcher Strikeouts: 6.5, over −140 / under +118"), unlike PrizePicks' old single flat line — the two-sided price is what lets this project compute a real no-vig market probability instead of just comparing against a bare number.
- **Cost:** Free. Pulled via Underdog's public pick'em feed (`api.underdogfantasy.com/beta/v6/over_under_lines`), which is **unofficial** — undocumented, not a published developer API, subject to their Terms of Service, and may change shape without notice.
- **Coverage:** Current/upcoming lines only, posted progressively through the morning as the slate firms up (a partial slate early in the day is normal, not a bug) — no historical backfill available this way.
- **Use in this project:** Primary (and only) line source for all pitcher props (strikeouts first, then pitching outs / earned runs allowed / walks allowed). `src/data/underdog_lines.py` converts each side's American odds to implied probability and no-vig-normalizes the pair to get `p_market`, which the model's own probability is compared against to compute edge.
- **Use responsibly:** personal/research use only, consistent with this repo's disclaimer — not a commercial scraping operation.

### PrizePicks (decommissioned, 2026-06-27 – 2026-08)
PrizePicks was the original line source. It was replaced after its public projections endpoint began permanently returning HTTP 403 (DataDome bot protection), and — separately — after it replaced its standard/goblin/demon ladder with unpublished-payout two-sided alternates that no longer fit this project's fixed-payout edge calculation. See `docs/decision_log.md` for the full history. The old `src/data/prizepicks_lines.py` module itself has been removed from the codebase — git history is the historical record — since nothing imported it and a dead, unreachable module in `src/data/` was more confusing than useful.

## What is explicitly out of scope for now

- **Historical odds backfill** — not needed for the current MVP since we're comparing live predictions to live lines, not re-deriving CLV from history.
- **Any paid data provider** (SportsDataIO, OpticOdds, Sportradar, The Odds API, etc.) — would only be considered later if the project needs deeper historical odds coverage, and would always be labeled as optional/paid before use.

## Storage policy

Raw and processed data files are never committed to GitHub (`data/` is gitignored). No API keys or credentials are stored in this repo.
