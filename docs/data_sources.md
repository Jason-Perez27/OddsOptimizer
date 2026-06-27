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

## Odds data

### The Odds API
- **What:** Aggregates sportsbook odds across multiple bookmakers.
- **Cost:** Free tier = 500 credits/month. The `pitcher_strikeouts` market is available on the free tier for **current** odds only.
- **Important limit:** Historical odds (for backtesting past seasons) require a paid plan. This project avoids that dependency by only pulling current lines, day-of, for live comparison against model predictions — not bulk historical backfill.
- **Credit cost:** usage = (number of markets) × (number of regions) per call. Querying `pitcher_strikeouts` for one region costs 1 credit per call.
- **Link:** https://the-odds-api.com/

## What is explicitly out of scope for now

- **Historical odds backfill** — paid tier only; not needed for the current MVP since we're comparing live predictions to live lines, not re-deriving CLV from history.
- **Any paid data provider** (SportsDataIO, OpticOdds, Sportradar, etc.) — would only be considered later if the project needs deeper historical odds coverage, and would always be labeled as optional/paid before use.

## Storage policy

Raw and processed data files are never committed to GitHub (`data/` is gitignored). API keys (e.g., for The Odds API) are never committed — they belong in a local `.env` file that is also gitignored.
