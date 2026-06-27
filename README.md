# OddsOptimizer

An analytics project that builds a baseline ML model for MLB pitcher strikeout props. This is a research and engineering portfolio project — it is **not** betting advice, and no part of it executes real wagers.

## Project goal

Predict a starting pitcher's expected strikeout count for an upcoming game, convert that prediction into a probability of going over/under the sportsbook's posted line, and honestly evaluate that probability against real outcomes over time (hit rate, calibration, and a flat-bet ROI backtest).

## Why MLB pitcher strikeout props

- **Sport:** MLB — in season now, and `pybaseball` / the MLB Stats API provide rich, free, no-key pitcher and batter data that updates same-day.
- **Market:** pitcher strikeouts — a count-based prop with a clean free-tier odds source (The Odds API includes `pitcher_strikeouts` on its free plan for current lines).
- **Target:** predicted strikeout distribution (Poisson / negative-binomial regression), not raw hit-rate or ROI. Hit rate and ROI are noisy evaluation metrics, not good training targets.

Full reasoning is logged in [`docs/decision_log.md`](docs/decision_log.md).

## Data sources

See [`docs/data_sources.md`](docs/data_sources.md) for the full list, what's free, and what's rate-limited.

- [pybaseball](https://github.com/jldbc/pybaseball) — Statcast pitch-level data, pitcher game logs (free, no key)
- [MLB-StatsAPI](https://github.com/toddrob99/MLB-StatsAPI) — probable pitchers, live boxscores, rosters (free, no key)
- [The Odds API](https://the-odds-api.com/) — current `pitcher_strikeouts` market on the free tier (current odds only, not historical)

No raw datasets are committed to this repo — `data/` is gitignored. No API keys or credentials are stored in this repo.

## Project structure

```
OddsOptimizer/
├── data/              # gitignored — raw/ and processed/
├── docs/              # data sources, decision log
├── src/
│   ├── data/          # ingestion (pybaseball, MLB-StatsAPI, odds feed)
│   ├── pipeline/       # pre-game refresh script
│   ├── features/       # feature engineering
│   ├── models/          # baseline regression model
│   ├── evaluation/       # calibration, Brier/log loss
│   └── backtest/         # ROI/hit-rate backtest
├── notebooks/          # exploratory work
├── tests/
├── configs/
└── reports/             # dated results writeups
```

## Status

Early planning/setup stage. See the project task list for the current implementation roadmap.

## Disclaimer

This project is for educational and analytics purposes only. It does not guarantee profitable betting outcomes, and nothing here should be treated as financial or gambling advice.