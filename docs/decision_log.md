# Decision Log

Dated record of major project decisions and the reasoning behind them. Newest entries at the top.

---

## 2026-06-26 — Sport, market, and target selection: MLB pitcher strikeout props

**Decision:** Build the first model around MLB pitcher strikeout props (over/under a sportsbook line), with a predicted strikeout-count distribution as the training target.

**Alternatives considered:**
- **NBA moneyline** — strong free data (`nba_api`, Kaggle historical odds), simple binary classification. Rejected because the NBA season had already ended, making live testing impossible at project start.
- **Esports (League of Legends)** — excellent free match data (Oracle's Elixir, updated after every pro match), but thin free odds/player-prop coverage and reliance on scraping less-official sources for some data. Kept as a candidate for a future second project.
- **MLB pitcher strikeout props (chosen)** — in season now, best-in-class free data tooling (`pybaseball`, MLB-StatsAPI), and The Odds API's free tier covers the exact prop market needed (`pitcher_strikeouts`) for current (not historical) odds.

**Why this wins:**
1. **Data availability:** `pybaseball` and MLB-StatsAPI require no API key and update same-day — fits the requirement to refresh pitcher stats after every outing and opponent batter stats as games are played.
2. **Live testability:** MLB is in season, so the pipeline can be validated against real upcoming games immediately, not just historical backtests.
3. **Difficulty:** A single-pitcher count-based regression (Poisson / negative binomial) is a practical, explainable baseline — no deep learning required.
4. **Portfolio value:** A working pre-game refresh pipeline plus an honestly evaluated probability model is a complete, demonstrable system end-to-end.

**Target metric reasoning:** Hit rate, ROI, and CLV are good *evaluation* metrics but poor *training* targets — they're noisy on small samples and don't directly optimize for prediction quality. The model trains on calibrated strikeout probability (via Poisson/NB regression), then hit rate / EV / ROI are computed afterward as honest evaluation, once results accumulate.

---

## 2026-06-26 — Skill/plugin requests not installable in-session

**Decision:** UI/UX Pro Max, Stop Slop, Superpowers, The Council (llm-council), and Playwright (as requested GitHub repos) cannot be installed directly into this session. `llm-council` in particular is not a packaged skill at all — it's a standalone FastAPI/React app requiring its own OpenRouter API key and local server, unrelated to the Claude Skills format.

**Resolution:** Applying the intent of these tools manually (direct, concise recommendations; multi-angle tradeoff comparisons) rather than installing them. Revisit if the user packages/installs them through Settings > Capabilities.
