# Runbook — Pre-game decision dashboard

A local, read-only web app that renders each morning's `refresh` output: per-pitcher
1+→10+ over probabilities, projected K, the two-sided Underdog line (over/under
American odds, no-vig market probability, vig), and decision/workload stats (avg
innings, expected batters, avg pitches, K-rate, opponent vs-hand, park factor, …).

It is **presentation only** — it never triggers a refresh and never writes to the
partitions (the morning snapshot stays authoritative; see `docs/decision_log.md`,
2026-06-29). Stdlib-only server: no `pip install` beyond the project's existing
requirements.

## Run it (VS Code)

1. Make sure a slate exists on disk (run the pipeline once if not):

   ```
   python -m src.pipeline.refresh
   ```

   This writes `data/processed/predictions/game_date=YYYY-MM-DD/`, now including the
   new `pitcher_cards.csv` (the decision/workload stats the dashboard reads).

2. Launch the dashboard:

   - Open `run_dashboard.py` and press Run/▶, or press F5 and pick the **Dashboard**
     config, or
   - Terminal: `python run_dashboard.py`

   It serves http://127.0.0.1:8000 and opens your browser. Ctrl+C to stop.

### Bundled VS Code launch config

The session couldn't write `.vscode/launch.json` directly (protected path). Create
it once with this content so F5 works:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Dashboard",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/run_dashboard.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    },
    {
      "name": "Refresh (today's slate)",
      "type": "debugpy",
      "request": "launch",
      "module": "src.pipeline.refresh",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}"
    }
  ]
}
```

(`debugpy` ships with the VS Code Python extension. If you don't debug, plain
`python run_dashboard.py` is all you need.)

## Options

```
python run_dashboard.py --port 8500            # different port
python run_dashboard.py --no-browser           # don't auto-open a tab
python run_dashboard.py --host 0.0.0.0         # expose on your LAN (deliberate)
python run_dashboard.py --processed-dir PATH   # point at another data root
```

## What you see

- KPI row: pitchers on slate, lines available, no line (sweep only), median vig,
  model age (with a stale warning past 7 days).
- Slate list: sortable by projected K, edge, P(over line), or tier. Click a pitcher
  to open the detail panel.
- Detail: the 1+→10+ probability ladder (the posted-line threshold is highlighted),
  the line block (over/under American odds, no-vig P(market), edge vs market, push,
  tier), and the decision-stat grid.
- If Underdog lines were unavailable at refresh time (`line_source_error` in the
  manifest), a banner says so and only the sweep shows.

## Endpoints (if you want the raw JSON)

- `GET /api/dates` — available slate dates
- `GET /api/slate?date=YYYY-MM-DD` — full slate dict (defaults to latest)
- `GET /healthz`

## Auto-refresh

The page re-reads the latest partition every 5 minutes (and on Reload). It reflects
whatever the scheduled morning `refresh` wrote — it does not re-pull lines itself.

## Optional: host it

The app is stateless and read-only over the partition files. A minimal container:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "run_dashboard.py", "--host", "0.0.0.0"]
```

The container needs `data/processed/predictions/` mounted or synced in. Hosting is
optional — local run is the supported path (keeps the project free/low-infra).
