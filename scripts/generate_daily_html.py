"""
scripts/generate_daily_html.py
Generate daily_results.html from pitcher_cards.csv + line_picks.csv.

Usage:
    python scripts/generate_daily_html.py [--date YYYY-MM-DD]

Defaults to today's date.  Output always written to daily_results.html
in the project root.
"""
import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=str(date.today()),
                   help="Game date YYYY-MM-DD (default: today)")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "daily_results.html"),
                   help="Output HTML path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(game_date: str):
    base = os.path.join(PROJECT_ROOT, "data", "processed", "predictions",
                        f"game_date={game_date}")
    cards_path = os.path.join(base, "pitcher_cards.csv")
    picks_path = os.path.join(base, "line_picks.csv")

    if not os.path.exists(cards_path):
        return None, None, f"No pitcher_cards.csv found for {game_date}"
    if not os.path.exists(picks_path):
        return None, None, f"No line_picks.csv found for {game_date}"

    cards = pd.read_csv(cards_path)
    picks = pd.read_csv(picks_path)
    return cards, picks, None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val, fmt=".1f", suffix="", na="—"):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return na
    try:
        return f"{val:{fmt}}{suffix}"
    except Exception:
        return na


def _pct(val, na="—"):
    return _fmt(val, ".1f", "%") if val is None or (isinstance(val, float) and math.isnan(val)) else f"{val*100:.1f}%"


def _weather_badge(row):
    if row.get("is_dome") == 1.0:
        return '<span class="env-badge env-dome">🏟 Dome</span>'
    temp = row.get("temp_f")
    wind = row.get("wind_mph")
    if temp is not None and not (isinstance(temp, float) and math.isnan(temp)):
        parts = [f"{temp:.0f}°F"]
        if wind is not None and not (isinstance(wind, float) and math.isnan(wind)):
            parts.append(f"{wind:.0f}mph wind")
        return f'<span class="env-badge env-outdoor">🌤 {" · ".join(parts)}</span>'
    return ""


def _game_ctx(row):
    total = row.get("game_total")
    if total is not None and not (isinstance(total, float) and math.isnan(total)):
        return f'<span class="ctx-tag">O/U {total:.1f}</span>'
    return ""


def _skill_html(row):
    items = []
    for label, col, mult in [
        ("SwStr%", "swstr_rate_last5", 100),
        ("CSW%",   "csw_rate_last5",   100),
        ("Putaway%","putaway_rate_last5",100),
    ]:
        v = row.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            items.append(f'<div class="stat-item"><span class="stat-label">{label} </span>'
                         f'<span class="stat-val">{v*mult:.1f}%</span></div>')
    return "".join(items)


def _opp_html(row):
    items = []
    lineup_k = row.get("opponent_lineup_k_rate_vs_hand")
    team_k   = row.get("opponent_k_rate_vs_hand_season")
    if lineup_k is not None and not (isinstance(lineup_k, float) and math.isnan(lineup_k)):
        items.append(f'<div class="stat-item"><span class="stat-label">Opp K% vs hand (lineup) </span>'
                     f'<span class="stat-val">{lineup_k*100:.1f}%</span></div>')
    elif team_k is not None and not (isinstance(team_k, float) and math.isnan(team_k)):
        items.append(f'<div class="stat-item"><span class="stat-label">Opp K% vs hand (team) </span>'
                     f'<span class="stat-val">{team_k*100:.1f}%</span></div>')
    ump = row.get("ump_k_factor")
    if ump is not None and not (isinstance(ump, float) and math.isnan(ump)) and abs(ump - 1.0) > 0.01:
        direction = "↑ K-friendly" if ump > 1.0 else "↓ K-suppressing"
        items.append(f'<div class="stat-item"><span class="stat-label">Ump factor </span>'
                     f'<span class="stat-val">{ump:.2f} {direction}</span></div>')
    return "".join(items)


def _fmt_american(v):
    v = int(round(v))
    return f"+{v}" if v > 0 else str(v)


def _market_price_tag(pick: dict) -> str:
    """
    Underdog posts a two-sided over/under line (no odds_type ladder to badge
    any more -- see tiering.py's 2026-08 migration). Show the two American
    prices plus the no-vig market probability instead.
    """
    over_am = pick.get("over_american")
    under_am = pick.get("under_american")
    has_over = over_am is not None and not (isinstance(over_am, float) and math.isnan(over_am))
    has_under = under_am is not None and not (isinstance(under_am, float) and math.isnan(under_am))
    if not (has_over and has_under):
        return '<span class="odds-type-tag ot-unknown">no line</span>'

    label = f"O {_fmt_american(over_am)} / U {_fmt_american(under_am)}"
    p_market = pick.get("p_market")
    if p_market is not None and not (isinstance(p_market, float) and math.isnan(p_market)):
        label += f" · mkt {p_market * 100:.0f}%"
    return f'<span class="odds-type-tag ot-priced">{label}</span>'


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

def _render_card(pick: dict, card: dict) -> str:
    lean    = pick.get("lean", "")
    p_over  = pick.get("p_over", 0.0) or 0.0
    p_under = pick.get("p_under", 0.0) or 0.0
    tier    = pick.get("tier", "low")
    line    = pick.get("line", 0)
    action  = pick.get("actionability", "no_action")
    conviction = pick.get("conviction")

    tier_cls = {"high": "tier-high", "medium": "tier-medium"}.get(tier, "tier-low")
    lean_pill = (
        '<span class="pill over">▲ OVER</span>' if lean == "over"
        else '<span class="pill under">▼ UNDER</span>'
    )
    bar_cls  = "bar-over" if lean == "over" else "bar-under"
    bar_w    = round(p_over * 100)

    # edge is p_over - p_market (no-vig, two-sided) when a market price was
    # matched; falls back to edge_vs_coinflip (p_over - 0.5) when it wasn't
    # -- see tiering.build_line_picks. Label reflects which one is showing.
    edge = pick.get("edge")
    has_edge = edge is not None and not (isinstance(edge, float) and math.isnan(edge))
    if has_edge:
        edge_label = "Edge vs market"
        edge_str = ("+" if edge >= 0 else "") + f"{edge*100:.1f}%"
    else:
        edge_label = "Edge vs coinflip (no market)"
        fallback = pick.get("edge_vs_coinflip") or 0.0
        edge_str = ("+" if fallback >= 0 else "") + f"{fallback*100:.1f}%"

    # actionability is "no_action" / "lean_over" / "lean_under" (tiering.py) --
    # there is no "actionable" value; any lean_* is the actionable case.
    act_badge = (
        '<span class="action-badge act-no">— No action</span>' if action == "no_action"
        else '<span class="action-badge act-yes">✓ Actionable</span>'
    )

    weather = _weather_badge(card)
    game_ctx = _game_ctx(card)
    skill    = _skill_html(card)
    opp      = _opp_html(card)
    mu       = card.get("mu")
    mu_str   = f"{mu:.2f}" if mu is not None and not (isinstance(mu, float) and math.isnan(mu)) else "—"

    top_meta = " ".join(filter(None, [weather, game_ctx]))

    conv_html = ""
    if conviction is not None and not (isinstance(conviction, float) and math.isnan(conviction)):
        conv_html = (f'<div class="stat-item"><span class="stat-label">Conviction </span>'
                     f'<span class="stat-val">{conviction:.2f}σ</span></div>')

    return f"""
    <div class="card">
      <div class="pitcher-name">{pick.get("pitcher_name", "?")} {_market_price_tag(pick)}</div>
      <div class="team">{pick.get("team", "?")} vs {card.get("opponent_team", "?")} · μ={mu_str} K</div>
      {f'<div class="top-meta">{top_meta}</div>' if top_meta else ""}
      <div class="line-row">
        <span class="line-val {tier_cls}">{line}K</span>
        {lean_pill}
        <span class="stat-val" style="font-size:0.72rem;color:#64748b;margin-left:auto;">{tier} · {act_badge}</span>
      </div>
      <div class="prob-bar-wrap"><div class="prob-bar {bar_cls}" style="width:{bar_w}%"></div></div>
      <div class="stats">
        <div class="stat-item"><span class="stat-label">P(over) </span><span class="stat-val">{p_over*100:.1f}%</span></div>
        <div class="stat-item"><span class="stat-label">P(under) </span><span class="stat-val">{p_under*100:.1f}%</span></div>
        <div class="stat-item"><span class="stat-label">{edge_label} </span><span class="stat-val">{edge_str}</span></div>
        <div class="stat-item"><span class="stat-label">Push mass </span><span class="stat-val">{_pct(pick.get("push_mass"))}</span></div>
        {conv_html}
        {skill}
        {opp}
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .subtitle { font-size: 0.8rem; color: #64748b; margin-bottom: 24px; }

  .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .section-header h2 { font-size: 1.05rem; font-weight: 700; color: #f8fafc; }
  .badge { font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 100px;
           text-transform: uppercase; letter-spacing: 0.05em; }
  .badge-ok { background: #14532d; color: #4ade80; }

  section { margin-bottom: 32px; }

  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px 16px; }
  .card .pitcher-name { font-size: 0.95rem; font-weight: 700; color: #f8fafc; }
  .card .team { font-size: 0.75rem; color: #64748b; margin-bottom: 6px; }
  .card .top-meta { font-size: 0.72rem; margin-bottom: 8px; display: flex; gap: 6px; flex-wrap: wrap; }
  .card .line-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .card .line-val { font-size: 1.3rem; font-weight: 800; }
  .pill { font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 100px;
          text-transform: uppercase; }
  .over  { background: #14532d; color: #4ade80; }
  .under { background: #450a0a; color: #fca5a5; }
  .tier-high   { color: #facc15; }
  .tier-medium { color: #fb923c; }
  .tier-low    { color: #94a3b8; }

  .card .prob-bar-wrap { height: 6px; background: #0f172a; border-radius: 3px;
                         margin-bottom: 8px; overflow: hidden; }
  .card .prob-bar { height: 100%; border-radius: 3px; }
  .bar-over  { background: #22c55e; }
  .bar-under { background: #ef4444; }

  .card .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .card .stat-item { font-size: 0.72rem; }
  .card .stat-label { color: #64748b; }
  .card .stat-val   { color: #cbd5e1; font-weight: 600; }

  .odds-type-tag { font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 3px;
                   text-transform: none; margin-left: 4px; }
  .ot-priced   { background: #1e3a5f; color: #60a5fa; }
  .ot-unknown  { background: #1e293b; color: #94a3b8; text-transform: uppercase; }

  .env-badge { display: inline-block; font-size: 0.68rem; font-weight: 600;
               padding: 2px 6px; border-radius: 4px; }
  .env-dome    { background: #1e293b; color: #7dd3fc; border: 1px solid #0ea5e9; }
  .env-outdoor { background: #1a2e1a; color: #86efac; border: 1px solid #22c55e; }

  .ctx-tag { display: inline-block; font-size: 0.68rem; font-weight: 600;
             padding: 2px 6px; border-radius: 4px; background: #292524; color: #d6d3d1;
             border: 1px solid #44403c; }

  .action-badge { font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 3px; }
  .act-yes { background: #14532d; color: #4ade80; }
  .act-no  { background: #1e293b; color: #64748b; }

  footer { text-align: center; font-size: 0.72rem; color: #334155; margin-top: 32px; }
"""


def _build_html(game_date: str, cards_html: str, n_picks: int, generated_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OddsOptimizer — {game_date}</title>
<style>{CSS}</style>
</head>
<body>

<h1>OddsOptimizer</h1>
<p class="subtitle">Pitcher Strikeout Projections — {game_date} · generated {generated_at}</p>

<section>
  <div class="section-header">
    <h2>{game_date} — Today's Picks</h2>
    <span class="badge badge-ok">{n_picks} picks</span>
  </div>
  <div class="cards">
{cards_html}
  </div>
</section>

<footer>OddsOptimizer · pitcher strikeout props · for personal research use</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    game_date = args.date

    cards_df, picks_df, err = _load(game_date)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    # Build a lookup: (pitcher, game_pk) → card row dict
    card_lookup = {
        (int(r["pitcher"]), int(r["game_pk"])): {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                                                  for k, v in r.items()}
        for _, r in cards_df.iterrows()
    }

    cards_html_parts = []
    for _, pick in picks_df.iterrows():
        key = (int(pick["pitcher"]), int(pick["game_pk"]))
        card = card_lookup.get(key, {})
        pick_dict = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                     for k, v in pick.items()}
        cards_html_parts.append(_render_card(pick_dict, card))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = _build_html(game_date, "\n".join(cards_html_parts), len(picks_df), generated_at)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {len(picks_df)} picks → {args.out}")


if __name__ == "__main__":
    main()
