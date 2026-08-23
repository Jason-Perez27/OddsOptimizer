"""
Closing-line resolution and closing-line-value (CLV) metrics (2026-08 CLV
feature). Reads the tick log src/data/underdog_ticks.py writes and turns it
into (a) a per-line closing snapshot and (b) CLV metrics comparing each
day's picked (open) line/price against its resolved close.

PURELY ADDITIVE / read-only: this module never writes line_picks.csv, the
predictions partition, or the frozen morning decision snapshot, and it never
triggers a model run or a refresh. It only reads a day's already-written
line_picks.csv (see src.predictions.tiering.LINE_PICKS_COLUMNS) and the
day's tick log (see src.data.underdog_ticks.TICK_COLUMNS).

Why CLV, and why per-game not per-slate closing (validation rationale, also
surfaced in src/backtest/report.py's CLV section and the README): a single
slate's first-pitch times can spread ~5-6 hours, so "the line at 19:00 ET"
is the close for a 19:05 first pitch and wide-open for a 23:10 one. Every
line's close is resolved independently, keyed off THAT line's own game's
`start_time`.

CRITICAL non-negotiable (per the design that commissioned this module -- do
not "fix" this by repricing across a line move): when the LINE itself moves
between open and close (e.g. 5.5 -> 6.5), the two no-vig market
probabilities are for DIFFERENT EVENTS (P(over 5.5) vs P(over 6.5)) and are
NOT comparable. `compute_clv` never blends them -- `price_move_toward_lean`
is left NaN whenever `line_move != 0`, and `market_agreed` is derived from
the DIRECTION of the line move instead. The model's own predicted
distribution is never used to reprice the open line at the close threshold
either -- doing so would let the model being validated contaminate its own
validation metric.

Honest caveat (see also src/backtest/report.py and the README): Underdog is
a DFS pick'em operator, not a sportsbook -- its lines move partly on its own
customer exposure/liability management, not purely on new information. CLV
measured against Underdog is therefore weaker evidence of model sharpness
than CLV against a sharp two-way sportsbook would be. src.data.vegas's ESPN
closing odds cover game totals/moneylines only (no player props), so they
cannot serve as a sharper CLV benchmark for pitcher K props either -- this
is documented, not silently worked around.
"""

import numpy as np
import pandas as pd

from src.predictions.tiering import normalize_name

# ---------------------------------------------------------------------------
# Closing-line resolver
# ---------------------------------------------------------------------------

# A closing capture is flagged "stale" (rather than "good") when the last
# eligible pre-game tick is more than this many minutes before first pitch --
# the poller may simply have missed the true close (e.g. a gap in cadence E's
# polling), and downstream analysis needs to be able to exclude those rather
# than silently treat a stale snapshot as "the close".
STALE_CLOSE_MINUTES = 60.0

CLOSING_LINES_COLUMNS = [
    "over_under_id", "pitcher", "stat_type", "line",
    "over_american", "under_american",
    "over_payout_multiplier", "under_payout_multiplier",
    "p_over_implied", "p_under_implied", "p_market",
    "game_id", "game_title", "start_time",
    "close_poll_at", "minutes_before_first_pitch", "close_quality",
]


def _parse_ts(v):
    """Parse an ISO-8601 timestamp (tz-aware or 'Z'-suffixed) to a pandas Timestamp; NaT on failure."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return pd.NaT
    try:
        return pd.Timestamp(v)
    except (ValueError, TypeError):
        return pd.NaT


def _to_bool(v) -> bool:
    """Robust bool coercion for a value that may have round-tripped through CSV as a string."""
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


def resolve_closing_lines(ticks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve one closing snapshot per over_under_id from a tick log (the
    concatenated ticks.csv rows for one game date -- see
    src.data.underdog_ticks.TICK_COLUMNS for the expected input schema).

    The close is the LAST tick satisfying:
        poll_at < start_time  AND  live_event == False  AND  game_status == "scheduled"
    i.e. the most recent genuinely pre-game snapshot for that market. Also
    emits `close_poll_at` (when that tick was captured) and
    `minutes_before_first_pitch` (how long before first pitch it was) so a
    stale capture is visible rather than silently treated as authoritative --
    `close_quality` is "stale" when that gap exceeds STALE_CLOSE_MINUTES,
    else "good".

    Returns an empty, correctly-columned frame for an empty/None input, or
    when no tick for a market ever satisfies the pre-game condition (e.g. the
    poller only ever caught that market live).
    """
    if ticks_df is None or ticks_df.empty:
        return pd.DataFrame(columns=CLOSING_LINES_COLUMNS)

    df = ticks_df.copy()
    df["_poll_dt"] = df["poll_at"].apply(_parse_ts)
    df["_start_dt"] = df["start_time"].apply(_parse_ts)
    df["_live_event"] = df["live_event"].apply(_to_bool)
    df["_game_status"] = df["game_status"].astype(str)

    pre_game = (
        df["_poll_dt"].notna() & df["_start_dt"].notna()
        & (df["_poll_dt"] < df["_start_dt"])
        & (~df["_live_event"])
        & (df["_game_status"] == "scheduled")
    )
    eligible = df[pre_game]
    if eligible.empty:
        return pd.DataFrame(columns=CLOSING_LINES_COLUMNS)

    idx = eligible.groupby("over_under_id")["_poll_dt"].idxmax()
    closing = eligible.loc[idx].copy()

    closing["close_poll_at"] = closing["poll_at"]
    minutes_before = (closing["_start_dt"] - closing["_poll_dt"]).dt.total_seconds() / 60.0
    closing["minutes_before_first_pitch"] = minutes_before
    closing["close_quality"] = np.where(minutes_before > STALE_CLOSE_MINUTES, "stale", "good")

    return closing[CLOSING_LINES_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Open (line_picks) <-> close join + per-pick CLV
# ---------------------------------------------------------------------------

CLV_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "team", "start_time",
    "tier", "lean", "actionability", "edge_open",
    "line_open", "line_close", "line_move",
    "p_market_open", "p_market_close",
    "price_move_toward_lean", "market_agreed",
    "close_quality", "minutes_before_first_pitch",
]


def compute_clv(line_picks_df: pd.DataFrame, closing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join a day's `line_picks.csv` (the OPEN -- the frozen morning pick) to
    `resolve_closing_lines`'s output (the CLOSE) and compute per-pick CLV.

    Join key (per the design's "over_under_id, fall back to pitcher +
    game_pk" instruction, adapted to what's actually on disk): `line_picks.csv`
    (see src.predictions.tiering.LINE_PICKS_COLUMNS) does NOT carry Underdog's
    raw `over_under_id` -- it was never added to that schema, and this module
    does not add it either (additive-only: line_picks.csv is untouched). The
    tick log's `game_id` is Underdog's own match UUID, not MLB's `game_pk`,
    so a true `(pitcher, game_pk)` fallback isn't available either. The join
    actually used, and the only one both frames can support today, is on
    NORMALIZED PITCHER NAME (via src.predictions.tiering.normalize_name --
    the same normalizer tiering.py itself uses to match Underdog rows to
    predicted pitchers) when `over_under_id` isn't present on both sides.
    If a future schema change adds `over_under_id` to line_picks.csv, this
    function prefers it automatically (checked at call time, not hardcoded).
    Within one game date this is unambiguous for the vast majority of
    slates; a same-day doubleheader starter appearing twice would collide on
    name alone -- a known, documented limitation, not silently papered over.

    CRITICAL: when `line_move != 0` (the line itself moved between open and
    close), `price_move_toward_lean` is left NaN -- the two no-vig market
    probabilities price DIFFERENT events and are not comparable (see module
    docstring). `market_agreed` is derived from the line-move DIRECTION
    instead in that case, and from the price-move SIGN only when the line
    was unchanged (line_move == 0).

    Returns an empty, correctly-columned frame if either input is empty, or
    if nothing matches.
    """
    if line_picks_df is None or line_picks_df.empty or closing_df is None or closing_df.empty:
        return pd.DataFrame(columns=CLV_COLUMNS)

    picks = line_picks_df.copy()
    closes = closing_df.copy()

    use_over_under_id = "over_under_id" in picks.columns and "over_under_id" in closes.columns
    if use_over_under_id:
        picks["_join_key"] = picks["over_under_id"].astype(str)
        closes["_join_key"] = closes["over_under_id"].astype(str)
    else:
        picks["_join_key"] = picks["pitcher_name"].apply(normalize_name)
        closes["_join_key"] = closes["pitcher"].apply(normalize_name)

    close_renamed = closes.rename(columns={
        "line": "_close_line",
        "p_market": "_close_p_market",
        "close_quality": "_close_quality",
        "minutes_before_first_pitch": "_close_minutes_before_first_pitch",
    })[[
        "_join_key", "_close_line", "_close_p_market",
        "_close_quality", "_close_minutes_before_first_pitch",
    ]]

    # A join key could in principle map to >1 closing row (e.g. a
    # doubleheader name collision on the name-based fallback) -- keep the
    # first deterministically rather than silently exploding the pick count
    # via a many-to-many merge.
    close_renamed = close_renamed.drop_duplicates(subset="_join_key", keep="first")

    merged = picks.merge(close_renamed, on="_join_key", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=CLV_COLUMNS)

    rows = []
    for _, row in merged.iterrows():
        line_open = float(row["line"]) if pd.notna(row["line"]) else np.nan
        line_close = float(row["_close_line"]) if pd.notna(row["_close_line"]) else np.nan
        line_move = (line_close - line_open) if (pd.notna(line_open) and pd.notna(line_close)) else np.nan

        p_open = row.get("p_market")
        p_close = row.get("_close_p_market")
        lean = row.get("lean")

        if not pd.notna(line_move):
            price_move_toward_lean = np.nan
            market_agreed = None
        elif line_move != 0:
            price_move_toward_lean = np.nan
            if lean == "over":
                market_agreed = "toward" if line_move > 0 else "against"
            else:
                market_agreed = "toward" if line_move < 0 else "against"
        else:
            if pd.notna(p_open) and pd.notna(p_close):
                direction = 1.0 if lean == "over" else -1.0
                price_move_toward_lean = (float(p_close) - float(p_open)) * direction
                if price_move_toward_lean > 1e-12:
                    market_agreed = "toward"
                elif price_move_toward_lean < -1e-12:
                    market_agreed = "against"
                else:
                    market_agreed = "unchanged"
            else:
                price_move_toward_lean = np.nan
                market_agreed = "unchanged"

        rows.append({
            "pitcher":                    row.get("pitcher"),
            "game_pk":                    row.get("game_pk"),
            "pitcher_name":               row.get("pitcher_name"),
            "team":                       row.get("team"),
            "start_time":                 row.get("start_time"),
            "tier":                       row.get("tier"),
            "lean":                       lean,
            "actionability":              row.get("actionability"),
            "edge_open":                  row.get("edge"),
            "line_open":                  line_open,
            "line_close":                 line_close,
            "line_move":                  line_move,
            "p_market_open":              p_open,
            "p_market_close":             p_close,
            "price_move_toward_lean":     price_move_toward_lean,
            "market_agreed":              market_agreed,
            "close_quality":              row.get("_close_quality"),
            "minutes_before_first_pitch": row.get("_close_minutes_before_first_pitch"),
        })

    return pd.DataFrame(rows, columns=CLV_COLUMNS)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def _stats_for(frame: pd.DataFrame) -> dict:
    """
    {"pct_market_agreed", "mean_price_move_toward_lean", "mean_abs_line_move"}
    each as {"value", "n"}, plus the bucket's total row count -- every
    statistic carries its own denominator (a percentage with no denominator
    is not a result).
    """
    n = len(frame)

    agreed_known = frame[frame["market_agreed"].notna()] if n else frame
    n_agreed_known = len(agreed_known)
    pct_agreed = (
        float((agreed_known["market_agreed"] == "toward").mean())
        if n_agreed_known else None
    )

    flat = frame[frame["line_move"] == 0] if n else frame
    price_vals = flat["price_move_toward_lean"].dropna() if len(flat) else flat
    mean_price_move = float(price_vals.mean()) if len(price_vals) else None

    line_move_vals = frame["line_move"].dropna().abs() if n else frame
    mean_abs_move = float(line_move_vals.mean()) if len(line_move_vals) else None

    return {
        "n": n,
        "pct_market_agreed": {"value": pct_agreed, "n": n_agreed_known},
        "mean_price_move_toward_lean": {"value": mean_price_move, "n": len(price_vals)},
        "mean_abs_line_move": {"value": mean_abs_move, "n": len(line_move_vals)},
    }


def clv_summary(clv_df: pd.DataFrame) -> dict:
    """
    Summarize a `compute_clv` output:
      - "overall": pct_market_agreed, mean_price_move_toward_lean (rows with
        line_move == 0 only), mean_abs_line_move -- each with its n.
      - "by_tier" / "by_actionability" / "by_edge_quartile": the same three
        stats, broken out by tiering.py's `tier`, `actionability`, and by
        quartile of |edge_open| (rows with a known edge_open only -- a
        matched line with no market price at open has no edge to bucket).
      - "n_total", "n_stale_excluded": every stat above is computed on the
        "good"-quality subset only; stale closes are counted, not silently
        dropped.

    Every percentage/mean is reported alongside its n (never bare) per the
    "a percentage with no denominator is not a result" requirement.
    """
    empty_stats = _stats_for(pd.DataFrame(columns=CLV_COLUMNS))
    if clv_df is None or clv_df.empty:
        return {
            "n_total": 0,
            "n_stale_excluded": 0,
            "n_edge_unknown_excluded_from_quartiles": 0,
            "overall": empty_stats,
            "by_tier": {},
            "by_actionability": {},
            "by_edge_quartile": {},
        }

    df = clv_df.copy()
    n_total = len(df)
    is_stale = df["close_quality"] == "stale"
    n_stale_excluded = int(is_stale.sum())
    good = df[~is_stale].copy()

    overall = _stats_for(good)
    by_tier = {str(k): _stats_for(g) for k, g in good.groupby("tier", dropna=False)}
    by_actionability = {str(k): _stats_for(g) for k, g in good.groupby("actionability", dropna=False)}

    edge_known = good[good["edge_open"].notna()].copy()
    n_edge_unknown = len(good) - len(edge_known)
    by_edge_quartile = {}
    if not edge_known.empty:
        edge_known["_abs_edge"] = edge_known["edge_open"].abs()
        try:
            edge_known["_quartile"] = pd.qcut(edge_known["_abs_edge"], 4, duplicates="drop")
            for q, g in edge_known.groupby("_quartile", observed=True):
                by_edge_quartile[str(q)] = _stats_for(g)
        except ValueError:
            # Too few distinct |edge_open| values for 4 quartiles (small
            # sample) -- fall back to one bucket rather than raising.
            by_edge_quartile["all"] = _stats_for(edge_known)

    return {
        "n_total": n_total,
        "n_stale_excluded": n_stale_excluded,
        "n_edge_unknown_excluded_from_quartiles": n_edge_unknown,
        "overall": overall,
        "by_tier": by_tier,
        "by_actionability": by_actionability,
        "by_edge_quartile": by_edge_quartile,
    }
