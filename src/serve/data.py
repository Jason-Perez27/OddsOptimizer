"""
Pure data-access / join layer for the pre-game decision dashboard.

Reads ONE date partition written by src/pipeline/refresh.py
(`data/processed/predictions/game_date=YYYY-MM-DD/`) and assembles a single
JSON-serializable slate dict the web layer (src/serve/server.py) serves as-is.

Deliberately import-light: pandas + stdlib only, NO scipy/model/network imports,
so it loads instantly and is unit-testable against a hand-built fixture partition
(tests/test_serve_data.py) -- the same network-free testing pattern as the rest of
the repo. Read-only: never writes, never triggers a refresh (the morning snapshot
is authoritative and frozen -- decision log 2026-06-29).

Canonical join key throughout is (pitcher, game_pk), matching the partition's
contract (doubleheader-safe). Pitchers with a threshold sweep but no posted line
are kept with `line: null`, never dropped. Each pitcher with a line carries its
two-sided market prices and no-vig market probability (`p_market`) so the UI
can show edge measured against the market, not a fixed 50% coinflip (2026-08
Underdog migration -- see src/predictions/tiering.py).
"""

import json
import math
import os
import statistics

import pandas as pd

PREDICTIONS_DIR = "predictions"

# Importing here to avoid circular imports -- data.py must stay model-import-free.
# We only need the DEFAULT_PROP string constant, not the full registry.
try:
    from src.props import DEFAULT_PROP as _DEFAULT_PROP
except ImportError:  # guard for environments where src is not on path
    _DEFAULT_PROP = "strikeouts"


def _partition_dir(processed_dir, game_date, prop=None):
    """Return the predictions partition directory for (date, prop).

    For the default prop ("strikeouts") the flat game_date=*/ path is used for
    backward compatibility.  Non-default props live under prop={key}/.
    """
    base = os.path.join(processed_dir, PREDICTIONS_DIR, f"game_date={game_date}")
    effective_prop = prop or _DEFAULT_PROP
    if effective_prop == _DEFAULT_PROP:
        return base
    return os.path.join(base, f"prop={effective_prop}")


def list_game_dates(processed_dir):
    """Sorted (ascending) list of game_date strings that have a partition on disk."""
    root = os.path.join(processed_dir, PREDICTIONS_DIR)
    if not os.path.isdir(root):
        return []
    dates = []
    for name in os.listdir(root):
        if name.startswith("game_date=") and os.path.isdir(os.path.join(root, name)):
            dates.append(name.split("=", 1)[1])
    return sorted(dates)


def latest_game_date(processed_dir):
    dates = list_game_dates(processed_dir)
    return dates[-1] if dates else None


def _read_csv(path):
    """Read a partition CSV, or an empty frame if it's absent/empty."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _clean(value):
    """Make a single value JSON-safe: NaN/NaT -> None, numpy scalars -> python."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _row_to_dict(row):
    return {k: _clean(v) for k, v in row.items()}


def _read_manifest(part_dir):
    path = os.path.join(part_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _diag_records(part_dir, filename):
    df = _read_csv(os.path.join(part_dir, "diagnostics", filename))
    if df.empty:
        return []
    return [_row_to_dict(r) for _, r in df.iterrows()]


def load_slate(processed_dir, game_date, prop=None):
    """
    Assemble the slate dict for one game_date (and optionally a prop).
    Raises FileNotFoundError if the partition doesn't exist (the caller maps
    that to a 404). `prop` defaults to the default prop ("strikeouts") which
    reads from the flat game_date=*/ path for backward compatibility.
    """
    part_dir = _partition_dir(processed_dir, game_date, prop)
    if not os.path.isdir(part_dir):
        raise FileNotFoundError(f"No partition for game_date={game_date} under {processed_dir}")

    predictions = _read_csv(os.path.join(part_dir, "predictions.csv"))
    thresholds = _read_csv(os.path.join(part_dir, "threshold_table.csv"))
    line_picks = _read_csv(os.path.join(part_dir, "line_picks.csv"))
    cards = _read_csv(os.path.join(part_dir, "pitcher_cards.csv"))
    manifest = _read_manifest(part_dir)

    ladders = {}
    if not thresholds.empty:
        for (pid, gpk), grp in thresholds.sort_values("threshold").groupby(["pitcher", "game_pk"]):
            ladders[(pid, gpk)] = [
                {"threshold": int(r["threshold"]),
                 "p_over": _clean(r["p_over"]),
                 "tier": _clean(r.get("tier"))}
                for _, r in grp.iterrows()
            ]

    lines = {}
    if not line_picks.empty:
        for _, r in line_picks.iterrows():
            lines[(r["pitcher"], r["game_pk"])] = {
                "line": _clean(r.get("line")),
                "line_threshold": _clean(r.get("line_threshold")),
                "lean": _clean(r.get("lean")),
                "edge": _clean(r.get("edge")),
                "edge_vs_coinflip": _clean(r.get("edge_vs_coinflip")),
                "p_over": _clean(r.get("p_over")),
                "p_under": _clean(r.get("p_under")),
                "p_market": _clean(r.get("p_market")),
                "vig": _clean(r.get("vig")),
                "over_american": _clean(r.get("over_american")),
                "under_american": _clean(r.get("under_american")),
                "over_payout_multiplier": _clean(r.get("over_payout_multiplier")),
                "under_payout_multiplier": _clean(r.get("under_payout_multiplier")),
                "push_mass": _clean(r.get("push_mass")),
                "tier": _clean(r.get("tier")),
                "start_time": _clean(r.get("start_time")),
                "p_over_lo": _clean(r.get("p_over_lo")),
                "p_over_hi": _clean(r.get("p_over_hi")),
                "conviction": _clean(r.get("conviction")),
                "actionability": _clean(r.get("actionability")),
            }

    stats = {}
    if not cards.empty:
        for _, r in cards.iterrows():
            d = _row_to_dict(r)
            stats[(r["pitcher"], r["game_pk"])] = d

    pitchers = []
    if not predictions.empty:
        for _, r in predictions.iterrows():
            key = (r["pitcher"], r["game_pk"])
            pitchers.append({
                "pitcher": _clean(r["pitcher"]),
                "game_pk": _clean(r["game_pk"]),
                "name": _clean(r.get("pitcher_name")),
                "team": _clean(r.get("pitcher_team")),
                "opponent": _clean(r.get("opponent_team")),
                "mu": _clean(r.get("mu")),
                "family": _clean(r.get("family")),
                "ladder": ladders.get(key, []),
                "line": lines.get(key),
                "stats": stats.get(key),
            })

    n_with_line = sum(1 for p in pitchers if p["line"])
    n_no_line = sum(1 for p in pitchers if not p["line"])
    vigs = [p["line"]["vig"] for p in pitchers if p["line"] and p["line"].get("vig") is not None]
    median_vig = statistics.median(vigs) if vigs else None

    return {
        "game_date": game_date,
        "available_dates": list_game_dates(processed_dir),
        "manifest": manifest,
        "has_stats": bool(stats),
        "kpis": {
            "n_pitchers": len(pitchers),
            "n_with_line": n_with_line,
            "n_no_line": n_no_line,
            "median_vig": median_vig,
            "model_age_days": manifest.get("model_age_days"),
            "model_stale": manifest.get("model_stale"),
            "line_source_error": manifest.get("line_source_error"),
        },
        "pitchers": pitchers,
        "diagnostics": {
            "skipped_pitchers": _diag_records(part_dir, "skipped_pitchers.csv"),
            "unmatched_lines": _diag_records(part_dir, "unmatched_lines.csv"),
            "predicted_no_line": _diag_records(part_dir, "predicted_no_line.csv"),
        },
    }
