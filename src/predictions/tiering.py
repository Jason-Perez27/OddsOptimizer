"""
Tiered per-threshold prop probabilities: turns the task #7 baseline model's
predicted strikeout-count distribution into the project's actual product
output -- a P(over) for every threshold 1+ through 10+, bucketed into 3
confidence tiers, plus (where an Underdog line exists) a single
line-specific pick row.

Line source migration (2026-08): Underdog Fantasy posts exactly one balanced
two-sided line per pitcher/stat (see src/data/underdog_lines.py) -- there is
no standard/goblin/demon ladder to collapse any more, so the old
_select_canonical_line fallback logic is gone. What Underdog's two-sided
prices buy instead: a real no-vig MARKET probability (p_market) to measure
edge against, rather than the old fixed "vs 50%" fallback. See
docs/design/specs/2026-06-29-standard-line-filter-design.md for the
(now-superseded) PrizePicks-era design this replaced.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

from src.data.underdog_lines import american_to_prob, no_vig_two_way
from src.models.baseline_model import THRESHOLDS, poisson_over_prob, nbinom_over_prob

try:
    from pybaseball import chadwick_register
except ImportError:
    chadwick_register = None


# ---------------------------------------------------------------------------
# Confidence tiering
# ---------------------------------------------------------------------------

HIGH_DISTANCE = 0.20
MEDIUM_DISTANCE = 0.10
_BOUNDARY_EPS = 1e-9


def tier(p: float) -> str:
    d = abs(float(p) - 0.5)
    if d >= HIGH_DISTANCE - _BOUNDARY_EPS:
        return "high"
    if d >= MEDIUM_DISTANCE - _BOUNDARY_EPS:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Line <-> threshold conversion / model survival function
# ---------------------------------------------------------------------------

def line_to_threshold(line: float) -> int:
    """floor(line) + 1 -- "over 6.5" means 7+ strikeouts."""
    return int(np.floor(float(line))) + 1


def _survival(family: str, mu: float, alpha, threshold: int) -> float:
    if family == "poisson":
        return float(poisson_over_prob(mu, threshold))
    if family == "negative_binomial":
        return float(nbinom_over_prob(mu, alpha, threshold))
    raise ValueError(f"Unknown family: {family!r}")


def prob_over_line(family: str, mu: float, alpha, line: float):
    threshold = line_to_threshold(line)
    p_over = _survival(family, mu, alpha, threshold)
    push_mass = 0.0
    if float(line).is_integer():
        p_at_or_above_line = _survival(family, mu, alpha, int(line))
        push_mass = p_at_or_above_line - p_over
    return p_over, push_mass


# ---------------------------------------------------------------------------
# Threshold sweep table
# ---------------------------------------------------------------------------

THRESHOLD_TABLE_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "team", "opponent_team", "game_date",
    "threshold", "p_over", "tier",
]


def build_threshold_table(predictions_df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(columns=THRESHOLD_TABLE_COLUMNS)
    rows = []
    for _, pred in predictions_df.iterrows():
        family = pred["family"]
        mu = float(pred["mu"])
        alpha = pred.get("alpha")
        for t in thresholds:
            p_over = _survival(family, mu, alpha, t)
            rows.append({
                "pitcher": pred["pitcher"],
                "game_pk": pred["game_pk"],
                "pitcher_name": pred["pitcher_name"],
                "team": pred["pitcher_team"],
                "opponent_team": pred["opponent_team"],
                "game_date": pred["game_date"],
                "threshold": t,
                "p_over": p_over,
                "tier": tier(p_over),
            })
    return pd.DataFrame(rows, columns=THRESHOLD_TABLE_COLUMNS)


# ---------------------------------------------------------------------------
# Name <-> id resolution
# ---------------------------------------------------------------------------

TEAM_CROSSWALK = {
    "WSH": "WAS",
    "CWS": "CHW",
}

_SUFFIX_PATTERN = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$")


def normalize_name(name) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace(".", "")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = _SUFFIX_PATTERN.sub("", text).strip()
    return text


def to_statcast_team(pp_team) -> str:
    if pp_team is None or (isinstance(pp_team, float) and pd.isna(pp_team)):
        return ""
    code = str(pp_team).upper()
    return TEAM_CROSSWALK.get(code, code)


def resolve_pitcher_ids(lines_df: pd.DataFrame, predictions_df: pd.DataFrame,
                         register_df: pd.DataFrame):
    """
    Match each line-source row to a predicted pitcher's MLBAM id by
    normalized name, disambiguating same-named candidates by team.

    Underdog carries no `team` field (see src/data/underdog_lines.py -- the
    feed has no `teams` key at all), so when `team` is null/missing the
    away_team/home_team pair from the line's game is used as the tiebreak
    instead: a candidate matches if the model's `pitcher_team` for that id is
    in {away_team, home_team}. The existing TEAM_CROSSWALK (WSH->WAS,
    CWS->CHW) is applied to whichever abbreviation(s) are used.
    """
    if lines_df.empty:
        empty = lines_df.copy()
        empty["pitcher_id"] = pd.Series(dtype="object")
        return empty, lines_df.copy()

    predicted_ids = set(predictions_df["pitcher"]) if not predictions_df.empty else set()
    id_to_team = (
        dict(zip(predictions_df["pitcher"], predictions_df["pitcher_team"]))
        if not predictions_df.empty else {}
    )

    name_to_ids = {}
    if not register_df.empty:
        for _, reg_row in register_df.iterrows():
            mlbam_id = reg_row["key_mlbam"]
            if mlbam_id not in predicted_ids:
                continue
            full_name = f"{reg_row['name_first']} {reg_row['name_last']}"
            norm = normalize_name(full_name)
            name_to_ids.setdefault(norm, []).append(mlbam_id)

    matched_rows = []
    unmatched_rows = []
    for _, line_row in lines_df.iterrows():
        norm_name = normalize_name(line_row.get("pitcher"))
        candidates = name_to_ids.get(norm_name, [])

        matched_id = None
        if len(candidates) == 1:
            matched_id = candidates[0]
        elif len(candidates) > 1:
            team_val = line_row.get("team")
            has_team = not (team_val is None or (isinstance(team_val, float) and pd.isna(team_val)))
            if has_team:
                line_team = to_statcast_team(team_val)
                team_matches = [cid for cid in candidates if id_to_team.get(cid) == line_team]
            else:
                # No `team` on this row (the Underdog case) -- disambiguate via
                # the game's away/home pair instead.
                away = to_statcast_team(line_row.get("away_team"))
                home = to_statcast_team(line_row.get("home_team"))
                game_teams = {t for t in (away, home) if t}
                team_matches = [cid for cid in candidates if id_to_team.get(cid) in game_teams]
            if len(team_matches) == 1:
                matched_id = team_matches[0]

        if matched_id is None:
            unmatched_rows.append(line_row)
        else:
            matched = line_row.copy()
            matched["pitcher_id"] = matched_id
            matched_rows.append(matched)

    resolved = (
        pd.DataFrame(matched_rows)
        if matched_rows
        else pd.DataFrame(columns=list(lines_df.columns) + ["pitcher_id"])
    )
    unmatched = (
        pd.DataFrame(unmatched_rows)
        if unmatched_rows
        else pd.DataFrame(columns=list(lines_df.columns))
    )
    return resolved.reset_index(drop=True), unmatched.reset_index(drop=True)


def fetch_register() -> pd.DataFrame:
    if chadwick_register is None:
        raise ImportError(
            "pybaseball is required to fetch a live player-id register (pip install pybaseball)"
        )
    register = chadwick_register()
    return register[["key_mlbam", "name_first", "name_last"]].dropna(subset=["key_mlbam"])


# ---------------------------------------------------------------------------
# Line picks (the actionable view)
# ---------------------------------------------------------------------------

# Conviction + no-action band (spec ①, 2026-06-30).
# Thresholds are PROVISIONAL — not yet ROI-validated on Track-B settled
# outcomes.  Validate per bucket (≥100 settled/bucket) via
# src/backtest/conviction.calibrate_no_action_band() before treating these
# as proven edge.
NO_ACTION_CONVICTION_THRESHOLD = 1.0   # provisional
NO_ACTION_EDGE_THRESHOLD = 0.05        # provisional (|p_over - p_market|, vs coinflip if no market)

LINE_PICKS_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "team", "start_time", "line",
    "line_threshold", "p_over", "p_under", "tier", "lean", "edge", "edge_vs_coinflip",
    "push_mass", "projection_id", "pulled_at",
    "over_american", "under_american", "over_payout_multiplier", "under_payout_multiplier",
    "p_over_implied", "p_under_implied", "vig", "p_market",
    "p_over_lo", "p_over_hi", "conviction", "actionability",
]


def _filter_pre_game(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    A line is pre-game when its game hasn't started (game_status ==
    "scheduled") and it isn't itself a live/in-play market (live_event is
    falsy). Replaces the old PrizePicks `status in {"pre_game"}` check --
    Underdog carries game state on `game_status` and line-level `live_event`
    separately rather than a single per-line "pre_game" status string.
    """
    if lines_df.empty:
        return lines_df
    mask = (lines_df["game_status"] == "scheduled") & (~lines_df["live_event"].fillna(False).astype(bool))
    return lines_df[mask].reset_index(drop=True)


def build_line_picks(predictions_df: pd.DataFrame, lines_df: pd.DataFrame,
                      register_df: pd.DataFrame):
    """One row per pitcher with a resolved, active pre-game Underdog line.
    Returns (line_picks, diagnostics)."""
    lines = _filter_pre_game(lines_df)

    resolved, unmatched = resolve_pitcher_ids(lines, predictions_df, register_df)

    if resolved.empty:
        line_picks = pd.DataFrame(columns=LINE_PICKS_COLUMNS)
    else:
        _pred_cols = ["pitcher", "game_pk", "pitcher_name", "pitcher_team", "family", "mu", "alpha"]
        if "mu_se" in predictions_df.columns:
            _pred_cols.append("mu_se")
        predictions_subset = predictions_df[_pred_cols].rename(
            columns={"pitcher": "_model_pitcher_id"}
        )
        merged = resolved.merge(
            predictions_subset, left_on="pitcher_id", right_on="_model_pitcher_id", how="left",
        )

        out_rows = []
        for _, row in merged.iterrows():
            family = row["family"]
            mu = float(row["mu"])
            alpha = row.get("alpha")
            line = float(row["line"])

            p_over, push_mass = prob_over_line(family, mu, alpha, line)
            p_under = 1.0 - p_over - push_mass

            # --- Market probability (2026-08 migration) ----------------------
            # Underdog posts two-sided American prices; convert each side to
            # its (vig-inclusive) implied probability, then no-vig-normalize
            # to get an honest market P(over). ~7.5% observed two-sided vig
            # means p_over_implied + p_under_implied sums to > 1.0 before
            # normalization.
            over_american = row.get("over_american")
            under_american = row.get("under_american")
            p_over_implied = p_under_implied = vig = p_market = np.nan
            if pd.notna(over_american) and pd.notna(under_american):
                p_over_implied = american_to_prob(over_american)
                p_under_implied = american_to_prob(under_american)
                vig = p_over_implied + p_under_implied - 1.0
                p_market = no_vig_two_way(p_over_implied, p_under_implied)

            # edge is measured against the no-vig market, not a fixed 50%
            # coinflip -- the whole point of a two-sided line. edge_vs_coinflip
            # is kept alongside so the historical (pre-migration) backtest,
            # which only ever had "vs 50%", stays comparable and the change is
            # auditable.
            edge_vs_coinflip = p_over - 0.5
            market_known = not pd.isna(p_market)
            edge = p_over - p_market if market_known else np.nan
            reference = p_market if market_known else 0.5
            lean = "over" if p_over > reference else "under"

            # --- Conviction (spec ①) ----------------------------------------
            # Propagate mu's estimation SE to a band on p_over via the delta
            # method: mu_lo/hi = mu * exp(∓eta_se), giving p_over_lo/hi, then
            # sd(p_over) ≈ (p_over_hi − p_over_lo) / 2.
            # eta_se is the SE of log(mu) (GLM linear-predictor SE).
            # This is parameter uncertainty, NOT count-distribution variance.
            eta_se = float(row.get("mu_se") or 0.0)
            if eta_se > 0:
                mu_lo = mu * np.exp(-eta_se)
                mu_hi = mu * np.exp(+eta_se)
                p_over_lo, _ = prob_over_line(family, mu_lo, alpha, line)
                p_over_hi, _ = prob_over_line(family, mu_hi, alpha, line)
            else:
                p_over_lo = p_over_hi = p_over
            p_over_sd = (p_over_hi - p_over_lo) / 2.0
            conviction = abs(p_over - reference) / max(p_over_sd, 1e-9)

            # Actionability: both lean and conviction must clear provisional
            # thresholds (labeled unvalidated until Track-B confirms).
            # Rebased on |p_over - p_market| (falls back to vs-coinflip when
            # the market isn't known, e.g. a matched line missing a price).
            edge_abs = abs(p_over - reference)
            if edge_abs < NO_ACTION_EDGE_THRESHOLD or conviction < NO_ACTION_CONVICTION_THRESHOLD:
                actionability = "no_action"
            elif lean == "over":
                actionability = "lean_over"
            else:
                actionability = "lean_under"

            out_rows.append({
                "pitcher":                 row["pitcher_id"],
                "game_pk":                 row["game_pk"],
                "pitcher_name":            row["pitcher_name"],
                "team":                    row["pitcher_team"],
                "start_time":              row["start_time"],
                "line":                    line,
                "line_threshold":          line_to_threshold(line),
                "p_over":                  p_over,
                "p_under":                 p_under,
                "tier":                    tier(p_over),
                "lean":                    lean,
                "edge":                    edge,
                "edge_vs_coinflip":        edge_vs_coinflip,
                "push_mass":               push_mass,
                "projection_id":           row["projection_id"],
                "pulled_at":               row["pulled_at"],
                "over_american":           over_american,
                "under_american":          under_american,
                "over_payout_multiplier":  row.get("over_payout_multiplier"),
                "under_payout_multiplier": row.get("under_payout_multiplier"),
                "p_over_implied":          p_over_implied,
                "p_under_implied":         p_under_implied,
                "vig":                     vig,
                "p_market":                p_market,
                "p_over_lo":               p_over_lo,
                "p_over_hi":               p_over_hi,
                "conviction":              conviction,
                "actionability":           actionability,
            })
        line_picks = pd.DataFrame(out_rows, columns=LINE_PICKS_COLUMNS)

    matched_ids = set(line_picks["pitcher"]) if not line_picks.empty else set()
    predicted_no_line = (
        predictions_df[~predictions_df["pitcher"].isin(matched_ids)].reset_index(drop=True)
        if not predictions_df.empty else predictions_df.copy()
    )

    diagnostics = {
        "unmatched_lines": unmatched.reset_index(drop=True),
        "predicted_no_line": predicted_no_line,
    }
    return line_picks, diagnostics
