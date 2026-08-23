"""
Daily pre-game refresh pipeline (task #9, module 4): turns "today's
probable starters" into predictions / threshold_table / line_picks /
diagnostics, written to disk.

Design: docs/design/specs/2026-06-27-pre-game-refresh-pipeline-design.md
("Outputs", "Partial failure", "Scheduling" sections); decision log,
2026-06-27 "Pre-game refresh pipeline" entry. The live-verification gate
(`run_dry_run` / `--dry-run`) is task #11 step 5, spec section 5 ("Live-data
verification gate (verify_live_sources)").

This module is orchestration only -- every step below is a thin call into an
already-built, independently-tested module (probable_pitchers, pitcher_logs,
underdog_lines, game_logs, predict_features, baseline_model, tiering).
Its own logic is: wiring, partial-failure handling, and output writing, via
dependency-injected fetchers so run_refresh() is unit-testable with
hand-built fixtures and no network -- matching the rest of this repo's test
style.

Design choice (statcast_fetcher, name vs. id -- flagged in the spec as an
integration risk, resolved here): the slate (probable_pitchers.
parse_probable_starters) hands us a pitcher's MLBAM id directly, but
pitcher_logs.get_pitcher_season_logs() only takes a NAME (it re-derives an
id internally via playerid_lookup) -- there's no id-based pull path in that
module. default_statcast_fetcher() below calls the name-based pull (trusting
the slate's name) and then CROSS-CHECKS the result against the slate's id
using Statcast's own `pitcher` column (the MLBAM id of who actually threw
each pitch) -- a mismatch (e.g. two same-named players) raises, which
run_refresh() catches and routes to skipped_pitchers rather than silently
feeding a wrong pitcher's history into the model.

Two distinct fetcher pairs exist for the two distinct live sources, with two
distinct return shapes -- do not mix them up:
- `default_schedule_fetcher` / `default_lines_fetcher`: ALREADY PARSED
  (DataFrame), what run_refresh()'s pipeline actually consumes downstream.
- `default_raw_schedule_fetcher` / `default_raw_lines_fetcher`: RAW payload
  (dict), what `run_dry_run` / `verify_live_sources` need, because the gate's
  job is to verify the PARSE step itself (parse_probable_starters /
  flatten_projections), not trust its already-parsed output.
"""

import json
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.data import probable_pitchers as pp_mod
from src.data import pitcher_logs
from src.data import underdog_lines
from src.data import lineups as lineup_mod
from src.data.umpires import load_ump_tendency, get_ump_k_factor
from src.data.weather import get_game_weather
from src.data.vegas import get_game_odds
from src.features.game_logs import aggregate_pitcher_games, OUTPUT_COLUMNS as GAME_LOG_COLUMNS
from src.features.opponent_features import build_lineup_weighted_opp_k
from src.features.predict_features import build_prediction_features
from src.models.baseline_model import (
    CORE_PITCHER_FORM_COLUMNS,
    IMPUTE_COLUMNS,
    transform_design_matrix,
    load_model,
    model_age_days,
)
from src.models.boosted_model import (
    DEFAULT_BOOSTED_MODEL_PATH,
    load_boosted_model,
    compute_agreement,
)
from src.pipeline.verify import verify_live_sources
from src.predictions.tiering import (
    LINE_PICKS_COLUMNS,
    THRESHOLD_TABLE_COLUMNS,
    build_threshold_table,
    build_line_picks,
    fetch_register,
)
from src.props import DEFAULT_PROP, get_prop

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
DEFAULT_MODEL_PATH = os.path.join("data", "models", "baseline_model.joblib")

# Surfaced not hidden (decision log): the run records model age and warns
# past this threshold, never auto-refits.
STALE_MODEL_WARNING_DAYS = 7

PREDICTIONS_COLUMNS = [
    "pitcher", "game_pk", "pitcher_name", "pitcher_team", "opponent_team",
    "game_date", "family", "mu", "alpha", "mu_se",
]

SKIPPED_PITCHERS_COLUMNS = ["pitcher", "pitcher_name", "reason"]

# Decision-context stats surfaced on the dashboard (src/serve). These are
# COMPUTED every refresh (on feature_rows) but were dropped before
# predictions.csv was written; pitcher_cards.csv persists them additively,
# keyed by (pitcher, game_pk), so the serving layer needs no feature recompute.
# `predictions.csv` / settle's grading contract are intentionally untouched.
# Windows are the existing validated ones (pitcher last-5, opponent vs-hand
# season + team last-10); bf_avg_last5 is the leakage-safe expected-batters
# proxy. `was_imputed` flags rows where a thin-sample opponent/park value fell
# back to the league mean (same meaning as the design matrix's was_imputed).
PITCHER_CARD_COLUMNS = [
    "pitcher", "game_pk", "game_date", "pitcher_name", "pitcher_team",
    "opponent_team", "pitcher_throws", "is_home", "rest_days", "mu",
    "k_rate_last5", "k_rate_season", "k_rate_vs_LHB", "k_rate_vs_RHB",
    "k_rate_home", "k_rate_away", "k_rate_vs_opponent_career",
    "ip_avg_last5", "pitch_count_avg_last5", "bf_avg_last5",
    "whiff_rate_last5", "velo_avg_last5",
    # plate-discipline skill features (spec ②, 2026-06-30) -- candidate regressors
    "swstr_rate_last5", "csw_rate_last5", "putaway_rate_last5",
    "whiff_rate_overall_last5", "k_minus_bb_rate_last5",
    "opponent_k_rate_vs_hand_season", "opponent_k_rate_vs_hand_last10",
    "opponent_k_rate_last10",
    "opponent_k_rate_home", "opponent_k_rate_away",
    "park_k_factor", "was_imputed",
    # spec ③ matchup + umpire features (2026-06-30) -- candidate regressors
    "opponent_lineup_k_rate_vs_hand", "opp_share_opposite_hand",
    "ump_k_factor", "ump_was_imputed", "lineup_source",
    # spec ④ weather + Vegas context (2026-06-30) -- candidate regressors
    "temp_f", "wind_mph", "humidity", "is_dome",
    "game_total", "is_favorite",
    "weather_was_imputed", "vegas_was_imputed",
    # spec ④ booster second-opinion (2026-06-30)
    "booster_mu", "glm_booster_agreement",
]


class EmptySlateError(RuntimeError):
    """
    Raised when no probable starters exist for `game_date` -- fatal-but-
    clean per the spec ("no probable pitchers at all -> nothing to
    predict"), distinct from every other failure mode below, all of which
    degrade to a partial result instead of raising.
    """


# ---------------------------------------------------------------------------
# Default (live) fetchers -- network-touching, never exercised by tests
# ---------------------------------------------------------------------------

def default_schedule_fetcher(game_date: str) -> pd.DataFrame:
    payload = pp_mod.fetch_schedule(game_date)
    return pp_mod.parse_probable_starters(payload, game_date)


def default_statcast_fetcher(pitcher_id, pitcher_name: str, season: int) -> pd.DataFrame:
    """See module docstring for the name/id cross-check rationale."""
    df = pitcher_logs.get_pitcher_season_logs(pitcher_name, season)
    if df.empty:
        return df
    actual_ids = set(df["pitcher"].unique())
    if pitcher_id not in actual_ids:
        raise ValueError(
            f"Statcast pull for {pitcher_name!r} returned pitcher id(s) {sorted(actual_ids)}, "
            f"none matching slate id {pitcher_id} -- likely a name-lookup mismatch "
            f"(e.g. a common name); treating as unresolvable."
        )
    return df[df["pitcher"] == pitcher_id].reset_index(drop=True)


def default_lines_fetcher() -> pd.DataFrame:
    payload = underdog_lines.fetch_over_under_lines()
    return underdog_lines.flatten_lines(payload)


def default_lines_fetcher_for_stat(stat: str) -> pd.DataFrame:
    """Like default_lines_fetcher but parameterized by Underdog stat key."""
    payload = underdog_lines.fetch_over_under_lines()
    return underdog_lines.flatten_lines(payload, stat=stat)


def default_raw_schedule_fetcher(game_date: str) -> dict:
    """RAW schedule payload (no parsing) -- for verify_live_sources/run_dry_run only."""
    return pp_mod.fetch_schedule(game_date)


def default_raw_lines_fetcher() -> dict:
    """RAW Underdog over_under_lines payload (no flattening) -- for verify_live_sources/run_dry_run only."""
    return underdog_lines.fetch_over_under_lines()


# ---------------------------------------------------------------------------
# Predictions assembly
# ---------------------------------------------------------------------------

def assemble_predictions(model, feature_rows: pd.DataFrame, slate: pd.DataFrame):
    """
    Turn predict_features.build_prediction_features() output into the
    predictions_df contract tiering.py expects (PREDICTIONS_COLUMNS).

    Two things this function exists to do that the feature/model layers
    deliberately don't:
    - Attach `pitcher_name` from the slate -- the game_logs-schema feature
      rows carry no name field (predict_features.py's docstring: name
      attachment is this pipeline's job, not derived on demand).
    - Decide, and SURFACE, which slate pitchers don't get a prediction at
      all -- a debutant/no-history starter whose row is all-NaN on
      CORE_PITCHER_FORM_COLUMNS would otherwise just silently vanish inside
      transform_design_matrix's internal dropna. Replicating that dropna
      here (rather than trusting transform_design_matrix's own row-dropping)
      keeps a stable index so the dropped pitchers can be named in
      diagnostics, not just counted.

    Returns (predictions_df, dropped) where `dropped` is a list of
    {"pitcher", "pitcher_name", "reason"} dicts, folded into
    diagnostics["skipped_pitchers"] by run_refresh().
    """
    if feature_rows.empty:
        return pd.DataFrame(columns=PREDICTIONS_COLUMNS), []

    name_by_id = dict(zip(slate["pitcher"], slate["pitcher_name"]))

    has_core = feature_rows[CORE_PITCHER_FORM_COLUMNS].notna().all(axis=1)
    kept_rows = feature_rows[has_core].reset_index(drop=True)
    dropped_rows = feature_rows[~has_core]

    dropped = [
        {
            "pitcher": row["pitcher"],
            "pitcher_name": name_by_id.get(row["pitcher"]),
            "reason": "no usable pre-game features (insufficient/no prior history)",
        }
        for _, row in dropped_rows.iterrows()
    ]

    if kept_rows.empty:
        return pd.DataFrame(columns=PREDICTIONS_COLUMNS), dropped

    X = transform_design_matrix(kept_rows, model.preprocessor)
    mu, mu_se = model.predict_mean_with_se(X)

    predictions = kept_rows[["pitcher", "game_pk", "pitcher_team", "opponent_team", "game_date"]].copy()
    predictions["pitcher_name"] = predictions["pitcher"].map(name_by_id)
    predictions["family"] = model.family
    predictions["mu"] = mu
    predictions["alpha"] = model.alpha
    predictions["mu_se"] = mu_se
    predictions = predictions[PREDICTIONS_COLUMNS].reset_index(drop=True)

    return predictions, dropped


def build_pitcher_cards(feature_rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Build the decision-context card frame (PITCHER_CARD_COLUMNS) the dashboard
    reads, one row per PREDICTED pitcher.

    Separate from assemble_predictions (rather than widening its return) so the
    predictions.csv contract and assemble_predictions' signature/tests stay
    unchanged. Starts from `predictions` (authoritative set of predicted
    pitchers, carrying pitcher_name + mu) and left-joins the already-computed
    feature columns off `feature_rows` on (pitcher, game_pk) -- no recompute, no
    model call. `is_home` is derived from home_away (engineered only inside the
    design matrix, not present on raw feature_rows); `was_imputed` mirrors the
    design matrix's flag: any IMPUTE_COLUMNS value missing pre-imputation.
    """
    if predictions.empty or feature_rows.empty:
        return pd.DataFrame(columns=PITCHER_CARD_COLUMNS)

    feat = feature_rows.copy()
    feat["is_home"] = (feat["home_away"] == "home").astype(float)
    feat["was_imputed"] = feat[IMPUTE_COLUMNS].isna().any(axis=1).astype(float)

    feature_pull_columns = [
        c for c in PITCHER_CARD_COLUMNS
        if c not in ("pitcher", "game_pk", "game_date", "pitcher_name",
                     "pitcher_team", "opponent_team", "mu")
    ]
    # Candidate columns added in later specs may be absent when the caller
    # doesn't compute them (e.g. tests that don't wire weather/vegas/booster).
    # Fill any missing columns with NaN so the final PITCHER_CARD_COLUMNS slice
    # always succeeds without requiring every caller to know about every column.
    for c in feature_pull_columns:
        if c not in feat.columns:
            feat[c] = np.nan
    feat_subset = feat[["pitcher", "game_pk"] + feature_pull_columns]

    cards = predictions[
        ["pitcher", "game_pk", "game_date", "pitcher_name",
         "pitcher_team", "opponent_team", "mu"]
    ].merge(feat_subset, on=["pitcher", "game_pk"], how="left")

    return cards[PITCHER_CARD_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_refresh(
    game_date: str = None,
    *,
    prop: str = DEFAULT_PROP,
    season: int = None,
    model_path=DEFAULT_MODEL_PATH,
    schedule_fetcher=None,
    statcast_fetcher=None,
    lines_fetcher=None,
    register_fetcher=None,
    model_loader=load_model,
    stale_warning_days: float = STALE_MODEL_WARNING_DAYS,
    # Spec ③ (2026-06-30): injectable lineup/ump fetchers for network-free tests.
    # lineup_fetcher(game_pk) -> DataFrame with LINEUP_COLUMNS (or empty on miss).
    # officials_fetcher(game_pk) -> raw dict for get_ump_k_factor (or None on miss).
    # batter_rolling_df: pre-computed per-batter rolling K-rates; None → NaN features.
    lineup_fetcher=None,
    officials_fetcher=None,
    batter_rolling_df=None,
    # Spec ④ (2026-06-30): injectable weather/vegas/booster fetchers.
    # weather_fetcher(home_team, game_date, first_pitch_hour) -> weather dict.
    # vegas_fetcher(home_team, game_date, pitcher_home_away) -> odds dict.
    # boosted_model_path: path to optional boosted_model.joblib; None skips booster.
    # boosted_model_loader: injectable for tests (path -> (BoostedModel, metadata)).
    weather_fetcher=None,
    vegas_fetcher=None,
    boosted_model_path=DEFAULT_BOOSTED_MODEL_PATH,
    boosted_model_loader=None,
) -> dict:
    """
    Orchestrate one day's refresh. Returns a results dict:
    {"game_date", "prop", "predictions", "threshold_table", "line_picks", "diagnostics"}.

    Partial-failure behavior (decision log, 2026-06-27 "Pre-game refresh
    pipeline" entry):
    - The line source (`lines_fetcher`, Underdog Fantasy) or the Chadwick
      register (`register_fetcher`) raising -> still produce the full
      `predictions` / `threshold_table` for everyone, an EMPTY `line_picks`,
      and diagnostics["line_source_error"] / ["register_error"] record why.
    - A single pitcher's Statcast pull failing, or a pitcher with no usable
      pre-game features at all -> excluded from `predictions`, recorded in
      diagnostics["skipped_pitchers"]; the run continues for everyone else.
    - No probable starters at all for `game_date` -> EmptySlateError (the
      only fatal case -- there is nothing to predict).
    """
    game_date = game_date or date.today().isoformat()
    season = season or pd.Timestamp(game_date).year
    prop_cfg = get_prop(prop)

    schedule_fetcher = schedule_fetcher or default_schedule_fetcher
    statcast_fetcher = statcast_fetcher or default_statcast_fetcher
    lines_fetcher = lines_fetcher or (
        lambda: default_lines_fetcher_for_stat(prop_cfg.underdog_stat)
    )
    register_fetcher = register_fetcher or fetch_register

    slate = schedule_fetcher(game_date)
    if slate is None or slate.empty:
        raise EmptySlateError(f"No probable starters found for {game_date}")
    slate = slate.reset_index(drop=True)

    model, model_metadata = model_loader(model_path)
    age_days = model_age_days(model_metadata)
    model_stale = age_days > stale_warning_days

    skipped_pitchers = []
    statcast_skipped_ids = set()
    pitch_level_frames = []
    for _, starter in slate.iterrows():
        try:
            pitches = statcast_fetcher(starter["pitcher"], starter["pitcher_name"], season)
        except Exception as exc:
            skipped_pitchers.append({
                "pitcher": starter["pitcher"],
                "pitcher_name": starter["pitcher_name"],
                "reason": f"Statcast pull failed: {exc}",
            })
            statcast_skipped_ids.add(starter["pitcher"])
            continue
        if pitches is None or pitches.empty:
            skipped_pitchers.append({
                "pitcher": starter["pitcher"],
                "pitcher_name": starter["pitcher_name"],
                "reason": "no Statcast rows returned (debutant or no prior starts this season)",
            })
            statcast_skipped_ids.add(starter["pitcher"])
            continue
        pitch_level_frames.append(pitches)

    history = (
        aggregate_pitcher_games(pd.concat(pitch_level_frames, ignore_index=True))
        if pitch_level_frames else pd.DataFrame(columns=GAME_LOG_COLUMNS)
    )

    # Pitchers already skipped above (failed/empty Statcast pull) never get a
    # feature row built for them -- without this filter, build_prediction_
    # features/assemble_predictions would independently re-discover that same
    # pitcher has no usable core features (since there's no history for them)
    # and append a SECOND, less specific skip entry for the same pitcher,
    # double-counting them in diagnostics["skipped_pitchers"].
    feature_slate = slate[~slate["pitcher"].isin(statcast_skipped_ids)].reset_index(drop=True)

    # Back-fill pitcher_throws from Statcast history when the StatsAPI probable-
    # pitcher payload omitted it (common -- see probable_pitchers.py docstring).
    # Without this, opponent_k_rate_vs_hand_* is NaN for every starter because
    # _opponent_k_rate_vs_hand skips rows where pd.isna(hand).
    if not history.empty and "pitcher_throws" in history.columns:
        last_hand = (
            history.dropna(subset=["pitcher_throws"])
            .sort_values("game_date")
            .groupby("pitcher")["pitcher_throws"]
            .last()
        )
        feature_slate = feature_slate.copy()
        missing = feature_slate["pitcher_throws"].isna()
        if missing.any():
            feature_slate.loc[missing, "pitcher_throws"] = (
                feature_slate.loc[missing, "pitcher"].map(last_hand)
            )

    feature_rows = build_prediction_features(history, feature_slate)

    # Spec ③ (2026-06-30): enrich feature_rows with per-game lineup-weighted
    # matchup + ump features.  Fully graceful: if no lineup is posted, the
    # CSV is absent, or the fetch fails → NaN / team_fallback / was_imputed=1.
    _lineup_fetcher = lineup_fetcher or lineup_mod.get_lineup
    _ump_tendency = load_ump_tendency()  # empty DataFrame if CSV doesn't exist yet
    for col in ["opponent_lineup_k_rate_vs_hand", "opp_share_opposite_hand",
                "ump_k_factor", "ump_was_imputed"]:
        feature_rows[col] = np.nan
    feature_rows["lineup_source"] = ""  # object dtype — will hold strings like "confirmed"

    for game_pk, game_rows in feature_rows.groupby("game_pk"):
        try:
            lineup_df = _lineup_fetcher(int(game_pk))
        except Exception:
            lineup_df = pd.DataFrame(columns=lineup_mod.LINEUP_COLUMNS)
        lineup_src = (
            lineup_df["lineup_source"].iloc[0]
            if not lineup_df.empty else "team_fallback"
        )
        try:
            ump_kf, ump_imputed = get_ump_k_factor(
                int(game_pk), _ump_tendency, fetcher=officials_fetcher
            )
        except Exception:
            ump_kf, ump_imputed = 1.0, True

        for idx, row in game_rows.iterrows():
            pitcher_side = row.get("home_away", "home")
            opp_side = "away" if pitcher_side == "home" else "home"
            opp_lineup = (
                lineup_df[lineup_df["team_side"] == opp_side]
                if not lineup_df.empty
                else pd.DataFrame(columns=lineup_mod.LINEUP_COLUMNS)
            )
            matchup = build_lineup_weighted_opp_k(
                opp_lineup, batter_rolling_df, row.get("pitcher_throws", "R")
            )
            feature_rows.loc[idx, "opponent_lineup_k_rate_vs_hand"] = matchup[
                "opponent_lineup_k_rate_vs_hand"
            ]
            feature_rows.loc[idx, "opp_share_opposite_hand"] = matchup[
                "opp_share_opposite_hand"
            ]
            feature_rows.loc[idx, "ump_k_factor"] = float(ump_kf)
            feature_rows.loc[idx, "ump_was_imputed"] = float(ump_imputed)
            feature_rows.loc[idx, "lineup_source"] = lineup_src

    # Spec ④ (2026-06-30): enrich feature_rows with weather + Vegas context
    # features.  Fully graceful: missing park / API failure → NaN + was_imputed=1.
    _weather_fetcher = weather_fetcher or get_game_weather
    _vegas_fetcher = vegas_fetcher or get_game_odds
    for col in ["temp_f", "wind_mph", "humidity", "is_dome",
                "game_total", "is_favorite"]:
        feature_rows[col] = np.nan
    feature_rows["weather_was_imputed"] = 1.0
    feature_rows["vegas_was_imputed"] = 1.0

    for game_pk, game_rows in feature_rows.groupby("game_pk"):
        first_row = game_rows.iloc[0]
        home_team = (
            first_row.get("pitcher_team")
            if first_row.get("home_away") == "home"
            else first_row.get("opponent_team")
        )
        try:
            wx = _weather_fetcher(str(home_team), str(game_date))
        except Exception:
            wx = {"temp_f": np.nan, "wind_mph": np.nan, "humidity": np.nan,
                  "is_dome": np.nan, "weather_was_imputed": True}
        for idx in game_rows.index:
            feature_rows.loc[idx, "temp_f"] = wx.get("temp_f", np.nan)
            feature_rows.loc[idx, "wind_mph"] = wx.get("wind_mph", np.nan)
            feature_rows.loc[idx, "humidity"] = wx.get("humidity", np.nan)
            feature_rows.loc[idx, "is_dome"] = wx.get("is_dome", np.nan)
            feature_rows.loc[idx, "weather_was_imputed"] = float(
                wx.get("weather_was_imputed", True)
            )

        for idx, row in game_rows.iterrows():
            pitcher_ha = row.get("home_away", "home")
            try:
                vx = _vegas_fetcher(str(home_team), str(game_date), pitcher_ha)
            except Exception:
                vx = {"game_total": np.nan, "is_favorite": np.nan,
                      "vegas_was_imputed": True}
            feature_rows.loc[idx, "game_total"] = vx.get("game_total", np.nan)
            feature_rows.loc[idx, "is_favorite"] = vx.get("is_favorite", np.nan)
            feature_rows.loc[idx, "vegas_was_imputed"] = float(
                vx.get("vegas_was_imputed", True)
            )

    # Spec ④ booster second-opinion: run BoostedModel alongside GLM if a
    # trained artifact exists; compute per-row agreement direction.
    # Missing artifact → booster_mu=NaN, agreement=NaN (graceful degradation).
    feature_rows["booster_mu"] = np.nan
    feature_rows["glm_booster_agreement"] = np.nan

    _booster_loader = boosted_model_loader or (
        load_boosted_model if os.path.exists(str(boosted_model_path)) else None
    )
    if _booster_loader is not None:
        try:
            booster, _ = _booster_loader(boosted_model_path)
            # transform_design_matrix calls _dropna_core (reset_index) internally,
            # so it may return fewer rows than feature_rows.  Prefilter to rows that
            # survive the core dropna, capture their ORIGINAL indices, then align
            # booster_mu_arr[i] back via those indices.
            has_core = feature_rows[CORE_PITCHER_FORM_COLUMNS].notna().all(axis=1)
            core_orig_indices = feature_rows.index[has_core].tolist()
            boostable = feature_rows[has_core].reset_index(drop=True)
            if not boostable.empty:
                X_boost = transform_design_matrix(boostable, booster.preprocessor)
                booster_mu_arr = booster.predict_mean(X_boost)
                # X_boost and boostable are aligned (same rows, same order) because
                # boostable already has no missing CORE_PITCHER_FORM_COLUMNS values,
                # so _dropna_core inside transform_design_matrix drops nothing more.
                for i, orig_idx in enumerate(core_orig_indices):
                    feature_rows.loc[orig_idx, "booster_mu"] = float(booster_mu_arr[i])
                # Agreement requires a line_score; stored later when line_picks joins.
        except Exception:
            pass  # leave booster_mu as NaN

    predictions, predict_dropped = assemble_predictions(model, feature_rows, feature_slate)
    skipped_pitchers.extend(predict_dropped)

    threshold_table = build_threshold_table(predictions)

    line_source_error = None
    register_error = None
    lines_df = pd.DataFrame()
    register_df = pd.DataFrame()
    try:
        lines_df = lines_fetcher()
    except Exception as exc:
        line_source_error = str(exc)
    try:
        register_df = register_fetcher()
    except Exception as exc:
        register_error = str(exc)

    if line_source_error or register_error:
        line_picks = pd.DataFrame(columns=LINE_PICKS_COLUMNS)
        pick_diagnostics = {
            "unmatched_lines": pd.DataFrame(),
            "predicted_no_line": predictions.copy(),
        }
    else:
        line_picks, pick_diagnostics = build_line_picks(predictions, lines_df, register_df)

    # Spec ④ booster agreement: compute per-row GLM-vs-booster agreement using
    # the matched line as the threshold.  Requires booster_mu to be
    # populated (non-NaN) and a matched line_score from line_picks.
    if not line_picks.empty and "line_score" in line_picks.columns:
        lp_idx = line_picks.set_index(["pitcher", "game_pk"])["line_score"]
        mu_idx = predictions.set_index(["pitcher", "game_pk"])["mu"]
        for idx, row in feature_rows.iterrows():
            key = (row.get("pitcher"), row.get("game_pk"))
            glm_mu_val = mu_idx.get(key, np.nan)
            booster_mu_val = feature_rows.loc[idx, "booster_mu"]
            line_val = lp_idx.get(key, np.nan)
            if not (np.isnan(glm_mu_val) or np.isnan(booster_mu_val)
                    or np.isnan(line_val)):
                feature_rows.loc[idx, "glm_booster_agreement"] = compute_agreement(
                    float(glm_mu_val), float(booster_mu_val), float(line_val)
                )

    # Rebuild pitcher_cards after agreement is populated in feature_rows.
    pitcher_cards = build_pitcher_cards(feature_rows, predictions)

    diagnostics = {
        "skipped_pitchers": pd.DataFrame(skipped_pitchers, columns=SKIPPED_PITCHERS_COLUMNS),
        "unmatched_lines": pick_diagnostics["unmatched_lines"],
        "predicted_no_line": pick_diagnostics["predicted_no_line"],
        "line_source_error": line_source_error,
        "register_error": register_error,
        "model_age_days": age_days,
        "model_stale": model_stale,
    }

    return {
        "game_date": game_date,
        "prop": prop,
        "predictions": predictions,
        "pitcher_cards": pitcher_cards,
        "threshold_table": threshold_table,
        "line_picks": line_picks,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Live-data verification gate (`refresh --dry-run`)
# ---------------------------------------------------------------------------

def run_dry_run(*, lines_fetcher=None, schedule_fetcher=None, game_date=None) -> dict:
    """
    `refresh --dry-run`: verify the live data sources' actual shape (see
    src.pipeline.verify.verify_live_sources) WITHOUT running the refresh or
    writing anything to disk -- spec section 5: "a refresh --dry-run mode
    that writes nothing." This function never calls run_refresh() or
    write_outputs(); it is a thin wrapper that supplies the RAW (unparsed)
    default live fetchers when the caller doesn't inject test fixtures.

    Returns the verify_live_sources report dict: {"passed": bool, "checks":
    [...]}.
    """
    lines_fetcher = lines_fetcher or default_raw_lines_fetcher
    schedule_fetcher = schedule_fetcher or default_raw_schedule_fetcher
    game_date = game_date or date.today().isoformat()

    return verify_live_sources(
        lines_fetcher=lines_fetcher,
        schedule_fetcher=schedule_fetcher,
        game_date=game_date,
    )


def print_dry_run_report(report: dict) -> None:
    """Print the verify_live_sources report as a pass/fail summary (spec: 'Print a pass/fail summary')."""
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")
    print("Overall: PASS" if report["passed"] else "Overall: FAIL -- do not trust a live run until this passes.")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(results: dict, processed_dir=DEFAULT_PROCESSED_DIR, overwrite: bool = True) -> str:
    """
    Write predictions.csv / threshold_table.csv / line_picks.csv /
    diagnostics/*.csv / run_manifest.json to the date partition.

    The default prop ("strikeouts") writes to the flat game_date=YYYY-MM-DD/
    path for backward compatibility. Non-default props write to
    game_date=YYYY-MM-DD/prop={key}/ so multiple props coexist per date.

    Re-running a date OVERWRITES by default (deterministic per morning's
    inputs -- appending would duplicate). `overwrite=False` raises
    FileExistsError if this date's manifest already exists, rather than
    silently appending/duplicating.
    """
    game_date = results["game_date"]
    prop = results.get("prop", DEFAULT_PROP)
    # Strikeouts (the default/legacy prop) writes to the flat game_date=*/
    # partition for backward compatibility with existing settle/serve/report
    # consumers and tests.  Non-default props gain a prop={key}/ sub-level
    # so multiple props can coexist under the same game_date.
    if prop == DEFAULT_PROP:
        out_dir = os.path.join(processed_dir, "predictions", f"game_date={game_date}")
    else:
        out_dir = os.path.join(
            processed_dir, "predictions", f"game_date={game_date}", f"prop={prop}"
        )
    manifest_path = os.path.join(out_dir, "run_manifest.json")

    if not overwrite and os.path.exists(manifest_path):
        raise FileExistsError(
            f"Output already exists for {game_date} at {out_dir} -- "
            f"pass overwrite=True to replace it."
        )

    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    results["predictions"].to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    results["threshold_table"].to_csv(os.path.join(out_dir, "threshold_table.csv"), index=False)
    results["line_picks"].to_csv(os.path.join(out_dir, "line_picks.csv"), index=False)

    # Additive decision-context file for the dashboard (src/serve). `.get` with
    # an empty well-formed frame keeps hand-built results dicts (and any caller
    # predating pitcher_cards) working without a KeyError.
    pitcher_cards = results.get("pitcher_cards")
    if pitcher_cards is None:
        pitcher_cards = pd.DataFrame(columns=PITCHER_CARD_COLUMNS)
    pitcher_cards.to_csv(os.path.join(out_dir, "pitcher_cards.csv"), index=False)

    diagnostics = results["diagnostics"]
    diagnostics["skipped_pitchers"].to_csv(os.path.join(diag_dir, "skipped_pitchers.csv"), index=False)
    diagnostics["unmatched_lines"].to_csv(os.path.join(diag_dir, "unmatched_lines.csv"), index=False)
    diagnostics["predicted_no_line"].to_csv(os.path.join(diag_dir, "predicted_no_line.csv"), index=False)

    manifest = {
        "game_date": game_date,
        "prop": prop,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_predictions": int(len(results["predictions"])),
        "n_pitcher_cards": int(len(pitcher_cards)),
        "n_threshold_rows": int(len(results["threshold_table"])),
        "n_line_picks": int(len(results["line_picks"])),
        "n_skipped_pitchers": int(len(diagnostics["skipped_pitchers"])),
        "n_unmatched_lines": int(len(diagnostics["unmatched_lines"])),
        "model_age_days": diagnostics["model_age_days"],
        "model_stale": diagnostics["model_stale"],
        "line_source_error": diagnostics["line_source_error"],
        "register_error": diagnostics["register_error"],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the daily pre-game refresh pipeline.")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD (default: today)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Abort instead of overwriting an existing run for this date",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify live data sources match the shape this pipeline's parsers expect; "
             "writes nothing and runs no refresh. Non-optional before the first real run "
             "(spec: 'never proceed to a real run on an unverified shape').",
    )
    args = parser.parse_args()

    if args.dry_run:
        report = run_dry_run(game_date=args.date)
        print_dry_run_report(report)
        return

    try:
        results = run_refresh(args.date, model_path=args.model_path)
    except EmptySlateError as exc:
        print(str(exc))
        return

    out_dir = write_outputs(
        results, processed_dir=args.processed_dir, overwrite=not args.no_overwrite,
    )
    diagnostics = results["diagnostics"]

    print(f"Wrote refresh output for {results['game_date']} to {out_dir}")
    if diagnostics["line_source_error"]:
        print(f"WARNING: line source fetch failed: {diagnostics['line_source_error']}")
    if diagnostics["register_error"]:
        print(f"WARNING: Chadwick register fetch failed: {diagnostics['register_error']}")
    if diagnostics["model_stale"]:
        print(f"WARNING: model is {diagnostics['model_age_days']:.1f} days old -- consider retraining.")
    n_skipped = len(diagnostics["skipped_pitchers"])
    if n_skipped:
        print(f"Skipped {n_skipped} pitcher(s) -- see diagnostics/skipped_pitchers.csv for details.")


if __name__ == "__main__":
    main()
