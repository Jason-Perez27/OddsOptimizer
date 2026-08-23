"""
Baseline strikeout count model: Poisson / Negative Binomial GLM.

Design: docs/design/specs/2026-06-27-baseline-poisson-nb-model-design.md
(task #7). This is the explainable baseline every later, more complex model
must justify beating.

Pipeline:
  1. temporal_train_test_split() -- chronological holdout by game_date
     (never random; see spec's "Train / test split" section).
  2. build_design_matrix() -- the v1 regressor allowlist plus preprocessing:
     winsorize rest_days, impute opponent/park nulls with the TRAIN mean
     and flag via `was_imputed`, standardize continuous regressors using
     TRAIN statistics, add a constant. Drops rows missing any
     CORE_PITCHER_FORM_COLUMNS value (genuinely un-predictable from
     history).
  3. select_family() / fit_baseline_model() -- fits Poisson via statsmodels
     GLM, tests the training residuals for overdispersion against a fitted
     Negative Binomial (dispersion ratio + likelihood-ratio test), and
     returns whichever family the training data actually supports.
  4. BaselineModel.predict_mean() / predict_over_prob(t) -- the fitted count
     mean and per-threshold P(K >= t), via scipy.stats.poisson/nbinom.

Exposure/offset: deliberately NOT used in v1 (see spec's "Exposure / offset"
section). pitch_count_avg_last5 is included as an ordinary regressor instead
of a log(expected_batters_faced) offset, because a pre-game batters-faced
projection would need its own calibrated model whose error would propagate
straight into the offset. Revisit in v2 once that projection exists.

rest_days handling: the spec's null-handling rules explicitly cover
"pitcher-form" (drop) and "opponent/park" (impute) columns but don't name
rest_days directly. It's treated here as an impute-with-flag column, not a
drop column -- a pitcher's literal first game in the dataset has no rest_days
value, and dropping every pitcher's debut start would throw away an
otherwise-usable row for the same reason the spec avoids dropping opponent/
park nulls. This interpretation is recorded here since the spec didn't
spell it out.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial
except ImportError:  # statsmodels may not be installed in every environment
    sm = None
    NegativeBinomial = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

try:
    import joblib
except ImportError:  # joblib ships transitively with scikit-learn; pinned explicitly too
    joblib = None


LABEL_COLUMN = "strikeouts"

# Same-game outcome / label columns that must NEVER appear as regressors --
# they're measured DURING the game being predicted (see spec's "Inputs").
LEAKAGE_COLUMNS = [
    "strikeouts", "walks", "batters_faced", "pitch_count", "whiff_rate",
    "fastball_velo_avg", "innings_pitched", "strikeouts_vs_LHB",
    "batters_faced_vs_LHB", "strikeouts_vs_RHB", "batters_faced_vs_RHB",
    # spec ② per-game skill stats (same-game, never a direct regressor)
    "csw_rate", "putaway_rate", "whiff_rate_overall", "k_minus_bb",
]

# v1 regressor allowlist (engineered names) -- see spec's "Regressor
# selection (v1)" section for the full list of what was held out and why
# (near-collinear k_rate_* variants, batter-hand splits needing lineup
# ingestion that doesn't exist yet, etc).
CORE_PITCHER_FORM_COLUMNS = [
    "k_rate_last5", "whiff_rate_last5", "velo_avg_last5", "pitch_count_avg_last5",
]
IMPUTE_COLUMNS = [
    "opponent_k_rate_last10", "opponent_k_rate_vs_hand_season", "park_k_factor",
    "rest_days",
]
CONTINUOUS_REGRESSOR_COLUMNS = CORE_PITCHER_FORM_COLUMNS + IMPUTE_COLUMNS
REGRESSOR_COLUMNS = CONTINUOUS_REGRESSOR_COLUMNS + ["is_home"]
DESIGN_MATRIX_EXTRA_COLUMNS = ["was_imputed", "const"]
DESIGN_MATRIX_COLUMNS = REGRESSOR_COLUMNS + DESIGN_MATRIX_EXTRA_COLUMNS

# Candidate regressors for the skill-features variant (spec ②, 2026-06-30).
# Walk-forward decides which survive; VIF-prune correlated subsets before
# promoting any into the v1 allowlist above.
SKILL_CANDIDATE_COLUMNS = [
    "swstr_rate_last5",       # SwStr% rolling (= whiff_rate_last5 with explicit name)
    "csw_rate_last5",         # CSW% rolling
    "putaway_rate_last5",     # K/2-strike pitches rolling
    "whiff_rate_overall_last5",  # swing-denominator whiff%
    "k_minus_bb_rate_last5",  # (K−BB)/BF rolling
    "swstr_rate_season",      # SwStr% season-to-date
    "csw_rate_season",        # CSW% season-to-date
]

# Candidate regressors for the matchup + umpire features variant (spec ③,
# 2026-06-30).  Sourced from lineup ingestion (lineups.py + batter_logs.py)
# and the umpire tendency CSV (umpires.py).  Walk-forward decides which
# survive; VIF-check against team-level opponent features before promoting
# any into the v1 IMPUTE_COLUMNS allowlist above.
MATCHUP_CANDIDATE_COLUMNS = [
    "opponent_lineup_k_rate_vs_hand",  # PA-weighted batter K% vs this pitcher's hand
    "opp_share_opposite_hand",         # platoon fraction of opposing lineup
    "ump_k_factor",                    # multiplicative umpire K-tendency (1.0 = neutral)
]

# Candidate regressors for the weather + Vegas context variant (spec ④,
# 2026-06-30).  Sourced from weather.py (open-meteo) and vegas.py (ESPN).
# Walk-forward decides which survive; team_total_for/against deferred (no
# free keyless source; see src/data/vegas.py docstring).
CONTEXT_CANDIDATE_COLUMNS = [
    "temp_f",        # temperature at first pitch (°F); NaN for domes
    "wind_mph",      # wind speed at first pitch (mph); NaN for domes
    "humidity",      # relative humidity at first pitch (0–100); NaN for domes
    "is_dome",       # 1.0 = domed/retractable park, 0.0 = outdoor
    "game_total",    # Vegas over/under for the game (e.g. 8.5 runs)
    "is_favorite",   # 1.0 = pitcher's team is the favourite; 0.0 = underdog
]

REST_DAYS_CAP = 7.0
DEFAULT_TEST_FRACTION = 0.275
DISPERSION_THRESHOLD = 1.25
ALPHA_PVALUE_THRESHOLD = 0.05
THRESHOLDS = range(1, 11)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_train_test_split(df: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION,
                               cutoff_date=None):
    """
    Chronological holdout by game_date: train = strictly before cutoff,
    test = on/after cutoff. Row order in the input is irrelevant -- both
    outputs are sorted by game_date. If cutoff_date is omitted, it's chosen
    so roughly `test_fraction` of distinct game DATES (not rows) fall in
    test, which keeps the split robust to days with very different numbers
    of games.
    """
    if df.empty:
        return df.copy(), df.copy()

    d = df.copy()
    d["game_date"] = pd.to_datetime(d["game_date"])
    dates = np.sort(d["game_date"].unique())

    if cutoff_date is not None:
        cutoff = pd.Timestamp(cutoff_date)
    elif len(dates) <= 1:
        # Nothing to split on -- everything is test, train is empty.
        cutoff = dates[0]
    else:
        train_n = max(1, int(np.floor(len(dates) * (1 - test_fraction))))
        train_n = min(train_n, len(dates) - 1)  # always leave >= 1 test date
        cutoff = dates[train_n]

    train = d[d["game_date"] < cutoff].sort_values("game_date").reset_index(drop=True)
    test = d[d["game_date"] >= cutoff].sort_values("game_date").reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# Design matrix construction
# ---------------------------------------------------------------------------

def _engineer_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_home"] = (out["home_away"] == "home").astype(float)
    out["rest_days"] = out["rest_days"].clip(upper=REST_DAYS_CAP)
    return out


def _dropna_core(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer raw columns, then drop rows missing any core pitcher-form value."""
    return _engineer_raw_columns(df).dropna(subset=CORE_PITCHER_FORM_COLUMNS).reset_index(drop=True)


def fit_preprocessor(train_df: pd.DataFrame, extra_columns: list = None) -> dict:
    """
    Compute TRAINING-SET-ONLY statistics needed to transform any frame
    (train or test) into the model's design matrix: per-column impute means
    (computed post-winsorize) and per-column standardization mean/std
    (computed post-impute). Fitting these on anything but the training
    partition is leakage.

    `extra_columns`: optional list of additional feature columns (e.g.
    SKILL_CANDIDATE_COLUMNS) to include alongside the v1 allowlist. Extra
    columns are treated like IMPUTE_COLUMNS (NaN filled with training mean)
    and CONTINUOUS_REGRESSOR_COLUMNS (standardized). Stored in the returned
    preprocessor dict so transform_design_matrix can reconstruct them without
    a separate parameter.
    """
    extra_columns = list(extra_columns or [])
    t = _dropna_core(train_df)

    all_impute = IMPUTE_COLUMNS + [c for c in extra_columns if c not in IMPUTE_COLUMNS]
    impute_means = {}
    for col in all_impute:
        if col in t.columns:
            mean_val = float(t[col].mean(skipna=True))
            # If every training row is NaN (e.g. new skill column, all debutants),
            # mean(skipna=True) returns NaN; fall back to 0 so fillna() works.
            impute_means[col] = mean_val if not np.isnan(mean_val) else 0.0
        else:
            impute_means[col] = 0.0  # absent column → impute as 0 at transform time

    t_imputed = t.copy()
    for col in all_impute:
        if col in t_imputed.columns:
            t_imputed[col] = t_imputed[col].fillna(impute_means[col])

    all_continuous = CONTINUOUS_REGRESSOR_COLUMNS + [c for c in extra_columns if c not in CONTINUOUS_REGRESSOR_COLUMNS]
    scale_stats = {}
    for col in all_continuous:
        if col in t_imputed.columns:
            mean = float(t_imputed[col].mean())
            std = float(t_imputed[col].std(ddof=0))
            scale_stats[col] = (mean, std if std > 0 else 1.0)
        else:
            scale_stats[col] = (0.0, 1.0)  # absent column → identity transform

    return {"impute_means": impute_means, "scale_stats": scale_stats, "extra_columns": extra_columns}


def transform_design_matrix(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """
    Apply a preprocessor fit on the training set (see fit_preprocessor) to
    build the model's design matrix from `df` (train or test alike). Rows
    missing any CORE_PITCHER_FORM_COLUMNS value are dropped; missing
    IMPUTE_COLUMNS values are filled with the training mean and flagged via
    a single shared `was_imputed` indicator.

    Extra columns stored in `preprocessor["extra_columns"]` are imputed and
    standardized using the same training statistics, appended after the v1
    columns. Columns absent from `df` are filled with the impute mean
    (backward compatibility with old feature tables lacking new columns).
    """
    extra_columns = preprocessor.get("extra_columns") or []
    out = _dropna_core(df)

    # Inject missing extra columns (backward compat: old cached feature tables
    # won't have skill columns; fill with impute mean so the model can still run)
    for col in extra_columns:
        if col not in out.columns:
            out[col] = preprocessor["impute_means"].get(col, 0.0)

    all_impute = IMPUTE_COLUMNS + [c for c in extra_columns if c not in IMPUTE_COLUMNS]
    was_imputed = out[all_impute].isna().any(axis=1).astype(float)
    for col in all_impute:
        out[col] = out[col].fillna(preprocessor["impute_means"].get(col, 0.0))

    all_continuous = CONTINUOUS_REGRESSOR_COLUMNS + [c for c in extra_columns if c not in CONTINUOUS_REGRESSOR_COLUMNS]
    for col in all_continuous:
        mean, std = preprocessor["scale_stats"].get(col, (0.0, 1.0))
        out[col] = (out[col] - mean) / std

    all_regressors = REGRESSOR_COLUMNS + [c for c in extra_columns if c not in REGRESSOR_COLUMNS]
    design = out[all_regressors].copy()
    design["was_imputed"] = was_imputed.values
    design["const"] = 1.0
    all_design_cols = DESIGN_MATRIX_COLUMNS + [c for c in extra_columns
                                                if c not in DESIGN_MATRIX_COLUMNS]
    return design[all_design_cols]


def build_design_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                         extra_columns: list = None):
    """
    Fit the preprocessor on train_df, then transform train_df (and test_df,
    if given) with it. Returns (X_train, y_train, X_test, y_test,
    preprocessor); y_* are row-aligned to the surviving rows of X_* (both
    derived from the same _dropna_core() call order, so positions match).

    `extra_columns`: see fit_preprocessor. Used by --variant backtest runs.
    """
    preprocessor = fit_preprocessor(train_df, extra_columns=extra_columns)

    train_kept = _dropna_core(train_df)
    X_train = transform_design_matrix(train_df, preprocessor)
    y_train = train_kept[LABEL_COLUMN].reset_index(drop=True)

    if test_df is None:
        return X_train, y_train, None, None, preprocessor

    test_kept = _dropna_core(test_df)
    X_test = transform_design_matrix(test_df, preprocessor)
    y_test = test_kept[LABEL_COLUMN].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, preprocessor


# ---------------------------------------------------------------------------
# Overdispersion test / family selection
# ---------------------------------------------------------------------------

def dispersion_ratio(y_true, fitted_mean, n_params: int) -> float:
    """
    Pearson chi-square / residual degrees of freedom, given already-fitted
    means (e.g. a Poisson GLM's fitted mu on its own training data). ~1
    means Poisson's mean=variance assumption holds; notably > 1 means
    overdispersion. Pure function -- no statsmodels/scipy dependency -- so
    it's directly unit-testable against hand-computed values.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(fitted_mean, dtype=float)
    if len(y) != len(mu):
        raise ValueError("y_true and fitted_mean must be the same length")
    pearson_resid = (y - mu) / np.sqrt(mu)
    chi2 = float(np.sum(pearson_resid ** 2))
    dof = len(y) - n_params
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive to compute a dispersion ratio")
    return chi2 / dof


def _active_design_columns(X_train: pd.DataFrame) -> list:
    """
    X_train.columns, minus any non-intercept column that's constant in the
    training data. A constant regressor (e.g. `was_imputed` when nothing in
    this particular batch needed imputing, or a `rest_days` column with no
    variation) carries zero information -- statsmodels can't estimate a
    coefficient for it, and including it makes the GLM/NB information matrix
    singular (`numpy.linalg.LinAlgError: Singular matrix` surfaced during
    real-data testing of this module). `const` is intentionally constant and
    always kept. Uses the actual X_train columns (not the module-level
    DESIGN_MATRIX_COLUMNS) so extra_columns passed via build_design_matrix
    are handled correctly without additional parameters.
    """
    keep = [col for col in X_train.columns
            if col == "const" or X_train[col].nunique(dropna=False) > 1]
    return keep


def select_family(X_train: pd.DataFrame, y_train: pd.Series,
                   dispersion_threshold: float = DISPERSION_THRESHOLD,
                   pvalue_threshold: float = ALPHA_PVALUE_THRESHOLD):
    """
    Fit Poisson on the training set, test its residuals for overdispersion,
    and decide whether Negative Binomial is warranted -- the decision is
    made on TRAINING data only (see spec's "Choosing Poisson vs Negative
    Binomial" section). Returns (chosen_family, poisson_result, nb_result,
    diagnostics) where chosen_family is "poisson" or "negative_binomial".
    `diagnostics["active_columns"]` records which DESIGN_MATRIX_COLUMNS were
    actually used (see _active_design_columns) -- callers must use the same
    subset at predict time.
    """
    if sm is None or NegativeBinomial is None:
        raise ImportError("statsmodels is required to fit the baseline model (pip install statsmodels)")

    active_columns = _active_design_columns(X_train)
    X_fit = X_train[active_columns]

    poisson_result = sm.GLM(y_train, X_fit, family=sm.families.Poisson()).fit()
    disp_ratio = float(poisson_result.pearson_chi2 / poisson_result.df_resid)

    nb_result = NegativeBinomial(y_train, X_fit, loglike_method="nb2").fit(disp=False)
    # nb_result.params is a label-indexed Series (e.g. ['const', ..., 'alpha']),
    # so positional access must go through .iloc, not [-1] (which is a label
    # lookup and raises KeyError since -1 isn't a real label).
    alpha_hat = float(nb_result.params.iloc[-1])  # NB2: params = [betas..., alpha]

    lr_stat = float(2 * (nb_result.llf - poisson_result.llf))
    if scipy_stats is None:
        raise ImportError("scipy is required to compute the LR-test p-value (pip install scipy)")
    p_value = float(scipy_stats.chi2.sf(max(lr_stat, 0.0), df=1))

    chosen = "negative_binomial" if (disp_ratio > dispersion_threshold and p_value < pvalue_threshold) else "poisson"

    diagnostics = {
        "dispersion_ratio": disp_ratio,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "alpha_hat": alpha_hat,
        "chosen_family": chosen,
        "active_columns": active_columns,
    }
    return chosen, poisson_result, nb_result, diagnostics


# ---------------------------------------------------------------------------
# Threshold probabilities
# ---------------------------------------------------------------------------

def poisson_over_prob(mu, threshold: int):
    """P(K >= threshold) under Poisson(mu). threshold=0 -> 1.0 by definition."""
    if scipy_stats is None:
        raise ImportError("scipy is required for threshold probabilities (pip install scipy)")
    if threshold <= 0:
        return np.ones_like(np.asarray(mu, dtype=float))
    return scipy_stats.poisson.sf(threshold - 1, mu)


def nbinom_over_prob(mu, alpha, threshold: int):
    """
    P(K >= threshold) under the NB2 parameterization Var = mu + alpha*mu^2
    (statsmodels' convention), converted to scipy.stats.nbinom's (n, p):
    n = 1/alpha, p = 1/(1 + alpha*mu).
    """
    if scipy_stats is None:
        raise ImportError("scipy is required for threshold probabilities (pip install scipy)")
    if threshold <= 0:
        return np.ones_like(np.asarray(mu, dtype=float))
    mu = np.asarray(mu, dtype=float)
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return scipy_stats.nbinom.sf(threshold - 1, n, p)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class BaselineModel:
    """Thin wrapper around a fitted statsmodels GLM/NB result for prediction."""

    def __init__(self, family: str, result, preprocessor: dict, active_columns: list = None,
                 alpha: float = None):
        self.family = family  # "poisson" or "negative_binomial"
        self.result = result
        self.preprocessor = preprocessor
        # Columns the result was actually fit on (see _active_design_columns) --
        # defaults to the full allowlist for callers that don't pass it.
        self.active_columns = list(active_columns) if active_columns is not None else list(DESIGN_MATRIX_COLUMNS)
        self.alpha = alpha

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.result.predict(X[self.active_columns]), dtype=float)

    def predict_mean_with_se(self, X: pd.DataFrame):
        """Return (mu, eta_se) where eta_se is the SE of the linear predictor
        (SE of log(mu)) from the GLM's get_prediction(linear=True).

        This is *estimation / parameter* uncertainty — how precisely we estimated
        μ given the training data — NOT the count distribution's spread (the
        Poisson or NB variance).  Keep the two distinct: eta_se → band on p_over
        → conviction; the count-distribution variance already drives p_over itself.

        Falls back to (mu, zeros) for model types that don't expose
        get_prediction(linear=True) (e.g. some NegativeBinomialResults builds)
        so callers always get a valid array pair regardless of family.
        """
        X_sub = X[self.active_columns]
        mu = np.asarray(self.result.predict(X_sub), dtype=float)
        try:
            pred = self.result.get_prediction(X_sub, linear=True)
            eta_se = np.asarray(pred.se_mean, dtype=float)
        except (AttributeError, TypeError, NotImplementedError):
            eta_se = np.zeros(len(mu))
        return mu, eta_se

    def predict_over_prob(self, X: pd.DataFrame, threshold: int) -> np.ndarray:
        mu = self.predict_mean(X)
        if self.family == "poisson":
            return poisson_over_prob(mu, threshold)
        return nbinom_over_prob(mu, self.alpha, threshold)

    def predict_over_prob_sweep(self, X: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
        """Per-threshold P(K >= t) for t in `thresholds`, one column per t."""
        mu = self.predict_mean(X)
        data = {}
        for t in thresholds:
            data[t] = (
                poisson_over_prob(mu, t) if self.family == "poisson"
                else nbinom_over_prob(mu, self.alpha, t)
            )
        return pd.DataFrame(data, index=X.index)


def fit_baseline_model(train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                        extra_columns: list = None, **selector_kwargs):
    """
    End-to-end fit: build the design matrix, select Poisson vs NB on the
    training data, and return (model, diagnostics, (X_test, y_test)).

    `extra_columns`: optional list of additional feature columns to include
    (see build_design_matrix / fit_preprocessor). Passed through from
    run_backtest --variant skill-features.
    """
    X_train, y_train, X_test, y_test, preprocessor = build_design_matrix(
        train_df, test_df, extra_columns=extra_columns
    )
    if len(X_train) == 0:
        raise ValueError("No training rows remain after dropping rows missing core pitcher-form features")

    chosen, poisson_result, nb_result, diagnostics = select_family(X_train, y_train, **selector_kwargs)
    active_columns = diagnostics["active_columns"]
    if chosen == "poisson":
        model = BaselineModel("poisson", poisson_result, preprocessor, active_columns=active_columns)
    else:
        model = BaselineModel("negative_binomial", nb_result, preprocessor, active_columns=active_columns,
                               alpha=diagnostics["alpha_hat"])

    return model, diagnostics, (X_test, y_test)


# ---------------------------------------------------------------------------
# Persistence (task #9, module 3)
# ---------------------------------------------------------------------------
#
# Closes a real gap: BaselineModel previously had no save/load story, which
# meant any consumer (the pre-game refresh pipeline) would have had to refit
# on every run -- slow, and re-couples daily *prediction* to the training
# data pull. The model is fit/evaluated occasionally (and reviewed); it
# should be predicted from far more often and far more cheaply. So: persist
# the fitted artifact, load it (never refit) for prediction, and surface
# staleness as data rather than silently going stale (see metadata below).
#
# A single joblib file holds both the model's reconstruction fields AND a
# metadata dict -- a "sidecar" in spirit (clearly separated, returned as its
# own object from load_model) without the bookkeeping cost of two files that
# could drift out of sync if only one were copied/moved.

MODEL_ARTIFACT_VERSION = 1  # bump if the saved-payload shape changes


def save_model(model: "BaselineModel", path, *, train_through_date=None) -> None:
    """
    Persist a fitted BaselineModel to `path` via joblib. `train_through_date`
    is the max `game_date` in the data the model was trained on -- pass it
    from the training script; it's recorded in metadata (not derivable from
    the model object itself, which only holds the fitted statsmodels result).

    Metadata captured: `trained_at` (UTC timestamp of this save call),
    `train_through_date`, `family`, `artifact_version` (this module's
    save/load payload shape, for forward-compatible loading).
    """
    if joblib is None:
        raise ImportError("joblib is required to save a model (pip install joblib)")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_through_date": (
            train_through_date.isoformat() if hasattr(train_through_date, "isoformat")
            else train_through_date
        ),
        "family": model.family,
        "artifact_version": MODEL_ARTIFACT_VERSION,
    }
    payload = {
        "family": model.family,
        "result": model.result,
        "preprocessor": model.preprocessor,
        "active_columns": model.active_columns,
        "alpha": model.alpha,
        "metadata": metadata,
    }
    # Every other writer in this codebase (persist_oos_frame, generate_report's
    # plot functions, etc.) creates its destination directory before writing --
    # save_model didn't, which meant a fresh checkout's models/ directory not
    # existing yet (the common case) raised a raw FileNotFoundError out of
    # joblib instead of just creating the directory like everywhere else does.
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(payload, path)


def load_model(path):
    """
    Load a model saved via save_model(). Returns (model, metadata) -- the
    model is loaded, never refit, per the task #9 design. Raises a clear
    error (not a bare exception from joblib/pickle internals) if the file is
    missing or unreadable, since a missing/corrupt model artifact is a fatal,
    "train and save a model first" condition for any caller (the refresh
    pipeline in particular).
    """
    if joblib is None:
        raise ImportError("joblib is required to load a model (pip install joblib)")
    try:
        payload = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No model artifact found at {path!r}"
            " -- train and save a model first (see fit_baseline_model / save_model)."
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load model artifact at {path!r}: {exc}"
        ) from exc
    model = BaselineModel(
        payload["family"],
        payload["result"],
        payload["preprocessor"],
        active_columns=payload.get("active_columns"),
        alpha=payload.get("alpha"),
    )
    return model, payload["metadata"]


def model_age_days(metadata: dict, as_of=None) -> float:
    """
    Return the age of a saved model artifact in fractional days, using the
    `trained_at` UTC ISO timestamp stored by save_model().  Returns float('inf')
    if `trained_at` is absent (old artifact without the field).

    `as_of`: optional ISO timestamp string or datetime to use as "now" instead
    of the actual wall clock -- used in tests for deterministic assertions.
    """
    trained_at_str = metadata.get("trained_at")
    if not trained_at_str:
        return float("inf")
    trained_at = datetime.fromisoformat(trained_at_str)
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)
    if as_of is None:
        now = datetime.now(tz=timezone.utc)
    elif isinstance(as_of, str):
        now = datetime.fromisoformat(as_of)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
    else:
        now = as_of
    return (now - trained_at).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Pure evaluation helpers (task #10, reused by src/evaluation/metrics.py)
# ---------------------------------------------------------------------------

def brier_score(y_true, y_prob) -> float:
    """Mean squared error between binary outcomes and predicted probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(y_true, y_prob, eps: float = 1e-15) -> float:
    """Binary cross-entropy / log-loss. Clips predictions away from 0/1 by eps."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error between predicted and actual counts."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float)
                                - np.asarray(y_pred, dtype=float))))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Root mean squared error between predicted and actual counts."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float)
                                  - np.asarray(y_pred, dtype=float)) ** 2)))


# ---------------------------------------------------------------------------
# Production-model fit (task #11 step 4)
# ---------------------------------------------------------------------------

def fit_production_model(feature_table, through_date=None, fit_fn=None, save_path=None):
    """
    Fit the go-live production model on all starter rows through `through_date`
    (inclusive), with no train/test holdout.

    `through_date`: max game_date to include (inclusive). If None, uses the
    latest date in feature_table.
    `fit_fn`: callable(train_df, test_df=None) -> (model, diagnostics, (X_test, y_test)).
    `save_path`: if given, the fitted model is persisted there via save_model().

    Raises ValueError if no rows survive the through_date filter.
    Returns the fitted BaselineModel.
    """
    fit_fn = fit_fn or fit_baseline_model

    ft = feature_table.copy()
    ft["game_date"] = pd.to_datetime(ft["game_date"])

    if through_date is None:
        through_date = ft["game_date"].max()
    through_date = pd.Timestamp(through_date)

    train_df = ft[ft["game_date"] <= through_date].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(
            f"No completed starts on or before through_date={through_date.date()} "
            "in the supplied feature table -- nothing to train on."
        )

    result = fit_fn(train_df, None)
    model = result[0]

    if save_path is not None:
        save_model(model, save_path, train_through_date=through_date)

    return model
