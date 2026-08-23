"""
src/models/boosted_model.py — gradient-boosted second model.

Part 3 of Spec ④ (2026-06-30): weather + Vegas context + boosted ensemble.

Design: docs/design/specs/2026-06-30-weather-vegas-ensemble-design.md
("Part 3 — Boosted ensemble + recalibration").

Architecture
------------
- A ``HistGradientBoostingRegressor`` (scikit-learn) predicts the K *mean* (µ)
  directly, just like the GLM.  The same Poisson/NB distribution is placed
  around µ to derive per-threshold P(K >= t), so the calibrated-count-model
  framework is preserved.
- Per-threshold ``IsotonicRegression`` maps the raw parametric P(K >= t)
  (derived from the HGB µ) to a calibrated probability.  Calibration is
  fitted on a dedicated *calibration fold* that is **not** the test set.
- ``BoostedModel`` exposes the same interface as ``BaselineModel``
  (``predict_mean``, ``predict_mean_with_se``, ``predict_over_prob``,
  ``predict_over_prob_sweep``) so callers can use either interchangeably.
- ``predict_mean_with_se`` returns zeros for the SE component — HGB doesn't
  natively expose prediction-SE, and the booster is used as a second-opinion
  conviction modifier rather than the primary uncertainty estimate.

Calibration fold
----------------
``fit_boosted_model()`` accepts an optional explicit ``val_df`` calibration
fold.  If omitted, it takes the chronologically *last* ``VAL_FRACTION`` of
``train_df`` rows as the calibration fold and fits the HGB on the remaining
rows.  The outer test set (passed via ``test_df``) is never touched during
calibration — it's kept for the walk-forward comparison gate.

Ensemble use (agreement signal)
--------------------------------
The ``compute_agreement`` helper compares the GLM's µ and the booster's µ to
a line (e.g. PrizePicks over/under), returning:
  +1 : both models project above the line (same-direction bullish)
  -1 : both models project below the line (same-direction bearish)
   0 : models disagree (one above, one below)
  NaN: line or either µ is NaN

This value feeds the conviction spec: same-direction divergence from the
line raises conviction; disagreement lowers it.

Persistence
-----------
Uses the same joblib save/load pattern as baseline_model.py.
Default path: data/models/boosted_model.joblib

Guard against missing sklearn: if sklearn is not installed, the module
imports cleanly but raises ImportError at fit/load time with a clear message.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.baseline_model import (
    THRESHOLDS,
    build_design_matrix,
    poisson_over_prob,
    nbinom_over_prob,
)

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    HistGradientBoostingRegressor = None
    IsotonicRegression = None

try:
    import joblib
except ImportError:
    joblib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BOOSTED_MODEL_PATH = os.path.join("data", "models", "boosted_model.joblib")
MODEL_ARTIFACT_VERSION = 1

# Fraction of train_df held back as calibration fold when val_df is not given.
VAL_FRACTION = 0.25

# HGB defaults — conservative to avoid overfitting on moderate MLB datasets.
HGB_MAX_ITER = 200
HGB_MAX_DEPTH = 4
HGB_LEARNING_RATE = 0.05
HGB_MIN_SAMPLES_LEAF = 20
HGB_L2_REG = 1.0


# ---------------------------------------------------------------------------
# Agreement signal
# ---------------------------------------------------------------------------

def compute_agreement(glm_mu: float, booster_mu: float, line: float) -> float:
    """
    Compare GLM µ and booster µ relative to a betting line.

    Returns
    -------
    +1.0  both models project ABOVE the line (bullish agreement)
    -1.0  both models project BELOW the line (bearish agreement)
     0.0  models disagree (one above, one below line)
     NaN  any input is NaN
    """
    if any(np.isnan(v) for v in [glm_mu, booster_mu, line]):
        return np.nan
    glm_dir = np.sign(glm_mu - line)
    boost_dir = np.sign(booster_mu - line)
    if glm_dir == 0 or boost_dir == 0:
        return 0.0
    if glm_dir == boost_dir:
        return float(glm_dir)  # +1 or -1
    return 0.0  # disagreement


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class BoostedModel:
    """
    Gradient-boosted K-rate model with per-threshold isotonic recalibration.

    Interface matches BaselineModel so callers can substitute either.
    """

    def __init__(
        self,
        gbr,
        feature_columns: list,
        preprocessor: dict,
        calibrators: dict = None,
        family: str = "poisson",
        alpha: float = None,
    ):
        """
        Parameters
        ----------
        gbr : HistGradientBoostingRegressor
            Fitted HGB regressor.
        feature_columns : list[str]
            Ordered list of feature column names used to fit ``gbr``
            (excludes the GLM-only ``const`` column).
        preprocessor : dict
            Same-format as ``BaselineModel.preprocessor``: keys
            ``impute_means``, ``scale_stats``, ``extra_columns``.
        calibrators : dict[int, IsotonicRegression], optional
            Per-threshold isotonic calibrators mapping raw P(over) → P_cal.
        family : str
            "poisson" or "negative_binomial" — parametric distribution used
            for raw P(over) before isotonic calibration.
        alpha : float, optional
            NB overdispersion parameter (only for family="negative_binomial").
        """
        self.gbr = gbr
        self.feature_columns = list(feature_columns)
        self.preprocessor = preprocessor
        self.calibrators = dict(calibrators) if calibrators else {}
        self.family = family
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _X_for_gbr(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns available in X; fill missing with NaN."""
        out = pd.DataFrame(index=X.index)
        for col in self.feature_columns:
            out[col] = X[col] if col in X.columns else np.nan
        return out

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict K mean µ via the HGB (clipped to >= 0)."""
        return np.maximum(0.0, self.gbr.predict(self._X_for_gbr(X)))

    def predict_mean_with_se(self, X: pd.DataFrame):
        """
        Return (mu, zeros).  HGB doesn't natively expose prediction SE;
        the booster's role is a second-opinion direction check, not the
        primary uncertainty estimate.
        """
        mu = self.predict_mean(X)
        return mu, np.zeros(len(mu))

    def _raw_over_prob(self, mu: np.ndarray, threshold: int) -> np.ndarray:
        """Parametric P(K >= threshold) from the µ array before calibration."""
        if self.family == "poisson":
            return poisson_over_prob(mu, threshold)
        return nbinom_over_prob(mu, self.alpha, threshold)

    def predict_over_prob(self, X: pd.DataFrame, threshold: int) -> np.ndarray:
        """
        Calibrated P(K >= threshold).  Raw parametric probability is mapped
        through the threshold's IsotonicRegression when available.
        """
        mu = self.predict_mean(X)
        raw = self._raw_over_prob(mu, threshold)
        calib = self.calibrators.get(threshold)
        if calib is None:
            return raw
        # IsotonicRegression.predict expects 1-D input.
        calibrated = calib.predict(raw)
        return np.clip(calibrated, 0.0, 1.0)

    def predict_over_prob_sweep(
        self, X: pd.DataFrame, thresholds=THRESHOLDS
    ) -> pd.DataFrame:
        """Per-threshold calibrated P(K >= t), one column per threshold.

        Enforces the probabilistic constraint that P(K >= t) is non-increasing
        as t increases.  Independent per-threshold IsotonicRegressors don't
        guarantee cross-threshold monotonicity, so a cummin is applied across
        the ascending-threshold axis after calibration.
        """
        mu = self.predict_mean(X)
        sorted_t = sorted(thresholds)
        # Compute calibrated probabilities in ascending threshold order.
        raw_cols = {}
        for t in sorted_t:
            raw = self._raw_over_prob(mu, t)
            calib = self.calibrators.get(t)
            if calib is not None:
                raw = np.clip(calib.predict(raw), 0.0, 1.0)
            raw_cols[t] = raw
        # Stack and apply cummin along threshold axis to enforce monotonicity.
        arr = np.column_stack([raw_cols[t] for t in sorted_t])
        arr = np.minimum.accumulate(arr, axis=1)
        calibrated = {t: arr[:, j] for j, t in enumerate(sorted_t)}
        # Return columns in the original requested threshold order.
        data = {t: calibrated[t] for t in thresholds}
        return pd.DataFrame(data, index=X.index)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_calibrators(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gbr,
    feature_columns: list,
    family: str,
    alpha: float,
    thresholds=THRESHOLDS,
) -> dict:
    """
    Fit one IsotonicRegression per threshold on the calibration fold.

    Raw parametric P(K >= t) is derived from the HGB's µ prediction on
    X_val; the isotonic regression maps these to calibrated probabilities
    using the actual y_val outcomes as labels.
    """
    if IsotonicRegression is None:
        raise ImportError("scikit-learn is required for isotonic calibration (pip install scikit-learn)")

    out_cols = pd.DataFrame(index=X_val.index)
    for col in feature_columns:
        out_cols[col] = X_val[col] if col in X_val.columns else np.nan
    mu_val = np.maximum(0.0, gbr.predict(out_cols))

    calibrators = {}
    y_arr = np.asarray(y_val, dtype=float)

    for t in thresholds:
        if family == "poisson":
            raw_prob = poisson_over_prob(mu_val, t)
        else:
            raw_prob = nbinom_over_prob(mu_val, alpha or 1.0, t)

        actual = (y_arr >= t).astype(float)

        # Skip if all labels are the same (isotonic would be trivial)
        if actual.std() == 0:
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_prob, actual)
        calibrators[t] = iso

    return calibrators


def fit_boosted_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    extra_columns: list = None,
    family: str = "poisson",
    alpha: float = None,
    hgb_params: dict = None,
) -> "BoostedModel":
    """
    Fit a BoostedModel on ``train_df``.

    Steps
    -----
    1. Build the same design matrix as the GLM (shared preprocessor).
    2. Optionally split a calibration fold from ``train_df`` if ``val_df``
       is not given.
    3. Fit HistGradientBoostingRegressor on the (inner) training rows.
    4. Fit per-threshold IsotonicRegression on the calibration fold.
    5. Return a ``BoostedModel`` with the fitted HGB + calibrators.

    Parameters
    ----------
    train_df : DataFrame
        Training data in game_logs schema (same as baseline_model).
    test_df : DataFrame, optional
        Held-out test set (NEVER touched for fitting or calibration — kept
        for the walk-forward comparison gate).
    val_df : DataFrame, optional
        Explicit calibration fold.  If None, the last VAL_FRACTION of
        train_df rows (chronologically) is used.
    extra_columns : list, optional
        Additional feature columns (skill/matchup/context candidates).
    family : str
        "poisson" or "negative_binomial" — parametric family for raw P(over).
    alpha : float, optional
        NB alpha (only used when family="negative_binomial").
    hgb_params : dict, optional
        Override default HistGradientBoostingRegressor hyperparameters.

    Returns
    -------
    BoostedModel
    """
    if HistGradientBoostingRegressor is None:
        raise ImportError("scikit-learn is required for BoostedModel (pip install scikit-learn)")

    # ---- Build design matrix (shared preprocessor with GLM) ----
    X_train_full, y_train_full, _, _, preprocessor = build_design_matrix(
        train_df, test_df=None, extra_columns=extra_columns
    )
    if len(X_train_full) == 0:
        raise ValueError("No training rows after dropping missing core pitcher-form features")

    # Drop the GLM-only constant column; HGB doesn't need an intercept term.
    feature_columns = [c for c in X_train_full.columns if c != "const"]
    X_train_full = X_train_full[feature_columns]

    # ---- Internal calibration split if val_df not provided ----
    if val_df is not None:
        X_val_dm, y_val_dm, _, _, _ = build_design_matrix(
            val_df, test_df=None, extra_columns=extra_columns
        )
        X_val_dm = X_val_dm[[c for c in feature_columns if c in X_val_dm.columns]]
        X_inner = X_train_full
        y_inner = y_train_full
    else:
        # Chronological split: last VAL_FRACTION of rows → calibration.
        n = len(X_train_full)
        n_inner = max(1, int(np.ceil(n * (1.0 - VAL_FRACTION))))
        X_inner = X_train_full.iloc[:n_inner]
        y_inner = y_train_full.iloc[:n_inner]
        X_val_dm = X_train_full.iloc[n_inner:]
        y_val_dm = y_train_full.iloc[n_inner:]

    # ---- Fit HGB ----
    params = {
        "max_iter": HGB_MAX_ITER,
        "max_depth": HGB_MAX_DEPTH,
        "learning_rate": HGB_LEARNING_RATE,
        "min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
        "l2_regularization": HGB_L2_REG,
        "random_state": 42,
    }
    if hgb_params:
        params.update(hgb_params)
    gbr = HistGradientBoostingRegressor(**params)
    gbr.fit(X_inner, y_inner)

    # ---- Isotonic calibration ----
    calibrators = {}
    if len(X_val_dm) >= 20:  # only calibrate with a meaningful sample
        calibrators = _fit_calibrators(
            X_val_dm, y_val_dm, gbr, feature_columns, family, alpha
        )

    return BoostedModel(
        gbr=gbr,
        feature_columns=feature_columns,
        preprocessor=preprocessor,
        calibrators=calibrators,
        family=family,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_boosted_model(model: BoostedModel, path=DEFAULT_BOOSTED_MODEL_PATH,
                       *, train_through_date=None) -> None:
    """Persist a fitted BoostedModel to ``path`` via joblib."""
    if joblib is None:
        raise ImportError("joblib is required to save a model (pip install joblib)")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_through_date": (
            train_through_date.isoformat()
            if hasattr(train_through_date, "isoformat") else train_through_date
        ),
        "family": model.family,
        "n_calibrators": len(model.calibrators),
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model_type": "boosted",
    }
    payload = {
        "gbr": model.gbr,
        "feature_columns": model.feature_columns,
        "preprocessor": model.preprocessor,
        "calibrators": model.calibrators,
        "family": model.family,
        "alpha": model.alpha,
        "metadata": metadata,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(payload, path)


def load_boosted_model(path=DEFAULT_BOOSTED_MODEL_PATH) -> tuple:
    """
    Load a BoostedModel from ``path``.

    Returns
    -------
    (BoostedModel, metadata_dict)
    """
    if joblib is None:
        raise ImportError("joblib is required to load a model (pip install joblib)")

    payload = joblib.load(path)
    model = BoostedModel(
        gbr=payload["gbr"],
        feature_columns=payload["feature_columns"],
        preprocessor=payload["preprocessor"],
        calibrators=payload.get("calibrators", {}),
        family=payload.get("family", "poisson"),
        alpha=payload.get("alpha"),
    )
    return model, payload["metadata"]
