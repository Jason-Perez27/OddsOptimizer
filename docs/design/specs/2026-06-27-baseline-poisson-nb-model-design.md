# Baseline Strikeout Count Model (Poisson / Negative Binomial) — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Related:** `docs/decision_log.md` (2026-06-26, 2026-06-27 entries),
`docs/design/specs/2026-06-27-strikeout-feature-engineering-design.md`,
task #7. Implementation target: `src/models/baseline_model.py`,
`tests/test_baseline_model.py`.

## Goal

Fit a baseline count-regression model that predicts a starting pitcher's
strikeout total for an upcoming game, using only the pre-game features in the
`build_features.build_training_table()` output. The model produces a full
predicted count distribution per pitcher-game, from which P(K ≥ t) for each
threshold t (1+ … 10+) is derived — the per-threshold probabilities the
tiered-confidence output (decision log, 2026-06-27) is built on.

This is deliberately the *explainable baseline*: a single GLM with interpretable
coefficients and a clear Poisson-vs-NB decision, against which any later model
(gradient boosting, hierarchical, etc.) must justify its added complexity.

## Inputs

The single training table from
`src.features.build_features.build_training_table(pitch_df)` — one row per
pitcher-game, `strikeouts` as the label. Column groups (from the feature spec):

- **Identifiers / label (never null):** `pitcher`, `game_pk`, `game_date`,
  `pitcher_team`, `opponent_team`, `home_away`, `strikeouts`.
- **Same-game outcomes (LABEL-ADJACENT — must NOT be used as regressors):**
  `batters_faced`, `pitch_count`, `whiff_rate`, `fastball_velo_avg`,
  `innings_pitched`, `strikeouts_vs_LHB`, `batters_faced_vs_LHB`,
  `strikeouts_vs_RHB`, `batters_faced_vs_RHB`. These are measured *during* the
  game being predicted; using them leaks the outcome.
- **Pre-game pitcher features:** `k_rate_last5`, `k_rate_season`,
  `k_rate_vs_LHB`, `k_rate_vs_RHB`, `k_rate_home`, `k_rate_away`,
  `k_rate_vs_opponent_career`, `ip_avg_last5`, `pitch_count_avg_last5`,
  `whiff_rate_last5`, `velo_avg_last5`.
- **Pre-game opponent features:** `opponent_k_rate_last10`,
  `opponent_k_rate_vs_hand_season`, `opponent_k_rate_home`,
  `opponent_k_rate_away`.
- **Pre-game context:** `rest_days`, `home_away`, `pitcher_throws`,
  `park_k_factor`, `day_night` (currently always null — excluded until the
  schedule backfill exists).

## Regressor selection (v1)

The pitcher-form columns are near-collinear by construction (every `k_rate_*` is
a slightly different slice of the same pitcher's strikeout rate), as are the
opponent splits. Throwing all of them into a GLM inflates standard errors and
makes coefficients uninterpretable — the opposite of what a baseline is for. v1
uses a parsimonious, low-collinearity set, with the redundant variants
consciously held out:

**Selected regressors (v1):**

1. `k_rate_last5` — recent pitcher strikeout rate (primary form signal).
2. `whiff_rate_last5` — swinging-strike rate; a less-noisy, more causal process
   signal for strikeout skill than the K-rate itself.
3. `velo_avg_last5` — fastball velocity trend (skill/health proxy).
4. `pitch_count_avg_last5` — workload proxy; stands in for how deep into games
   the pitcher typically goes, i.e. how many strikeout *opportunities* a start
   tends to carry. (See "Exposure" below — this is how v1 captures opportunity
   without an explicit offset.)
5. `opponent_k_rate_last10` — opponent's recent strikeout-proneness.
6. `opponent_k_rate_vs_hand_season` — opponent K-rate specifically vs this
   pitcher's throwing hand (the matchup-specific opponent signal).
7. `park_k_factor` — ballpark strikeout environment.
8. `rest_days` — days of rest (winsorized; see below).
9. `home_away` — encoded as a single dummy (`is_home`, 1 = home).

**Held out of v1 (documented, not forgotten):**

- `k_rate_season`, `k_rate_home`, `k_rate_away`, `k_rate_vs_opponent_career` —
  redundant with `k_rate_last5`; revisit via regularization (sklearn path) or
  explicit collinearity check (VIF) in a later iteration.
- `k_rate_vs_LHB` / `k_rate_vs_RHB` — these split by the *batter's* stand, but
  pre-game we don't yet ingest the confirmed opposing lineup's L/R composition,
  so we can't weight them correctly. Deferred until lineup ingestion exists.
- `opponent_k_rate_home` / `opponent_k_rate_away` — partially captured by
  `is_home` plus the opponent recent/matchup terms; held out to limit collinearity.
- `pitcher_throws` — its predictive content largely flows through
  `opponent_k_rate_vs_hand_season` already; can be added as a dummy in a later
  iteration if residuals show a hand effect.

**Preprocessing:**

- **Standardize** the continuous regressors (z-score) so coefficients are
  comparable and the optimizer is well-conditioned (`park_k_factor` is ~100-scale
  while the K-rates are ~0.2-scale). Mean/SD are fit on the **training set only**
  and reused for the test set — fitting the scaler on the full set is leakage.
- **Winsorize `rest_days`** (e.g. cap at ~6–7 days) so post–All-Star-break or
  IL-return gaps don't act as high-leverage points.
- **Add an explicit intercept** (statsmodels does not add one automatically).
- **Null handling:** a pitcher's early starts have null rolling features. After
  the temporal split, drop rows missing any *selected pitcher-form* regressor
  (listwise) — these are genuinely un-predictable from history. For missing
  *opponent/park* features specifically (a team or park with no prior history
  yet), impute the training-set mean and add a binary `was_imputed` indicator
  rather than dropping, so we don't throw away an otherwise-usable pitcher row.
  Document the row count dropped vs imputed in the fit summary.

### Exposure / offset (design decision, recorded)

Strikeout count is mechanically bounded by opportunity: a pitcher who faces 28
batters has more chances than one who faces 18. The textbook formulation is a
rate model with an exposure **offset**: `K ~ Poisson(μ)` with
`log(μ) = offset + Xβ`, `offset = log(expected_batters_faced)`. The catch is
that *actual* `batters_faced` for the game being predicted is itself an outcome
(it's in the leakage list), so the offset would have to be built from a
**pre-game projection** of batters faced (e.g. from `ip_avg_last5` ×
~4.3 BF/inning, or `pitch_count_avg_last5` ÷ ~3.9 pitches/PA).

**v1 decision:** do **not** use an explicit offset. Instead include
`pitch_count_avg_last5` as an ordinary regressor and let the model learn the
workload→count relationship. Rationale: it avoids committing to a hand-built
exposure projection (whose error would propagate straight into the offset), and
keeps the baseline a single clean GLM. The offset formulation is the first thing
to try in a v2 iteration once a calibrated expected-BF projection exists — note
it explicitly in the model module docstring so it isn't lost.

## Train / test split (temporal, not random)

The split must be **chronological by `game_date`**, never a random K-fold.
Reasons:

1. The deployment task is to predict *future* games from *past* games; a random
   split lets the model train on games that occur chronologically *after* test
   games, which can't happen in production and inflates measured performance.
2. Even though each feature row is individually leakage-free (built only from
   prior games), random splitting still lets information about the test period's
   general environment (league-wide K trends, a pitcher's mid-season form)
   bleed into training. A temporal holdout is the honest analogue of live use.

**v1 split:** a single chronological holdout at a parameterizable cutoff date.
Train = games with `game_date < cutoff`; test = games with `game_date ≥ cutoff`.
Default cutoff chosen so roughly the most recent ~25–30% of games (by date) are
held out. The boundary is strict: `max(train.game_date) < min(test.game_date)`.

It is expected and acceptable for the **same pitcher to appear in both** train
and test — that mirrors reality (you repeatedly predict the same pitchers).
Grouping by pitcher is therefore *not* required; grouping by *date* is what
matters.

**Noted for a later iteration (not v1):** an expanding-window / walk-forward
backtest (retrain at each date step, predict the next slice) is the more
rigorous evaluation and is what task #10's backtest will effectively do; the
baseline uses the simpler single holdout to keep the first model tractable.

## Choosing Poisson vs Negative Binomial

Fit Poisson first, then test the **training residuals for overdispersion** and
escalate to NB only if warranted. The decision is made on **training data only**
(never peeking at the test set).

1. Fit `GLM(strikeouts ~ Xβ, family=Poisson)` on the training set.
2. **Dispersion statistic:** Pearson χ² ÷ residual degrees of freedom. ≈ 1 means
   Poisson's mean = variance assumption holds; >> 1 means overdispersion.
3. **Formal confirmation:** fit a Negative Binomial on the same training data and
   either (a) inspect its dispersion parameter α and its confidence interval, or
   (b) run a likelihood-ratio test of `α = 0` (Poisson) vs `α > 0` (NB). NB nests
   Poisson at α → 0.
4. **Decision rule:** use NB if the dispersion ratio exceeds ~1.25 **and** the
   LR test rejects α = 0 at p < 0.05; otherwise keep Poisson. Both the statistic
   and the chosen family are recorded in the fit output so the choice is auditable
   and re-checkable as more season data arrives.

Note the realistic possibility that per-start strikeout counts are only mildly
over- or even slightly *under*-dispersed (the batters-faced cap truncates the
upper tail). If so, Poisson is the correct, simpler choice and the spec's bias is
toward keeping it. (Quasi-Poisson is mentioned only as an alternative variance
adjustment; it doesn't give a proper count likelihood for the threshold
probabilities, so it is not the primary path.)

## Library: statsmodels (not sklearn) for the baseline

Use **statsmodels**. The baseline's purpose is interpretability and the
Poisson-vs-NB decision, both of which statsmodels supports natively:

- `statsmodels.api.GLM` with `families.Poisson` (and offset support for the v2
  exposure path), and `families.NegativeBinomial` / `discrete.NegativeBinomial`
  with an **estimated** α.
- Full inferential output the decision needs: coefficient standard errors and
  p-values, residual **deviance**, **Pearson χ²**, log-likelihood, AIC — i.e. the
  dispersion statistic and LR test come for free.
- Deterministic IRLS fit (no random seed needed), which makes the model testable
  for reproducibility.

`sklearn`'s `PoissonRegressor` is prediction-oriented (L2-penalized), has **no
negative-binomial** estimator and **no** built-in dispersion test or inferential
summary, so it can't drive the Poisson-vs-NB choice. It stays a candidate for a
*later* regularized/cross-validated variant (where its `Pipeline`/penalty
tooling helps), not the baseline. Both libraries are already pinned in
`requirements.txt`.

## From count model to threshold probabilities

The model module exposes the predicted distribution, not just a point estimate.
For each test row with fitted mean μ (Poisson) or (μ, α) (NB):

- `predict_mean()` → μ (expected strikeouts).
- `predict_over_prob(threshold t)` → P(K ≥ t) = 1 − F(t − 1), using
  `scipy.stats.poisson` or `scipy.stats.nbinom` (NB's (n, p) derived from
  (μ, α)). `scipy` is a transitive dependency of the pinned stack; pin it
  explicitly in `requirements.txt` as part of this task.
- A convenience that sweeps t = 1 … 10 to produce the per-threshold P(over)
  vector the tiered-confidence output consumes.

This is what connects task #7 to the project's actual deliverable; it is part of
the baseline module, not a later add-on.

## Evaluation metrics (reported on the temporal test set)

Three layers, because a good betting model must be right *and* calibrated:

1. **Point accuracy of the count** (μ vs actual `strikeouts`): **MAE** and
   **RMSE**. Reported alongside two naive baselines the model must beat —
   (a) predict each pitcher's `k_rate_season × expected_BF`, and (b) predict the
   global mean strikeout count. A baseline that can't beat "the pitcher's own
   recent average" isn't worth shipping.
2. **Distributional fit:** mean **Poisson/NB deviance** and **mean
   log-likelihood per game** on the test set, plus the **test-set dispersion
   ratio** as a sanity check that the train-chosen family still holds out of
   sample.
3. **Calibration against actual strikeout outcomes** (the metric that matters
   most for the tiered output):
   - **Reliability of P(K ≥ t):** bucket predicted over-probabilities and plot
     predicted vs. empirical hit rate, at the thresholds that actually appear as
     PrizePicks lines (≈ 4.5–7.5). Report a reliability diagram plus a summary
     **calibration error** (e.g. ECE).
   - **Brier score** and **log loss** on the binary over/under at a couple of
     representative thresholds (e.g. 5.5, 6.5).
   - A **PIT histogram** (probability integral transform for counts) as an
     overall distributional-calibration check — flat ≈ well-calibrated.

Plots/tables land in `reports/` per the repo layout; numeric metrics are
returned by an evaluation helper so tests can assert on them.

## Module layout

- `src/models/baseline_model.py` — feature-matrix construction (the regressor
  allowlist + preprocessing), train/test temporal split, Poisson/NB fit, the
  dispersion-based family selector, and the predict / threshold-probability
  methods.
- Evaluation metric helpers (MAE, RMSE, deviance, Brier, log loss, calibration)
  live either in this module or a thin `src/evaluation/` helper it imports —
  either is fine, but they must be importable by the test without fitting a full
  model (pass-in predictions, get-back numbers).

## Deferred (conscious scope cuts)

- Explicit exposure **offset** model (needs a calibrated expected-BF projection).
- Regularized / cross-validated sklearn variant and full collinearity (VIF)
  pruning of the held-out `k_rate_*` family.
- Batter-hand-weighted pitcher splits (needs confirmed-lineup ingestion).
- Walk-forward / expanding-window evaluation (arrives with task #10's backtest).
- `day_night`, umpire, weather, bullpen features (per the feature spec's
  deferral list).

## Testing approach — what `tests/test_baseline_model.py` must assert

Deterministic, no-network unit tests on small hand-built DataFrames, mirroring
the existing `tests/test_*` style:

1. **Temporal split is leak-free:** for a toy frame spanning several dates, the
   split function puts every test-row `game_date` on or after the cutoff and
   strictly after every train-row date (`max(train.date) < min(test.date)`), and
   a random shuffle of input order doesn't change the partition.
2. **Regressor matrix excludes leakage columns:** the built design matrix
   contains exactly the documented v1 allowlist and **none** of the label /
   same-game-outcome columns (`strikeouts`, `batters_faced`, `whiff_rate`,
   `innings_pitched`, `pitch_count`, the `*_vs_LHB/RHB` raw counts). Assert by
   set comparison against the allowlist.
3. **No NaNs reach the fitter:** after preprocessing, rows missing core
   pitcher-form features are dropped and opponent/park nulls are imputed with the
   `was_imputed` flag set — assert the resulting matrix has no NaNs and that a
   known-incomplete early row was handled the specified way.
4. **Scaler is fit on train only:** the standardizer's mean/SD come from the
   training partition (assert test-set transform uses train statistics, e.g. a
   column constant in train but not test scales as expected).
5. **Model fits and predicts:** on a small synthetic set the fit returns a fitted
   model; `predict_mean` returns finite, **non-negative** μ of the right length.
6. **Family selector logic:** on a constructed **overdispersed** count sample the
   selector returns "NB"; on a clean Poisson-generated sample it returns
   "Poisson". At minimum, assert the dispersion ratio is computed correctly on a
   tiny hand-checkable case and that the selector returns a valid family for both.
7. **Threshold probabilities are well-formed:** P(K ≥ t) is in [0, 1],
   **monotonically non-increasing in t**, equals 1 at t = 0, and matches a
   `scipy.stats` reference value for a known μ (and known α for the NB path).
8. **Metric helpers are correct:** MAE / RMSE / log loss / Brier match
   hand-computed values on a tiny fixed prediction-vs-actual array.
9. **Reproducibility:** fitting twice on identical data yields identical
   coefficients (statsmodels IRLS is deterministic) — guards against accidental
   nondeterminism in preprocessing.
10. **Edge cases:** empty input and a single-row input are handled gracefully
    (clear error or empty result, not a crash).
