# Compression-fix gate — 2026-06-30

**Decision: **REJECTED****

## Gate reasoning

  (a) Top-decile bias_pct: baseline=-0.4%  fix=-2.8%  ✗ DID NOT IMPROVE
  (b) MAE: baseline=1.7943  fix=1.7879  ✓ OK
      log-loss@7: baseline=0.5171  fix=0.5145  ✓ OK

---

## Aggregate metrics (walk-forward OOS)

| metric | baseline | compression-fix | Δ |
|--------|----------|-----------------|---|
| n starts | 4268 | 4268 | — |
| MAE | 1.7943 | 1.7879 | -0.0064 |
| RMSE | 2.2358 | 2.2276 | -0.0081 |
| ECE | 0.0108 | 0.0135 | +0.0027 |
| Brier@7 | 0.1707 | 0.1698 | -0.0009 |
| log-loss@7 | 0.5171 | 0.5145 | -0.0027 |

---

## μ-decile table — BASELINE

*bias = mean_pred − mean_actual; negative = model under-predicted (actual > predicted)*

| decile | μ range | n | mean_pred | mean_actual | bias | bias_pct |
|--------|---------|---|-----------|-------------|------|----------|
| 1 | 1.73–3.74 | 427 | 3.212 | 3.279 | -0.066 | -2.0% |
| 2 | 3.74–4.15 | 427 | 3.961 | 3.981 | -0.020 | -0.5% |
| 3 | 4.15–4.44 | 427 | 4.303 | 4.222 | +0.081 | 1.9% |
| 4 | 4.44–4.69 | 426 | 4.563 | 4.484 | +0.080 | 1.8% |
| 5 | 4.69–4.94 | 427 | 4.814 | 4.614 | +0.201 | 4.4% |
| 6 | 4.94–5.18 | 427 | 5.055 | 5.330 | -0.275 | -5.2% |
| 7 | 5.18–5.43 | 426 | 5.297 | 5.263 | +0.034 | 0.6% |
| 8 | 5.43–5.74 | 427 | 5.583 | 5.405 | +0.178 | 3.3% |
| 9 | 5.74–6.18 | 427 | 5.954 | 6.068 | -0.114 | -1.9% |
| 10 | 6.18–9.54 | 427 | 6.760 | 6.696 | +0.064 | 1.0% |


## μ-decile table — COMPRESSION FIX

| decile | μ range | n | mean_pred | mean_actual | bias | bias_pct |
|--------|---------|---|-----------|-------------|------|----------|
| 1 | 1.39–3.72 | 427 | 3.208 | 3.340 | -0.131 | -3.9% |
| 2 | 3.72–4.14 | 427 | 3.957 | 3.796 | +0.161 | 4.2% |
| 3 | 4.14–4.41 | 427 | 4.284 | 4.300 | -0.016 | -0.4% |
| 4 | 4.41–4.67 | 426 | 4.540 | 4.538 | +0.002 | 0.1% |
| 5 | 4.67–4.91 | 427 | 4.791 | 4.614 | +0.177 | 3.8% |
| 6 | 4.91–5.13 | 427 | 5.024 | 5.173 | -0.149 | -2.9% |
| 7 | 5.13–5.37 | 426 | 5.244 | 5.275 | -0.031 | -0.6% |
| 8 | 5.37–5.69 | 427 | 5.528 | 5.396 | +0.132 | 2.5% |
| 9 | 5.69–6.12 | 427 | 5.891 | 6.185 | -0.294 | -4.7% |
| 10 | 6.13–8.88 | 427 | 6.671 | 6.726 | -0.055 | -0.8% |


---

## VIF on candidate K-skill cluster (compression-fix variant)

*VIF > 10 = problematic collinearity; drop the highest-VIF redundant feature*

| feature | VIF |
|---------|-----|
| k_stab_last5 | 33.56 |
| whiff_rate_last5 | 29.72 |
| velo_avg_last5 | 135.12 |
| pitch_count_avg_last5 | 34.69 |
| csw_rate_season | 162.25 |

---

## baseline_model.py edits

```
  CORE_PITCHER_FORM_COLUMNS: k_rate_last5 → k_stab_last5
  IMPUTE_COLUMNS: opponent_k_rate_last10 removed
```

---

## Feature sets compared

**Baseline (current production):**
  CORE: ['k_rate_last5', 'whiff_rate_last5', 'velo_avg_last5', 'pitch_count_avg_last5']
  IMPUTE: ['opponent_k_rate_last10', 'opponent_k_rate_vs_hand_season', 'park_k_factor', 'rest_days']

**Compression-fix candidate:**
  CORE: ['k_stab_last5', 'whiff_rate_last5', 'velo_avg_last5', 'pitch_count_avg_last5']
  IMPUTE: ['opponent_k_rate_vs_hand_season', 'park_k_factor', 'rest_days']
  EXTRA: ['csw_rate_season']

---

## Next steps

Gate rejected — production model unchanged. Investigate:
- If top-decile bias didn't shrink: k_stab_last5 may need a different C constant (tune EB_K_CONSTANT in rolling_features.py).
- If aggregate metrics regressed: widened-corpus retrain (the PRIMARY lever from the spec) may be needed before the feature changes show their full benefit.
