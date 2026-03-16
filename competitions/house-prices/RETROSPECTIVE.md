# House Prices: Retrospective

**Competition:** House Prices - Advanced Regression Techniques
**Final Score:** Kaggle RMSLE 0.127 (top 25%, position ~996/4035)
**Best Model:** Tuned XGBoost (max_depth=3, lr=0.08, full data)
**Date:** 2026-03-15 to 2026-03-16

---

## Score Progression

| Step | What changed | CV RMSLE | Kaggle |
|------|-------------|----------|--------|
| 2 | Ridge (raw, unscaled) | 0.236 | 0.184 |
| 3 | + StandardScaler + log target | 0.151 | — |
| 4 | + Tuned alpha (ElasticNet) | 0.138 | — |
| 6 | + Outlier removal (4 houses) | 0.112* | 0.136 |
| 10 | XGBoost baseline (full data) | 0.130 | — |
| 11 | XGBoost tuned | 0.128 | 0.127 |
| 13 | Blend (95% XGB + 5% EN) | 0.130 | — |

*\*inflated — CV evaluated on cleaned data, Kaggle on full distribution*

## What Drove Each Improvement

1. **Log transform (Step 3): 0.236 → 0.151 (36% better)**
   The single biggest win. Training on log1p(SalePrice) directly optimizes RMSLE, prevents negative predictions, and fixes heteroscedasticity (uneven error scatter). This transfers to any regression problem with a right-skewed target.

2. **Regularization tuning (Step 4): 0.151 → 0.138 (9% better)**
   Alpha=1.0 was arbitrary. With StandardScaler, the optimal ElasticNet alpha was 0.0285. Alpha's scale depends on feature scale and count — you always tune it.

3. **Switching to XGBoost (Step 10): 0.138 → 0.130 (6% better)**
   Trees capture nonlinear patterns that linear models structurally cannot: quality-size interactions, neighborhood-specific slopes, outlier isolation. No feature engineering needed — trees learn interactions automatically.

4. **XGBoost tuning (Step 11): 0.130 → 0.129 (1% better)**
   Conservative trees (depth 3), heavy randomization (50% columns, 70% rows), moderate regularization. Small gains — the defaults were already good.

## What Didn't Work

- **Feature engineering for linear models (Step 6):** TotalSF, HouseAge, QualxSF — none improved ElasticNet. Linear models already compute linear combinations through their coefficients. These features were redundant.

- **Encoding strategies (Step 7):** One-hot, ordinal, and target-mean encoding scored within 0.0004 RMSLE of each other. The equal-spacing assumption was "close enough."

- **Blending (Step 13):** ElasticNet added negligible signal on top of XGBoost. Error correlation was 0.67 — both models fail on the same houses. The optimizer gave ElasticNet only 4.5% weight.

- **Outlier removal as a strategy (Step 6):** The CV improvement was real (0.138 → 0.112) but misleading — Kaggle only moved to 0.136 because the test set has the full distribution. This was a validation artifact, not a modeling improvement.

## What Surprised Me

- **Four outliers caused 19% of the linear model's error** — but XGBoost handled them natively without removal. The "fix" depended on the model type.

- **The 2-feature model was worse than predicting the median** (RMSLE 0.55 vs 0.40). Training loss ≠ evaluation metric: OLS minimizes dollar error while RMSLE measures proportional error. Misaligned objectives can make a "real" model worse than a constant prediction.

- **Scaling hurt when alpha wasn't re-tuned** (Step 3, Model A: 0.362, worse than unscaled Ridge at 0.236). Scaling and regularization strength are coupled — you can't change one without reconsidering the other.

- **The realistic leaderboard range is narrow.** Scores 0.12 to 0.13 contain ~30% of all competitors. The gap between top 25% and top 5% is only 0.007 RMSLE (~$2K per house on a $300K home).

## CV-Kaggle Correlation

| Model | CV | Kaggle | Gap |
|-------|------|--------|-----|
| Median baseline | 0.400 | 0.417 | CV slightly optimistic |
| Ridge baseline | 0.236 | 0.184 | CV pessimistic |
| ElasticNet (cleaned data) | 0.112 | 0.136 | **CV badly optimistic** (validation artifact) |
| XGBoost tuned (full data) | 0.129 | 0.127 | CV slightly pessimistic |

**Lesson:** CV must evaluate on the same distribution as deployment. Removing hard cases from validation inflates scores. The XGBoost full-data CV (0.129 vs 0.127 Kaggle) is the healthiest gap — slightly pessimistic is the safe direction.

## Key Differences from Titanic

| Aspect | Titanic | House Prices |
|--------|---------|-------------|
| Target transform | Not needed (binary) | Essential (log transform was 36% improvement) |
| Feature engineering | Huge impact (ticket groups, surname rules) | Zero impact for linear models; trees didn't need it |
| Model type | Logistic regression was competitive | Linear models hit a ceiling; trees needed |
| Outlier handling | Not applicable | Important — but model-dependent (remove for linear, keep for trees) |
| Encoding choices | Mattered (ordinal Pclass) | Didn't matter |
| Hard cases | Bubble passengers (P≈0.5) | Non-normal sales (circumstances > features) |
| Validation risk | Grouped CV needed for ticket leakage | Full-data CV needed to avoid outlier-removal artifact |

## Concepts Learned

- **Regression fundamentals:** skewness, kurtosis, heteroscedasticity, RMSLE
- **Log transforms:** why they fix skewed targets and align training loss with evaluation metric
- **Regularization:** Ridge vs Lasso vs ElasticNet, alpha tuning, feature selection behavior
- **Residual analysis:** diagnosing model failures, identifying outliers, guiding feature engineering
- **Tree vs linear models:** when nonlinearity matters, why trees handle outliers and interactions natively
- **Validation design:** why evaluating on cleaned data inflates scores, early stopping inside CV folds
- **Blending:** when it helps (uncorrelated errors) and when it doesn't (0.67 correlation)

## Files

```
scripts/preprocessing/01_build_features.py    — EDA + preprocessing (238 features)
scripts/modeling/02_baselines.py              — 3 baselines (median, 2-feature, Ridge)
scripts/modeling/03_log_transform.py          — log transform experiment
scripts/modeling/04_regularization.py         — Ridge/Lasso/ElasticNet alpha sweep
scripts/modeling/05_residual_analysis.py      — residual analysis + diagnostic plots
scripts/modeling/06_feature_engineering.py     — feature engineering experiments
scripts/modeling/07_encoding_experiment.py     — ordinal encoding deep dive
scripts/modeling/08_error_audit.py            — final linear model error audit
scripts/modeling/10_tree_baselines.py         — XGBoost + LightGBM vs ElasticNet
scripts/modeling/11_tree_tuning.py            — XGBoost hyperparameter search (60 trials)
scripts/modeling/13_blending.py               — OOF blending with constrained weights
results/models/*.txt                          — logged output from each step
results/analysis/05_residual_plots.png        — residual diagnostic plots
results/analysis/08_error_audit.png           — error distribution plots
submissions/xgboost_tuned.csv                 — best submission (Kaggle 0.127)
```
