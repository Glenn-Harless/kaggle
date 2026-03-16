# House Prices: Plan of Attack

**Competition:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
**Goal:** Learning-focused — master regression concepts, not chase leaderboard
**Approach:** Technique-focused (Approach B) — structured experiments targeting specific regression skills
**Date:** 2026-03-15

---

## The Dataset

- 1,460 training rows, 1,459 test rows
- 79 features (38 numeric, 43 categorical)
- Target: `SalePrice` (continuous, $34.9K–$755K, right-skewed)
- Metric: RMSLE (root mean squared log error) — penalizes underestimating cheap houses more
- Heavy missing values, but most are meaningful ("no pool", "no alley", not broken data)

## Key Differences from Titanic

| | Titanic | House Prices |
|---|---|---|
| Problem type | Classification (survived/died) | Regression (continuous price) |
| Target | Binary | Right-skewed continuous |
| Features | 12 raw, ~15 engineered | 79 raw |
| Metric | Accuracy (% correct) | RMSLE (log-scale error) |
| Missing data | Mostly broken | Mostly meaningful ("none") |
| Key challenge | Small N, noisy | High dimensionality, encoding |

---

## Steps

### Phase 1: Baseline (Steps 1-2)

#### Step 1: Quick EDA + Preprocessing

Understand just enough to get a model running.

- Examine target distribution (SalePrice) — expect right skew
- Handle missing values with simple rules:
  - Quality/categorical NaN = "None" (no feature present)
  - Numeric NaN = median imputation
- Encode categoricals: ordinal quality features → numeric, nominal → one-hot
- Output: `data/train_processed.csv`, `data/test_processed.csv`
- Script: `scripts/preprocessing/01_build_features.py`

#### Step 2: Dumb Baseline → Reasonable Baseline

Three baselines to establish reference points:

1. **Median baseline:** predict median price for everything (floor score)
2. **2-feature model:** linear regression on `GrLivArea` + `OverallQual`
3. **All-features Ridge:** Ridge regression on all processed features

Each gets: CV score (5-fold), Kaggle submission, and comparison.

- Script: `scripts/modeling/02_baselines.py`
- Submissions: `submissions/median_baseline.csv`, `submissions/two_feature.csv`, `submissions/ridge_baseline.csv`

**Primer — RMSLE:** Root Mean Squared Log Error = `sqrt(mean((log(pred) - log(actual))²))`. Lower is better. The log means a $10K error on a $50K house is worse than a $10K error on a $500K house — it measures *proportional* accuracy. If we train on `log(SalePrice)`, RMSLE becomes plain RMSE on the log target.

**What you learn:** How RMSLE works, what a "good" score looks like, how much a simple model captures.

### Phase 2: Regression Concepts (Steps 3-5)

#### Step 3: The Log Transform Experiment

Train two models on the same features:
- Model A: predict raw `SalePrice`
- Model B: predict `log(SalePrice)`, exponentiate predictions back

Compare CV scores. Then plot residuals of each to see *why* one is better.

- Script: `scripts/modeling/03_log_transform.py`

**Primer — Log transforms and heteroscedasticity:**

When the target is right-skewed (few very expensive houses, many moderate ones), a model trained on raw prices will make huge errors on expensive houses and small errors on cheap ones. This uneven error pattern is called *heteroscedasticity* — literally "different scatter."

Why it happens: the model minimizes squared error, so it focuses on getting the big numbers less wrong (since those contribute most to the loss). But it sacrifices accuracy on the bulk of houses to do so.

Log-transforming the target compresses the scale: $50K → 10.8, $200K → 12.2, $750K → 13.5. Now the model sees a much more uniform spread, and errors become proportional rather than absolute. A $10K error on a $50K house (20% off) and a $40K error on a $200K house (20% off) become roughly equal in log space.

The residual plots will make this concrete — we'll see the heteroscedasticity disappear.

**What you learn:** Why log transforms fix skewed regression targets, what heteroscedasticity looks like, how to diagnose it.

#### Step 4: Regularization — Ridge vs Lasso vs ElasticNet

With 79+ features, ordinary least squares overfits. Test three regularized approaches:

- **Ridge (L2):** adds penalty on the *sum of squared* coefficients. Shrinks all coefficients toward zero but never to exactly zero. Good when most features contribute a little.
- **Lasso (L1):** adds penalty on the *sum of absolute* coefficients. Drives useless coefficients to exactly zero — built-in feature selection. Good when many features are irrelevant.
- **ElasticNet (L1+L2):** mix of both. Good when features are correlated (which they are here — `GrLivArea` correlates with `TotRmsAbvGrd`, `GarageCars` correlates with `GarageArea`).

Sweep alpha values for each. Compare: CV scores, number of non-zero coefficients, which features each model keeps/drops.

- Script: `scripts/modeling/04_regularization.py`

**Primer — Regularization:**

Regularization adds a penalty to the loss function that discourages large coefficients. Without it, a model with 79 features can assign large weights to noise and overfit the training data.

The penalty strength is controlled by `alpha`:
- alpha = 0: no penalty → ordinary least squares
- alpha = very large: huge penalty → all coefficients crushed toward zero → underfitting
- alpha = just right: enough penalty to prevent overfitting but not so much that it kills real signal

**Ridge** penalizes `sum(coef²)`. Because squaring amplifies large values, Ridge aggressively shrinks the biggest coefficients while barely touching small ones. It keeps all features but reduces their influence.

**Lasso** penalizes `sum(|coef|)`. The absolute value creates a "diamond" constraint that has corners at zero — mathematically, this is why coefficients get driven to exactly zero. Lasso answers: "which features can I completely ignore?"

**ElasticNet** penalizes `l1_ratio * sum(|coef|) + (1 - l1_ratio) * sum(coef²)`. When correlated features exist (e.g., `GarageCars` and `GarageArea`), Lasso randomly picks one and drops the other. ElasticNet keeps both but shrinks them together.

The key experiment: compare which features Lasso zeros out vs which Ridge keeps. This directly tells you which of the 79 features matter.

**What you learn:** How each regularizer works, how to tune alpha, how to use Lasso as feature selection.

#### Step 5: Residual Analysis

After the best regularized model, analyze what's left — the errors.

- Plot residuals vs predicted values (should be random scatter if model is well-specified)
- Plot residuals vs key features (patterns = the model is missing something)
- Identify the worst predictions — what makes those houses hard?
- Check for outliers that may be distorting the model

- Script: `scripts/modeling/05_residual_analysis.py`

**Primer — Residual analysis:**

A residual is `actual - predicted`. If the model is perfect, residuals are zero. In practice, they form a cloud. The *shape* of that cloud tells you what the model is missing:

- **Funnel shape** (residuals get bigger for larger predictions): heteroscedasticity — the log transform from Step 3 should have fixed this, but check.
- **Curved pattern** (residuals form a U or arc): the model is missing a nonlinear relationship. Maybe `YearBuilt` has a quadratic effect (very old houses are charming, middle-aged houses are worst).
- **Clusters of large residuals** in one feature range: the model is systematically wrong for a subgroup. This tells you exactly where to engineer a new feature.
- **Random scatter**: the model has captured what it can. Remaining error is noise.

**What you learn:** How to read residual plots, how to use errors to guide feature engineering instead of guessing.

### Phase 3: Feature Engineering (Steps 6-7)

#### Step 6: Residual-Guided Feature Engineering

Use the patterns from Step 5 to create targeted features. For each:
1. Identify the residual pattern (e.g., "model underpredicts houses with finished basements")
2. Engineer a feature to address it
3. Validate: did CV improve? Did the residual pattern shrink?

Likely candidates (based on typical patterns in this dataset):
- Total square footage (combining basement + above ground)
- Age at sale (YrSold - YearBuilt)
- Remodel recency (YrSold - YearRemodAdd)
- Total bathrooms (combining full/half, basement/above ground)
- Interactions between quality and size

- Script: `scripts/modeling/06_feature_engineering.py`

**What you learn:** Disciplined feature engineering driven by data, not intuition. Each feature has a measurable reason to exist.

#### Step 7: Ordinal Encoding Deep Dive

Many features are quality ratings (Ex/Gd/TA/Fa/Po). Test three encoding strategies:

1. **One-hot:** treats each level as independent (ignores ordering)
2. **Ordinal numeric:** Ex=5, Gd=4, TA=3, Fa=2, Po=1 (assumes equal spacing)
3. **Target-mean encoding:** replace each level with the average SalePrice for that level (captures non-equal spacing, but risks overfitting — must use OOF encoding)

Compare CV scores across strategies. This is a technique you'll use on every tabular dataset.

- Script: `scripts/modeling/07_encoding_experiment.py`

**What you learn:** When ordinal encoding is appropriate vs when it loses information, and the risks of target encoding.

### Phase 4: Wrap-up (Steps 8-9)

#### Step 8: Final Model + Error Audit

- Fit the best model from the experiments above
- Submit to Kaggle
- Analyze remaining errors: what kinds of houses are hardest to predict?
- Are they outliers, unusual property types, or missing-data cases?

- Script: `scripts/modeling/08_error_audit.py`

**What you learn:** How to close the loop — understand what's left and whether it's fixable.

#### Step 9: Retrospective

Short document covering:
- Which regression concepts were most valuable
- What surprised you
- How CV and Kaggle correlated (better or worse than Titanic?)
- What would you do differently next time

- File: `RETROSPECTIVE.md`

### Phase 5: Beyond Linear Models (Steps 10-14)

**Motivation:** The Phase 2-4 work hit the linear model ceiling at ~0.136 Kaggle.
A third-party audit showed that basic XGBoost (0.126) already beats our tuned
ElasticNet (0.139) on full data without any tuning. The residual analysis (Step 5)
identified the remaining errors as nonlinear patterns — exactly where trees win.

#### Phase 5 Validation Protocol

All experiments in Phase 5 follow these rules to prevent information leakage:

1. **One canonical splitter.** All CV uses `KFold(n_splits=5, shuffle=True, random_state=42)`. Models are compared on the same folds. Repeated CV (5x5) for final comparison only.

2. **Early stopping inside folds.** For boosted models, early stopping uses a held-out portion of the *training fold* (e.g., 80/20 split within each training fold). The validation fold is never touched for early stopping — it is for scoring only.

   ```
   Full training set (1460 rows)
   └── CV fold split
       ├── Training fold (1168 rows)
       │   ├── Fit portion (934 rows) ← model trains here
       │   └── Early-stop portion (234 rows) ← monitors overfitting
       └── Validation fold (292 rows) ← scores here, never seen during training
   ```

3. **OOF predictions for blending.** Blend weights are fit on out-of-fold predictions only. Each sample's blend input comes from a model that never saw that sample during training. Never tune blend weights on Kaggle feedback.

4. **Full-data CV always.** No outlier removal during validation. The Step 6 outlier-removal CV (0.138 → 0.112) was a validation artifact — Kaggle only moved to 0.136. Models must score well on the full distribution.

5. **Leaderboard hygiene.** Require a local CV improvement before submitting. Cap at 1-2 submissions per step. Phase 5 goal is ~4 total submissions (tree baseline, tuned tree, blend, final).

#### Step 10: Tree Model Baselines (XGBoost + LightGBM)

Train XGBoost and LightGBM on all 1,460 rows with log1p(SalePrice) target.
Conservative hyperparameters, early stopping per the protocol above.
Compare to ElasticNet on the SAME full-data CV.

- XGBoost: max_depth=3, learning_rate=0.05, n_estimators=1000, early stopping
- LightGBM: num_leaves=31, learning_rate=0.05, n_estimators=1000, early stopping
- Both use the current one-hot/ordinal processed data (238 features)
- No StandardScaler needed (trees are scale-invariant)
- Script: `scripts/modeling/10_tree_baselines.py`

**Primer — Why trees handle what linear models can't:**

A linear model learns one coefficient per feature — `price = coef1 * GrLivArea + coef2 * OverallQual + ...`. It can't learn "the effect of GrLivArea depends on OverallQual" without an explicit interaction feature.

Trees learn decision rules: "IF GrLivArea > 3000 AND OverallQual >= 9 THEN predict $400K." Each split creates a region where the model can learn a different relationship. This means:
- Quality-size interactions are captured automatically
- Neighborhood-specific price slopes happen naturally
- Outliers get isolated into their own leaf nodes instead of distorting global coefficients
- No need for StandardScaler (trees are scale-invariant)

**XGBoost vs LightGBM:** Both are gradient-boosted tree frameworks. XGBoost grows trees level-by-level (balanced). LightGBM grows leaf-by-leaf (potentially unbalanced but faster and often more accurate). LightGBM also supports native categorical splits, but that requires raw category columns — our data is already one-hot/ordinal encoded, so both frameworks see the same numeric matrix here. Native categoricals are out of scope for this phase.

**What you learn:** How tree models compare to linear models on the same data, and whether XGBoost or LightGBM wins on this dataset.

#### Step 11: Tree Hyperparameter Tuning

Tune the better tree model from Step 10. Key hyperparameters:

- `max_depth` / `num_leaves`: tree complexity
- `min_child_weight` / `min_child_samples`: minimum samples per leaf
- `subsample` / `bagging_fraction`: row sampling per tree
- `colsample_bytree` / `feature_fraction`: column sampling per tree
- `reg_alpha` (L1) and `reg_lambda` (L2): tree-level regularization
- `n_estimators` + `learning_rate`: boosting rounds vs step size

Use randomized search over these ranges. Early stopping per the validation protocol.
The inner early-stopping split must be deterministic per outer fold: use
`random_state=fold_index` so that each fold's 80/20 split is fixed across
hyperparameter trials. This ensures trial-to-trial comparisons within a fold
are not confounded by different early-stopping splits.

- Script: `scripts/modeling/11_tree_tuning.py`

**What you learn:** How boosting hyperparameters control the bias-variance tradeoff differently from linear regularization.

#### Step 12: Outliers as Features, Not Deletions

Instead of removing the 4 large homes, handle them as features:

- `is_large_home` = GrLivArea > 4000 (binary flag)
- `GrLivArea_capped` = min(GrLivArea, 4000) (clip extreme values)
- Potentially: `SaleCondition_Partial * GrLivArea` interaction

Test each approach with both ElasticNet and the best tree model.
Trees should handle the raw outliers without any treatment — verify this.

- Script: `scripts/modeling/12_outlier_features.py`

**What you learn:** When to remove outliers vs encode them, and how model type affects that decision.

#### Step 13: OOF Blending

Blend ElasticNet and tree model(s) using OOF predictions:

1. Collect OOF predictions from each model (per the validation protocol)
2. Simple average: `0.5 * elasticnet_oof + 0.5 * xgboost_oof`
3. Weighted average: fit weights via constrained linear regression on OOF predictions
   (never on Kaggle scores). Weights must be non-negative and sum to 1 —
   no negative weights, no extrapolation beyond the individual models.
4. If three models (ElasticNet + XGBoost + LightGBM), test 3-way blend

Why blending works: ElasticNet captures the broad linear trend well,
tree models capture interactions and weird segments. Their errors are
partially uncorrelated, so averaging cancels some noise.

- Script: `scripts/modeling/13_blending.py`

**What you learn:** Why ensembling works, how to find blend weights without leaking, and when simple averaging beats complex stacking.

#### Step 14: Final Submission + Updated Retrospective

- Best blend submitted to Kaggle
- Updated RETROSPECTIVE.md with Phase 5 learnings
- Compare: linear-only score vs tree-only vs blend
- What would Phase 6 look like? (stacking, neural nets, or diminishing returns?)

---

## Principles (carried from Titanic)

1. **Every change gets validated** — CV score before and after
2. **Small changes, measured impact** — one thing at a time
3. **Keep submissions small** — submit often, track what each change does
4. **Understand before optimizing** — the point is learning, not leaderboard
5. **Clean code, clear output** — scripts should be readable and self-documenting
6. **Explain deeply** — every step includes the *why*, not just the *what*

## File Structure

```
competitions/house-prices/
├── data/                     # Raw data (gitignored)
├── docs/
│   └── PLAN.md              # This file
├── scripts/
│   ├── preprocessing/
│   │   └── 01_build_features.py
│   └── modeling/
│       ├── 02_baselines.py
│       ├── 03_log_transform.py
│       ├── 04_regularization.py
│       ├── 05_residual_analysis.py
│       ├── 06_feature_engineering.py
│       ├── 07_encoding_experiment.py
│       ├── 08_error_audit.py
│       ├── 10_tree_baselines.py
│       ├── 11_tree_tuning.py
│       ├── 12_outlier_features.py
│       ├── 13_blending.py
│       └── 14_final_submission.py
├── results/models/           # CV scores, logs, analysis output
├── submissions/              # Kaggle submission CSVs
└── RETROSPECTIVE.md          # Final reflection
```
