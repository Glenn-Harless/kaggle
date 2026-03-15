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
│       └── 08_error_audit.py
├── results/models/           # CV scores, logs, analysis output
├── submissions/              # Kaggle submission CSVs
└── RETROSPECTIVE.md          # Final reflection
```
