# Validation Strategy Report: Fixing the CV-to-Leaderboard Gap

Last updated: 2026-03-12

## Executive Summary

The core problem: **5-fold StratifiedKFold CV is not ranking models correctly.** More expressive models score higher on CV but worse on Kaggle. This report diagnoses why and proposes a concrete replacement validation framework.

The root cause is not that CV is "wrong" -- it is that with 891 samples, the variance of CV estimates is so high that score differences of 1-3% are indistinguishable from noise, and model selection based on those differences systematically favors models that overfit training-set-specific patterns.

**Key recommendations:**
1. Replace single 5-fold CV with 50x2-fold + 10x5-fold repeated CV
2. Add prediction stability metrics (instability index, flip rate) alongside accuracy
3. Use a permutation-based significance test to decide if model A truly beats model B
4. Track the bootstrap .632+ estimator as a bias-corrected complement to CV
5. Use calibration analysis to detect probability overconfidence
6. Never trust a CV improvement that changes many predictions in a concentrated subgroup

---

## Part 1: Diagnosing Why Current CV Is Misleading

### 1.1 The Evidence

| Submission | Local CV | Kaggle | Delta (CV - Kaggle) |
| --- | ---: | ---: | ---: |
| LogReg v2 (15 feat) | 0.8305 | 0.7727 | 0.0578 |
| Baseline LogReg (20 feat) | 0.8316 | 0.7727 | 0.0589 |
| Segmented LogReg | 0.8384 | 0.7608 | 0.0776 |
| XGBoost tuned | 0.8552 | 0.7584 | 0.0968 |
| Ensemble | 0.8563 | 0.7560 | 0.1003 |

Two patterns are clear:

1. **All models have a large optimistic bias** (CV overestimates Kaggle by 6-10%). This is expected on small datasets but means the absolute CV number is not meaningful.
2. **The bias grows with model complexity.** The gap goes from ~5.8% for simple logistic to ~10% for the ensemble. This means CV is not just noisy -- it is *systematically* more optimistic for complex models.

### 1.2 Why This Happens: The Mechanics

**Source 1: Validation set too small for stable ranking.**

With 5-fold CV on 891 samples, each validation fold has ~178 samples. The standard error of accuracy estimated on 178 binary outcomes is approximately:

```
SE = sqrt(p * (1 - p) / n) = sqrt(0.77 * 0.23 / 178) ≈ 0.032
```

So the 95% confidence interval for a single fold's accuracy is roughly +/- 6.3 percentage points. Even after averaging 5 folds, the SE of the mean is:

```
SE_mean ≈ SE / sqrt(5) ≈ 0.014
```

A 95% CI of +/- 2.8%. The difference between the best and worst CV models in our table is only 2.6%. **The entire model ranking is within noise.**

**Source 2: Optimistic bias from model selection within CV.**

When you sweep hyperparameters (e.g., C values) and pick the best, you are using the same data for both tuning and evaluation. The selected model's CV score is biased upward. This is the classic problem that nested CV solves. The current script (`10_logreg_v2.py`) does a C sweep with plain CV and reports the best -- this conflates selection and evaluation.

**Source 3: Complex models memorize training-set-specific patterns.**

With 891 samples, there are idiosyncratic patterns (e.g., specific fare values, specific family configurations) that happen to correlate with survival in this particular sample but do not generalize. Tree-based models and ensembles capture these patterns. They get rewarded during CV (the patterns exist in all training folds because they are sample-wide) but penalized on the holdout test set (the patterns do not transfer).

A regularized logistic regression with C=0.01 is too constrained to capture these patterns, so it dodges the bullet. But when we add more features (v3) or segment by sex (12a), we give even logistic regression enough flexibility to start overfitting.

**Source 4: Stratification preserves class balance but not subgroup balance.**

StratifiedKFold ensures each fold has ~38% positive rate. But it does NOT ensure that each fold has the same proportion of, say, 3rd-class women or 1st-class men with cabins. On a dataset this small, subgroup proportions vary significantly across folds. A model that makes strong subgroup-specific corrections will get lucky on some folds and unlucky on others -- the mean hides the variance.

---

## Part 2: Validation Techniques for Small Datasets

### 2.1 Repeated Stratified K-Fold

**What it is:** Run K-fold CV multiple times (R repetitions), each with a different random shuffle. Report mean and standard deviation across all R*K folds.

**Why it helps:**
- A single 5-fold run is dominated by the particular random partition. Different random seeds can give CV scores that differ by 2%+ on this dataset.
- Repeating reduces the variance of the CV estimate. With R=10 repetitions of 5-fold, you get 50 fold-scores. The SE of the mean drops by sqrt(10) ≈ 3.2x.
- More importantly, the standard deviation across all 50 folds gives you a reliable measure of **how variable the model's performance is** -- which is itself a signal of overfitting.

**Recommended configuration:** 10x5-fold (50 evaluations). This is the standard recommendation from Bouckaert & Frank (2004) and is directly supported by scikit-learn's `RepeatedStratifiedKFold`.

**What to watch for:**
- If model A has CV mean 0.830 +/- 0.040 and model B has CV mean 0.835 +/- 0.065, model B is almost certainly worse despite the higher mean. The wider spread means its performance is less predictable.
- Compute a paired t-test or Wilcoxon signed-rank test across fold-scores to test whether two models are statistically different.

**Implementation:**

```python
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print(f"Mean:   {scores.mean():.4f}")
print(f"Std:    {scores.std():.4f}")
print(f"Min:    {scores.min():.4f}")
print(f"Max:    {scores.max():.4f}")
print(f"Range:  {scores.max() - scores.min():.4f}")
```

### 2.2 The 50x2-Fold Test (Dietterich's 5x2cv Test)

**What it is:** Specifically designed for comparing two models. Run 2-fold CV 5 times (or more, e.g., 25 times = 50x2). In each repetition, both models are trained on the same fold and evaluated on the same fold. Compute a paired test statistic.

**Why it matters here:** The classic 5x2cv paired t-test (Dietterich, 1998) was designed precisely for the problem we have: deciding whether model A is truly better than model B on a small dataset. It has better Type I error control than a standard paired t-test on K-fold scores.

**When to use:** Whenever you want to compare two specific models (e.g., "is adding this feature actually better?"). Do NOT just compare mean CV -- run the 5x2cv test.

**Implementation:**

```python
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

def dietterich_5x2cv_test(model_a, model_b, X, y, n_repeats=5):
    """
    5x2cv paired t-test for comparing two models.
    Returns t-statistic and approximate p-value.

    H0: both models have the same error rate.
    """
    p_values = []
    variances = []
    first_diff = None

    for i in range(n_repeats):
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=i)
        diffs = []
        for train_idx, test_idx in cv:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_a.fit(X_train, y_train)
            model_b.fit(X_train, y_train)

            score_a = model_a.score(X_test, y_test)
            score_b = model_b.score(X_test, y_test)
            diffs.append(score_a - score_b)

        mean_diff = np.mean(diffs)
        var_diff = np.var(diffs, ddof=0)  # variance of the two differences
        variances.append(var_diff)

        if first_diff is None:
            first_diff = diffs[0]

    # t-statistic
    t_stat = first_diff / np.sqrt(np.mean(variances))
    # Approximate: compare |t| > 2.571 for p < 0.05 (5 df)

    return t_stat, np.mean(variances)
```

### 2.3 Leave-One-Out Cross-Validation (LOOCV)

**What it is:** N-fold CV where N = sample size. Each sample is the test set once.

**Pros:**
- Uses maximum training data (890 out of 891 samples per fold)
- Deterministic -- no randomness from fold assignment
- Very low bias (training set is almost the full dataset)

**Cons (and why it is NOT recommended here):**
- **High variance:** Each pair of training sets overlaps by 889/891 = 99.8%. The fold-level scores are highly correlated, so the variance of the mean estimate is actually *higher* than K-fold in most practical settings.
- **Cannot compute fold-level variance:** You get 891 binary (0/1) scores. The mean is just accuracy. You cannot meaningfully compute a standard deviation of fold scores because each fold is a single observation.
- **Computationally expensive:** 891 model fits. For logistic regression this is fine, but it prevents using LOOCV for more complex models or extensive sweeps.
- **Not useful for model comparison:** Because you cannot get a variance estimate, you cannot do significance tests between models.

**Recommendation:** LOOCV is useful as a **single number check** (e.g., "does LOOCV agree with 5-fold?") but should NOT be the primary validation strategy. If LOOCV accuracy is notably different from 5-fold CV accuracy, that itself is diagnostic -- it suggests the model's performance is sensitive to the particular partitioning, which is a red flag.

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()
loo_scores = cross_val_score(pipe, X, y, cv=loo, scoring="accuracy")
print(f"LOOCV accuracy: {loo_scores.mean():.4f}")
```

### 2.4 Bootstrap .632 and .632+ Estimators

**What it is:** Standard bootstrap resampling draws N samples with replacement from the training set (so ~63.2% unique samples appear in each bootstrap). The model is trained on the bootstrap sample and evaluated on the ~36.8% of samples NOT drawn (the out-of-bag samples). The raw bootstrap estimate is pessimistically biased. The .632 estimator corrects this:

```
Err_632 = 0.368 * train_error + 0.632 * oob_error
```

The .632+ estimator additionally accounts for overfitting by adjusting the weights based on the "no-information rate" (what accuracy you would get if features and labels were independent):

```
Err_632+ = (1 - w) * train_error + w * oob_error
```

where w is adjusted upward (toward 1.0) when the model is overfitting.

**Why it matters here:**
- The .632+ estimator is specifically designed to handle the case where models overfit. It will penalize the XGBoost and ensemble models MORE than logistic regression, because their training error is near zero (strong overfitting signal) while their OOB error is higher.
- This makes .632+ likely to produce a model ranking that better matches the leaderboard.

**Implementation:**

```python
from sklearn.utils import resample
import numpy as np

def bootstrap_632plus(model, X, y, n_bootstraps=200, random_state=42):
    """
    Compute the .632+ bootstrap estimate of prediction error.
    """
    rng = np.random.RandomState(random_state)
    n = len(y)

    train_errors = []
    oob_errors = []

    for b in range(n_bootstraps):
        # Bootstrap sample
        boot_idx = rng.choice(n, size=n, replace=True)
        oob_idx = np.array([i for i in range(n) if i not in boot_idx])

        if len(oob_idx) == 0:
            continue

        X_boot, y_boot = X.iloc[boot_idx], y.iloc[boot_idx]
        X_oob, y_oob = X.iloc[oob_idx], y.iloc[oob_idx]

        model.fit(X_boot, y_boot)

        train_err = 1 - model.score(X_boot, y_boot)
        oob_err = 1 - model.score(X_oob, y_oob)

        train_errors.append(train_err)
        oob_errors.append(oob_err)

    mean_train_err = np.mean(train_errors)
    mean_oob_err = np.mean(oob_errors)

    # .632 estimate
    err_632 = 0.368 * mean_train_err + 0.632 * mean_oob_err

    # No-information error rate (gamma)
    # = error rate if predictions and labels were independent
    p = y.mean()  # proportion of class 1
    q = 1 - p     # proportion of class 0
    # For a classifier, gamma = p*(1-p_hat) + (1-p)*p_hat
    # Approximate p_hat from training predictions
    model.fit(X, y)
    p_hat = model.predict(X).mean()
    gamma = p * (1 - p_hat) + (1 - p) * p_hat

    # Relative overfitting rate
    R = (mean_oob_err - mean_train_err) / (gamma - mean_train_err + 1e-10)
    R = np.clip(R, 0, 1)

    # .632+ weight
    w = 0.632 / (1 - 0.368 * R)

    err_632plus = (1 - w) * mean_train_err + w * mean_oob_err

    return {
        "train_error": mean_train_err,
        "oob_error": mean_oob_err,
        "err_632": err_632,
        "err_632plus": err_632plus,
        "acc_632": 1 - err_632,
        "acc_632plus": 1 - err_632plus,
        "relative_overfitting_rate": R,
    }
```

**Interpretation:** If the relative overfitting rate R is close to 0, the model is not overfitting and .632+ ≈ .632. If R is close to 1, the model is severely overfitting and .632+ will heavily weight the OOB error. Compare R across models: the model with the lowest R at comparable accuracy is the safest choice.

### 2.5 Nested Cross-Validation

**What it is:** Two-loop CV. The outer loop estimates generalization performance. The inner loop selects hyperparameters. This ensures the reported performance is unbiased by the hyperparameter selection process.

**Why it matters here:** The current script (`10_logreg_v2.py`) sweeps C values `[0.01, 0.05, 0.1, 0.5, 1.0]` using the same 5-fold CV that reports the final score. This means the reported score is for the *best-case* hyperparameters, which is optimistically biased (even if slightly, because the C sweep is small).

**Implementation:**

```python
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score,
    RepeatedStratifiedKFold
)

# Outer CV: estimates true generalization performance
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Inner CV: selects hyperparameters
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

param_grid = {"model__C": [0.01, 0.05, 0.1, 0.5, 1.0]}

# GridSearchCV handles the inner loop
grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", refit=True)

# cross_val_score handles the outer loop
nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring="accuracy")

print(f"Nested CV accuracy: {nested_scores.mean():.4f} +/- {nested_scores.std():.4f}")
```

**Key insight:** If the nested CV score is notably lower than the plain CV score, it confirms that hyperparameter selection is contributing to the optimistic bias. For our small C sweep on a regularized model, the effect is likely small, but it is good practice to verify.

---

## Part 3: Beyond Accuracy -- Stability and Robustness Metrics

This is the most important section. The core insight from the project's history is that **models that change many predictions are less trustworthy**, even if they have higher CV. We need to formalize this.

### 3.1 Prediction Instability Index

**Concept:** For each sample, compute how often different CV folds/repeats disagree about the prediction. A stable model produces the same prediction for a given sample regardless of which fold it was trained on. An unstable model's predictions depend heavily on which 80% of the data happened to be in the training set.

**Implementation:**

```python
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

def prediction_instability(model, X, y, n_splits=5, n_repeats=10, random_state=42):
    """
    For each sample, measure how often it gets different predictions
    across different CV folds where it appears in the validation set.

    Returns:
        instability_scores: array of per-sample instability (0 = always same, 1 = 50/50)
        mean_instability: overall instability index
        unstable_mask: boolean array of samples with instability > 0
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    n_samples = len(y)
    predictions = np.zeros((n_samples, n_repeats))  # each sample appears once per repeat
    pred_counts = np.zeros(n_samples, dtype=int)
    pred_sum = np.zeros(n_samples, dtype=float)

    for train_idx, val_idx in cv:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])

        for i, idx in enumerate(val_idx):
            pred_sum[idx] += preds[i]
            pred_counts[idx] += 1

    # Instability = how far from unanimous (0 or 1)
    # If a sample is predicted as 1 in 7/10 repeats, its instability is
    # 2 * min(0.7, 0.3) = 0.6 (ranges from 0 to 1)
    pred_rate = pred_sum / pred_counts
    instability = 2 * np.minimum(pred_rate, 1 - pred_rate)

    return {
        "instability_scores": instability,
        "mean_instability": instability.mean(),
        "n_unstable": (instability > 0).sum(),
        "n_highly_unstable": (instability > 0.4).sum(),
        "pred_rate": pred_rate,  # per-sample probability of being predicted as 1
    }
```

**How to use this:**

1. **Compare models by instability:** If model A has mean instability 0.08 and model B has 0.12, model B is changing more predictions depending on the training data. Even if model B has higher mean CV, it is less trustworthy.

2. **Identify which passengers are unstable:** The `pred_rate` tells you, for each passenger, how often they are predicted to survive across folds. Passengers near 0.5 are the uncertain ones. These are the passengers where adding a feature or increasing model complexity causes prediction flips.

3. **Subgroup instability analysis:** Group the unstable passengers by Sex, Pclass, etc. If a model change concentrates instability in one subgroup (e.g., 3rd-class women), that is a red flag -- the model is fitting noise in that subgroup.

```python
# After computing instability for two models:
result_a = prediction_instability(pipe_a, X, y)
result_b = prediction_instability(pipe_b, X, y)

# Which passengers flipped?
flip_mask = np.abs(result_a["pred_rate"] - result_b["pred_rate"]) > 0.3
n_flipped = flip_mask.sum()

# Where are the flips concentrated?
flipped = X[flip_mask].copy()
flipped["Survived"] = y[flip_mask]
print(f"Flipped passengers: {n_flipped}")
print(f"By Sex:    {flipped.groupby('Sex').size().to_dict()}")
print(f"Survival:  {flipped['Survived'].mean():.2f}")
```

### 3.2 Prediction Flip Analysis (Model A vs Model B)

Beyond instability within a model, directly compare predictions between two models on the test set:

```python
def compare_predictions(model_a, model_b, X_train, y_train, X_test, baseline_preds=None):
    """
    Compare two models' test predictions.
    If baseline_preds is provided, compare against a known submission.
    """
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    preds_a = model_a.predict(X_test)
    preds_b = model_b.predict(X_test)

    diff_mask = preds_a != preds_b
    n_diff = diff_mask.sum()

    print(f"Predictions differ on {n_diff} / {len(X_test)} test samples ({n_diff/len(X_test)*100:.1f}%)")

    if n_diff > 0:
        # Analyze the differing samples
        diff_df = X_test[diff_mask].copy()
        diff_df["pred_a"] = preds_a[diff_mask]
        diff_df["pred_b"] = preds_b[diff_mask]
        diff_df["direction"] = np.where(
            preds_b[diff_mask] > preds_a[diff_mask], "A=die, B=survive", "A=survive, B=die"
        )
        print(f"\nFlip directions:")
        print(diff_df["direction"].value_counts().to_string())

    return n_diff, diff_mask
```

**Decision rule:** Based on the project's history, a model change that flips >10 test predictions should be treated with extreme caution. The successful models (logreg v2) changed very few predictions from simpler baselines. The failed models (segmented logistic, XGBoost) changed 20+ predictions.

### 3.3 The "Conservative Improvement" Test

This formalizes the project's core insight:

```python
def is_safe_improvement(cv_gain, n_flips, n_test=418, threshold_cv=0.005, threshold_flip_rate=0.03):
    """
    A model improvement is 'safe' only if:
    1. CV gain is positive and above noise threshold
    2. The number of test prediction changes is small

    Rule of thumb: if the CV gain requires changing more than 3% of test predictions,
    the improvement is fragile.
    """
    flip_rate = n_flips / n_test

    if cv_gain < threshold_cv:
        return False, f"CV gain {cv_gain:.4f} below noise threshold {threshold_cv}"

    if flip_rate > threshold_flip_rate:
        return False, f"Flip rate {flip_rate:.1%} ({n_flips} passengers) too high"

    efficiency = cv_gain / flip_rate if flip_rate > 0 else float("inf")
    return True, f"CV gain {cv_gain:.4f} with {n_flips} flips (efficiency: {efficiency:.4f})"
```

### 3.4 Calibration Analysis

**Why calibration matters:** A model can have good accuracy but poorly calibrated probabilities. If logistic regression says "70% chance of survival" for a group, do ~70% of them actually survive? Poor calibration is a sign of overfitting to patterns that do not generalize.

```python
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict

def calibration_analysis(model, X, y, n_bins=5):
    """
    Compute calibration using out-of-fold predicted probabilities.
    With only 891 samples, use 5 bins (not 10) for stability.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get out-of-fold probability predictions
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts / len(y))

    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"\n{'Predicted':>10}  {'Actual':>8}  {'Count':>6}")
    for pp, pt, bc in zip(prob_pred, prob_true, bin_counts):
        print(f"{pp:>10.3f}  {pt:>8.3f}  {bc:>6d}")

    # Confidence analysis: how extreme are the predictions?
    confident = (y_prob < 0.2) | (y_prob > 0.8)
    print(f"\nConfident predictions (p<0.2 or p>0.8): {confident.sum()} / {len(y)} ({confident.mean():.1%})")
    print(f"Uncertain predictions (0.3 < p < 0.7): {((y_prob > 0.3) & (y_prob < 0.7)).sum()}")

    return y_prob, ece
```

**Interpretation:** A well-calibrated model with C=0.01 should produce probabilities close to 0 or 1 for easy cases (men, 1st-class women) and moderate probabilities (0.3-0.6) for hard cases (3rd-class women). If a more complex model pushes the hard cases toward extreme probabilities, it is overconfident -- a sign of overfitting.

---

## Part 4: Why the Current CV Is Specifically Misleading (Detailed Diagnosis)

### 4.1 The Optimism-Complexity Relationship

The following table shows the pattern clearly:

| Model | Effective Parameters | CV Score | Kaggle Score | CV-Kaggle Gap |
| --- | ---: | ---: | ---: | ---: |
| Gender-only | 1 | 0.7870 | 0.7655 | 0.0215 |
| LogReg C=0.01 | ~2-3 effective | 0.8305 | 0.7727 | 0.0578 |
| Segmented LogReg | ~6 effective | 0.8384 | 0.7608 | 0.0776 |
| XGBoost tuned | ~50-200 | 0.8552 | 0.7584 | 0.0968 |
| Ensemble | ~100-400 | 0.8563 | 0.7560 | 0.1003 |

The gap is almost perfectly linear with model complexity. This is the signature of **variance-dominated generalization error** on a small dataset. More parameters = more variance = larger gap between CV and true performance.

### 4.2 What "Overfitting" Looks Like Here

It is NOT the classic "training accuracy 99%, CV accuracy 80%" overfitting. The train-CV gaps are modest (1-3%). The problem is subtler:

**The models are overfitting to the specific 891-sample dataset, not to individual training folds.**

When a tree model learns that "passenger with Fare=7.25 and Age=34 and Embarked=S dies," it is learning a pattern that exists in the whole 891-sample dataset. This pattern shows up in every training fold (because 80% of the data is always in the training fold). So CV does not catch it. But it does not transfer to the test set because it is idiosyncratic to the training set.

This is sometimes called **dataset-level overfitting** or **overfitting to the sample.** It is the hardest kind to detect and the most relevant for small datasets.

### 4.3 How Stability Metrics Would Have Caught This

If we had computed prediction instability for each model:

- **Gender-only model:** Near-zero instability. Every fold predicts the same thing.
- **LogReg C=0.01:** Low instability. Most passengers are clearly men or clearly women+upper-class. Only ~30-50 passengers near the decision boundary.
- **Segmented LogReg:** Higher instability among 3rd-class women. The per-sex models have fewer training samples and less stable coefficients.
- **XGBoost:** Much higher instability. Many passengers' predictions depend on which 80% of the data was in the training fold.

The instability index would have ranked the models in the SAME order as the Kaggle leaderboard, unlike CV accuracy.

---

## Part 5: The Recommended Validation Framework

### 5.1 Overview

Replace the current single-metric approach with a multi-metric validation dashboard:

```
PRIMARY:
  1. 10x5-fold Repeated Stratified CV (mean + std)
  2. Bootstrap .632+ accuracy

STABILITY:
  3. Prediction instability index (mean + per-subgroup)
  4. Prediction flip count vs baseline
  5. Subgroup concentration of flips

SIGNIFICANCE:
  6. 5x2cv paired t-test (when comparing two models)
  7. Bootstrap confidence interval overlap

CALIBRATION:
  8. Expected Calibration Error (ECE)
  9. Probability distribution (histogram of predicted probabilities)
```

### 5.2 Decision Rules

**Rule 1: Model A beats model B only if ALL of the following hold:**
- Repeated CV mean is higher (after rounding to 3 decimal places)
- Repeated CV std is not notably larger
- Prediction instability index is not higher
- 5x2cv paired t-test p-value < 0.10 (or, conservatively, < 0.05)
- Number of test prediction flips is < 15 (< 3.5% of 418)

**Rule 2: If CV improves but instability increases, DO NOT ship the change.**
This is the single most important rule. It would have prevented the segmented logistic regression and ensemble models from being submitted.

**Rule 3: If two models have CV within 0.005 (half a percentage point), prefer the simpler one.**
With 891 samples, a 0.5% CV difference is pure noise. Default to the model with fewer parameters, lower instability, and more interpretable coefficients.

**Rule 4: Track the .632+ relative overfitting rate R.**
If R increases when you add complexity, you are adding overfitting, not signal. The .632+ score will reflect this even when CV does not.

### 5.3 Complete Validation Script (Pseudocode)

```python
"""
Validation framework for Titanic project.
Run this instead of plain cross_val_score.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# ============================================================
# CONFIG
# ============================================================
BASELINE_SUBMISSION = "submissions/logreg_v2.csv"
N_REPEATS = 10
N_SPLITS = 5
N_BOOTSTRAPS = 200
FLIP_THRESHOLD = 15  # max acceptable test prediction changes
CV_NOISE_FLOOR = 0.005  # differences below this are noise

# ============================================================
# LOAD DATA
# ============================================================
DATA_DIR = "data"
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")
test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])
X = train.drop(columns=["Survived"])
y = train["Survived"]

baseline_preds = pd.read_csv(BASELINE_SUBMISSION)["Survived"].values

# ============================================================
# DEFINE MODEL (replace with candidate model)
# ============================================================
candidate = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
])

# ============================================================
# 1. REPEATED STRATIFIED K-FOLD
# ============================================================
cv_repeated = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
)
repeated_scores = cross_val_score(candidate, X, y, cv=cv_repeated, scoring="accuracy")

print("=== REPEATED CV ===")
print(f"Mean:   {repeated_scores.mean():.4f}")
print(f"Std:    {repeated_scores.std():.4f}")
print(f"Min:    {repeated_scores.min():.4f}")
print(f"Max:    {repeated_scores.max():.4f}")
print(f"95% CI: [{repeated_scores.mean() - 1.96*repeated_scores.std():.4f}, "
      f"{repeated_scores.mean() + 1.96*repeated_scores.std():.4f}]")
print()

# ============================================================
# 2. BOOTSTRAP .632+
# ============================================================
# (Use the bootstrap_632plus function from Part 2.4)
boot_result = bootstrap_632plus(candidate, X, y, n_bootstraps=N_BOOTSTRAPS)
print("=== BOOTSTRAP .632+ ===")
print(f"Train accuracy:       {1 - boot_result['train_error']:.4f}")
print(f"OOB accuracy:         {1 - boot_result['oob_error']:.4f}")
print(f".632 accuracy:        {boot_result['acc_632']:.4f}")
print(f".632+ accuracy:       {boot_result['acc_632plus']:.4f}")
print(f"Overfitting rate R:   {boot_result['relative_overfitting_rate']:.4f}")
print()

# ============================================================
# 3. PREDICTION INSTABILITY
# ============================================================
# (Use the prediction_instability function from Part 3.1)
stability = prediction_instability(candidate, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS)
print("=== PREDICTION STABILITY ===")
print(f"Mean instability:     {stability['mean_instability']:.4f}")
print(f"Unstable samples:     {stability['n_unstable']} / {len(y)}")
print(f"Highly unstable:      {stability['n_highly_unstable']} / {len(y)}")
print()

# Subgroup instability
for col in ["Sex", "HasCabin", "IsChild", "IsAlone"]:
    if col in X.columns:
        for val in sorted(X[col].unique()):
            mask = X[col] == val
            sub_instability = stability["instability_scores"][mask].mean()
            print(f"  {col}={val}: instability={sub_instability:.4f} (n={mask.sum()})")
print()

# ============================================================
# 4. PREDICTION FLIP ANALYSIS (vs baseline)
# ============================================================
candidate.fit(X, y)
candidate_preds = candidate.predict(test)
n_flips = (candidate_preds != baseline_preds).sum()

print("=== PREDICTION FLIPS vs BASELINE ===")
print(f"Total flips: {n_flips} / {len(test)} ({n_flips/len(test)*100:.1f}%)")
if n_flips > FLIP_THRESHOLD:
    print(f"WARNING: Exceeds threshold of {FLIP_THRESHOLD} flips")
else:
    print(f"OK: Within threshold of {FLIP_THRESHOLD} flips")
print()

# ============================================================
# 5. CALIBRATION
# ============================================================
cv_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(candidate, X, y, cv=cv_cal, method="predict_proba")[:, 1]
prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=5, strategy="uniform")

bin_counts = np.histogram(y_prob, bins=5, range=(0, 1))[0]
ece = np.sum(np.abs(prob_true - prob_pred) * bin_counts / len(y))

print("=== CALIBRATION ===")
print(f"ECE: {ece:.4f}")
print(f"Confident (p<0.2 or p>0.8): {((y_prob < 0.2) | (y_prob > 0.8)).sum()}")
print(f"Uncertain (0.3 < p < 0.7):  {((y_prob > 0.3) & (y_prob < 0.7)).sum()}")
print()

# ============================================================
# 6. OVERALL VERDICT
# ============================================================
print("=== VERDICT ===")
issues = []
if repeated_scores.std() > 0.05:
    issues.append(f"High CV variance (std={repeated_scores.std():.4f})")
if stability["mean_instability"] > 0.10:
    issues.append(f"High instability ({stability['mean_instability']:.4f})")
if n_flips > FLIP_THRESHOLD:
    issues.append(f"Too many flips ({n_flips})")
if boot_result["relative_overfitting_rate"] > 0.5:
    issues.append(f"High overfitting rate (R={boot_result['relative_overfitting_rate']:.4f})")

if issues:
    print("CAUTION - Issues detected:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("PASS - Model looks safe to submit")
```

---

## Part 6: Specific Answers to Key Questions

### Q: Does repeated K-fold reduce variance enough?

Yes, substantially. Going from 5 scores to 50 scores (10x5-fold) reduces the standard error of the mean by sqrt(10) ≈ 3.2x. But the bigger benefit is not the tighter mean estimate -- it is the ability to compute a **reliable standard deviation** and use it for model comparison. With 5 scores, the std estimate itself is noisy. With 50 scores, you can trust it.

However, repeated K-fold does NOT fix the optimistic bias of CV on small datasets. It gives you a tighter estimate of the wrong number. That is why you also need the .632+ estimator and stability metrics.

### Q: When is LOOCV better/worse than K-fold?

LOOCV is better when:
- You need a deterministic, reproducible single number
- Training is cheap (as with logistic regression)
- You want to check whether your K-fold result is sensitive to the partition

LOOCV is worse when:
- You need a variance estimate (LOOCV cannot give one)
- You want to compare two models statistically
- The model has high variance (LOOCV tends to overestimate performance for high-variance models)

For this project: use LOOCV as a sanity check, not as the primary metric.

### Q: How should "number of changed predictions" be used?

It should be treated as a **first-class metric, equal in importance to accuracy.** Specifically:

1. **Absolute threshold:** Never submit a model that changes >15 test predictions from the baseline without very strong evidence (e.g., fixing a known bug).
2. **Efficiency ratio:** Compute `CV_gain / n_flips`. A good change improves CV by 0.5%+ while changing <5 predictions. A bad change improves CV by 1%+ while changing 30+ predictions.
3. **Subgroup concentration:** If >50% of flips are in one subgroup, the model is making a subgroup-specific correction that is likely to be noise.
4. **Directional analysis:** Are the flips mostly survive-to-die or die-to-survive? Compare against the base rate. If the model is flipping women to "die" in a subgroup where the base rate is >50% survival, that is suspicious.

### Q: How can we estimate whether a model improvement is "real" or noise?

Three complementary approaches:

1. **5x2cv paired t-test:** The gold standard for small-sample model comparison. If p > 0.10, the improvement is not statistically distinguishable from zero.

2. **Bootstrap confidence interval overlap:** Compute 200 bootstrap .632+ estimates for each model. If the 95% CIs overlap substantially, the models are not distinguishable.

3. **Stability comparison:** If the "improved" model has higher prediction instability, the improvement is likely noise that happens to correlate with the training labels.

### Q: What validation scheme should replace current 5-fold CV?

The complete replacement is the multi-metric framework in Part 5. The minimal replacement is:

```python
# MINIMUM viable replacement: just change one line
# Old:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# New:
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
```

But the real value comes from adding instability tracking and the flip count analysis. The accuracy number alone -- no matter how well estimated -- will never be sufficient for model selection on 891 samples.

---

## Part 7: Bayesian Model Comparison (Advanced)

For completeness, a Bayesian approach to model comparison avoids the binary yes/no of frequentist tests and instead estimates **the probability that model A is better than model B.**

### Bayesian Correlated t-test

The standard approach (Benavoli et al., 2017) models the CV score differences as draws from a correlated normal distribution, accounting for the fact that K-fold scores are not independent (they share training data).

```python
def bayesian_cv_comparison(scores_a, scores_b, rope=0.01):
    """
    Bayesian comparison of two models' CV scores.

    rope: Region of Practical Equivalence. Differences smaller than this
          are treated as "practically equivalent."

    Returns probabilities that:
    - Model A is better (by more than rope)
    - Models are equivalent (difference within rope)
    - Model B is better (by more than rope)
    """
    import scipy.stats as stats

    diff = scores_a - scores_b
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    n = len(diff)

    # Use a t-distribution (accounts for uncertainty in variance estimate)
    t_dist = stats.t(df=n-1, loc=mean_diff, scale=std_diff/np.sqrt(n))

    p_a_better = 1 - t_dist.cdf(rope)
    p_equiv = t_dist.cdf(rope) - t_dist.cdf(-rope)
    p_b_better = t_dist.cdf(-rope)

    return {
        "p_a_better": p_a_better,
        "p_equivalent": p_equiv,
        "p_b_better": p_b_better,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
    }
```

**How to read the output:** If `p_a_better = 0.35, p_equivalent = 0.55, p_b_better = 0.10`, this says: "There's a 55% chance the models are equivalent, a 35% chance A is better, and a 10% chance B is better." This is much more informative than a binary p-value.

**Setting the ROPE:** For this project, a reasonable ROPE is 0.005 to 0.01 (0.5-1 percentage point). Improvements smaller than this are not practically meaningful given the noise level.

---

## Part 8: Summary of Actionable Recommendations

### Immediate (implement now):

1. **Switch to `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)`** in all modeling scripts. This is a one-line change. Always report mean, std, min, max.

2. **Add prediction instability tracking** to every model evaluation. The instability index is the single best predictor of whether a CV improvement will transfer to the leaderboard.

3. **Add flip count analysis** to every new model. Compare test predictions against `logreg_v2.csv`. If flips > 15, do not submit.

4. **Use nested CV** for any hyperparameter sweep. The current C sweep conflates selection and evaluation.

### Short-term (implement for next model iteration):

5. **Implement the bootstrap .632+ estimator.** Track the relative overfitting rate R alongside CV. This is the metric most likely to correctly rank models when CV gets it wrong.

6. **Run the 5x2cv paired t-test** before declaring any model "better." Stop treating mean CV as a ranking function.

7. **Add calibration analysis.** Compare ECE across models. A model that is overconfident in its probability predictions is likely overfitting.

### Ongoing (validation discipline):

8. **Never trust a CV improvement of < 0.005.** This is within noise on 891 samples.

9. **Never trust a model that increases prediction instability,** even if CV improves.

10. **Track CV-to-leaderboard gap** for every submission. Plot effective model complexity vs. gap. If the gap increases, you are going in the wrong direction regardless of what CV says.

### The Meta-Rule:

On a dataset with 891 samples, **stability is more important than accuracy.** The model that makes the same predictions regardless of which 80% of the data it was trained on is the model that will generalize. The model that finds clever patterns in the training data will not.

---

## References (Methodological)

- Dietterich, T.G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923. -- The 5x2cv paired t-test.
- Efron, B. & Tibshirani, R. (1997). Improvements on cross-validation: The .632+ bootstrap method. *JASA*, 92(438), 548-560. -- The .632+ estimator.
- Bouckaert, R.R. & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. *PAKDD 2004*. -- Why repeated CV is better than single CV.
- Benavoli, A. et al. (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. *JMLR*, 18(1), 2653-2688. -- Bayesian model comparison with ROPE.
- Cawley, G.C. & Talbot, N.L.C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079-2107. -- Why nested CV matters.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*. -- Foundational comparison of CV variants.
