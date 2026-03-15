"""
Titanic Model 16: RBF SVM Nonlinear Probe

Tests whether a nonlinear model class (RBF kernel SVM) can improve on
the logistic regression baseline using the same v2 feature set.

If RBF SVM ties logistic regression, the ceiling is in the features,
not the model class. If it wins, there are interaction patterns worth
capturing that the linear model misses.

Approach:
  1. Grid search over C and gamma using inner 5-fold CV
  2. Evaluate best model through the full evaluation harness
  3. Compare paired against baseline CV scores

Constraints (from NEXT_STEPS.md):
  - One bounded nonlinear probe, not multiple families
  - Use repeated CV, flip audits, subgroup change analysis
  - Reject if it changes many passengers without a strong case
"""

import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import evaluate_model, reconstruct_v2_features

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# ---- Load data ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

# Reconstruct v2 features
train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)

y = train_v2["Survived"]
X = train_v2.drop(columns=["Survived"])
test_ids = test_v2["PassengerId"]
X_test = test_v2.drop(columns=["PassengerId"])

# Load baseline CV scores from step 13
baseline_cv = np.load(f"{BASE}/results/models/13_v2_baseline_cv_scores.npy")

print("=" * 60)
print("STEP 16: RBF SVM NONLINEAR PROBE")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print(f"Features: {list(X.columns)}")
print(f"Feature count: {X.shape[1]}")
print()


# ================================================================
# PHASE 1: Grid Search for best C and gamma
# ================================================================
print("--- PHASE 1: Hyperparameter Grid Search ---")
print()

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
])

param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1],
}

grid = GridSearchCV(
    pipe, param_grid,
    cv=inner_cv, scoring="accuracy",
    n_jobs=-1, return_train_score=True,
)

print(f"Grid: {len(param_grid['svm__C'])} C values x {len(param_grid['svm__gamma'])} gamma values = 25 combinations")
print(f"Inner CV: 5-fold stratified")
print("Searching...")
print()

grid.fit(X, y)

best_C = grid.best_params_["svm__C"]
best_gamma = grid.best_params_["svm__gamma"]
best_inner_score = grid.best_score_

print(f"Best inner CV score: {best_inner_score:.4f}")
print(f"Best parameters: C={best_C}, gamma={best_gamma}")
print()

# Show full grid results
print("--- Full Grid Results (sorted by mean test score) ---")
results_df = pd.DataFrame(grid.cv_results_)
results_df = results_df.sort_values("mean_test_score", ascending=False)

print(f"{'C':>6} {'gamma':>8} {'CV Mean':>8} {'CV Std':>7} {'Train':>7} {'Gap':>7}")
print("-" * 50)
for _, row in results_df.head(10).iterrows():
    c = row["param_svm__C"]
    g = row["param_svm__gamma"]
    mean = row["mean_test_score"]
    std = row["std_test_score"]
    train = row["mean_train_score"]
    gap = train - mean
    print(f"{c:>6} {str(g):>8} {mean:>8.4f} {std:>7.4f} {train:>7.4f} {gap:>+7.4f}")
print()

# Check for overfitting signals
best_train = results_df.iloc[0]["mean_train_score"]
best_gap = best_train - best_inner_score
print(f"Best model train-CV gap: {best_gap:+.4f}")
if best_gap > 0.05:
    print("  *** WARNING: Gap > 5% — potential overfitting ***")
elif best_gap > 0.03:
    print("  CAUTION: Gap 3-5% — moderate overfitting risk")
else:
    print("  Gap < 3% — looks healthy")
print()


# ================================================================
# PHASE 2: Evaluate best SVM through the harness
# ================================================================
print("--- PHASE 2: Full Evaluation Harness ---")
print()

# Build fresh pipeline with best params for the harness
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf", C=best_C, gamma=best_gamma,
        probability=True, random_state=42,
    )),
])

results_svm = evaluate_model(
    svm_pipeline,
    X, y,
    X_test, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=None,
    report_path=f"{BASE}/results/models/16_rbf_svm_probe.txt",
    label=f"16_RBF_SVM_C{best_C}_g{best_gamma}",
)


# ================================================================
# PHASE 3: Comparative Summary
# ================================================================
print("\n" + "=" * 60)
print("COMPARATIVE SUMMARY: LOGISTIC REGRESSION vs RBF SVM")
print("=" * 60)
print()

paired = results_svm.get("paired", {})
delta = paired.get("mean_delta", 0)
wins = paired.get("n_candidate_wins", 0)
losses = paired.get("n_baseline_wins", 0)
ties = paired.get("n_ties", 0)
flips = results_svm["flips"]["n_flips"]

print(f"{'Model':<35} {'CV Mean':>8} {'CV Std':>7} {'Delta':>7} {'Flips':>6} {'W/L/T':>10}")
print("-" * 75)
print(f"{'v2 LogReg (C=0.01)':<35} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'0':>6} {'--':>10}")
print(f"{'RBF SVM (C=' + str(best_C) + ', g=' + str(best_gamma) + ')':<35} {results_svm['cv_mean']:>8.4f} {results_svm['cv_std']:>7.4f} {delta:>+7.4f} {flips:>6} {f'{wins}/{losses}/{ties}':>10}")
print()

# Interpretation
print("--- Interpretation ---")
if delta > 0.005:
    print("  POSITIVE: RBF SVM shows meaningful improvement over logistic regression.")
    print("  This suggests nonlinear interactions exist in the current feature set.")
    if flips <= 20:
        print(f"  Test flips ({flips}) are within acceptable range.")
    else:
        print(f"  *** WARNING: {flips} test flips — high risk of leaderboard regression ***")
elif delta > 0.001:
    print("  MARGINAL: Tiny positive delta. Not convincing for a model class change.")
    print("  The decision boundary appears mostly linear on these features.")
elif delta > -0.001:
    print("  FLAT: RBF SVM essentially ties logistic regression.")
    print("  The ceiling is in the features, not the model class.")
else:
    print("  NEGATIVE: RBF SVM is worse. The linear model is already optimal for this feature set.")
    print("  Nonlinearity adds complexity without improving generalization.")
print()

# Save submission if compelling
if delta > 0.002 and flips <= 20:
    sub_path = f"{BASE}/submissions/logreg_16_rbf_svm.csv"
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": results_svm["test_preds"],
    })
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")
    print(f"Predicted survival rate: {results_svm['test_preds'].mean():.3f}")
else:
    print("No submission saved (delta too small or too many flips).")

print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")
