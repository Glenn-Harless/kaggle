"""
Titanic Model 9: Ensemble (Soft Voting)

Combines tuned XGBoost, Random Forest, and Logistic Regression.
Tests both equal weighting and performance-based weighting.
"""

import pandas as pd
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/data"
RESULTS_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/results/models"


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


tee = Tee(f"{RESULTS_DIR}/09_ensemble.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 9: ENSEMBLE (SOFT VOTING)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Define tuned models (best params from previous scripts) ----

# XGBoost tuned (from script 06)
xgb = XGBClassifier(
    n_estimators=568,
    max_depth=3,
    learning_rate=0.0492,
    subsample=0.8929,
    colsample_bytree=0.8257,
    min_child_weight=6,
    reg_alpha=0.2637,
    reg_lambda=1.0330,
    gamma=0.4149,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)

# Random Forest tuned (from script 07)
rf = RandomForestClassifier(
    n_estimators=543,
    max_depth=8,
    min_samples_leaf=6,
    min_samples_split=12,
    max_features=None,
    criterion="entropy",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Logistic Regression tuned (from script 08)
# Needs scaling, so wrap in a pipeline
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=0.1138,
        penalty="l2",
        solver="saga",
        max_iter=2000,
        random_state=42
    ))
])

print("--- Component Models ---")
print(f"  1. XGBoost (tuned):              CV=0.8552")
print(f"  2. Random Forest (tuned):        CV=0.8462")
print(f"  3. Logistic Regression (tuned):  CV=0.8406")
print()

# ---- Cross-validation setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# ENSEMBLE 1: Equal weights
# ============================================================
print("=" * 60)
print("ENSEMBLE A: EQUAL WEIGHTS (1/3 each)")
print("=" * 60)

ensemble_equal = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)],
    voting="soft",
    weights=[1, 1, 1]
)

scores_equal = cross_val_score(ensemble_equal, X, y, cv=cv, scoring="accuracy")

print("\n--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores_equal):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores_equal.mean():.4f}")
print(f"  Std:   {scores_equal.std():.4f}")
print()

# ============================================================
# ENSEMBLE 2: Performance-weighted
# ============================================================
print("=" * 60)
print("ENSEMBLE B: PERFORMANCE-WEIGHTED (proportional to CV score)")
print("=" * 60)

# Weight by CV accuracy (normalized)
w_xgb, w_rf, w_lr = 0.8552, 0.8462, 0.8406
total = w_xgb + w_rf + w_lr
w_xgb, w_rf, w_lr = w_xgb/total, w_rf/total, w_lr/total
print(f"\n  XGBoost weight:  {w_xgb:.3f}")
print(f"  RF weight:       {w_rf:.3f}")
print(f"  LogReg weight:   {w_lr:.3f}")

ensemble_weighted = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)],
    voting="soft",
    weights=[w_xgb, w_rf, w_lr]
)

scores_weighted = cross_val_score(ensemble_weighted, X, y, cv=cv, scoring="accuracy")

print("\n--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores_weighted):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores_weighted.mean():.4f}")
print(f"  Std:   {scores_weighted.std():.4f}")
print()

# ============================================================
# ENSEMBLE 3: XGBoost-heavy
# ============================================================
print("=" * 60)
print("ENSEMBLE C: XGBOOST-HEAVY (50/30/20)")
print("=" * 60)

ensemble_xgb_heavy = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)],
    voting="soft",
    weights=[0.50, 0.30, 0.20]
)

scores_xgb_heavy = cross_val_score(ensemble_xgb_heavy, X, y, cv=cv, scoring="accuracy")

print("\n--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores_xgb_heavy):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores_xgb_heavy.mean():.4f}")
print(f"  Std:   {scores_xgb_heavy.std():.4f}")
print()

# ============================================================
# Also try hard voting for comparison
# ============================================================
print("=" * 60)
print("ENSEMBLE D: HARD VOTING (majority rules)")
print("=" * 60)

ensemble_hard = VotingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)],
    voting="hard"
)

scores_hard = cross_val_score(ensemble_hard, X, y, cv=cv, scoring="accuracy")

print("\n--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores_hard):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores_hard.mean():.4f}")
print(f"  Std:   {scores_hard.std():.4f}")
print()

# ============================================================
# FINAL COMPARISON
# ============================================================
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

results = {
    "Gender-only":              0.7870,
    "LogReg (original)":        0.8316,
    "Random Forest (orig)":     0.8316,
    "Neural Network":           0.8204,
    "LightGBM":                 0.8384,
    "LogReg (tuned)":           0.8406,
    "XGBoost (original)":       0.8451,
    "Random Forest (tuned)":    0.8462,
    "XGBoost (tuned)":          0.8552,
    "Ensemble (equal)":         scores_equal.mean(),
    "Ensemble (weighted)":      scores_weighted.mean(),
    "Ensemble (XGB-heavy)":     scores_xgb_heavy.mean(),
    "Ensemble (hard vote)":     scores_hard.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print()
for name, score in sorted_results:
    marker = " <-- BEST" if score == max(results.values()) else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Pick the best ensemble and train on full data ----
best_name = max(
    [("equal", scores_equal.mean()), ("weighted", scores_weighted.mean()),
     ("xgb_heavy", scores_xgb_heavy.mean()), ("hard", scores_hard.mean())],
    key=lambda x: x[1]
)
print(f"Best ensemble variant: {best_name[0]} ({best_name[1]:.4f})")
print()

# Use the best one for submission
best_ensembles = {
    "equal": ensemble_equal,
    "weighted": ensemble_weighted,
    "xgb_heavy": ensemble_xgb_heavy,
    "hard": ensemble_hard,
}
best_ensemble = best_ensembles[best_name[0]]
best_ensemble.fit(X, y)

# ---- Confusion matrix ----
y_pred = best_ensemble.predict(X)
cm = confusion_matrix(y, y_pred)
print(f"--- Confusion Matrix (best ensemble, on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Overfitting check ----
train_accuracy = best_ensemble.score(X, y)
best_cv = best_name[1]
print(f"--- Overfitting Check ---")
print(f"  Training accuracy:  {train_accuracy:.4f}")
print(f"  CV accuracy:        {best_cv:.4f}")
print(f"  Gap:                {train_accuracy - best_cv:.4f}")
if train_accuracy - best_cv > 0.05:
    print(f"  Warning: Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Generate submission ----
test_pred = best_ensemble.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/ensemble.csv", index=False)
print(f"Submission saved: submissions/ensemble.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

print(f"\nResults saved to: results/models/09_ensemble.txt")

tee.close()
