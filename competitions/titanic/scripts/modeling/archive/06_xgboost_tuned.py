"""
Titanic Model 6: XGBoost (Hyperparameter Tuned)

Uses RandomizedSearchCV to explore hyperparameter space.
Searches across tree depth, learning rate, regularization, and sampling params.
"""

import pandas as pd
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform, randint

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


tee = Tee(f"{RESULTS_DIR}/06_xgboost_tuned.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 6: XGBOOST (HYPERPARAMETER TUNED)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Define hyperparameter search space ----
param_distributions = {
    "n_estimators":      randint(100, 600),       # number of trees
    "max_depth":         randint(2, 8),            # tree depth
    "learning_rate":     uniform(0.01, 0.19),      # 0.01 to 0.20
    "subsample":         uniform(0.6, 0.4),        # 0.6 to 1.0
    "colsample_bytree":  uniform(0.5, 0.5),        # 0.5 to 1.0
    "min_child_weight":  randint(1, 10),
    "reg_alpha":         uniform(0, 1.0),          # L1 regularization
    "reg_lambda":        uniform(0.5, 2.0),        # L2 regularization
    "gamma":             uniform(0, 0.5),          # min loss reduction for split
}

print("--- Search Space ---")
for param, dist in param_distributions.items():
    if hasattr(dist, "a") and hasattr(dist, "b"):
        # uniform distribution
        print(f"  {param:20s}  [{dist.a:.2f}, {dist.a + dist.b:.2f}]")
    else:
        # randint
        print(f"  {param:20s}  [{dist.a}, {dist.b})")
print()

# ---- Run randomized search ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_distributions,
    n_iter=200,           # try 200 random combinations
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Searching 200 random hyperparameter combinations...")
print("(5-fold CV for each = 1,000 model fits total)")
print()

search.fit(X, y)

print("--- Search Results ---")
print(f"  Best CV accuracy: {search.best_score_:.4f}")
print()

# ---- Best hyperparameters ----
print("--- Best Hyperparameters ---")
best_params = search.best_params_
for param, value in sorted(best_params.items()):
    if isinstance(value, float):
        print(f"  {param:20s}  {value:.4f}")
    else:
        print(f"  {param:20s}  {value}")
print()

# ---- Compare: original vs tuned ----
print("--- Original vs Tuned XGBoost ---")
original_params = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
}
print(f"\n  {'Parameter':20s}  {'Original':>10s}  {'Tuned':>10s}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
for param in sorted(original_params.keys()):
    orig = original_params[param]
    tuned = best_params.get(param, "N/A")
    if isinstance(orig, float):
        print(f"  {param:20s}  {orig:10.4f}  {tuned:10.4f}")
    else:
        print(f"  {param:20s}  {orig:>10}  {tuned:>10}")
if "gamma" in best_params:
    print(f"  {'gamma':20s}  {'N/A':>10}  {best_params['gamma']:10.4f}")
print()

# ---- Validate best model with fresh CV ----
best_model = search.best_estimator_
fresh_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")

print("--- Fresh 5-Fold CV (best model) ---")
for i, score in enumerate(fresh_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {fresh_scores.mean():.4f}")
print(f"  Std:   {fresh_scores.std():.4f}")
print()

# ---- Full model comparison ----
print("--- Full Model Comparison ---")
results = {
    "Gender-only":           0.7870,
    "Logistic Regression":   0.8316,
    "Random Forest":         0.8316,
    "LightGBM":              0.8384,
    "XGBoost (original)":    0.8451,
    "Neural Network":        0.8204,
    "XGBoost (tuned)":       fresh_scores.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    marker = " <-- best" if score == max(results.values()) else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Train best model on full data ----
best_model.fit(X, y)

# Feature importance
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("--- Feature Importance (Tuned Model) ---")
print()
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:20s}  {row['importance']:.4f}  {bar}")
print()

# ---- Confusion matrix ----
y_pred = best_model.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Overfitting check ----
train_accuracy = best_model.score(X, y)
print(f"--- Overfitting Check ---")
print(f"  Training accuracy:  {train_accuracy:.4f}")
print(f"  CV accuracy:        {fresh_scores.mean():.4f}")
print(f"  Gap:                {train_accuracy - fresh_scores.mean():.4f}")
original_gap = 0.9091 - 0.8451
print(f"  Original XGB gap:   {original_gap:.4f}")
if train_accuracy - fresh_scores.mean() < original_gap:
    print(f"  Improvement: less overfitting than original")
print()

# ---- Top 10 search results ----
print("--- Top 10 Hyperparameter Combinations ---")
results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values("rank_test_score").head(10)
for _, row in results_df.iterrows():
    print(f"  Rank {int(row['rank_test_score']):2d}: {row['mean_test_score']:.4f} "
          f"(+/- {row['std_test_score']:.4f})")
print()

# ---- Generate submission ----
test_pred = best_model.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/xgboost_tuned.csv", index=False)
print(f"Submission saved: submissions/xgboost_tuned.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save results ----
importance_df.to_csv(f"{RESULTS_DIR}/06_xgboost_tuned_importance.csv", index=False)
print(f"\nResults saved to: results/models/06_xgboost_tuned.txt")
print(f"Importance saved to: results/models/06_xgboost_tuned_importance.csv")

tee.close()
