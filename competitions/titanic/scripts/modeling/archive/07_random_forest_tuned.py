"""
Titanic Model 7: Random Forest (Hyperparameter Tuned)

Uses RandomizedSearchCV to explore hyperparameter space.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
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


tee = Tee(f"{RESULTS_DIR}/07_random_forest_tuned.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 7: RANDOM FOREST (HYPERPARAMETER TUNED)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Define hyperparameter search space ----
param_distributions = {
    "n_estimators":      randint(100, 800),
    "max_depth":         randint(3, 12),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 15),
    "max_features":      ["sqrt", "log2", None],
    "bootstrap":         [True],
    "criterion":         ["gini", "entropy"],
}

print("--- Search Space ---")
for param, dist in param_distributions.items():
    if hasattr(dist, "a") and hasattr(dist, "b"):
        print(f"  {param:20s}  [{dist.a}, {dist.b})")
    elif isinstance(dist, list):
        print(f"  {param:20s}  {dist}")
print()

# ---- Run randomized search ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_distributions,
    n_iter=200,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Searching 200 random hyperparameter combinations...")
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
print("--- Original vs Tuned Random Forest ---")
original_params = {
    "n_estimators": 500, "max_depth": 6, "min_samples_leaf": 5,
    "max_features": "sqrt", "criterion": "gini",
}
print(f"\n  {'Parameter':20s}  {'Original':>10s}  {'Tuned':>10s}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
for param in sorted(original_params.keys()):
    orig = original_params[param]
    tuned = best_params.get(param, "N/A")
    print(f"  {param:20s}  {str(orig):>10}  {str(tuned):>10}")
for param in sorted(best_params.keys()):
    if param not in original_params:
        print(f"  {param:20s}  {'N/A':>10}  {str(best_params[param]):>10}")
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
print("--- Model Comparison ---")
results = {
    "Gender-only":           0.7870,
    "Logistic Regression":   0.8316,
    "Random Forest (orig)":  0.8316,
    "Neural Network":        0.8204,
    "LightGBM":              0.8384,
    "XGBoost (original)":    0.8451,
    "XGBoost (tuned)":       0.8552,
    "Random Forest (tuned)": fresh_scores.mean(),
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
if train_accuracy - fresh_scores.mean() > 0.05:
    print(f"  Warning: Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Generate submission ----
test_pred = best_model.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/random_forest_tuned.csv", index=False)
print(f"Submission saved: submissions/random_forest_tuned.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save results ----
importance_df.to_csv(f"{RESULTS_DIR}/07_random_forest_tuned_importance.csv", index=False)
print(f"\nResults saved to: results/models/07_random_forest_tuned.txt")
print(f"Importance saved to: results/models/07_random_forest_tuned_importance.csv")

tee.close()
