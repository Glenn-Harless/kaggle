"""
Titanic Model 8: Logistic Regression (Hyperparameter Tuned)

Uses RandomizedSearchCV to explore regularization strength, penalty type,
and solver combinations. Requires scaling.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform, loguniform

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


tee = Tee(f"{RESULTS_DIR}/08_logistic_regression_tuned.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 8: LOGISTIC REGRESSION (HYPERPARAMETER TUNED)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Define hyperparameter search space ----
# Logistic regression's main knob is C (inverse regularization strength)
# Lower C = stronger regularization
param_distributions = {
    "model__C":           loguniform(0.001, 100),    # wide range on log scale
    "model__penalty":     ["l1", "l2", "elasticnet"],
    "model__solver":      ["saga"],                   # saga supports all penalties
    "model__l1_ratio":    uniform(0, 1),              # only used with elasticnet
    "model__max_iter":    [2000],
}

print("--- Search Space ---")
print(f"  C (regularization):    [0.001, 100] (log-uniform)")
print(f"  penalty:               [l1, l2, elasticnet]")
print(f"  l1_ratio:              [0, 1] (for elasticnet)")
print(f"  solver:                saga")
print()

# ---- Run randomized search ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=200,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=0,
    error_score="raise"
)

print("Searching 200 random hyperparameter combinations...")
print()

# Need to handle elasticnet needing l1_ratio and l1/l2 not using it
# saga solver handles all penalty types
search.fit(X, y)

print("--- Search Results ---")
print(f"  Best CV accuracy: {search.best_score_:.4f}")
print()

# ---- Best hyperparameters ----
print("--- Best Hyperparameters ---")
best_params = search.best_params_
for param, value in sorted(best_params.items()):
    param_short = param.replace("model__", "")
    if isinstance(value, float):
        print(f"  {param_short:20s}  {value:.4f}")
    else:
        print(f"  {param_short:20s}  {value}")
print()

# ---- Compare: original vs tuned ----
print("--- Original vs Tuned Logistic Regression ---")
print(f"\n  {'Parameter':20s}  {'Original':>12s}  {'Tuned':>12s}")
print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
print(f"  {'C':20s}  {'1.0 (default)':>12s}  {best_params['model__C']:12.4f}")
print(f"  {'penalty':20s}  {'l2':>12s}  {best_params['model__penalty']:>12s}")
if best_params["model__penalty"] == "elasticnet":
    print(f"  {'l1_ratio':20s}  {'N/A':>12s}  {best_params['model__l1_ratio']:12.4f}")
print()

# ---- Validate best model with fresh CV ----
best_pipeline = search.best_estimator_
fresh_scores = cross_val_score(best_pipeline, X, y, cv=cv, scoring="accuracy")

print("--- Fresh 5-Fold CV (best model) ---")
for i, score in enumerate(fresh_scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {fresh_scores.mean():.4f}")
print(f"  Std:   {fresh_scores.std():.4f}")
print()

# ---- Full model comparison ----
print("--- Model Comparison ---")
results = {
    "Gender-only":            0.7870,
    "LogReg (original)":      0.8316,
    "Random Forest (orig)":   0.8316,
    "Neural Network":         0.8204,
    "LightGBM":               0.8384,
    "XGBoost (original)":     0.8451,
    "XGBoost (tuned)":        0.8552,
    "LogReg (tuned)":         fresh_scores.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    marker = " <-- best" if score == max(results.values()) else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Train on full data and inspect coefficients ----
best_pipeline.fit(X, y)
model = best_pipeline.named_steps["model"]

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0],
    "abs_coefficient": np.abs(model.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

print("--- Feature Coefficients (sorted by importance) ---")
print("Positive = increases survival, Negative = decreases survival\n")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    bar = "█" * int(row["abs_coefficient"] * 10)
    print(f"  {direction} {row['feature']:20s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Confusion matrix ----
y_pred = best_pipeline.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Overfitting check ----
train_accuracy = best_pipeline.score(X, y)
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
test_pred = best_pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logistic_regression_tuned.csv", index=False)
print(f"Submission saved: submissions/logistic_regression_tuned.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save results ----
coef_df.to_csv(f"{RESULTS_DIR}/08_logistic_regression_tuned_coefficients.csv", index=False)
print(f"\nResults saved to: results/models/08_logistic_regression_tuned.txt")
print(f"Coefficients saved to: results/models/08_logistic_regression_tuned_coefficients.csv")

tee.close()
