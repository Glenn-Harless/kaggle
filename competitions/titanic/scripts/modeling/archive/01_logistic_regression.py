"""
Titanic Model 1: Logistic Regression (Baseline)

Uses cross-validation to evaluate performance.
Requires scaling since logistic regression is sensitive to feature magnitude.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


tee = Tee(f"{RESULTS_DIR}/01_logistic_regression.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Cross-validation ----
# StratifiedKFold preserves the survival ratio in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline: scale first, then logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores.mean():.4f}")
print(f"  Std:   {scores.std():.4f}")
print()

# ---- Compare to baselines ----
print("--- Baseline Comparison ---")
gender_baseline = 0.787
print(f"  Gender-only model:    {gender_baseline:.4f}")
print(f"  Logistic Regression:  {scores.mean():.4f}")
print(f"  Improvement:          {scores.mean() - gender_baseline:+.4f}")
print()

# ---- Train on full data and inspect coefficients ----
pipeline.fit(X, y)
model = pipeline.named_steps["model"]
scaler = pipeline.named_steps["scaler"]

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

# ---- Confusion matrix on full training set (for inspection) ----
y_pred = pipeline.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Generate submission ----
test_pred = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logistic_regression.csv", index=False)
print(f"Submission saved: submissions/logistic_regression.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save coefficients as CSV ----
coef_df.to_csv(f"{RESULTS_DIR}/01_logistic_regression_coefficients.csv", index=False)
print(f"\nResults saved to: results/models/01_logistic_regression.txt")
print(f"Coefficients saved to: results/models/01_logistic_regression_coefficients.csv")

tee.close()
