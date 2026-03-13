"""
Titanic Model 2: Random Forest

An ensemble of decision trees. Each tree sees a random subset of data and features,
then they vote on the prediction. Handles non-linear relationships naturally.
No scaling needed — trees split on thresholds.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
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


tee = Tee(f"{RESULTS_DIR}/02_random_forest.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=500,      # number of trees
    max_depth=6,           # limit depth to prevent overfitting on small data
    min_samples_leaf=5,    # each leaf needs at least 5 samples
    max_features="sqrt",   # each tree considers sqrt(n_features) at each split
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores.mean():.4f}")
print(f"  Std:   {scores.std():.4f}")
print()

# ---- Compare to baselines ----
print("--- Baseline Comparison ---")
gender_baseline = 0.787
logreg_baseline = 0.8316
print(f"  Gender-only model:    {gender_baseline:.4f}")
print(f"  Logistic Regression:  {logreg_baseline:.4f}")
print(f"  Random Forest:        {scores.mean():.4f}")
print(f"  vs Gender:            {scores.mean() - gender_baseline:+.4f}")
print(f"  vs LogReg:            {scores.mean() - logreg_baseline:+.4f}")
print()

# ---- Train on full data and inspect feature importance ----
model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("--- Feature Importance (Mean Decrease in Impurity) ---")
print()
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:20s}  {row['importance']:.4f}  {bar}")
print()

# ---- Confusion matrix on full training set ----
y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Check for overfitting ----
train_accuracy = model.score(X, y)
print(f"--- Overfitting Check ---")
print(f"  Training accuracy:  {train_accuracy:.4f}")
print(f"  CV accuracy:        {scores.mean():.4f}")
print(f"  Gap:                {train_accuracy - scores.mean():.4f}")
if train_accuracy - scores.mean() > 0.05:
    print(f"  ⚠ Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Generate submission ----
test_pred = model.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/random_forest.csv", index=False)
print(f"Submission saved: submissions/random_forest.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save importance as CSV ----
importance_df.to_csv(f"{RESULTS_DIR}/02_random_forest_importance.csv", index=False)
print(f"\nResults saved to: results/models/02_random_forest.txt")
print(f"Importance saved to: results/models/02_random_forest_importance.csv")

tee.close()
