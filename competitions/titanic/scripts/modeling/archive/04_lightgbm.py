"""
Titanic Model 4: LightGBM

Microsoft's gradient boosting framework. Similar to XGBoost but uses
leaf-wise tree growth (vs level-wise), which can be more efficient.
Often faster and competitive on accuracy.
"""

import pandas as pd
import numpy as np
import sys
from lightgbm import LGBMClassifier
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


tee = Tee(f"{RESULTS_DIR}/04_lightgbm.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 4: LIGHTGBM")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,            # leaf-wise growth — controls complexity
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1              # suppress training logs
)

scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores.mean():.4f}")
print(f"  Std:   {scores.std():.4f}")
print()

# ---- Compare to all models ----
print("--- Model Comparison ---")
results = {
    "Gender-only":          0.7870,
    "Logistic Regression":  0.8316,
    "Random Forest":        0.8316,
    "XGBoost":              0.8451,
    "LightGBM":             scores.mean(),
}
for name, score in results.items():
    marker = " <-- best" if score == max(results.values()) else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Train on full data and inspect feature importance ----
model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

# Normalize to percentages for comparability
total = importance_df["importance"].sum()
importance_df["importance_pct"] = importance_df["importance"] / total

print("--- Feature Importance (Split count, normalized) ---")
print()
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance_pct"] * 100)
    print(f"  {row['feature']:20s}  {row['importance_pct']:.4f}  {bar}")
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

# ---- Overfitting check ----
train_accuracy = model.score(X, y)
print(f"--- Overfitting Check ---")
print(f"  Training accuracy:  {train_accuracy:.4f}")
print(f"  CV accuracy:        {scores.mean():.4f}")
print(f"  Gap:                {train_accuracy - scores.mean():.4f}")
if train_accuracy - scores.mean() > 0.05:
    print(f"  Warning: Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Prediction confidence distribution ----
y_proba = model.predict_proba(X)[:, 1]
print("--- Prediction Confidence ---")
print(f"  Predictions > 0.9 (very confident survive): {(y_proba > 0.9).sum()}")
print(f"  Predictions < 0.1 (very confident die):     {(y_proba < 0.1).sum()}")
print(f"  Predictions 0.4-0.6 (uncertain):            {((y_proba >= 0.4) & (y_proba <= 0.6)).sum()}")
print()

# ---- Generate submission ----
test_pred = model.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/lightgbm.csv", index=False)
print(f"Submission saved: submissions/lightgbm.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save importance as CSV ----
importance_df.to_csv(f"{RESULTS_DIR}/04_lightgbm_importance.csv", index=False)
print(f"\nResults saved to: results/models/04_lightgbm.txt")
print(f"Importance saved to: results/models/04_lightgbm_importance.csv")

tee.close()
