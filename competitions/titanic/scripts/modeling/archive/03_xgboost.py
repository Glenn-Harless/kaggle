"""
Titanic Model 3: XGBoost

Gradient boosted trees — builds trees sequentially, with each new tree
correcting the mistakes of the previous ones. The dominant model type
for tabular Kaggle competitions.
"""

import pandas as pd
import numpy as np
import sys
from xgboost import XGBClassifier
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


tee = Tee(f"{RESULTS_DIR}/03_xgboost.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,              # shallow trees — boosting works best with weak learners
    learning_rate=0.05,       # slow learning rate for better generalization
    subsample=0.8,            # use 80% of data per tree (reduces overfitting)
    colsample_bytree=0.8,    # use 80% of features per tree
    min_child_weight=3,       # minimum samples in a leaf
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    random_state=42,
    eval_metric="logloss",
    verbosity=0
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
gender_baseline = 0.787
logreg = 0.8316
rf = 0.8316
print(f"  Gender-only model:    {gender_baseline:.4f}")
print(f"  Logistic Regression:  {logreg:.4f}")
print(f"  Random Forest:        {rf:.4f}")
print(f"  XGBoost:              {scores.mean():.4f}")
print(f"  vs Gender:            {scores.mean() - gender_baseline:+.4f}")
print(f"  vs LogReg:            {scores.mean() - logreg:+.4f}")
print(f"  vs Random Forest:     {scores.mean() - rf:+.4f}")
print()

# ---- Train on full data and inspect feature importance ----
model.fit(X, y)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("--- Feature Importance (Gain) ---")
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
submission.to_csv(f"{DATA_DIR}/../submissions/xgboost.csv", index=False)
print(f"Submission saved: submissions/xgboost.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save importance as CSV ----
importance_df.to_csv(f"{RESULTS_DIR}/03_xgboost_importance.csv", index=False)
print(f"\nResults saved to: results/models/03_xgboost.txt")
print(f"Importance saved to: results/models/03_xgboost_importance.csv")

tee.close()
