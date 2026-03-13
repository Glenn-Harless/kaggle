"""
Titanic Model 10: Logistic Regression on v2 Feature Pipeline

Uses the cleaned v2 feature set (15 features, no leakage, no redundancy,
EDA-aligned buckets). Tests multiple regularization strengths.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

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


tee = Tee(f"{RESULTS_DIR}/10_logreg_v2.txt")
sys.stdout = tee

# ---- Load processed data (v2 pipeline) ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 10: LOGISTIC REGRESSION (v2 FEATURES)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print(f"\nFeature list: {list(X.columns)}")
print()

# ---- Cross-validation setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Test multiple regularization strengths ----
print("--- Regularization Sweep ---")
print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}  {'Gap':>8}")
print("-" * 40)

best_C = None
best_score = 0
for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    pipe.fit(X, y)
    train_acc = pipe.score(X, y)
    gap = train_acc - scores.mean()
    marker = ""
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_C = C
        marker = " <-- best"
    print(f"{C:>8.2f}  {scores.mean():>8.4f}  {scores.std():>8.4f}  {gap:>+8.4f}{marker}")

print(f"\nSelected C={best_C}")
print()

# ---- Train best model ----
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=best_C, max_iter=2000, random_state=42))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
pipe.fit(X, y)
train_acc = pipe.score(X, y)

print("--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores.mean():.4f}")
print(f"  Std:   {scores.std():.4f}")
print()

# ---- Coefficients ----
model = pipe.named_steps["model"]
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0],
    "abs_coefficient": np.abs(model.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

print("--- Feature Coefficients (sorted by importance) ---")
print("Positive = increases survival, Negative = decreases survival\n")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {direction} {row['feature']:>15s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Confusion matrix ----
y_pred = pipe.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Overfitting check ----
print("--- Overfitting Check ---")
print(f"  Training accuracy:  {train_acc:.4f}")
print(f"  CV accuracy:        {scores.mean():.4f}")
print(f"  Gap:                {train_acc - scores.mean():.4f}")
if train_acc - scores.mean() > 0.05:
    print(f"  Warning: Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Model comparison (v1 vs v2) ----
print("--- v1 vs v2 Comparison ---")
results = {
    "Gender-only":              0.7870,
    "v1 LogReg (20 feat)":      0.8339,
    "v1 XGBoost (tuned)":       0.8552,
    "v1 Ensemble (hard)":       0.8563,
    "v2 LogReg (15 feat)":      scores.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    marker = " <-- v2" if "v2" in name else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Generate submission ----
test_pred = pipe.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logreg_v2.csv", index=False)
print(f"Submission saved: submissions/logreg_v2.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Save coefficients ----
coef_df.to_csv(f"{RESULTS_DIR}/10_logreg_v2_coefficients.csv", index=False)
print(f"\nResults saved to: results/models/10_logreg_v2.txt")
print(f"Coefficients saved to: results/models/10_logreg_v2_coefficients.csv")

tee.close()
