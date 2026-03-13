"""
Titanic Model 11a: Logistic Regression v3 — Stage A (Interactions)

Adds Sex x Pclass interaction features on top of the v2 processed data.
Pclass is one-hot encoded here (not in build_features.py) so that
interaction terms are clean binary indicators.

Interaction features:
  - Pclass_2, Pclass_3 (Pclass_1 is reference)
  - Sex_x_Pclass_2 = Sex * Pclass_2
  - Sex_x_Pclass_3 = Sex * Pclass_3
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


tee = Tee(f"{RESULTS_DIR}/11a_logreg_v3_interactions.txt")
sys.stdout = tee

# ---- Load processed data (v2 pipeline — unchanged) ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]


def add_interaction_features(df):
    """Add Pclass one-hot and Sex x Pclass interactions.

    Replaces ordinal Pclass with dummies (Pclass_1 dropped as reference).
    """
    df = df.copy()
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
    df["Sex_x_Pclass_2"] = df["Sex"] * df["Pclass_2"]
    df["Sex_x_Pclass_3"] = df["Sex"] * df["Pclass_3"]
    df = df.drop(columns=["Pclass"])
    return df


X = add_interaction_features(X)
test = add_interaction_features(test)

print("=" * 60)
print("MODEL 11a: LOGISTIC REGRESSION v3 — INTERACTIONS")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print(f"\nFeature list: {list(X.columns)}")
print()

# ---- Cross-validation setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Regularization sweep ----
print("--- Regularization Sweep ---")
print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}  {'Gap':>8}")
print("-" * 40)

best_C = None
best_score = 0
all_results = []
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
    all_results.append((C, scores.mean(), scores.std(), gap))
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
    print(f"  {direction} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Interaction coefficient interpretation ----
print("--- Interaction Interpretation ---")
coefs = dict(zip(X.columns, model.coef_[0]))
intercept = model.intercept_[0]

# Show effective Sex coefficient for each class (on scaled features)
print("Effective Sex penalty by class (scaled coefficients):")
sex_base = coefs.get("Sex", 0)
sex_p2 = sex_base + coefs.get("Sex_x_Pclass_2", 0)
sex_p3 = sex_base + coefs.get("Sex_x_Pclass_3", 0)
print(f"  1st class (reference): Sex coef = {sex_base:+.4f}")
print(f"  2nd class:             Sex coef = {sex_p2:+.4f}  (Sex + Sex_x_Pclass_2)")
print(f"  3rd class:             Sex coef = {sex_p3:+.4f}  (Sex + Sex_x_Pclass_3)")
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

# ---- Subgroup analysis on training data ----
print("--- Subgroup Behavior (Training Data) ---")
raw_train = pd.read_csv(f"{DATA_DIR}/train.csv")
train_with_pred = pd.DataFrame({
    "Sex": raw_train["Sex"],
    "Pclass": raw_train["Pclass"],
    "Survived": y.values,
    "Predicted": y_pred,
})

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (train_with_pred["Sex"] == sex) & (train_with_pred["Pclass"] == pclass)
        sub = train_with_pred[mask]
        n = len(sub)
        actual_rate = sub["Survived"].mean()
        pred_rate = sub["Predicted"].mean()
        correct = (sub["Survived"] == sub["Predicted"]).mean()
        print(f"  {sex:6s} Pclass {pclass}: n={n:3d}  actual={actual_rate:.3f}  pred={pred_rate:.3f}  acc={correct:.3f}")
print()

# ---- Model comparison ----
print("--- v2 vs v3a Comparison ---")
results = {
    "Gender-only":              0.7870,
    "v1 LogReg (20 feat)":      0.8339,
    "v2 LogReg (15 feat)":      0.8305,
    "v3a Interactions":         scores.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    marker = " <-- current" if "v3a" in name else ""
    kaggle = ""
    if "v2" in name:
        kaggle = " (Kaggle: 0.7727)"
    print(f"  {name:25s}  {score:.4f}{kaggle}{marker}")
print()

# ---- Generate submission ----
test_pred = pipe.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logreg_v3a_interactions.csv", index=False)
print(f"Submission saved: submissions/logreg_v3a_interactions.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Test prediction profile ----
print()
print("--- Test Prediction Profile ---")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")
test_profile = pd.DataFrame({
    "Sex": raw_test["Sex"],
    "Pclass": raw_test["Pclass"],
    "Predicted": test_pred,
})

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (test_profile["Sex"] == sex) & (test_profile["Pclass"] == pclass)
        sub = test_profile[mask]
        n = len(sub)
        pred_surv = sub["Predicted"].sum()
        pred_rate = sub["Predicted"].mean()
        print(f"  {sex:6s} Pclass {pclass}: n={n:3d}  predicted_survived={pred_surv:3.0f}  rate={pred_rate:.3f}")

total_female_dead = ((test_profile["Sex"] == "female") & (test_profile["Predicted"] == 0)).sum()
print(f"\n  Total female predicted dead: {total_female_dead}")
print()

# ---- Compare vs v2 predictions ----
v2_sub = pd.read_csv(f"{DATA_DIR}/../submissions/logreg_v2.csv")
diff = (submission["Survived"] != v2_sub["Survived"])
n_diff = diff.sum()
print(f"--- Diff vs v2 ---")
print(f"  Passengers changed: {n_diff} / {len(submission)}")
if n_diff > 0:
    changed_ids = submission.loc[diff, "PassengerId"].values
    changed_v2 = v2_sub.loc[diff, "Survived"].values
    changed_v3 = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v2, changed_v3):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        print(f"    PassengerId {pid}: {old} -> {new}  ({sex}, Pclass {pclass})")
print()

# ---- Save coefficients ----
coef_df.to_csv(f"{RESULTS_DIR}/11a_logreg_v3_interactions_coefficients.csv", index=False)
print(f"Results saved to: results/models/11a_logreg_v3_interactions.txt")
print(f"Coefficients saved to: results/models/11a_logreg_v3_interactions_coefficients.csv")

tee.close()
