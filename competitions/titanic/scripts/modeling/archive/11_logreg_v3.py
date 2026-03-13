"""
Titanic Model 11: Logistic Regression v3 — Final Assembly

Combines features that proved useful in Stages A-C experimentation:
  - Stage A: Sex x Pclass interactions (Pclass one-hot, interaction terms)
  - Stage B: Grouped deck features (Deck_ABC, Deck_DE, Deck_FG)
  - Stage B: Raw Fare kept (log1p did not help)
  - Stage C: No surname features (all variants hurt CV)

Changes from v2:
  - Pclass: ordinal → one-hot (Pclass_2, Pclass_3; Pclass_1 is reference)
  - Added: Sex_x_Pclass_2, Sex_x_Pclass_3 interaction terms
  - Added: Deck_ABC, Deck_DE, Deck_FG (grouped cabin deck)
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
SUBMISSIONS_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/submissions"


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


tee = Tee(f"{RESULTS_DIR}/11_logreg_v3.txt")
sys.stdout = tee

# ---- Load data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")
raw_train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]


# ---- Feature engineering (modeling script only, build_features.py unchanged) ----

def add_v3_features(df, cabin_series):
    """Add v3 features: Pclass interactions + grouped deck.

    Applied in the modeling script — does NOT modify build_features.py.
    """
    df = df.copy()

    # Stage A: Pclass one-hot + Sex x Pclass interactions
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
    df["Sex_x_Pclass_2"] = df["Sex"] * df["Pclass_2"]
    df["Sex_x_Pclass_3"] = df["Sex"] * df["Pclass_3"]
    df = df.drop(columns=["Pclass"])

    # Stage B: Grouped deck from cabin letter
    deck = cabin_series.fillna("U").str[0].replace("T", "U")
    df["Deck_ABC"] = deck.isin(["A", "B", "C"]).astype(int)
    df["Deck_DE"] = deck.isin(["D", "E"]).astype(int)
    df["Deck_FG"] = deck.isin(["F", "G"]).astype(int)

    return df


X = add_v3_features(X, raw_train["Cabin"])
test = add_v3_features(test, raw_test["Cabin"])

print("=" * 60)
print("MODEL 11: LOGISTIC REGRESSION v3 (FINAL)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print(f"\nFeature list: {list(X.columns)}")
print()

print("--- v3 Changes from v2 ---")
print("  Added: Pclass_2, Pclass_3 (one-hot, replaces ordinal Pclass)")
print("  Added: Sex_x_Pclass_2, Sex_x_Pclass_3 (interaction terms)")
print("  Added: Deck_ABC, Deck_DE, Deck_FG (grouped cabin deck)")
print("  Dropped: Pclass (replaced by one-hot)")
print("  Rejected: log1p(Fare) — did not improve CV")
print("  Rejected: SurnameGroupSize, SurnameSurvHint — hurt CV")
print()

# ---- Cross-validation setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Regularization sweep ----
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
    print(f"  {direction} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Interaction interpretation ----
print("--- Interaction Interpretation ---")
coefs = dict(zip(X.columns, model.coef_[0]))
print("Effective Sex penalty by Pclass (scaled coefficients):")
sex_base = coefs.get("Sex", 0)
sex_p2 = sex_base + coefs.get("Sex_x_Pclass_2", 0)
sex_p3 = sex_base + coefs.get("Sex_x_Pclass_3", 0)
print(f"  1st class (reference): Sex = {sex_base:+.4f}")
print(f"  2nd class:             Sex = {sex_p2:+.4f}  (base + interaction)")
print(f"  3rd class:             Sex = {sex_p3:+.4f}  (base + interaction)")
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
print("--- Full Model Comparison ---")
results = {
    "Gender-only":              (0.7870, "0.7655"),
    "v1 LogReg (20 feat)":      (0.8339, "0.7727"),
    "v1 XGBoost (tuned)":       (0.8552, "0.7584"),
    "v1 Ensemble (hard)":       (0.8563, "0.7560"),
    "v2 LogReg (15 feat)":      (0.8305, "0.7727"),
    "v3 LogReg (21 feat)":      (scores.mean(), "TBD"),
}
print(f"  {'Model':30s}  {'CV':>8}  {'Kaggle':>8}")
print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for name, (cv_score, kaggle) in sorted_results:
    marker = " <--" if "v3" in name else ""
    print(f"  {name:30s}  {cv_score:.4f}  {kaggle:>8}{marker}")
print()

# ---- Generate submission ----
test_pred = pipe.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{SUBMISSIONS_DIR}/logreg_v3.csv", index=False)
print(f"Submission saved: submissions/logreg_v3.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")
print()

# ---- Test prediction profile ----
print("--- Test Prediction Profile ---")
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

# ---- Passenger-level diff vs v2 ----
v2_sub = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_v2.csv")
diff = (submission["Survived"] != v2_sub["Survived"])
n_diff = diff.sum()
print(f"--- Passenger-Level Diff vs v2 ---")
print(f"  Passengers changed: {n_diff} / {len(submission)}")
if n_diff > 0:
    # Count direction of changes
    v2_to_survived = ((v2_sub.loc[diff, "Survived"] == 0) & (submission.loc[diff, "Survived"] == 1)).sum()
    v2_to_died = ((v2_sub.loc[diff, "Survived"] == 1) & (submission.loc[diff, "Survived"] == 0)).sum()
    print(f"  Flipped 0→1 (now survived): {v2_to_survived}")
    print(f"  Flipped 1→0 (now died):     {v2_to_died}")
    print()
    changed_ids = submission.loc[diff, "PassengerId"].values
    changed_v2 = v2_sub.loc[diff, "Survived"].values
    changed_v3 = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v2, changed_v3):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
        cabin_str = cabin if pd.notna(cabin) else "unknown"
        print(f"    PID {pid}: {old}→{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
print()

# ---- Save coefficients ----
coef_df.to_csv(f"{RESULTS_DIR}/11_logreg_v3_coefficients.csv", index=False)
print(f"Results saved to: results/models/11_logreg_v3.txt")
print(f"Coefficients saved to: results/models/11_logreg_v3_coefficients.csv")
print(f"Submission saved to: submissions/logreg_v3.csv")

tee.close()
