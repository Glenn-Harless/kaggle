"""
Titanic Experiment 12a: Segmented Logistic Regression by Sex

The v3 global LogReg (83.05% CV, 77.27% Kaggle) uses one decision boundary
across all subgroups. Biggest misfits:
  - Female 3rd class: actual 50%, predicted 81.2%
  - Male 1st class: actual 36.9%, predicted 13.1%

This experiment trains separate LogReg per sex segment, allowing each to
have its own coefficients and decision boundary.

Female model (17 features):
  DROP: Sex (=0), Title_Mr (=0), Sex_x_Pclass_2 (=0), Sex_x_Pclass_3 (=0)

Male model (16 features):
  DROP: Sex (=1), Title_Miss (=0), Title_Mrs (=0),
        Sex_x_Pclass_2 (redundant w/ Pclass_2), Sex_x_Pclass_3 (redundant w/ Pclass_3)
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
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


tee = Tee(f"{RESULTS_DIR}/12a_segment_by_sex.txt")
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

sex_train = raw_train["Sex"].values
sex_test = raw_test["Sex"].values

female_train = sex_train == "female"
male_train = sex_train == "male"
female_test = sex_test == "female"
male_test = sex_test == "male"

# ---- Segment-specific feature sets ----
female_drop = ["Sex", "Title_Mr", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]
male_drop = ["Sex", "Title_Miss", "Title_Mrs", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]

X_female = X.drop(columns=female_drop)
X_male = X.drop(columns=male_drop)
test_female = test.drop(columns=female_drop)
test_male = test.drop(columns=male_drop)

print("=" * 60)
print("EXPERIMENT 12a: SEGMENTED LOGREG BY SEX")
print("=" * 60)
print(f"\nFemale: {female_train.sum()} train, {female_test.sum()} test")
print(f"Male:   {male_train.sum()} train, {male_test.sum()} test")
print(f"\nFemale features ({X_female.shape[1]}): {list(X_female.columns)}")
print(f"Male features ({X_male.shape[1]}): {list(X_male.columns)}")
print(f"\nFemale dropped: {female_drop}")
print(f"Male dropped: {male_drop}")
print()

# ---- CV setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X, y))
C_values = [0.01, 0.05, 0.1, 0.5, 1.0]


def make_pipe(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
    ])


# ---- Per-segment C sweep (using overall CV folds) ----
def segment_c_sweep(name, X_seg, seg_mask):
    """Sweep C values for a segment, evaluated within the overall CV folds."""
    print(f"--- {name} Model: C Sweep ---")
    print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}")
    print("-" * 30)

    best_C = None
    best_score = 0

    for C in C_values:
        fold_accs = []
        for train_idx, val_idx in splits:
            s_train = train_idx[seg_mask[train_idx]]
            s_val = val_idx[seg_mask[val_idx]]

            pipe = make_pipe(C)
            pipe.fit(X_seg.iloc[s_train], y.iloc[s_train])
            pred = pipe.predict(X_seg.iloc[s_val])
            fold_accs.append(accuracy_score(y.iloc[s_val], pred))

        scores = np.array(fold_accs)
        marker = ""
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
            marker = " <-- best"
        print(f"{C:>8.2f}  {scores.mean():>8.4f}  {scores.std():>8.4f}{marker}")

    print(f"\nSelected C={best_C} (CV={best_score:.4f})")
    print()
    return best_C


best_C_f = segment_c_sweep("Female", X_female, female_train)
best_C_m = segment_c_sweep("Male", X_male, male_train)

# ---- Combined segmented CV ----
print("--- Combined Segmented CV (5-Fold) ---")
print(f"  Female C={best_C_f}, Male C={best_C_m}")
print()

fold_accs = []
oof_preds = np.empty(len(X), dtype=int)

for fold_idx, (train_idx, val_idx) in enumerate(splits):
    # Female segment
    f_tr = train_idx[female_train[train_idx]]
    f_va = val_idx[female_train[val_idx]]
    pipe_f = make_pipe(best_C_f)
    pipe_f.fit(X_female.iloc[f_tr], y.iloc[f_tr])
    y_f_pred = pipe_f.predict(X_female.iloc[f_va])
    f_acc = accuracy_score(y.iloc[f_va], y_f_pred)

    # Male segment
    m_tr = train_idx[male_train[train_idx]]
    m_va = val_idx[male_train[val_idx]]
    pipe_m = make_pipe(best_C_m)
    pipe_m.fit(X_male.iloc[m_tr], y.iloc[m_tr])
    y_m_pred = pipe_m.predict(X_male.iloc[m_va])
    m_acc = accuracy_score(y.iloc[m_va], y_m_pred)

    # Combine predictions in val_idx order
    val_pos = {idx: pos for pos, idx in enumerate(val_idx)}
    y_val_pred = np.empty(len(val_idx), dtype=int)
    for i, idx in enumerate(f_va):
        y_val_pred[val_pos[idx]] = y_f_pred[i]
    for i, idx in enumerate(m_va):
        y_val_pred[val_pos[idx]] = y_m_pred[i]

    # Store OOF predictions
    for i, idx in enumerate(f_va):
        oof_preds[idx] = y_f_pred[i]
    for i, idx in enumerate(m_va):
        oof_preds[idx] = y_m_pred[i]

    fold_acc = accuracy_score(y.iloc[val_idx], y_val_pred)
    fold_accs.append(fold_acc)
    print(f"  Fold {fold_idx+1}: {fold_acc:.4f}  (female={f_acc:.4f}, male={m_acc:.4f})")

overall_cv = np.mean(fold_accs)
overall_std = np.std(fold_accs)
print(f"\n  Mean:  {overall_cv:.4f}")
print(f"  Std:   {overall_std:.4f}")
print(f"  vs v3: {overall_cv:.4f} vs 0.8305 ({overall_cv - 0.8305:+.4f})")
print()

# ---- Train final models on full data ----
pipe_f_final = make_pipe(best_C_f)
pipe_f_final.fit(X_female[female_train], y[female_train])

pipe_m_final = make_pipe(best_C_m)
pipe_m_final.fit(X_male[male_train], y[male_train])

# In-sample predictions
y_pred = np.empty(len(X), dtype=int)
y_pred[female_train] = pipe_f_final.predict(X_female[female_train])
y_pred[male_train] = pipe_m_final.predict(X_male[male_train])

train_acc = accuracy_score(y, y_pred)

# ---- Coefficients per segment ----
print("--- Female Model Coefficients ---")
print("Positive = increases survival, Negative = decreases survival\n")
model_f = pipe_f_final.named_steps["model"]
coef_f = pd.DataFrame({
    "feature": X_female.columns,
    "coefficient": model_f.coef_[0],
    "abs_coefficient": np.abs(model_f.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

for _, row in coef_f.iterrows():
    d = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {d} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")
print()

print("--- Male Model Coefficients ---")
print("Positive = increases survival, Negative = decreases survival\n")
model_m = pipe_m_final.named_steps["model"]
coef_m = pd.DataFrame({
    "feature": X_male.columns,
    "coefficient": model_m.coef_[0],
    "abs_coefficient": np.abs(model_m.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

for _, row in coef_m.iterrows():
    d = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {d} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Intercept comparison ----
print("--- Decision Boundaries (Intercepts) ---")
print(f"  Female intercept: {model_f.intercept_[0]:+.4f}")
print(f"  Male intercept:   {model_m.intercept_[0]:+.4f}")
print(f"  (Positive = biased toward survival, Negative = biased toward death)")
print()

# ---- Confusion matrix per segment ----
print("--- Confusion Matrix: Female (training data) ---")
cm_f = confusion_matrix(y[female_train], y_pred[female_train])
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm_f[0][0]:4d}    {cm_f[0][1]:4d}")
print(f"  Actual Surv:   {cm_f[1][0]:4d}    {cm_f[1][1]:4d}")
print()

print("--- Confusion Matrix: Male (training data) ---")
cm_m = confusion_matrix(y[male_train], y_pred[male_train])
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm_m[0][0]:4d}    {cm_m[0][1]:4d}")
print(f"  Actual Surv:   {cm_m[1][0]:4d}    {cm_m[1][1]:4d}")
print()

# ---- Overfitting check ----
print("--- Overfitting Check ---")
print(f"  Training accuracy:  {train_acc:.4f}")
print(f"  CV accuracy:        {overall_cv:.4f}")
print(f"  Gap:                {train_acc - overall_cv:.4f}")
if train_acc - overall_cv > 0.05:
    print("  Warning: Gap > 5% — potential overfitting")
else:
    print("  Gap < 5% — looks healthy")
print()

# ---- Subgroup behavior (in-sample) ----
print("--- Subgroup Behavior (Training Data, In-Sample) ---")
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

# ---- Subgroup behavior (OOF — more honest) ----
print("--- Subgroup Behavior (OOF Predictions) ---")
oof_with_pred = pd.DataFrame({
    "Sex": raw_train["Sex"],
    "Pclass": raw_train["Pclass"],
    "Survived": y.values,
    "Predicted": oof_preds,
})

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (oof_with_pred["Sex"] == sex) & (oof_with_pred["Pclass"] == pclass)
        sub = oof_with_pred[mask]
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
    "v3 LogReg (21 feat)":      (0.8305, "0.7727"),
    "12a Segmented by Sex":     (overall_cv, "TBD"),
}
print(f"  {'Model':30s}  {'CV':>8}  {'Kaggle':>8}")
print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for name, (cv_score, kaggle) in sorted_results:
    marker = " <--" if "12a" in name else ""
    print(f"  {name:30s}  {cv_score:.4f}  {kaggle:>8}{marker}")
print()

# ---- Generate submission ----
test_pred = np.empty(len(test), dtype=int)
test_pred[female_test] = pipe_f_final.predict(test_female[female_test])
test_pred[male_test] = pipe_m_final.predict(test_male[male_test])

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{SUBMISSIONS_DIR}/logreg_12a_segment_sex.csv", index=False)
print(f"Submission saved: submissions/logreg_12a_segment_sex.csv")
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

# ---- Passenger-level diff vs v3 ----
v3_sub = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_v3.csv")
diff = (submission["Survived"] != v3_sub["Survived"])
n_diff = diff.sum()
print(f"--- Passenger-Level Diff vs v3 ---")
print(f"  Passengers changed: {n_diff} / {len(submission)}")
if n_diff > 0:
    v3_to_survived = ((v3_sub.loc[diff, "Survived"] == 0) & (submission.loc[diff, "Survived"] == 1)).sum()
    v3_to_died = ((v3_sub.loc[diff, "Survived"] == 1) & (submission.loc[diff, "Survived"] == 0)).sum()
    print(f"  Flipped 0->1 (now survived): {v3_to_survived}")
    print(f"  Flipped 1->0 (now died):     {v3_to_died}")
    print()
    changed_ids = submission.loc[diff, "PassengerId"].values
    changed_v3 = v3_sub.loc[diff, "Survived"].values
    changed_12a = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v3, changed_12a):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
        cabin_str = cabin if pd.notna(cabin) else "unknown"
        print(f"    PID {pid}: {old}->{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
print()

# ---- Save coefficients ----
coef_f.to_csv(f"{RESULTS_DIR}/12a_female_coefficients.csv", index=False)
coef_m.to_csv(f"{RESULTS_DIR}/12a_male_coefficients.csv", index=False)
print(f"Results saved to: results/models/12a_segment_by_sex.txt")
print(f"Female coefficients: results/models/12a_female_coefficients.csv")
print(f"Male coefficients: results/models/12a_male_coefficients.csv")
print(f"Submission: submissions/logreg_12a_segment_sex.csv")

tee.close()
