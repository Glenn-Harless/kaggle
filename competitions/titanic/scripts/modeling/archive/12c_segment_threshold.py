"""
Titanic Experiment 12c: Segmented LogReg + Per-Sex Threshold Tuning

Takes 12a's segmented architecture (Female C=0.5, Male C=0.05) and sweeps
decision thresholds per sex on OOF probabilities.

Goal: keep most of the female-3rd correction while avoiding an overly low
overall survival rate (12a test rate=0.335, v3 was 0.373).

This is a calibration experiment, not feature discovery.
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


tee = Tee(f"{RESULTS_DIR}/12c_segment_threshold.txt")
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

# ---- Segment-specific feature sets (same as 12a) ----
female_drop = ["Sex", "Title_Mr", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]
male_drop = ["Sex", "Title_Miss", "Title_Mrs", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]

X_female = X.drop(columns=female_drop)
X_male = X.drop(columns=male_drop)
test_female = test.drop(columns=female_drop)
test_male = test.drop(columns=male_drop)

# ---- CV setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X, y))

# C values from 12a (deterministic — same data, same splits, same seed)
BEST_C_F = 0.5
BEST_C_M = 0.05


def make_pipe(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
    ])


print("=" * 60)
print("EXPERIMENT 12c: SEGMENTED LOGREG + THRESHOLD TUNING")
print("=" * 60)
print(f"\nUsing 12a C values: Female C={BEST_C_F}, Male C={BEST_C_M}")
print(f"Female: {female_train.sum()} train, {female_test.sum()} test")
print(f"Male:   {male_train.sum()} train, {male_test.sum()} test")
print()

# ---- Collect OOF probabilities ----
print("--- Collecting OOF Probabilities ---")
oof_probs = np.full(len(X), np.nan)

for fold_idx, (train_idx, val_idx) in enumerate(splits):
    # Female
    f_tr = train_idx[female_train[train_idx]]
    f_va = val_idx[female_train[val_idx]]
    pipe_f = make_pipe(BEST_C_F)
    pipe_f.fit(X_female.iloc[f_tr], y.iloc[f_tr])
    oof_probs[f_va] = pipe_f.predict_proba(X_female.iloc[f_va])[:, 1]

    # Male
    m_tr = train_idx[male_train[train_idx]]
    m_va = val_idx[male_train[val_idx]]
    pipe_m = make_pipe(BEST_C_M)
    pipe_m.fit(X_male.iloc[m_tr], y.iloc[m_tr])
    oof_probs[m_va] = pipe_m.predict_proba(X_male.iloc[m_va])[:, 1]

assert not np.isnan(oof_probs).any(), "Missing OOF predictions"

# Baseline: 0.50/0.50 (12a result)
baseline_preds = (oof_probs >= 0.50).astype(int)
baseline_acc = accuracy_score(y, baseline_preds)
print(f"  Baseline (0.50/0.50): {baseline_acc:.4f}")
print()

# ---- OOF probability distribution by subgroup ----
print("--- OOF Probability Distribution ---")
for sex, mask in [("female", female_train), ("male", male_train)]:
    for pclass in [1, 2, 3]:
        pc_mask = mask & (raw_train["Pclass"].values == pclass)
        probs = oof_probs[pc_mask]
        survived = y.values[pc_mask]
        print(f"  {sex:6s} Pclass {pclass}: n={pc_mask.sum():3d}  "
              f"prob_mean={probs.mean():.3f}  "
              f"prob_median={np.median(probs):.3f}  "
              f"[{probs.min():.3f} - {probs.max():.3f}]  "
              f"actual={survived.mean():.3f}")
print()

# ---- Threshold sweep ----
female_thresholds = [0.45, 0.50, 0.55]
male_thresholds = [0.45, 0.50]

print("--- Threshold Sweep (OOF) ---")
print(f"  {'F_thresh':>8}  {'M_thresh':>8}  {'CV Acc':>8}  {'Surv Rate':>10}  "
      f"{'F3_acc':>7}  {'M1_acc':>7}  {'F3_pred':>8}  {'M1_pred':>8}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")

best_combo = None
best_acc = 0
sweep_results = []

for t_f in female_thresholds:
    for t_m in male_thresholds:
        preds = np.empty(len(X), dtype=int)
        preds[female_train] = (oof_probs[female_train] >= t_f).astype(int)
        preds[male_train] = (oof_probs[male_train] >= t_m).astype(int)

        acc = accuracy_score(y, preds)
        surv_rate = preds.mean()

        # Subgroup metrics
        f3_mask = female_train & (raw_train["Pclass"].values == 3)
        f3_acc = accuracy_score(y[f3_mask], preds[f3_mask])
        f3_pred = preds[f3_mask].mean()

        m1_mask = male_train & (raw_train["Pclass"].values == 1)
        m1_acc = accuracy_score(y[m1_mask], preds[m1_mask])
        m1_pred = preds[m1_mask].mean()

        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_combo = (t_f, t_m)
            marker = " <-- best"

        sweep_results.append({
            "t_f": t_f, "t_m": t_m, "acc": acc, "surv_rate": surv_rate,
            "f3_acc": f3_acc, "m1_acc": m1_acc, "f3_pred": f3_pred, "m1_pred": m1_pred,
        })
        print(f"  {t_f:>8.2f}  {t_m:>8.2f}  {acc:>8.4f}  {surv_rate:>10.3f}  "
              f"{f3_acc:>7.3f}  {m1_acc:>7.3f}  {f3_pred:>8.3f}  {m1_pred:>8.3f}{marker}")

print(f"\n  Best: female={best_combo[0]}, male={best_combo[1]} (CV={best_acc:.4f})")
print(f"  vs 12a (0.50/0.50): {best_acc - baseline_acc:+.4f}")
print()

# ---- Full subgroup breakdown for best threshold ----
best_t_f, best_t_m = best_combo
best_preds = np.empty(len(X), dtype=int)
best_preds[female_train] = (oof_probs[female_train] >= best_t_f).astype(int)
best_preds[male_train] = (oof_probs[male_train] >= best_t_m).astype(int)

print(f"--- Subgroup Behavior (OOF, threshold f={best_t_f}/m={best_t_m}) ---")
for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (raw_train["Sex"] == sex) & (raw_train["Pclass"] == pclass)
        sub_y = y[mask].values
        sub_pred = best_preds[mask.values]
        n = mask.sum()
        actual_rate = sub_y.mean()
        pred_rate = sub_pred.mean()
        correct = accuracy_score(sub_y, sub_pred)
        print(f"  {sex:6s} Pclass {pclass}: n={n:3d}  actual={actual_rate:.3f}  "
              f"pred={pred_rate:.3f}  acc={correct:.3f}")
print()

# ---- Compare with 12a and v3 subgroup behavior ----
print("--- Subgroup Accuracy Comparison (OOF) ---")
# 12a = threshold 0.50/0.50 on same OOF probs
preds_12a = (oof_probs >= 0.50).astype(int)
print(f"  {'Subgroup':20s}  {'12a (0.50)':>10}  {'12c best':>10}  {'Delta':>8}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = ((raw_train["Sex"] == sex) & (raw_train["Pclass"] == pclass)).values
        acc_12a = accuracy_score(y[mask], preds_12a[mask])
        acc_12c = accuracy_score(y[mask], best_preds[mask])
        delta = acc_12c - acc_12a
        label = f"{sex} Pclass {pclass}"
        print(f"  {label:20s}  {acc_12a:>10.3f}  {acc_12c:>10.3f}  {delta:>+8.3f}")

acc_12a_all = accuracy_score(y, preds_12a)
acc_12c_all = accuracy_score(y, best_preds)
print(f"  {'OVERALL':20s}  {acc_12a_all:>10.4f}  {acc_12c_all:>10.4f}  {acc_12c_all - acc_12a_all:>+8.4f}")
print()

# ---- Train final models on full data ----
pipe_f_final = make_pipe(BEST_C_F)
pipe_f_final.fit(X_female[female_train], y[female_train])

pipe_m_final = make_pipe(BEST_C_M)
pipe_m_final.fit(X_male[male_train], y[male_train])

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

# ---- Decision boundaries ----
print("--- Decision Boundaries ---")
print(f"  Female: intercept={model_f.intercept_[0]:+.4f}, threshold={best_t_f}")
print(f"  Male:   intercept={model_m.intercept_[0]:+.4f}, threshold={best_t_m}")
print()

# ---- In-sample predictions with best thresholds ----
train_probs_f = pipe_f_final.predict_proba(X_female[female_train])[:, 1]
train_probs_m = pipe_m_final.predict_proba(X_male[male_train])[:, 1]

y_pred = np.empty(len(X), dtype=int)
y_pred[female_train] = (train_probs_f >= best_t_f).astype(int)
y_pred[male_train] = (train_probs_m >= best_t_m).astype(int)
train_acc = accuracy_score(y, y_pred)

# ---- Overfitting check ----
print("--- Overfitting Check ---")
print(f"  Training accuracy:  {train_acc:.4f}")
print(f"  CV accuracy:        {best_acc:.4f}")
print(f"  Gap:                {train_acc - best_acc:.4f}")
if train_acc - best_acc > 0.05:
    print("  Warning: Gap > 5% — potential overfitting")
else:
    print("  Gap < 5% — looks healthy")
print()

# ---- Confusion matrices ----
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
        print(f"  {sex:6s} Pclass {pclass}: n={n:3d}  actual={actual_rate:.3f}  "
              f"pred={pred_rate:.3f}  acc={correct:.3f}")
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
    "12a Segment (0.50/0.50)":  (baseline_acc, "TBD"),
    "12c Segment+Thresh":       (best_acc, "TBD"),
}
print(f"  {'Model':30s}  {'CV':>8}  {'Kaggle':>8}")
print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for name, (cv_score, kaggle) in sorted_results:
    marker = " <--" if "12c" in name else ""
    print(f"  {name:30s}  {cv_score:.4f}  {kaggle:>8}{marker}")
print()

# ---- Generate submission ----
test_probs_f = pipe_f_final.predict_proba(test_female[female_test])[:, 1]
test_probs_m = pipe_m_final.predict_proba(test_male[male_test])[:, 1]

test_pred = np.empty(len(test), dtype=int)
test_pred[female_test] = (test_probs_f >= best_t_f).astype(int)
test_pred[male_test] = (test_probs_m >= best_t_m).astype(int)

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{SUBMISSIONS_DIR}/logreg_12c_threshold.csv", index=False)
print(f"Submission saved: submissions/logreg_12c_threshold.csv")
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

# ---- Test prediction profile comparison ----
print("--- Test Prediction Profile Comparison ---")
sub_v3 = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_v3.csv")
sub_12a = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_12a_segment_sex.csv")

print(f"  {'Subgroup':20s}  {'v3':>6}  {'12a':>6}  {'12c':>6}")
print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*6}")
for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (raw_test["Sex"] == sex) & (raw_test["Pclass"] == pclass)
        v3_rate = sub_v3.loc[mask.values, "Survived"].mean()
        a_rate = sub_12a.loc[mask.values, "Survived"].mean()
        c_rate = test_pred[mask.values].mean()
        label = f"{sex} Pclass {pclass}"
        print(f"  {label:20s}  {v3_rate:>6.3f}  {a_rate:>6.3f}  {c_rate:>6.3f}")

print(f"  {'TOTAL surv rate':20s}  {sub_v3['Survived'].mean():>6.3f}  "
      f"{sub_12a['Survived'].mean():>6.3f}  {test_pred.mean():>6.3f}")
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
    changed_12c = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v3, changed_12c):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
        cabin_str = cabin if pd.notna(cabin) else "unknown"
        print(f"    PID {pid}: {old}->{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
print()

# ---- Passenger-level diff vs 12a ----
diff_12a = (submission["Survived"] != sub_12a["Survived"])
n_diff_12a = diff_12a.sum()
print(f"--- Passenger-Level Diff vs 12a ---")
print(f"  Passengers changed: {n_diff_12a} / {len(submission)}")
if n_diff_12a > 0:
    a_to_survived = ((sub_12a.loc[diff_12a, "Survived"] == 0) & (submission.loc[diff_12a, "Survived"] == 1)).sum()
    a_to_died = ((sub_12a.loc[diff_12a, "Survived"] == 1) & (submission.loc[diff_12a, "Survived"] == 0)).sum()
    print(f"  Flipped 0->1 (now survived): {a_to_survived}")
    print(f"  Flipped 1->0 (now died):     {a_to_died}")
    print()
    changed_ids = submission.loc[diff_12a, "PassengerId"].values
    changed_12a_vals = sub_12a.loc[diff_12a, "Survived"].values
    changed_12c_vals = submission.loc[diff_12a, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_12a_vals, changed_12c_vals):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
        cabin_str = cabin if pd.notna(cabin) else "unknown"
        print(f"    PID {pid}: {old}->{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
print()

# ---- Save coefficients ----
coef_f.to_csv(f"{RESULTS_DIR}/12c_female_coefficients.csv", index=False)
coef_m.to_csv(f"{RESULTS_DIR}/12c_male_coefficients.csv", index=False)
print(f"Results saved to: results/models/12c_segment_threshold.txt")
print(f"Female coefficients: results/models/12c_female_coefficients.csv")
print(f"Male coefficients: results/models/12c_male_coefficients.csv")
print(f"Submission: submissions/logreg_12c_threshold.csv")

tee.close()
