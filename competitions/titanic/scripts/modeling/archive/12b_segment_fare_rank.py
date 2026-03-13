"""
Titanic Experiment 12b: Segmented LogReg + FareRankInClass

Same segmented approach as 12a. Adds one feature: FareRankInClass
(percentile rank of Fare within Pclass, 0-1 scale).

Leakage-safe: within each CV fold, rank Fare within Pclass on the train
fold only. For val fold, use scipy.stats.percentileofscore against the
train fold's Pclass fare distribution.

Tests two variants:
  A) + FareRankInClass (keep Fare)
  B) + FareRankInClass (drop Fare)

Picks the better variant. Compares directly to 12a.
"""

import pandas as pd
import numpy as np
import sys
from scipy.stats import percentileofscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
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


tee = Tee(f"{RESULTS_DIR}/12b_segment_fare_rank.txt")
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
pclass_train = raw_train["Pclass"].values
pclass_test = raw_test["Pclass"].values
fare_train = X["Fare"].values
fare_test = test["Fare"].values

female_train = sex_train == "female"
male_train = sex_train == "male"
female_test = sex_test == "female"
male_test = sex_test == "male"

# ---- Segment-specific feature sets (same as 12a) ----
female_drop = ["Sex", "Title_Mr", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]
male_drop = ["Sex", "Title_Miss", "Title_Mrs", "Sex_x_Pclass_2", "Sex_x_Pclass_3"]

X_female_base = X.drop(columns=female_drop)
X_male_base = X.drop(columns=male_drop)
test_female_base = test.drop(columns=female_drop)
test_male_base = test.drop(columns=male_drop)

# ---- CV setup ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X, y))
C_values = [0.01, 0.05, 0.1, 0.5, 1.0]


def make_pipe(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
    ])


# ---- FareRankInClass computation (leakage-safe) ----
def batch_percentile_rank(values, reference):
    """Vectorized percentile rank: (count_below + 0.5*count_equal) / N."""
    ref_sorted = np.sort(reference)
    below = np.searchsorted(ref_sorted, values, side="left")
    above = np.searchsorted(ref_sorted, values, side="right")
    count_equal = above - below
    return (below + 0.5 * count_equal) / len(ref_sorted)


def compute_fare_rank_fold(fare, pclass, train_idx, val_idx):
    """Compute FareRankInClass per-fold (leakage-safe).

    Returns full-length array with values at train_idx and val_idx positions.
    Train uses its own Pclass fare distribution. Val maps against train distribution.
    """
    result = np.full(len(fare), np.nan)

    for pc in [1, 2, 3]:
        ref_fares = fare[train_idx[pclass[train_idx] == pc]]
        if len(ref_fares) == 0:
            continue

        # Train fold: rank against own Pclass distribution
        train_pc_idx = train_idx[pclass[train_idx] == pc]
        result[train_pc_idx] = batch_percentile_rank(fare[train_pc_idx], ref_fares)

        # Val fold: rank against train Pclass distribution
        val_pc_idx = val_idx[pclass[val_idx] == pc]
        if len(val_pc_idx) > 0:
            result[val_pc_idx] = batch_percentile_rank(fare[val_pc_idx], ref_fares)

    return result


def compute_fare_rank_final(fare_src, pclass_src, fare_tgt, pclass_tgt):
    """Compute FareRankInClass: rank src against itself, rank tgt against src."""
    src_rank = np.zeros(len(fare_src))
    tgt_rank = np.zeros(len(fare_tgt))

    for pc in [1, 2, 3]:
        ref_fares = fare_src[pclass_src == pc]
        if len(ref_fares) == 0:
            continue
        src_rank[pclass_src == pc] = batch_percentile_rank(fare_src[pclass_src == pc], ref_fares)
        tgt_mask = pclass_tgt == pc
        if tgt_mask.any():
            tgt_rank[tgt_mask] = batch_percentile_rank(fare_tgt[tgt_mask], ref_fares)

    return src_rank, tgt_rank


print("=" * 60)
print("EXPERIMENT 12b: SEGMENTED LOGREG + FareRankInClass")
print("=" * 60)
print(f"\nFemale: {female_train.sum()} train, {female_test.sum()} test")
print(f"Male:   {male_train.sum()} train, {male_test.sum()} test")
print(f"\nBase female features ({X_female_base.shape[1]}): {list(X_female_base.columns)}")
print(f"Base male features ({X_male_base.shape[1]}): {list(X_male_base.columns)}")
print()

# ---- Run both variants ----
# Variant A: + FareRankInClass (keep Fare)
# Variant B: + FareRankInClass (drop Fare)

variant_results = {}

for variant_name, drop_fare in [("A: +FareRank, keep Fare", False),
                                 ("B: +FareRank, drop Fare", True)]:
    print("=" * 60)
    print(f"VARIANT {variant_name}")
    print("=" * 60)
    print()

    # ---- Per-segment C sweep ----
    def segment_c_sweep_with_rank(seg_name, X_seg_base, seg_mask, drop_fare_col):
        """Sweep C for a segment with per-fold FareRankInClass."""
        print(f"--- {seg_name} Model: C Sweep ---")
        print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}")
        print("-" * 30)

        best_C = None
        best_score = 0

        for C in C_values:
            fold_accs = []
            for train_idx, val_idx in splits:
                # Compute FareRankInClass for this fold
                fare_rank = compute_fare_rank_fold(fare_train, pclass_train,
                                                   train_idx, val_idx)

                # Segment indices
                s_train = train_idx[seg_mask[train_idx]]
                s_val = val_idx[seg_mask[val_idx]]

                # Build augmented features
                X_tr = X_seg_base.iloc[s_train].copy()
                X_tr["FareRankInClass"] = fare_rank[s_train]
                X_va = X_seg_base.iloc[s_val].copy()
                X_va["FareRankInClass"] = fare_rank[s_val]

                if drop_fare_col:
                    X_tr = X_tr.drop(columns=["Fare"])
                    X_va = X_va.drop(columns=["Fare"])

                pipe = make_pipe(C)
                pipe.fit(X_tr, y.iloc[s_train])
                pred = pipe.predict(X_va)
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

    best_C_f = segment_c_sweep_with_rank("Female", X_female_base, female_train, drop_fare)
    best_C_m = segment_c_sweep_with_rank("Male", X_male_base, male_train, drop_fare)

    # ---- Combined segmented CV ----
    print(f"--- Combined Segmented CV (5-Fold) ---")
    print(f"  Female C={best_C_f}, Male C={best_C_m}")
    print()

    fold_accs = []
    oof_preds = np.empty(len(X), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fare_rank = compute_fare_rank_fold(fare_train, pclass_train,
                                           train_idx, val_idx)

        # Female segment
        f_tr = train_idx[female_train[train_idx]]
        f_va = val_idx[female_train[val_idx]]

        X_f_tr = X_female_base.iloc[f_tr].copy()
        X_f_tr["FareRankInClass"] = fare_rank[f_tr]
        X_f_va = X_female_base.iloc[f_va].copy()
        X_f_va["FareRankInClass"] = fare_rank[f_va]
        if drop_fare:
            X_f_tr = X_f_tr.drop(columns=["Fare"])
            X_f_va = X_f_va.drop(columns=["Fare"])

        pipe_f = make_pipe(best_C_f)
        pipe_f.fit(X_f_tr, y.iloc[f_tr])
        y_f_pred = pipe_f.predict(X_f_va)
        f_acc = accuracy_score(y.iloc[f_va], y_f_pred)

        # Male segment
        m_tr = train_idx[male_train[train_idx]]
        m_va = val_idx[male_train[val_idx]]

        X_m_tr = X_male_base.iloc[m_tr].copy()
        X_m_tr["FareRankInClass"] = fare_rank[m_tr]
        X_m_va = X_male_base.iloc[m_va].copy()
        X_m_va["FareRankInClass"] = fare_rank[m_va]
        if drop_fare:
            X_m_tr = X_m_tr.drop(columns=["Fare"])
            X_m_va = X_m_va.drop(columns=["Fare"])

        pipe_m = make_pipe(best_C_m)
        pipe_m.fit(X_m_tr, y.iloc[m_tr])
        y_m_pred = pipe_m.predict(X_m_va)
        m_acc = accuracy_score(y.iloc[m_va], y_m_pred)

        # Combine predictions
        val_pos = {idx: pos for pos, idx in enumerate(val_idx)}
        y_val_pred = np.empty(len(val_idx), dtype=int)
        for i, idx in enumerate(f_va):
            y_val_pred[val_pos[idx]] = y_f_pred[i]
        for i, idx in enumerate(m_va):
            y_val_pred[val_pos[idx]] = y_m_pred[i]

        for i, idx in enumerate(f_va):
            oof_preds[idx] = y_f_pred[i]
        for i, idx in enumerate(m_va):
            oof_preds[idx] = y_m_pred[i]

        fold_acc = accuracy_score(y.iloc[val_idx], y_val_pred)
        fold_accs.append(fold_acc)
        print(f"  Fold {fold_idx+1}: {fold_acc:.4f}  (female={f_acc:.4f}, male={m_acc:.4f})")

    var_cv = np.mean(fold_accs)
    var_std = np.std(fold_accs)
    print(f"\n  Mean:  {var_cv:.4f}")
    print(f"  Std:   {var_std:.4f}")
    print()

    variant_results[variant_name] = {
        "cv": var_cv,
        "std": var_std,
        "best_C_f": best_C_f,
        "best_C_m": best_C_m,
        "drop_fare": drop_fare,
        "oof_preds": oof_preds.copy(),
    }

# ---- Pick best variant ----
print("=" * 60)
print("VARIANT COMPARISON")
print("=" * 60)
print(f"\n  {'Variant':35s}  {'CV':>8}  {'Std':>8}")
print(f"  {'-'*35}  {'--------':>8}  {'--------':>8}")
best_variant = None
best_variant_cv = 0
for name, res in variant_results.items():
    marker = ""
    if res["cv"] > best_variant_cv:
        best_variant_cv = res["cv"]
        best_variant = name
        marker = " <-- best"
    print(f"  {name:35s}  {res['cv']:>8.4f}  {res['std']:>8.4f}{marker}")
print(f"\n  Selected: {best_variant}")
print()

# ---- Train final models with best variant ----
best = variant_results[best_variant]
best_C_f = best["best_C_f"]
best_C_m = best["best_C_m"]
drop_fare = best["drop_fare"]
oof_preds = best["oof_preds"]

# Compute FareRankInClass on full train and test
fare_rank_train_final, fare_rank_test_final = compute_fare_rank_final(
    fare_train, pclass_train, fare_test, pclass_test
)

# Build final augmented feature sets
X_f_final = X_female_base[female_train].copy()
X_f_final["FareRankInClass"] = fare_rank_train_final[female_train]
X_m_final = X_male_base[male_train].copy()
X_m_final["FareRankInClass"] = fare_rank_train_final[male_train]

test_f_final = test_female_base[female_test].copy()
test_f_final["FareRankInClass"] = fare_rank_test_final[female_test]
test_m_final = test_male_base[male_test].copy()
test_m_final["FareRankInClass"] = fare_rank_test_final[male_test]

if drop_fare:
    X_f_final = X_f_final.drop(columns=["Fare"])
    X_m_final = X_m_final.drop(columns=["Fare"])
    test_f_final = test_f_final.drop(columns=["Fare"])
    test_m_final = test_m_final.drop(columns=["Fare"])

print(f"--- Final Model Features ---")
print(f"  Female ({X_f_final.shape[1]}): {list(X_f_final.columns)}")
print(f"  Male ({X_m_final.shape[1]}): {list(X_m_final.columns)}")
print()

pipe_f_final = make_pipe(best_C_f)
pipe_f_final.fit(X_f_final, y[female_train])

pipe_m_final = make_pipe(best_C_m)
pipe_m_final.fit(X_m_final, y[male_train])

# ---- In-sample predictions ----
y_pred = np.empty(len(X), dtype=int)
# For in-sample, need full-train FareRankInClass
X_f_all = X_female_base[female_train].copy()
X_f_all["FareRankInClass"] = fare_rank_train_final[female_train]
X_m_all = X_male_base[male_train].copy()
X_m_all["FareRankInClass"] = fare_rank_train_final[male_train]
if drop_fare:
    X_f_all = X_f_all.drop(columns=["Fare"])
    X_m_all = X_m_all.drop(columns=["Fare"])

y_pred[female_train] = pipe_f_final.predict(X_f_all)
y_pred[male_train] = pipe_m_final.predict(X_m_all)
train_acc = accuracy_score(y, y_pred)

# ---- Coefficients per segment ----
print("--- Female Model Coefficients ---")
print("Positive = increases survival, Negative = decreases survival\n")
model_f = pipe_f_final.named_steps["model"]
coef_f = pd.DataFrame({
    "feature": X_f_final.columns,
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
    "feature": X_m_final.columns,
    "coefficient": model_m.coef_[0],
    "abs_coefficient": np.abs(model_m.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

for _, row in coef_m.iterrows():
    d = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {d} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")
print()

# ---- Decision boundaries ----
print("--- Decision Boundaries (Intercepts) ---")
print(f"  Female intercept: {model_f.intercept_[0]:+.4f}")
print(f"  Male intercept:   {model_m.intercept_[0]:+.4f}")
print()

# ---- Overfitting check ----
overall_cv = best["cv"]
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

# ---- Subgroup behavior (OOF) ----
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

# ---- FareRankInClass distribution check ----
print("--- FareRankInClass Distribution (Train, Full) ---")
for pc in [1, 2, 3]:
    mask = pclass_train == pc
    ranks = fare_rank_train_final[mask]
    print(f"  Pclass {pc}: n={mask.sum():3d}  min={ranks.min():.3f}  median={np.median(ranks):.3f}  max={ranks.max():.3f}")
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
    "12b Segment+FareRank":     (overall_cv, "TBD"),
}

# Try to read 12a result from its output
try:
    with open(f"{RESULTS_DIR}/12a_segment_by_sex.txt") as f:
        for line in f:
            if "vs v3:" in line:
                cv_12a = float(line.strip().split("vs v3:")[0].strip().split()[-1])
                results["12a Segmented by Sex"] = (cv_12a, "TBD")
                break
except Exception:
    pass

print(f"  {'Model':30s}  {'CV':>8}  {'Kaggle':>8}")
print(f"  {'-'*30}  {'--------':>8}  {'--------':>8}")
sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
for name, (cv_score, kaggle) in sorted_results:
    marker = " <--" if "12b" in name else ""
    print(f"  {name:30s}  {cv_score:.4f}  {kaggle:>8}{marker}")
print()

# ---- Generate submission ----
test_pred = np.empty(len(test), dtype=int)
test_pred[female_test] = pipe_f_final.predict(test_f_final)
test_pred[male_test] = pipe_m_final.predict(test_m_final)

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{SUBMISSIONS_DIR}/logreg_12b_fare_rank.csv", index=False)
print(f"Submission saved: submissions/logreg_12b_fare_rank.csv")
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
    changed_12b = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v3, changed_12b):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
        cabin_str = cabin if pd.notna(cabin) else "unknown"
        print(f"    PID {pid}: {old}->{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
print()

# ---- Passenger-level diff vs 12a ----
try:
    sub_12a = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_12a_segment_sex.csv")
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
        changed_12a = sub_12a.loc[diff_12a, "Survived"].values
        changed_12b = submission.loc[diff_12a, "Survived"].values
        for pid, old, new in zip(changed_ids, changed_12a, changed_12b):
            sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
            pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
            name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
            cabin = raw_test.loc[raw_test["PassengerId"] == pid, "Cabin"].values[0]
            cabin_str = cabin if pd.notna(cabin) else "unknown"
            print(f"    PID {pid}: {old}->{new}  {sex:6s} Pclass {pclass}  Cabin={cabin_str:8s}  {name}")
    print()
except FileNotFoundError:
    print("--- 12a submission not found, skipping diff ---")
    print()

# ---- Save coefficients ----
coef_f.to_csv(f"{RESULTS_DIR}/12b_female_coefficients.csv", index=False)
coef_m.to_csv(f"{RESULTS_DIR}/12b_male_coefficients.csv", index=False)
print(f"Results saved to: results/models/12b_segment_fare_rank.txt")
print(f"Female coefficients: results/models/12b_female_coefficients.csv")
print(f"Male coefficients: results/models/12b_male_coefficients.csv")
print(f"Submission: submissions/logreg_12b_fare_rank.csv")

tee.close()
