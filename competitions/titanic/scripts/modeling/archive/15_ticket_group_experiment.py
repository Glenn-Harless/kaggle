"""
Titanic Model 15: Ticket Group Size Experiment

Evaluates ticket-based group features against the locked v2 baseline.
TicketGroupSize is computed from training data only (no test leakage).

Candidates:
  15a. TicketGroupSize (raw)   — raw count added as continuous feature
  15b. Ticket group buckets    — IsSmallTicketGroup + IsLargeTicketGroup
  15c. HasNonFamilyCompanions  — shares ticket but has no family (SibSp+Parch==0)

Each adds features to the v2 15-feature set and runs through the evaluation harness.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import evaluate_model, reconstruct_v2_features

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# ---- Load data ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_train = pd.read_csv(f"{BASE}/data/train.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

# Reconstruct v2 features
train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)

y = train_v2["Survived"]
X_base = train_v2.drop(columns=["Survived"])
test_ids = test_v2["PassengerId"]
X_test_base = test_v2.drop(columns=["PassengerId"])

# Load baseline CV scores from step 13
baseline_cv = np.load(f"{BASE}/results/models/13_v2_baseline_cv_scores.npy")

# Pipeline template (same as v2 baseline)
def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
    ])


# ================================================================
# COMPUTE TICKET GROUP SIZE (train-only, no leakage)
# ================================================================
# v1 computed this on train+test combined — that was the leakage.
# Here we compute from train only. Test passengers with tickets
# not seen in train get size 1 (assumed solo).
# ================================================================

train_ticket_counts = raw_train["Ticket"].value_counts()

# Map to train passengers
train_tgs = raw_train["Ticket"].map(train_ticket_counts).values

# Map to test passengers (unseen tickets default to 1)
test_tgs = raw_test["Ticket"].map(train_ticket_counts).fillna(1).astype(int).values

# Raw family info for candidate 15c
train_family_size = (raw_train["SibSp"] + raw_train["Parch"]).values
test_family_size = (raw_test["SibSp"] + raw_test["Parch"]).values

print("=" * 60)
print("STEP 15: TICKET GROUP SIZE EXPERIMENTS")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print()

# ---- Diagnostic: ticket group distribution ----
print("--- Ticket Group Size Distribution (train only) ---")
tgs_series = pd.Series(train_tgs)
for size in sorted(tgs_series.unique()):
    n = (tgs_series == size).sum()
    surv_rate = y[tgs_series == size].mean()
    print(f"  Size {int(size):>2}: {n:>4} passengers, survival {surv_rate:.3f}")
print()

# How many test passengers have unseen tickets?
n_test_unseen = raw_test["Ticket"].map(train_ticket_counts).isna().sum()
n_test_total = len(raw_test)
print(f"Test passengers with ticket NOT in train: {n_test_unseen} / {n_test_total}")
print(f"Test passengers with ticket in train:     {n_test_total - n_test_unseen} / {n_test_total}")
print()

# Overlap between ticket groups and family groups
has_family = train_family_size > 0
in_ticket_group = train_tgs > 1
both = (has_family & in_ticket_group).sum()
family_only = (has_family & ~in_ticket_group).sum()
ticket_only = (~has_family & in_ticket_group).sum()
neither = (~has_family & ~in_ticket_group).sum()
print("--- Ticket Group vs Family Overlap (train) ---")
print(f"  Both family + ticket group:      {both}")
print(f"  Family only (no shared ticket):  {family_only}")
print(f"  Ticket group only (no family):   {ticket_only}  <-- unique signal")
print(f"  Neither (solo on both):          {neither}")
print()

# Survival by overlap category
print("--- Survival by Ticket/Family Category (train) ---")
cats = {
    "Both family + ticket":      has_family & in_ticket_group,
    "Family only":               has_family & ~in_ticket_group,
    "Ticket group only":         ~has_family & in_ticket_group,
    "Solo (neither)":            ~has_family & ~in_ticket_group,
}
for name, mask in cats.items():
    n = mask.sum()
    if n > 0:
        surv = y[mask].mean()
        print(f"  {name:<30s}  n={n:>4}  survival={surv:.3f}")
print()


# ==================================================================
# 15a: TicketGroupSize (raw)
# ==================================================================
print("\n" + "=" * 60)
print("15a: TicketGroupSize (raw continuous feature)")
print("=" * 60)
print("Adds raw ticket group count. StandardScaler normalizes it.")
print()

X_train_a = X_base.copy()
X_test_a = X_test_base.copy()
X_train_a["TicketGroupSize"] = train_tgs
X_test_a["TicketGroupSize"] = test_tgs

print(f"Features: {X_train_a.shape[1]} (v2 15 + 1 new)")
print()

results_a = evaluate_model(
    make_pipeline(),
    X_train_a, y,
    X_test_a, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=None,
    label="15a_TicketGroupSize_raw",
)


# ==================================================================
# 15b: Ticket Group Buckets
# ==================================================================
print("\n" + "=" * 60)
print("15b: Ticket Group Buckets (matching FamilySize pattern)")
print("=" * 60)
print("EDA pattern: solo=low, small(2-3)=high, large(4+)=very low")
print()

X_train_b = X_base.copy()
X_test_b = X_test_base.copy()

X_train_b["IsSmallTicketGroup"] = ((train_tgs >= 2) & (train_tgs <= 3)).astype(int)
X_train_b["IsLargeTicketGroup"] = (train_tgs >= 4).astype(int)
X_test_b["IsSmallTicketGroup"] = ((test_tgs >= 2) & (test_tgs <= 3)).astype(int)
X_test_b["IsLargeTicketGroup"] = (test_tgs >= 4).astype(int)

print(f"Features: {X_train_b.shape[1]} (v2 15 + 2 new)")
print(f"Train IsSmallTicketGroup: {X_train_b['IsSmallTicketGroup'].sum()} passengers")
print(f"Train IsLargeTicketGroup: {X_train_b['IsLargeTicketGroup'].sum()} passengers")
print()

results_b = evaluate_model(
    make_pipeline(),
    X_train_b, y,
    X_test_b, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=None,
    label="15b_TicketGroup_buckets",
)


# ==================================================================
# 15c: HasNonFamilyCompanions
# ==================================================================
print("\n" + "=" * 60)
print("15c: HasNonFamilyCompanions")
print("=" * 60)
print("Captures unique signal: shares ticket but has no family (SibSp+Parch==0)")
print(f"Targets the {ticket_only} train passengers in ticket groups without family ties")
print()

X_train_c = X_base.copy()
X_test_c = X_test_base.copy()

X_train_c["HasNonFamilyComp"] = ((train_tgs > 1) & (train_family_size == 0)).astype(int)
X_test_c["HasNonFamilyComp"] = ((test_tgs > 1) & (test_family_size == 0)).astype(int)

print(f"Features: {X_train_c.shape[1]} (v2 15 + 1 new)")
print(f"Train HasNonFamilyComp: {X_train_c['HasNonFamilyComp'].sum()} passengers")
print(f"Test HasNonFamilyComp:  {X_test_c['HasNonFamilyComp'].sum()} passengers")
print()

results_c = evaluate_model(
    make_pipeline(),
    X_train_c, y,
    X_test_c, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=None,
    label="15c_NonFamilyCompanions",
)


# ==================================================================
# COMPARATIVE SUMMARY
# ==================================================================
print("\n" + "=" * 60)
print("COMPARATIVE SUMMARY")
print("=" * 60)
print()
print(f"{'Model':<30} {'CV Mean':>8} {'CV Std':>7} {'Delta':>7} {'Flips':>6} {'W/L/T':>10}")
print("-" * 70)

print(f"{'v2 baseline':<30} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'0':>6} {'--':>10}")

all_results = [
    ("15a TGS raw", results_a),
    ("15b TGS buckets", results_b),
    ("15c NonFamilyComp", results_c),
]

for label, res in all_results:
    paired = res.get("paired", {})
    delta = paired.get("mean_delta", 0)
    wins = paired.get("n_candidate_wins", 0)
    losses = paired.get("n_baseline_wins", 0)
    ties = paired.get("n_ties", 0)
    flips = res["flips"]["n_flips"]
    print(f"{label:<30} {res['cv_mean']:>8.4f} {res['cv_std']:>7.4f} {delta:>+7.4f} {flips:>6} {f'{wins}/{losses}/{ties}':>10}")

print()
print("Legend: Delta = candidate - baseline (positive = better)")
print("        W/L/T = candidate wins / baseline wins / ties (out of 50 folds)")
print()

# ---- Save submission for best candidate if compelling ----
best_label = None
best_delta = 0
best_res = None

for label, res in all_results:
    paired = res.get("paired", {})
    delta = paired.get("mean_delta", 0)
    flips = res["flips"]["n_flips"]
    if delta > best_delta and flips <= 20:
        best_delta = delta
        best_label = label
        best_res = res

if best_label and best_delta > 0.001:
    sub_name = best_label.lower().replace(" ", "_")
    sub_path = f"{BASE}/submissions/logreg_15_{sub_name}.csv"
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": best_res["test_preds"],
    })
    submission.to_csv(sub_path, index=False)
    print(f"Best candidate: {best_label} (delta={best_delta:+.4f})")
    print(f"Submission saved: {sub_path}")
    print(f"Predicted survival rate: {best_res['test_preds'].mean():.3f}")
else:
    print("No candidate showed a compelling improvement over baseline.")
    print("No submission saved.")

print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")
