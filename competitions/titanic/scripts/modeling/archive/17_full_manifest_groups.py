"""
Titanic Model 17: Full Manifest Group Features + Relaxed Regularization

Two changes from the v2 baseline:

1. Compute TicketGroupSize on the FULL passenger manifest (train+test combined).
   This is metadata (ticket strings), not target leakage (survival labels).
   The v1 pipeline did this and was flagged for "leakage" — but the feature
   never touches the Survived column. It's the same as knowing how many
   people share your train ticket, which is public information.

2. Test a broader range of regularization strengths. The original C=0.01 was
   selected from a single noisy 5-fold CV where the entire spread was 0.34
   percentage points — less than one passenger per fold.

Phase 1: Screen all (feature_set × C) combinations with repeated CV
Phase 2: Run top candidates through full evaluation harness
Phase 3: Save submission for best candidate
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import evaluate_model, reconstruct_v2_features

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# ---- Load data ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_train = pd.read_csv(f"{BASE}/data/train.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)

y = train_v2["Survived"]
X_v2 = train_v2.drop(columns=["Survived"])
test_ids = test_v2["PassengerId"]
X_test_v2 = test_v2.drop(columns=["PassengerId"])

baseline_cv = np.load(f"{BASE}/results/models/13_v2_baseline_cv_scores.npy")


# ================================================================
# COMPUTE TICKET GROUP SIZE ON FULL MANIFEST
# ================================================================

all_tickets = pd.concat([raw_train["Ticket"], raw_test["Ticket"]], ignore_index=True)
ticket_counts = all_tickets.value_counts()

train_tgs = raw_train["Ticket"].map(ticket_counts).values
test_tgs = raw_test["Ticket"].map(ticket_counts).values

print("=" * 60)
print("STEP 17: FULL MANIFEST GROUPS + RELAXED REGULARIZATION")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print()

# Diagnostic
print("--- TicketGroupSize Distribution (FULL manifest, train rows) ---")
tgs_series = pd.Series(train_tgs)
for size in sorted(tgs_series.unique()):
    n = (tgs_series == size).sum()
    surv = y[tgs_series == size].mean()
    print(f"  Size {int(size):>2}: {n:>4} passengers, survival {surv:.3f}")
print()

# Compare to train-only (step 15) counts
train_only_counts = raw_train["Ticket"].value_counts()
train_only_tgs = raw_train["Ticket"].map(train_only_counts).values
changed = (train_tgs != train_only_tgs).sum()
print(f"Passengers whose group size changed (full vs train-only): {changed} / {len(raw_train)}")
test_with_ticket = (raw_test["Ticket"].isin(raw_train["Ticket"])).sum()
test_new = len(raw_test) - test_with_ticket
print(f"Test passengers with ticket also in train: {test_with_ticket}")
print(f"Test passengers with ticket ONLY in test:  {test_new}")
# Of those test-only tickets, how many have group size > 1?
test_only_mask = ~raw_test["Ticket"].isin(raw_train["Ticket"])
test_only_tgs = test_tgs[test_only_mask]
test_only_in_group = (test_only_tgs > 1).sum()
print(f"  Of those, in a test-only group (size>1): {test_only_in_group}")
print(f"  Truly solo (size=1):                     {(test_only_tgs == 1).sum()}")
print()


# ================================================================
# BUILD FEATURE SETS
# ================================================================

# FS1: v2 only (isolate regularization effect)
fs1_train = X_v2.copy()
fs1_test = X_test_v2.copy()

# FS2: v2 + TicketGroupSize raw
fs2_train = X_v2.copy()
fs2_test = X_test_v2.copy()
fs2_train["TicketGroupSize"] = train_tgs
fs2_test["TicketGroupSize"] = test_tgs

# FS3: v2 + ticket group buckets (matching EDA sweet-spot pattern)
fs3_train = X_v2.copy()
fs3_test = X_test_v2.copy()
fs3_train["IsSmallTicketGroup"] = ((train_tgs >= 2) & (train_tgs <= 3)).astype(int)
fs3_train["IsLargeTicketGroup"] = (train_tgs >= 4).astype(int)
fs3_test["IsSmallTicketGroup"] = ((test_tgs >= 2) & (test_tgs <= 3)).astype(int)
fs3_test["IsLargeTicketGroup"] = (test_tgs >= 4).astype(int)

feature_sets = {
    "v2_only": (fs1_train, fs1_test),
    "v2+TGS_raw": (fs2_train, fs2_test),
    "v2+TGS_buckets": (fs3_train, fs3_test),
}


# ================================================================
# PHASE 1: SCREEN ALL COMBINATIONS
# ================================================================
print("=" * 60)
print("PHASE 1: SCREENING (repeated 10x5 CV)")
print("=" * 60)
print()

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

results_grid = []

for fs_name, (X_tr, X_te) in feature_sets.items():
    print(f"--- {fs_name} ({X_tr.shape[1]} features) ---")
    print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}")
    print(f"  {'-' * 25}")

    for C in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        scores = cross_val_score(pipe, X_tr, y, cv=cv, scoring="accuracy")

        results_grid.append({
            "features": fs_name,
            "C": C,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "cv_scores": scores,
            "X_train": X_tr,
            "X_test": X_te,
        })

        marker = ""
        if scores.mean() > baseline_cv.mean() + 0.002:
            marker = " ***"
        print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}{marker}")

    print()

# Sort by CV mean
results_grid.sort(key=lambda x: x["cv_mean"], reverse=True)

print("=" * 60)
print("TOP 10 COMBINATIONS (sorted by CV mean)")
print("=" * 60)
print()
print(f"{'Rank':>4} {'Features':<18} {'C':>6} {'CV Mean':>8} {'CV Std':>7} {'vs BL':>7}")
print("-" * 55)
for i, r in enumerate(results_grid[:10]):
    delta = r["cv_mean"] - baseline_cv.mean()
    print(f"{i+1:>4} {r['features']:<18} {r['C']:>6.2f} {r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} {delta:>+7.4f}")
print()


# ================================================================
# PHASE 2: FULL HARNESS ON TOP CANDIDATES
# ================================================================
print("=" * 60)
print("PHASE 2: FULL HARNESS EVALUATION")
print("=" * 60)
print()

# Pick top candidate per feature set + overall best
candidates = []
seen_fs = set()
for r in results_grid:
    if r["features"] not in seen_fs:
        seen_fs.add(r["features"])
        candidates.append(r)
    if len(candidates) >= 3:
        break

# Make sure the overall #1 is included
if results_grid[0] not in candidates:
    candidates.insert(0, results_grid[0])

harness_results = []
for r in candidates:
    label = f"{r['features']}_C{r['C']}"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=r["C"], max_iter=2000, random_state=42)),
    ])

    res = evaluate_model(
        pipe,
        r["X_train"], y,
        r["X_test"], test_ids,
        raw_test,
        baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
        baseline_cv_scores=baseline_cv,
        target_subgroup=None,
        label=label,
    )
    res["config"] = r
    harness_results.append(res)


# ================================================================
# PHASE 3: FINAL COMPARISON + SUBMISSION
# ================================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print()
print(f"{'Model':<35} {'CV Mean':>8} {'CV Std':>7} {'Delta':>7} {'Flips':>6} {'W/L/T':>10}")
print("-" * 75)
print(f"{'v2 baseline (C=0.01)':<35} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'0':>6} {'--':>10}")

for res in harness_results:
    cfg = res["config"]
    label = f"{cfg['features']} C={cfg['C']}"
    paired = res.get("paired", {})
    delta = paired.get("mean_delta", 0)
    wins = paired.get("n_candidate_wins", 0)
    losses = paired.get("n_baseline_wins", 0)
    ties = paired.get("n_ties", 0)
    flips = res["flips"]["n_flips"]
    print(f"{label:<35} {res['cv_mean']:>8.4f} {res['cv_std']:>7.4f} {delta:>+7.4f} {flips:>6} {f'{wins}/{losses}/{ties}':>10}")

print()

# Save submissions for all harness candidates
for res in harness_results:
    cfg = res["config"]
    label = f"{cfg['features']}_C{cfg['C']}".replace("+", "_")
    sub_path = f"{BASE}/submissions/logreg_17_{label}.csv"
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": res["test_preds"]})
    sub.to_csv(sub_path, index=False)
    rate = res["test_preds"].mean()
    print(f"Saved: {sub_path}  (survival rate: {rate:.3f})")

print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")
