"""
Titanic Model 13: v2 Rebaseline

Reconstructs the exact v2 feature matrix (15 features, ordinal Pclass) from
the current v3 processed data. Verifies reproduction against logreg_v2.csv,
then runs through the evaluation harness to establish baseline metrics.

Saves baseline repeated CV scores for paired comparison in step 14.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add project root so shared module is importable
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import evaluate_model, reconstruct_v2_features

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# ---- Load and reconstruct v2 features ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)

y = train_v2["Survived"]
X = train_v2.drop(columns=["Survived"])
test_ids = test_v2["PassengerId"]
X_test = test_v2.drop(columns=["PassengerId"])

print("=" * 60)
print("STEP 13: v2 REBASELINE")
print("=" * 60)
print(f"Reconstructed v2 features: {list(X.columns)}")
print(f"Feature count: {X.shape[1]}")
print()

# ---- Verify exact reproduction ----
print("--- Verification: reproduce logreg_v2.csv ---")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
])
pipe.fit(X, y)
test_preds = pipe.predict(X_test)

baseline = pd.read_csv(f"{BASE}/submissions/logreg_v2.csv")
baseline_preds = baseline["Survived"].values

n_diffs = (test_preds != baseline_preds).sum()
print(f"  Predictions match logreg_v2.csv: {n_diffs == 0}")
print(f"  Differences: {n_diffs}")

if n_diffs > 0:
    diff_mask = test_preds != baseline_preds
    diff_ids = test_ids[diff_mask].values
    print(f"  Differing PassengerIds: {diff_ids.tolist()}")
    print()
    print("  *** VERIFICATION FAILED ***")
    print("  Cannot proceed until reconstruction matches exactly.")
    sys.exit(1)

print("  Reconstruction verified: exact match.")
print()

# ---- Run full evaluation harness ----
pipe_unfitted = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
])

results = evaluate_model(
    pipe_unfitted,
    X, y,
    X_test, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=None,  # No baseline yet — this IS the baseline
    target_subgroup=None,
    report_path=f"{BASE}/results/models/13_v2_rebaseline.txt",
    label="v2_baseline",
)

# ---- Save baseline CV scores for step 14 ----
np.save(
    f"{BASE}/results/models/13_v2_baseline_cv_scores.npy",
    results["cv_scores"],
)
print(f"Baseline CV scores saved ({len(results['cv_scores'])} folds)")
print(f"Results saved to: results/models/13_v2_rebaseline.txt")
