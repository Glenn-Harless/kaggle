"""
Titanic Model 14: Male Pclass=1 Interaction Features

Evaluates three narrow interaction features targeting 1st-class male
bubble passengers, each independently against the locked v2 baseline.

Candidates:
  14a. Male_Pclass1_HasCabin  — 1st-class men with known cabin (near lifeboats)
  14b. Male_Pclass1_Deck_DE   — 1st-class men on D/E decks (strongest survival signal)
  14c. Male_Pclass1_Emb_C     — 1st-class men embarked at Cherbourg (wealthy travellers)

Each adds exactly one feature to the v2 15-feature set (-> 16 features).
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

# Target subgroup for outside-target warnings
target_subgroup = {"Sex": "male", "Pclass": 1}

# Pipeline template (same as v2 baseline)
def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
    ])

# ---- Helper to derive raw Pclass for feature engineering ----
# We need Pclass for creating interactions but it's already in X_base.
# For Sex, we use the encoded column (1=male, 0=female in v2 encoding).

# Also need Deck_DE from v3 for candidate 14b
train_deck_de = train_v3["Deck_DE"].values
test_deck_de = test_v3["Deck_DE"].values

print("=" * 60)
print("STEP 14: MALE PCLASS=1 INTERACTION EXPERIMENTS")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print(f"Target subgroup: {target_subgroup}")
print()

# ==================================================================
# 14a: Male_Pclass1_HasCabin
# ==================================================================
print("\n" + "=" * 60)
print("14a: Male_Pclass1_HasCabin")
print("=" * 60)
print("Mechanism: 1st-class men with known cabins were on upper decks near lifeboats")
print()

X_train_a = X_base.copy()
X_test_a = X_test_base.copy()
X_train_a["Male_Pc1_Cabin"] = ((X_train_a["Sex"] == 1) & (X_train_a["Pclass"] == 1) & (X_train_a["HasCabin"] == 1)).astype(int)
X_test_a["Male_Pc1_Cabin"] = ((X_test_a["Sex"] == 1) & (X_test_a["Pclass"] == 1) & (X_test_a["HasCabin"] == 1)).astype(int)

print(f"Train counts: {X_train_a['Male_Pc1_Cabin'].sum()} positive out of {len(X_train_a)}")
print(f"Test counts:  {X_test_a['Male_Pc1_Cabin'].sum()} positive out of {len(X_test_a)}")
print()

results_a = evaluate_model(
    make_pipeline(),
    X_train_a, y,
    X_test_a, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=target_subgroup,
    label="14a_Male_Pc1_Cabin",
)

# ==================================================================
# 14b: Male_Pclass1_Deck_DE
# ==================================================================
print("\n" + "=" * 60)
print("14b: Male_Pclass1_Deck_DE")
print("=" * 60)
print("Mechanism: D/E decks had strongest survival coefficient in v3 analysis")
print()

X_train_b = X_base.copy()
X_test_b = X_test_base.copy()
X_train_b["Male_Pc1_DeckDE"] = ((X_train_b["Sex"] == 1) & (X_train_b["Pclass"] == 1) & (train_deck_de == 1)).astype(int)
X_test_b["Male_Pc1_DeckDE"] = ((X_test_b["Sex"] == 1) & (X_test_b["Pclass"] == 1) & (test_deck_de == 1)).astype(int)

print(f"Train counts: {X_train_b['Male_Pc1_DeckDE'].sum()} positive out of {len(X_train_b)}")
print(f"Test counts:  {X_test_b['Male_Pc1_DeckDE'].sum()} positive out of {len(X_test_b)}")
print()

results_b = evaluate_model(
    make_pipeline(),
    X_train_b, y,
    X_test_b, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=target_subgroup,
    label="14b_Male_Pc1_DeckDE",
)

# ==================================================================
# 14c: Male_Pclass1_Emb_C
# ==================================================================
print("\n" + "=" * 60)
print("14c: Male_Pclass1_Emb_C")
print("=" * 60)
print("Mechanism: Cherbourg 1st-class men were wealthy, potentially better cabin locations")
print()

X_train_c = X_base.copy()
X_test_c = X_test_base.copy()
X_train_c["Male_Pc1_EmbC"] = ((X_train_c["Sex"] == 1) & (X_train_c["Pclass"] == 1) & (X_train_c["Emb_C"] == 1)).astype(int)
X_test_c["Male_Pc1_EmbC"] = ((X_test_c["Sex"] == 1) & (X_test_c["Pclass"] == 1) & (X_test_c["Emb_C"] == 1)).astype(int)

print(f"Train counts: {X_train_c['Male_Pc1_EmbC'].sum()} positive out of {len(X_train_c)}")
print(f"Test counts:  {X_test_c['Male_Pc1_EmbC'].sum()} positive out of {len(X_test_c)}")
print()

results_c = evaluate_model(
    make_pipeline(),
    X_train_c, y,
    X_test_c, test_ids,
    raw_test,
    baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
    baseline_cv_scores=baseline_cv,
    target_subgroup=target_subgroup,
    label="14c_Male_Pc1_EmbC",
)

# ==================================================================
# COMPARATIVE SUMMARY
# ==================================================================
print("\n" + "=" * 60)
print("COMPARATIVE SUMMARY")
print("=" * 60)
print()
print(f"{'Model':<25} {'CV Mean':>8} {'CV Std':>7} {'Delta':>7} {'Flips':>6} {'Outside':>8}")
print("-" * 65)

baseline_row = f"{'v2 baseline':<25} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'0':>6} {'--':>8}"
print(baseline_row)

for label, res in [("14a HasCabin", results_a), ("14b Deck DE", results_b), ("14c Emb C", results_c)]:
    delta = res["paired"]["mean_delta"] if "paired" in res else 0
    flips = res["flips"]["n_flips"]
    outside = res["flips"]["outside_target"]
    print(f"{label:<25} {res['cv_mean']:>8.4f} {res['cv_std']:>7.4f} {delta:>+7.4f} {flips:>6} {outside:>8}")

print()
print("Note: Positive delta = candidate better than baseline")
print("      Outside = flips outside target subgroup (male Pclass=1)")
print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject.")
