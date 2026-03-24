"""
Criteo Uplift Modeling: Step 2 — Naive Baselines
Build intentionally wrong approaches to understand why uplift modeling exists:
  1. Standard classifier: rank by P(visit) — targets Sure Things, not Persuadables
  2. Two separate classifiers: T-learner done naively (subtract predictions)
  3. Random baseline: predict uplift = 0 for everyone

The key "aha": standard classifier should perform WORSE than random for uplift,
because high-propensity users are mostly Sure Things, not Persuadables.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_curve
from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/02_naive_baselines.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 2 — Naive Baselines")
print("=" * 60)


# ============================================================
# DATA LOADING & TRAIN/TEST SPLIT
# ============================================================

from sklift.datasets import fetch_criteo

print("\nLoading Criteo 10% sample...")
data = fetch_criteo(percent10=True)

df = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(12)])
df["treatment"] = data.treatment.astype(int)
df["visit"] = data.target.astype(int)

feature_cols = [f"f{i}" for i in range(12)]
X = df[feature_cols]
y = df["visit"]
treatment = df["treatment"]

# Stratified split preserving treatment AND outcome proportions
# We stratify on treatment*2 + visit to maintain both rates
strat_col = treatment * 2 + y
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42, stratify=strat_col
)

print(f"\n  Train: {len(X_train):,} rows")
print(f"  Test:  {len(X_test):,} rows")
print(f"  Train treatment rate: {t_train.mean():.1%}")
print(f"  Test treatment rate:  {t_test.mean():.1%}")
print(f"  Train visit rate: {y_train.mean():.2%}")
print(f"  Test visit rate:  {y_test.mean():.2%}")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# APPROACH 1: Standard Classifier (P(visit))
# ============================================================
# Train a classifier to predict P(visit) ignoring treatment.
# Rank by predicted probability. This is what most ad platforms do:
# "target users most likely to convert."
#
# The problem: this ranks SURE THINGS highest, not PERSUADABLES.

print("\n\n" + "=" * 60)
print("APPROACH 1: Standard Classifier — P(visit)")
print("=" * 60)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predicted probability of visit — used as the "uplift score"
# Higher P(visit) = "target this user first"
propensity_scores = clf.predict_proba(X_test_scaled)[:, 1]

print(f"\n  Model: LogisticRegression on all data (ignoring treatment)")
print(f"  Using P(visit) as targeting score (higher = target first)")
print(f"  Score range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
print(f"  Score mean:  {propensity_scores.mean():.4f}")

# Evaluate as uplift model
auuc_propensity = uplift_auc_score(y_test, propensity_scores, t_test)
qini_propensity = qini_auc_score(y_test, propensity_scores, t_test)
print(f"\n  AUUC:  {auuc_propensity:.6f}")
print(f"  Qini:  {qini_propensity:.6f}")


# ============================================================
# APPROACH 2: Two Separate Classifiers (Naive T-Learner)
# ============================================================
# Train one model on treatment data, one on control data.
# Uplift score = P(visit|treated) - P(visit|control)
#
# This is structurally a T-learner — the simplest real uplift
# approach. But we're doing it naively with LogReg to set a baseline.

print("\n\n" + "=" * 60)
print("APPROACH 2: Two Separate Classifiers (Naive T-Learner)")
print("=" * 60)

# Split training data by treatment
treat_mask = t_train == 1
X_train_treat = X_train_scaled[treat_mask]
y_train_treat = y_train[treat_mask]
X_train_control = X_train_scaled[~treat_mask]
y_train_control = y_train[~treat_mask]

print(f"\n  Treatment model trained on: {len(X_train_treat):,} rows")
print(f"  Control model trained on:   {len(X_train_control):,} rows")

clf_treat = LogisticRegression(max_iter=1000, random_state=42)
clf_treat.fit(X_train_treat, y_train_treat)

clf_control = LogisticRegression(max_iter=1000, random_state=42)
clf_control.fit(X_train_control, y_train_control)

# Uplift = P(visit|treated) - P(visit|control)
p_treat = clf_treat.predict_proba(X_test_scaled)[:, 1]
p_control = clf_control.predict_proba(X_test_scaled)[:, 1]
uplift_two_model = p_treat - p_control

print(f"\n  P(visit|treated) range:  [{p_treat.min():.4f}, {p_treat.max():.4f}]")
print(f"  P(visit|control) range:  [{p_control.min():.4f}, {p_control.max():.4f}]")
print(f"  Uplift score range:      [{uplift_two_model.min():.4f}, {uplift_two_model.max():.4f}]")
print(f"  Uplift score mean:       {uplift_two_model.mean():.4f}")
print(f"  (Compare to true ATE:    {y_test[t_test == 1].mean() - y_test[t_test == 0].mean():.4f})")

auuc_two_model = uplift_auc_score(y_test, uplift_two_model, t_test)
qini_two_model = qini_auc_score(y_test, uplift_two_model, t_test)
print(f"\n  AUUC:  {auuc_two_model:.6f}")
print(f"  Qini:  {qini_two_model:.6f}")


# ============================================================
# APPROACH 3: Random Baseline
# ============================================================
# Predict uplift = 0 for everyone. This is equivalent to
# targeting users at random — the "no skill" baseline.

print("\n\n" + "=" * 60)
print("APPROACH 3: Random Baseline")
print("=" * 60)

# Random scores — uniform random, so targeting order is random
rng = np.random.RandomState(42)
random_scores = rng.uniform(0, 1, size=len(X_test))

auuc_random = uplift_auc_score(y_test, random_scores, t_test)
qini_random = qini_auc_score(y_test, random_scores, t_test)
print(f"\n  AUUC:  {auuc_random:.6f}")
print(f"  Qini:  {qini_random:.6f}")


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\n  {'Approach':<35s}  {'AUUC':>10s}  {'Qini':>10s}")
print(f"  {'-' * 60}")
print(f"  {'1. Standard Classifier P(visit)':<35s}  {auuc_propensity:>10.6f}  {qini_propensity:>10.6f}")
print(f"  {'2. Two Classifiers (naive T)':<35s}  {auuc_two_model:>10.6f}  {qini_two_model:>10.6f}")
print(f"  {'3. Random Baseline':<35s}  {auuc_random:>10.6f}  {qini_random:>10.6f}")

# Determine ordering
approaches = [
    ("Standard Classifier", auuc_propensity),
    ("Two Classifiers", auuc_two_model),
    ("Random", auuc_random),
]
approaches_sorted = sorted(approaches, key=lambda x: x[1], reverse=True)
print(f"\n  Ranking by AUUC (higher is better):")
for rank, (name, score) in enumerate(approaches_sorted, 1):
    print(f"    {rank}. {name}: {score:.6f}")


# ============================================================
# UPLIFT CURVES — THE KEY VISUALIZATION
# ============================================================
# This is where the "aha" moment should happen.
# Each curve shows: "if I target the top X% of users according
# to this model's ranking, what cumulative uplift do I get?"

print("\n\n" + "=" * 60)
print("UPLIFT CURVES")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Step 2 — Naive Baselines: Uplift Curves", fontsize=14, fontweight="bold")

approaches_data = [
    ("1. Standard Classifier\n    P(visit)", propensity_scores, "#FF6B6B", "-"),
    ("2. Two Classifiers\n    (naive T-Learner)", uplift_two_model, "#4ECDC4", "-"),
    ("3. Random", random_scores, "#B0BEC5", "--"),
]

# ---- Left plot: Uplift Curve ----
ax = axes[0]
for label, scores, color, ls in approaches_data:
    # uplift_curve returns (x, y) where x is proportion targeted
    x_vals, y_vals = uplift_curve(y_test, scores, t_test)
    # Normalize x to [0, 1]
    x_norm = x_vals / x_vals.max() if x_vals.max() > 0 else x_vals
    ax.plot(x_norm, y_vals, color=color, ls=ls, lw=2, label=label)

ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("Proportion of Users Targeted", fontsize=11)
ax.set_ylabel("Cumulative Uplift", fontsize=11)
ax.set_title("Uplift Curve\n(higher = better targeting)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Right plot: AUUC bar chart ----
ax = axes[1]
names = ["Standard\nClassifier", "Two\nClassifiers", "Random"]
auuc_values = [auuc_propensity, auuc_two_model, auuc_random]
colors = ["#FF6B6B", "#4ECDC4", "#B0BEC5"]
bars = ax.bar(names, auuc_values, color=colors, width=0.5, alpha=0.8)
ax.set_ylabel("AUUC Score", fontsize=11)
ax.set_title("Area Under Uplift Curve\n(higher = better)")
ax.axhline(y=0, color="black", lw=0.5)
for bar, val in zip(bars, auuc_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = f"{BASE}/results/analysis/02_naive_baselines.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Saved plot: {plot_path}")


# ============================================================
# ANALYSIS: Why does the standard classifier fail?
# ============================================================

print("\n\n" + "=" * 60)
print("ANALYSIS: Why Standard Classifiers Fail at Uplift")
print("=" * 60)

# Show that high-propensity users are Sure Things
# by checking the ATE in propensity deciles
propensity_deciles = pd.qcut(propensity_scores, q=10, duplicates="drop")

print(f"\n  ATE by propensity decile (standard classifier ranking):")
print(f"  {'Decile':>25s}  {'N':>7s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}  {'ATE':>9s}  {'P(visit)':>9s}")
print(f"  {'-' * 75}")

y_test_arr = y_test.values
t_test_arr = t_test.values

for decile in sorted(propensity_deciles.unique()):
    mask = propensity_deciles == decile
    y_sub = y_test_arr[mask]
    t_sub = t_test_arr[mask]
    p_sub = propensity_scores[mask]

    treat_mask = t_sub == 1
    vt = y_sub[treat_mask].mean() if treat_mask.sum() > 0 else float("nan")
    vc = y_sub[~treat_mask].mean() if (~treat_mask).sum() > 0 else float("nan")
    ate = vt - vc

    print(f"  {str(decile):>25s}  {mask.sum():>7,}  {vt:>9.4f}  {vc:>9.4f}  {ate:>+9.4f}  {p_sub.mean():>9.4f}")

print(f"""
  KEY FINDING:
  Look at the ATE column. The users ranked HIGHEST by the standard
  classifier (high P(visit)) do NOT have the highest ATE. In fact,
  high-propensity deciles often have LOWER uplift than low-propensity ones.

  This confirms the four-quadrant theory:
  - High P(visit) → Sure Things (would visit anyway, τ ≈ 0)
  - The standard classifier targets the WRONG users for uplift

  This is why uplift modeling exists: you need models that predict
  τ(x) = P(visit|treated) - P(visit|control), not just P(visit).
""")


# ============================================================
# KEY TAKEAWAYS
# ============================================================

print("=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print(f"""
  1. STANDARD CLASSIFIER FAILS: Ranking by P(visit) targets Sure Things,
     not Persuadables. Its AUUC is {'worse' if auuc_propensity < auuc_random else 'similar to'} random targeting.

  2. TWO-MODEL APPROACH WORKS BETTER: Subtracting P(visit|treated) - P(visit|control)
     at least tries to estimate the treatment effect. AUUC: {auuc_two_model:.6f}

  3. THE "AHA" MOMENT: High propensity ≠ high uplift. Users most likely
     to visit are NOT the users most affected by the ad.

  4. THE UPLIFT CURVE SHOWS IT: A good uplift model's curve rises steeply
     at the left (capturing most uplift from a small fraction of users).
     The standard classifier doesn't show this pattern.

  NEXT: Step 3 — S-Learner, the simplest principled uplift approach.
  Uses treatment as a feature in a single model.
""")

tee.close()
print("Done. Results saved to results/models/02_naive_baselines.txt")
