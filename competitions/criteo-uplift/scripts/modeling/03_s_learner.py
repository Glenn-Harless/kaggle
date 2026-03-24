"""
Criteo Uplift Modeling: Step 3 — S-Learner
The simplest real uplift model: train ONE model with treatment as a feature.
  - P(visit) = f(f0, ..., f11, treatment)
  - τ(x) = f(x, treatment=1) - f(x, treatment=0)

Uses LightGBM as the base learner (can capture feature × treatment interactions).
Compare against Step 2 naive baselines.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_curve
from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/03_s_learner.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 3 — S-Learner")
print("=" * 60)


# ============================================================
# DATA LOADING & TRAIN/TEST SPLIT
# ============================================================
# Same split as Step 2 for fair comparison.

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

strat_col = treatment * 2 + y
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42, stratify=strat_col
)

print(f"\n  Train: {len(X_train):,} rows")
print(f"  Test:  {len(X_test):,} rows")


# ============================================================
# S-LEARNER: TRAIN
# ============================================================
# The S-learner adds treatment as the 13th feature column.
# The model learns how treatment interacts with other features
# to affect the visit outcome.

print("\n\n" + "=" * 60)
print("S-LEARNER: Training")
print("=" * 60)

# Add treatment as a feature
X_train_s = X_train.copy()
X_train_s["treatment"] = t_train.values

X_test_s = X_test.copy()
X_test_s["treatment"] = t_test.values

s_features = feature_cols + ["treatment"]

# LightGBM parameters — moderate complexity, tuned for uplift signal
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}
n_rounds = 300

train_data = lgb.Dataset(X_train_s[s_features], label=y_train)

model_s = lgb.train(
    params,
    train_data,
    num_boost_round=n_rounds,
)

print(f"\n  Model: LightGBM S-learner")
print(f"  Features: {s_features}")
print(f"  Rounds: {n_rounds}")
print(f"  Num leaves: {params['num_leaves']}")


# ============================================================
# S-LEARNER: PREDICT UPLIFT
# ============================================================
# For each test user, predict twice:
#   1. With treatment = 1 → P(visit | treated)
#   2. With treatment = 0 → P(visit | control)
#   3. Uplift = (1) - (2)

print("\n\n" + "=" * 60)
print("S-LEARNER: Predicting Uplift")
print("=" * 60)

# Create two copies of test features: one with treatment=1, one with treatment=0
X_test_t1 = X_test.copy()
X_test_t1["treatment"] = 1

X_test_t0 = X_test.copy()
X_test_t0["treatment"] = 0

p_treated = model_s.predict(X_test_t1[s_features])
p_control = model_s.predict(X_test_t0[s_features])
uplift_s = p_treated - p_control

print(f"\n  P(visit|treated) range:  [{p_treated.min():.4f}, {p_treated.max():.4f}]")
print(f"  P(visit|control) range:  [{p_control.min():.4f}, {p_control.max():.4f}]")
print(f"  Uplift score range:      [{uplift_s.min():.4f}, {uplift_s.max():.4f}]")
print(f"  Uplift score mean:       {uplift_s.mean():.4f}")
print(f"  (True ATE:               {y_test[t_test == 1].mean() - y_test[t_test == 0].mean():.4f})")

# What fraction of users have negative predicted uplift?
n_negative = (uplift_s < 0).sum()
print(f"\n  Users with negative predicted uplift: {n_negative:,} ({n_negative/len(uplift_s):.1%})")
print(f"  These are predicted Sleeping Dogs — users the ad may hurt.")


# ============================================================
# S-LEARNER: FEATURE IMPORTANCE
# ============================================================
# Critical question: how important is the treatment feature?
# If it's ranked low, the model might be ignoring it (regularization bias).

print("\n\n" + "=" * 60)
print("S-LEARNER: Feature Importance")
print("=" * 60)

importance = model_s.feature_importance(importance_type="gain")
feature_imp = pd.DataFrame({
    "feature": s_features,
    "importance": importance,
}).sort_values("importance", ascending=False)

print(f"\n  Feature importance (gain):")
print(f"  {'Feature':>12s}  {'Importance':>12s}  {'Rank':>6s}")
print(f"  {'-' * 35}")
for rank, (_, row) in enumerate(feature_imp.iterrows(), 1):
    marker = " ← treatment" if row["feature"] == "treatment" else ""
    print(f"  {row['feature']:>12s}  {row['importance']:>12.1f}  {rank:>6d}{marker}")

treatment_rank = feature_imp.index.tolist().index(
    feature_imp[feature_imp["feature"] == "treatment"].index[0]
) + 1
print(f"\n  Treatment feature rank: {treatment_rank} out of {len(s_features)}")
if treatment_rank <= 3:
    print(f"  Treatment is a top feature — the model IS learning the treatment effect.")
elif treatment_rank <= 7:
    print(f"  Treatment is mid-ranked — the model captures SOME treatment effect.")
else:
    print(f"  Treatment is low-ranked — possible regularization bias (S-learner weakness).")


# ============================================================
# EVALUATION: AUUC and Qini
# ============================================================

print("\n\n" + "=" * 60)
print("EVALUATION: AUUC and Qini")
print("=" * 60)

auuc_s = uplift_auc_score(y_test, uplift_s, t_test)
qini_s = qini_auc_score(y_test, uplift_s, t_test)

print(f"\n  S-Learner AUUC:  {auuc_s:.6f}")
print(f"  S-Learner Qini:  {qini_s:.6f}")

# Step 2 baselines for comparison (recompute for consistency)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline 1: Standard classifier
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)
propensity_scores = clf.predict_proba(X_test_scaled)[:, 1]
auuc_propensity = uplift_auc_score(y_test, propensity_scores, t_test)
qini_propensity = qini_auc_score(y_test, propensity_scores, t_test)

# Baseline 2: Two classifiers (naive T-learner)
treat_mask = t_train == 1
clf_t = LogisticRegression(max_iter=1000, random_state=42)
clf_t.fit(X_train_scaled[treat_mask], y_train[treat_mask])
clf_c = LogisticRegression(max_iter=1000, random_state=42)
clf_c.fit(X_train_scaled[~treat_mask], y_train[~treat_mask])
uplift_two = clf_t.predict_proba(X_test_scaled)[:, 1] - clf_c.predict_proba(X_test_scaled)[:, 1]
auuc_two = uplift_auc_score(y_test, uplift_two, t_test)
qini_two = qini_auc_score(y_test, uplift_two, t_test)

# Baseline 3: Random
rng = np.random.RandomState(42)
random_scores = rng.uniform(0, 1, size=len(X_test))
auuc_random = uplift_auc_score(y_test, random_scores, t_test)
qini_random = qini_auc_score(y_test, random_scores, t_test)

print(f"\n  {'Approach':<35s}  {'AUUC':>10s}  {'Qini':>10s}")
print(f"  {'-' * 60}")
print(f"  {'S-Learner (LightGBM)':<35s}  {auuc_s:>10.6f}  {qini_s:>10.6f}")
print(f"  {'Standard Classifier (LogReg)':<35s}  {auuc_propensity:>10.6f}  {qini_propensity:>10.6f}")
print(f"  {'Two Classifiers (LogReg)':<35s}  {auuc_two:>10.6f}  {qini_two:>10.6f}")
print(f"  {'Random':<35s}  {auuc_random:>10.6f}  {qini_random:>10.6f}")

# Improvement over best baseline
best_baseline = max(auuc_propensity, auuc_two)
improvement = (auuc_s - best_baseline) / abs(best_baseline) * 100 if best_baseline != 0 else float("inf")
print(f"\n  S-Learner improvement over best baseline: {improvement:+.1f}%")


# ============================================================
# UPLIFT DECILE ANALYSIS
# ============================================================
# Bin users by predicted uplift and check if actual ATE follows.

print("\n\n" + "=" * 60)
print("UPLIFT DECILE ANALYSIS")
print("=" * 60)

uplift_deciles = pd.qcut(uplift_s, q=10, duplicates="drop")

print(f"\n  ATE by S-learner predicted uplift decile:")
print(f"  {'Decile':>30s}  {'N':>7s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}  {'ATE':>9s}  {'Pred τ':>9s}")
print(f"  {'-' * 80}")

y_test_arr = y_test.values
t_test_arr = t_test.values

for decile in sorted(uplift_deciles.unique()):
    mask = uplift_deciles == decile
    y_sub = y_test_arr[mask]
    t_sub = t_test_arr[mask]
    pred_sub = uplift_s[mask]

    treat_mask = t_sub == 1
    vt = y_sub[treat_mask].mean() if treat_mask.sum() > 0 else float("nan")
    vc = y_sub[~treat_mask].mean() if (~treat_mask).sum() > 0 else float("nan")
    ate = vt - vc

    print(f"  {str(decile):>30s}  {mask.sum():>7,}  {vt:>9.4f}  {vc:>9.4f}  {ate:>+9.4f}  {pred_sub.mean():>+9.4f}")

print(f"""
  KEY CHECK: If the model is working, the actual ATE should INCREASE
  from bottom decile to top decile — the model should rank users with
  genuinely higher treatment effects at the top.
""")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Step 3 — S-Learner: Uplift Performance", fontsize=14, fontweight="bold")

# ---- Plot 1: Uplift curves ----
ax = axes[0]
approaches_data = [
    ("S-Learner (LightGBM)", uplift_s, "#2196F3", "-", 2.5),
    ("Standard Classifier", propensity_scores, "#FF6B6B", "--", 1.5),
    ("Two Classifiers", uplift_two, "#4ECDC4", "--", 1.5),
    ("Random", random_scores, "#B0BEC5", ":", 1.0),
]

for label, scores, color, ls, lw in approaches_data:
    x_vals, y_vals = uplift_curve(y_test, scores, t_test)
    x_norm = x_vals / x_vals.max() if x_vals.max() > 0 else x_vals
    ax.plot(x_norm, y_vals, color=color, ls=ls, lw=lw, label=label)

ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("Proportion of Users Targeted")
ax.set_ylabel("Cumulative Uplift")
ax.set_title("Uplift Curves")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Plot 2: AUUC bar chart ----
ax = axes[1]
names = ["S-Learner\n(LightGBM)", "Standard\nClassifier", "Two\nClassifiers", "Random"]
auuc_values = [auuc_s, auuc_propensity, auuc_two, auuc_random]
colors = ["#2196F3", "#FF6B6B", "#4ECDC4", "#B0BEC5"]
bars = ax.bar(names, auuc_values, color=colors, width=0.5, alpha=0.8)
ax.set_ylabel("AUUC Score")
ax.set_title("AUUC Comparison")
ax.axhline(y=0, color="black", lw=0.5)
for bar, val in zip(bars, auuc_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# ---- Plot 3: Predicted uplift distribution ----
ax = axes[2]
ax.hist(uplift_s, bins=80, color="#2196F3", alpha=0.7, edgecolor="white", linewidth=0.3)
ax.axvline(x=0, color="red", lw=1.5, ls="--", label="Zero uplift")
ax.axvline(x=uplift_s.mean(), color="black", lw=1.5, ls="-",
           label=f"Mean uplift ({uplift_s.mean():.4f})")
ax.set_xlabel("Predicted Uplift Score")
ax.set_ylabel("Count")
ax.set_title("Distribution of Predicted Uplift")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = f"{BASE}/results/analysis/03_s_learner.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved plot: {plot_path}")


# ============================================================
# KEY TAKEAWAYS
# ============================================================

print("\n\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print(f"""
  1. S-LEARNER AUUC: {auuc_s:.6f} (vs best baseline: {best_baseline:.6f})
     {'Improvement' if auuc_s > best_baseline else 'No improvement'} over naive baselines.

  2. TREATMENT FEATURE RANK: {treatment_rank}/{len(s_features)} by importance (gain).
     {'Model IS learning treatment interactions.' if treatment_rank <= 5 else 'Possible regularization bias — treatment may be underweighted.'}

  3. UPLIFT DISTRIBUTION: Range [{uplift_s.min():.4f}, {uplift_s.max():.4f}].
     {n_negative:,} users ({n_negative/len(uplift_s):.1%}) predicted as Sleeping Dogs (negative uplift).

  4. The uplift decile table shows whether predicted uplift correlates
     with actual treatment effect — the key validation of any uplift model.

  NEXT: Step 4 — T-Learner with LightGBM. Forces treatment effect capture
  by training separate models on treatment and control data.
""")

tee.close()
print("Done. Results saved to results/models/03_s_learner.txt")
