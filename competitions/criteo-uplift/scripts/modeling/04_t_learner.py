"""
Criteo Uplift Modeling: Step 4 — T-Learner
Two separate LightGBM models: one on treatment data, one on control data.
  - Model_T: P(visit | treated) = f(f0, ..., f11)  trained on treatment group
  - Model_C: P(visit | control) = f(f0, ..., f11)  trained on control group
  - τ(x) = Model_T(x) - Model_C(x)

Unlike the S-learner, the treatment effect CAN'T be regularized away —
it's baked into the structure (two separate models).

Tradeoff: the control model only gets ~15% of the data.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_curve
from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/04_t_learner.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 4 — T-Learner")
print("=" * 60)


# ============================================================
# DATA LOADING & TRAIN/TEST SPLIT
# ============================================================
# Same split as Steps 2-3 for fair comparison.

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
# T-LEARNER: TRAIN TWO MODELS
# ============================================================

print("\n\n" + "=" * 60)
print("T-LEARNER: Training Two Separate Models")
print("=" * 60)

# Split training data by treatment assignment
treat_mask = t_train == 1
X_train_treat = X_train[treat_mask]
y_train_treat = y_train[treat_mask]
X_train_control = X_train[~treat_mask]
y_train_control = y_train[~treat_mask]

print(f"\n  Treatment training set: {len(X_train_treat):,} rows ({len(X_train_treat)/len(X_train):.1%})")
print(f"  Control training set:   {len(X_train_control):,} rows ({len(X_train_control)/len(X_train):.1%})")
print(f"  Treatment visit rate:   {y_train_treat.mean():.4%}")
print(f"  Control visit rate:     {y_train_control.mean():.4%}")

# Same LightGBM params as S-learner for fair comparison
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

# Model T: trained on treatment group only
print(f"\n  Training treatment model ({len(X_train_treat):,} rows)...")
train_data_t = lgb.Dataset(X_train_treat[feature_cols], label=y_train_treat)
model_t = lgb.train(params, train_data_t, num_boost_round=n_rounds)

# Model C: trained on control group only
print(f"  Training control model ({len(X_train_control):,} rows)...")
train_data_c = lgb.Dataset(X_train_control[feature_cols], label=y_train_control)
model_c = lgb.train(params, train_data_c, num_boost_round=n_rounds)


# ============================================================
# T-LEARNER: PREDICT UPLIFT
# ============================================================

print("\n\n" + "=" * 60)
print("T-LEARNER: Predicting Uplift")
print("=" * 60)

# Both models see the SAME test features — just trained on different data
p_treated = model_t.predict(X_test[feature_cols])
p_control = model_c.predict(X_test[feature_cols])
uplift_t = p_treated - p_control

print(f"\n  P(visit|treated) range:  [{p_treated.min():.4f}, {p_treated.max():.4f}]")
print(f"  P(visit|control) range:  [{p_control.min():.4f}, {p_control.max():.4f}]")
print(f"  Uplift score range:      [{uplift_t.min():.4f}, {uplift_t.max():.4f}]")
print(f"  Uplift score mean:       {uplift_t.mean():.4f}")
print(f"  (True ATE:               {y_test[t_test == 1].mean() - y_test[t_test == 0].mean():.4f})")

n_negative = (uplift_t < 0).sum()
print(f"\n  Users with negative predicted uplift: {n_negative:,} ({n_negative/len(uplift_t):.1%})")


# ============================================================
# T-LEARNER: FEATURE IMPORTANCE COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("T-LEARNER: Feature Importance (Treatment vs Control)")
print("=" * 60)

imp_t = model_t.feature_importance(importance_type="gain")
imp_c = model_c.feature_importance(importance_type="gain")

print(f"\n  {'Feature':>8s}  {'Treat Imp':>12s}  {'Rank(T)':>8s}  {'Control Imp':>12s}  {'Rank(C)':>8s}")
print(f"  {'-' * 55}")

rank_t = np.argsort(-imp_t) + 1
rank_c = np.argsort(-imp_c) + 1

for i, col in enumerate(feature_cols):
    rt = np.where(np.argsort(-imp_t) == i)[0][0] + 1
    rc = np.where(np.argsort(-imp_c) == i)[0][0] + 1
    print(f"  {col:>8s}  {imp_t[i]:>12.1f}  {rt:>8d}  {imp_c[i]:>12.1f}  {rc:>8d}")

print(f"""
  Features with DIFFERENT importance rankings between treatment and
  control models suggest interaction with the treatment — these features
  moderate the treatment effect.
""")


# ============================================================
# EVALUATION: AUUC and Qini + ALL BASELINES
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION: Full Comparison")
print("=" * 60)

auuc_t = uplift_auc_score(y_test, uplift_t, t_test)
qini_t = qini_auc_score(y_test, uplift_t, t_test)

# --- Recompute all prior approaches for fair comparison ---

# S-Learner (LightGBM)
X_train_s = X_train.copy()
X_train_s["treatment"] = t_train.values
s_features = feature_cols + ["treatment"]
train_data_s = lgb.Dataset(X_train_s[s_features], label=y_train)
model_s = lgb.train(params, train_data_s, num_boost_round=n_rounds)
X_test_t1 = X_test.copy(); X_test_t1["treatment"] = 1
X_test_t0 = X_test.copy(); X_test_t0["treatment"] = 0
uplift_s = model_s.predict(X_test_t1[s_features]) - model_s.predict(X_test_t0[s_features])
auuc_s = uplift_auc_score(y_test, uplift_s, t_test)
qini_s = qini_auc_score(y_test, uplift_s, t_test)

# Standard Classifier (LogReg)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_sc, y_train)
propensity_scores = clf.predict_proba(X_test_sc)[:, 1]
auuc_prop = uplift_auc_score(y_test, propensity_scores, t_test)
qini_prop = qini_auc_score(y_test, propensity_scores, t_test)

# Two Classifiers (LogReg — naive T-learner)
clf_t_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_t_lr.fit(X_train_sc[treat_mask], y_train[treat_mask])
clf_c_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_c_lr.fit(X_train_sc[~treat_mask], y_train[~treat_mask])
uplift_lr = clf_t_lr.predict_proba(X_test_sc)[:, 1] - clf_c_lr.predict_proba(X_test_sc)[:, 1]
auuc_lr = uplift_auc_score(y_test, uplift_lr, t_test)
qini_lr = qini_auc_score(y_test, uplift_lr, t_test)

# Random
rng = np.random.RandomState(42)
random_scores = rng.uniform(0, 1, size=len(X_test))
auuc_rand = uplift_auc_score(y_test, random_scores, t_test)
qini_rand = qini_auc_score(y_test, random_scores, t_test)

print(f"\n  {'Approach':<35s}  {'AUUC':>10s}  {'Qini':>10s}")
print(f"  {'-' * 60}")
print(f"  {'T-Learner (LightGBM)':<35s}  {auuc_t:>10.6f}  {qini_t:>10.6f}")
print(f"  {'S-Learner (LightGBM)':<35s}  {auuc_s:>10.6f}  {qini_s:>10.6f}")
print(f"  {'Standard Classifier (LogReg)':<35s}  {auuc_prop:>10.6f}  {qini_prop:>10.6f}")
print(f"  {'Two Classifiers (LogReg)':<35s}  {auuc_lr:>10.6f}  {qini_lr:>10.6f}")
print(f"  {'Random':<35s}  {auuc_rand:>10.6f}  {qini_rand:>10.6f}")

# Rank
all_approaches = [
    ("T-Learner (LightGBM)", auuc_t),
    ("S-Learner (LightGBM)", auuc_s),
    ("Standard Classifier (LogReg)", auuc_prop),
    ("Two Classifiers (LogReg)", auuc_lr),
    ("Random", auuc_rand),
]
all_sorted = sorted(all_approaches, key=lambda x: x[1], reverse=True)
print(f"\n  Ranking by AUUC (higher is better):")
for rank, (name, score) in enumerate(all_sorted, 1):
    print(f"    {rank}. {name}: {score:.6f}")


# ============================================================
# UPLIFT DECILE ANALYSIS
# ============================================================

print("\n\n" + "=" * 60)
print("UPLIFT DECILE ANALYSIS: T-Learner vs S-Learner")
print("=" * 60)

y_test_arr = y_test.values
t_test_arr = t_test.values

for model_name, uplift_scores in [("T-Learner", uplift_t), ("S-Learner", uplift_s)]:
    deciles = pd.qcut(uplift_scores, q=10, duplicates="drop")

    print(f"\n  {model_name} — ATE by predicted uplift decile:")
    print(f"  {'Decile':>30s}  {'N':>7s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}  {'ATE':>9s}  {'Pred τ':>9s}")
    print(f"  {'-' * 80}")

    for decile in sorted(deciles.unique()):
        mask = deciles == decile
        y_sub = y_test_arr[mask]
        t_sub = t_test_arr[mask]
        pred_sub = uplift_scores[mask]

        tm = t_sub == 1
        vt = y_sub[tm].mean() if tm.sum() > 0 else float("nan")
        vc = y_sub[~tm].mean() if (~tm).sum() > 0 else float("nan")
        ate = vt - vc

        print(f"  {str(decile):>30s}  {mask.sum():>7,}  {vt:>9.4f}  {vc:>9.4f}  {ate:>+9.4f}  {pred_sub.mean():>+9.4f}")

print(f"""
  KEY COMPARISON:
  Look at the ATE spread from bottom to top decile for each model.
  A wider spread means the model better separates high-uplift from
  low-uplift users.
""")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Step 4 — T-Learner: Uplift Performance", fontsize=14, fontweight="bold")

# ---- Plot 1: Uplift curves ----
ax = axes[0]
approaches_plot = [
    ("T-Learner (LightGBM)", uplift_t, "#FF9800", "-", 2.5),
    ("S-Learner (LightGBM)", uplift_s, "#2196F3", "--", 1.5),
    ("Standard Classifier", propensity_scores, "#FF6B6B", "--", 1.5),
    ("Two Classifiers (LogReg)", uplift_lr, "#4ECDC4", "--", 1.5),
    ("Random", random_scores, "#B0BEC5", ":", 1.0),
]

for label, scores, color, ls, lw in approaches_plot:
    x_vals, y_vals = uplift_curve(y_test, scores, t_test)
    x_norm = x_vals / x_vals.max() if x_vals.max() > 0 else x_vals
    ax.plot(x_norm, y_vals, color=color, ls=ls, lw=lw, label=label)

ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("Proportion of Users Targeted")
ax.set_ylabel("Cumulative Uplift")
ax.set_title("Uplift Curves (all approaches)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---- Plot 2: AUUC bar chart ----
ax = axes[1]
names = ["T-Learner\n(LGBM)", "S-Learner\n(LGBM)", "Std\nClassifier", "Two Clf\n(LogReg)", "Random"]
auuc_vals = [auuc_t, auuc_s, auuc_prop, auuc_lr, auuc_rand]
colors = ["#FF9800", "#2196F3", "#FF6B6B", "#4ECDC4", "#B0BEC5"]
bars = ax.bar(names, auuc_vals, color=colors, width=0.5, alpha=0.8)
ax.set_ylabel("AUUC Score")
ax.set_title("AUUC Comparison")
ax.axhline(y=0, color="black", lw=0.5)
for bar, val in zip(bars, auuc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ---- Plot 3: Uplift distribution comparison ----
ax = axes[2]
ax.hist(uplift_t, bins=80, alpha=0.6, color="#FF9800", label="T-Learner", density=True)
ax.hist(uplift_s, bins=80, alpha=0.6, color="#2196F3", label="S-Learner", density=True)
ax.axvline(x=0, color="red", lw=1.5, ls="--", label="Zero uplift")
ax.set_xlabel("Predicted Uplift Score")
ax.set_ylabel("Density")
ax.set_title("Predicted Uplift Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = f"{BASE}/results/analysis/04_t_learner.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved plot: {plot_path}")


# ============================================================
# KEY TAKEAWAYS
# ============================================================

print("\n\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

t_vs_s = "better" if auuc_t > auuc_s else "worse" if auuc_t < auuc_s else "equal"
best_auuc = max(auuc_t, auuc_s, auuc_prop, auuc_lr)
best_name = [n for n, v in all_approaches if v == best_auuc][0]

print(f"""
  1. T-LEARNER AUUC: {auuc_t:.6f} — {t_vs_s} than S-Learner ({auuc_s:.6f})

  2. BEST APPROACH SO FAR: {best_name} (AUUC = {best_auuc:.6f})

  3. CONTROL MODEL had {len(X_train_control):,} rows to learn from
     (vs {len(X_train_treat):,} for treatment). The 85/15 imbalance
     means the control predictions may be noisier.

  4. UPLIFT DISTRIBUTION: Compare the rightmost plot — does the T-learner
     spread predictions more than the S-learner's spike at zero?

  5. DECILE TABLES: Compare ATE spread (top vs bottom decile) between
     T-learner and S-learner. Wider spread = better discrimination.

  NEXT: Step 5 — X-Learner, designed specifically for treatment group
  imbalance like our 85/15 split.
""")

tee.close()
print("Done. Results saved to results/models/04_t_learner.txt")
