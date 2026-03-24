"""
Criteo Uplift Modeling: Step 5 — X-Learner
Three-stage meta-learner designed for imbalanced treatment groups (85/15):

  Stage 1: Train outcome models (same as T-learner)
    Model_T on treatment data, Model_C on control data

  Stage 2: Cross-impute individual treatment effects
    D_treated = actual_outcome - Model_C(features)  (for treated users)
    D_control = Model_T(features) - actual_outcome   (for control users)

  Stage 3: Model the imputed effects + combine with propensity weights
    τ_model_1 predicts D_treated from features
    τ_model_0 predicts D_control from features
    τ_final = g(x) · τ_model_0(x) + (1 - g(x)) · τ_model_1(x)
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

tee = Tee(f"{BASE}/results/models/05_x_learner.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 5 — X-Learner")
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

strat_col = treatment * 2 + y
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42, stratify=strat_col
)

print(f"\n  Train: {len(X_train):,} rows")
print(f"  Test:  {len(X_test):,} rows")

treat_mask = t_train == 1
X_train_treat = X_train[treat_mask]
y_train_treat = y_train[treat_mask]
X_train_control = X_train[~treat_mask]
y_train_control = y_train[~treat_mask]


# ============================================================
# STAGE 1: Outcome Models (same as T-Learner)
# ============================================================

print("\n\n" + "=" * 60)
print("STAGE 1: Outcome Models (T-Learner base)")
print("=" * 60)

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

print(f"\n  Training treatment outcome model ({len(X_train_treat):,} rows)...")
model_t = lgb.train(
    params,
    lgb.Dataset(X_train_treat[feature_cols], label=y_train_treat),
    num_boost_round=n_rounds,
)

print(f"  Training control outcome model ({len(X_train_control):,} rows)...")
model_c = lgb.train(
    params,
    lgb.Dataset(X_train_control[feature_cols], label=y_train_control),
    num_boost_round=n_rounds,
)

print(f"  Stage 1 complete — same as T-learner so far.")


# ============================================================
# STAGE 2: Cross-Impute Individual Treatment Effects
# ============================================================
# This is the new part. For each training user, impute their
# individual treatment effect using real outcome + model prediction.

print("\n\n" + "=" * 60)
print("STAGE 2: Cross-Impute Treatment Effects")
print("=" * 60)

# For treated users: D = actual_outcome - Model_C(features)
# "What did happen" minus "what would have happened without the ad"
d_treated = y_train_treat.values - model_c.predict(X_train_treat[feature_cols])

# For control users: D = Model_T(features) - actual_outcome
# "What would have happened with the ad" minus "what did happen"
d_control = model_t.predict(X_train_control[feature_cols]) - y_train_control.values

print(f"\n  D_treated (imputed effects for treatment group):")
print(f"    N:      {len(d_treated):,}")
print(f"    Mean:   {d_treated.mean():.4f}")
print(f"    Std:    {d_treated.std():.4f}")
print(f"    Range:  [{d_treated.min():.4f}, {d_treated.max():.4f}]")

print(f"\n  D_control (imputed effects for control group):")
print(f"    N:      {len(d_control):,}")
print(f"    Mean:   {d_control.mean():.4f}")
print(f"    Std:    {d_control.std():.4f}")
print(f"    Range:  [{d_control.min():.4f}, {d_control.max():.4f}]")

print(f"\n  Both means should be close to the true ATE (~0.0103):")
print(f"    D_treated mean: {d_treated.mean():.4f}")
print(f"    D_control mean: {d_control.mean():.4f}")


# ============================================================
# STAGE 3: Model the Imputed Effects
# ============================================================
# Train regression models to predict D from features.
# These models learn the CATE — how treatment effect varies with features.

print("\n\n" + "=" * 60)
print("STAGE 3: Model the Imputed Effects")
print("=" * 60)

# Switch to regression objective since D values are continuous
params_reg = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# τ_model_1: predict D_treated from features (trained on treatment users)
print(f"\n  Training τ_model_1 on treatment users' imputed effects ({len(d_treated):,} rows)...")
tau_model_1 = lgb.train(
    params_reg,
    lgb.Dataset(X_train_treat[feature_cols], label=d_treated),
    num_boost_round=n_rounds,
)

# τ_model_0: predict D_control from features (trained on control users)
print(f"  Training τ_model_0 on control users' imputed effects ({len(d_control):,} rows)...")
tau_model_0 = lgb.train(
    params_reg,
    lgb.Dataset(X_train_control[feature_cols], label=d_control),
    num_boost_round=n_rounds,
)

print(f"  Stage 3 complete — four models total.")


# ============================================================
# PREDICT: Combine with Propensity Weights
# ============================================================

print("\n\n" + "=" * 60)
print("PREDICTION: Propensity-Weighted Combination")
print("=" * 60)

# Propensity score: P(treatment=1)
# In this RCT, it's roughly constant at ~0.85 for everyone.
# We use the empirical rate rather than modeling it.
g = t_train.mean()
print(f"\n  Propensity score g = P(treatment=1) = {g:.4f}")

# Get both τ predictions for test users
tau_1_pred = tau_model_1.predict(X_test[feature_cols])  # treatment-group perspective
tau_0_pred = tau_model_0.predict(X_test[feature_cols])  # control-group perspective

# Weighted combination: τ = g · τ_model_0 + (1-g) · τ_model_1
uplift_x = g * tau_0_pred + (1 - g) * tau_1_pred

print(f"\n  τ_model_1 predictions (treatment perspective):")
print(f"    Range: [{tau_1_pred.min():.4f}, {tau_1_pred.max():.4f}]")
print(f"    Mean:  {tau_1_pred.mean():.4f}")

print(f"\n  τ_model_0 predictions (control perspective):")
print(f"    Range: [{tau_0_pred.min():.4f}, {tau_0_pred.max():.4f}]")
print(f"    Mean:  {tau_0_pred.mean():.4f}")

print(f"\n  Final X-learner uplift (weighted combination):")
print(f"    Range: [{uplift_x.min():.4f}, {uplift_x.max():.4f}]")
print(f"    Mean:  {uplift_x.mean():.4f}")
print(f"    (True ATE: {y_test[t_test == 1].mean() - y_test[t_test == 0].mean():.4f})")

n_negative = (uplift_x < 0).sum()
print(f"\n  Users with negative predicted uplift: {n_negative:,} ({n_negative/len(uplift_x):.1%})")


# ============================================================
# EVALUATION: Full Comparison
# ============================================================

print("\n\n" + "=" * 60)
print("EVALUATION: Full Comparison (All Approaches)")
print("=" * 60)

auuc_x = uplift_auc_score(y_test, uplift_x, t_test)
qini_x = qini_auc_score(y_test, uplift_x, t_test)

# --- Recompute all prior approaches ---

# T-Learner
uplift_tl = model_t.predict(X_test[feature_cols]) - model_c.predict(X_test[feature_cols])
auuc_tl = uplift_auc_score(y_test, uplift_tl, t_test)
qini_tl = qini_auc_score(y_test, uplift_tl, t_test)

# S-Learner
X_train_s = X_train.copy()
X_train_s["treatment"] = t_train.values
s_features = feature_cols + ["treatment"]
model_s = lgb.train(params, lgb.Dataset(X_train_s[s_features], label=y_train), num_boost_round=n_rounds)
X_t1 = X_test.copy(); X_t1["treatment"] = 1
X_t0 = X_test.copy(); X_t0["treatment"] = 0
uplift_sl = model_s.predict(X_t1[s_features]) - model_s.predict(X_t0[s_features])
auuc_sl = uplift_auc_score(y_test, uplift_sl, t_test)
qini_sl = qini_auc_score(y_test, uplift_sl, t_test)

# Standard Classifier (LogReg)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr_sc, y_train)
propensity_scores = clf.predict_proba(X_te_sc)[:, 1]
auuc_prop = uplift_auc_score(y_test, propensity_scores, t_test)
qini_prop = qini_auc_score(y_test, propensity_scores, t_test)

# Two Classifiers (LogReg)
clf_t_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_t_lr.fit(X_tr_sc[treat_mask], y_train[treat_mask])
clf_c_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_c_lr.fit(X_tr_sc[~treat_mask], y_train[~treat_mask])
uplift_lr = clf_t_lr.predict_proba(X_te_sc)[:, 1] - clf_c_lr.predict_proba(X_te_sc)[:, 1]
auuc_lr = uplift_auc_score(y_test, uplift_lr, t_test)
qini_lr = qini_auc_score(y_test, uplift_lr, t_test)

# Random
rng = np.random.RandomState(42)
random_scores = rng.uniform(0, 1, size=len(X_test))
auuc_rand = uplift_auc_score(y_test, random_scores, t_test)
qini_rand = qini_auc_score(y_test, random_scores, t_test)

print(f"\n  {'Approach':<35s}  {'AUUC':>10s}  {'Qini':>10s}")
print(f"  {'-' * 60}")
print(f"  {'X-Learner (LightGBM)':<35s}  {auuc_x:>10.6f}  {qini_x:>10.6f}")
print(f"  {'T-Learner (LightGBM)':<35s}  {auuc_tl:>10.6f}  {qini_tl:>10.6f}")
print(f"  {'S-Learner (LightGBM)':<35s}  {auuc_sl:>10.6f}  {qini_sl:>10.6f}")
print(f"  {'Standard Classifier (LogReg)':<35s}  {auuc_prop:>10.6f}  {qini_prop:>10.6f}")
print(f"  {'Two Classifiers (LogReg)':<35s}  {auuc_lr:>10.6f}  {qini_lr:>10.6f}")
print(f"  {'Random':<35s}  {auuc_rand:>10.6f}  {qini_rand:>10.6f}")

all_approaches = [
    ("X-Learner (LightGBM)", auuc_x),
    ("T-Learner (LightGBM)", auuc_tl),
    ("S-Learner (LightGBM)", auuc_sl),
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
print("UPLIFT DECILE ANALYSIS: X-Learner vs S-Learner")
print("=" * 60)

y_test_arr = y_test.values
t_test_arr = t_test.values

for model_name, uplift_scores in [("X-Learner", uplift_x), ("S-Learner", uplift_sl)]:
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


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Step 5 — X-Learner: Uplift Performance", fontsize=14, fontweight="bold")

# ---- Plot 1: Uplift curves ----
ax = axes[0]
approaches_plot = [
    ("X-Learner", uplift_x, "#9C27B0", "-", 2.5),
    ("T-Learner", uplift_tl, "#FF9800", "--", 1.5),
    ("S-Learner", uplift_sl, "#2196F3", "--", 1.5),
    ("Std Classifier", propensity_scores, "#FF6B6B", ":", 1.0),
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
names = ["X-Lrnr", "T-Lrnr", "S-Lrnr", "Std Clf", "2-Clf\nLogReg", "Random"]
auuc_vals = [auuc_x, auuc_tl, auuc_sl, auuc_prop, auuc_lr, auuc_rand]
colors = ["#9C27B0", "#FF9800", "#2196F3", "#FF6B6B", "#4ECDC4", "#B0BEC5"]
bars = ax.bar(names, auuc_vals, color=colors, width=0.5, alpha=0.8)
ax.set_ylabel("AUUC Score")
ax.set_title("AUUC Comparison")
ax.axhline(y=0, color="black", lw=0.5)
for bar, val in zip(bars, auuc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

# ---- Plot 3: Uplift distributions ----
ax = axes[2]
ax.hist(uplift_x, bins=80, alpha=0.6, color="#9C27B0", label="X-Learner", density=True)
ax.hist(uplift_sl, bins=80, alpha=0.6, color="#2196F3", label="S-Learner", density=True)
ax.hist(uplift_tl, bins=80, alpha=0.5, color="#FF9800", label="T-Learner", density=True)
ax.axvline(x=0, color="red", lw=1.5, ls="--", label="Zero uplift")
ax.set_xlabel("Predicted Uplift Score")
ax.set_ylabel("Density")
ax.set_title("Predicted Uplift Distributions")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.15, 0.25)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = f"{BASE}/results/analysis/05_x_learner.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n\nSaved plot: {plot_path}")


# ============================================================
# KEY TAKEAWAYS
# ============================================================

print("\n\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

best_auuc = max(auuc_vals)
best_name = [n for n, v in all_approaches if v == best_auuc][0]

print(f"""
  1. X-LEARNER AUUC: {auuc_x:.6f}
     {'Best uplift model so far!' if auuc_x > max(auuc_tl, auuc_sl) else 'Did not beat all prior approaches.'}

  2. RANKING:
     {'  >  '.join(f'{n} ({v:.4f})' for n, v in all_sorted[:3])}

  3. CALIBRATION: Check decile table — does predicted τ match actual ATE
     better than S-learner and T-learner?

  4. NEGATIVE UPLIFT: {n_negative:,} users ({n_negative/len(uplift_x):.1%}) predicted
     as Sleeping Dogs.

  NEXT: Step 6 — Head-to-head model comparison with uplift@K% analysis
  (practical targeting decisions).
""")

tee.close()
print("Done. Results saved to results/models/05_x_learner.txt")
