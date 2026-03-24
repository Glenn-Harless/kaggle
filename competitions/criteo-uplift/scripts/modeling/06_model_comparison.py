"""
Criteo Uplift Modeling: Step 6 — Model Comparison
Head-to-head comparison of all approaches with practical targeting analysis:
  - Uplift@K%: "if you can only target X% of users, how many extra visits?"
  - Consolidated uplift and Qini curves
  - Decile tables side-by-side
  - Business recommendation: which model, at what targeting threshold?
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

from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_curve, qini_curve
from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/06_model_comparison.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 6 — Model Comparison")
print("=" * 60)


# ============================================================
# DATA LOADING & SPLIT (same as all prior steps)
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

treat_mask = t_train == 1
y_test_arr = y_test.values
t_test_arr = t_test.values
n_test = len(X_test)

true_ate = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
print(f"\n  Train: {len(X_train):,}  |  Test: {n_test:,}")
print(f"  True ATE on test set: {true_ate:.4f}")


# ============================================================
# BUILD ALL MODELS
# ============================================================

print("\n\n" + "=" * 60)
print("BUILDING ALL MODELS")
print("=" * 60)

lgb_params = {
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
lgb_reg_params = {**lgb_params, "objective": "regression", "metric": "rmse"}
n_rounds = 300

# --- Standard Classifier (LogReg) ---
print("\n  1. Standard Classifier (LogReg)...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr_sc, y_train)
scores_std = clf.predict_proba(X_te_sc)[:, 1]

# --- S-Learner (LightGBM) ---
print("  2. S-Learner (LightGBM)...")
X_train_s = X_train.copy(); X_train_s["treatment"] = t_train.values
s_features = feature_cols + ["treatment"]
model_s = lgb.train(lgb_params, lgb.Dataset(X_train_s[s_features], label=y_train),
                    num_boost_round=n_rounds)
X_t1 = X_test.copy(); X_t1["treatment"] = 1
X_t0 = X_test.copy(); X_t0["treatment"] = 0
scores_s = model_s.predict(X_t1[s_features]) - model_s.predict(X_t0[s_features])

# --- T-Learner (LightGBM) ---
print("  3. T-Learner (LightGBM)...")
X_tr_t = X_train[treat_mask]; y_tr_t = y_train[treat_mask]
X_tr_c = X_train[~treat_mask]; y_tr_c = y_train[~treat_mask]
model_t = lgb.train(lgb_params, lgb.Dataset(X_tr_t[feature_cols], label=y_tr_t),
                    num_boost_round=n_rounds)
model_c = lgb.train(lgb_params, lgb.Dataset(X_tr_c[feature_cols], label=y_tr_c),
                    num_boost_round=n_rounds)
scores_t = model_t.predict(X_test[feature_cols]) - model_c.predict(X_test[feature_cols])

# --- X-Learner (LightGBM) ---
print("  4. X-Learner (LightGBM)...")
# Stage 2: cross-impute
d_treated = y_tr_t.values - model_c.predict(X_tr_t[feature_cols])
d_control = model_t.predict(X_tr_c[feature_cols]) - y_tr_c.values
# Stage 3: model effects
tau_1 = lgb.train(lgb_reg_params, lgb.Dataset(X_tr_t[feature_cols], label=d_treated),
                  num_boost_round=n_rounds)
tau_0 = lgb.train(lgb_reg_params, lgb.Dataset(X_tr_c[feature_cols], label=d_control),
                  num_boost_round=n_rounds)
g = t_train.mean()
scores_x = g * tau_0.predict(X_test[feature_cols]) + (1 - g) * tau_1.predict(X_test[feature_cols])

# --- Random ---
rng = np.random.RandomState(42)
scores_rand = rng.uniform(0, 1, size=n_test)

print("  All models built.")


# ============================================================
# AUUC AND QINI SCORES
# ============================================================

print("\n\n" + "=" * 60)
print("AUUC AND QINI SCORES")
print("=" * 60)

models = {
    "X-Learner":            scores_x,
    "S-Learner":            scores_s,
    "T-Learner":            scores_t,
    "Standard Classifier":  scores_std,
    "Random":               scores_rand,
}

results = {}
print(f"\n  {'Model':<25s}  {'AUUC':>10s}  {'Qini':>10s}")
print(f"  {'-' * 50}")
for name, scores in models.items():
    auuc = uplift_auc_score(y_test, scores, t_test)
    qini = qini_auc_score(y_test, scores, t_test)
    results[name] = {"auuc": auuc, "qini": qini, "scores": scores}
    print(f"  {name:<25s}  {auuc:>10.6f}  {qini:>10.6f}")


# ============================================================
# UPLIFT@K% — THE PRACTICAL TARGETING METRIC
# ============================================================
# "If I can only target the top K% of users, what's the
# average treatment effect in that targeted group?"

print("\n\n" + "=" * 60)
print("UPLIFT@K% — Practical Targeting Analysis")
print("=" * 60)

target_pcts = [5, 10, 15, 20, 30, 40, 50, 100]

print(f"\n  Average treatment effect (ATE) when targeting top K% of users:")
print(f"  {'Model':<25s}", end="")
for pct in target_pcts:
    print(f"  {'@' + str(pct) + '%':>8s}", end="")
print()
print(f"  {'-' * (25 + 10 * len(target_pcts))}")

for name, info in results.items():
    scores = info["scores"]
    print(f"  {name:<25s}", end="")

    for pct in target_pcts:
        k = int(n_test * pct / 100)
        # Get top-k users by model score
        top_k_idx = np.argsort(-scores)[:k]
        y_topk = y_test_arr[top_k_idx]
        t_topk = t_test_arr[top_k_idx]

        # ATE in the targeted group
        tm = t_topk == 1
        if tm.sum() > 0 and (~tm).sum() > 0:
            ate_k = y_topk[tm].mean() - y_topk[~tm].mean()
        else:
            ate_k = float("nan")

        print(f"  {ate_k:>+8.4f}", end="")

    print()

print(f"\n  True overall ATE: {true_ate:>+.4f} (this is what @100% should show)")

# Incremental visits version
print(f"\n\n  Estimated incremental visits when targeting top K% of users:")
print(f"  (Incremental = ATE × number of users targeted)")
print(f"  {'Model':<25s}", end="")
for pct in target_pcts:
    print(f"  {'@' + str(pct) + '%':>8s}", end="")
print()
print(f"  {'-' * (25 + 10 * len(target_pcts))}")

for name, info in results.items():
    scores = info["scores"]
    print(f"  {name:<25s}", end="")

    for pct in target_pcts:
        k = int(n_test * pct / 100)
        top_k_idx = np.argsort(-scores)[:k]
        y_topk = y_test_arr[top_k_idx]
        t_topk = t_test_arr[top_k_idx]

        tm = t_topk == 1
        if tm.sum() > 0 and (~tm).sum() > 0:
            ate_k = y_topk[tm].mean() - y_topk[~tm].mean()
            incremental = ate_k * k
        else:
            incremental = float("nan")

        print(f"  {incremental:>8.0f}", end="")

    print()


# ============================================================
# DECILE TABLES SIDE-BY-SIDE (Top 3 models only)
# ============================================================

print("\n\n" + "=" * 60)
print("DECILE ANALYSIS — Top 3 Models")
print("=" * 60)

top_models = ["X-Learner", "S-Learner", "Standard Classifier"]

for name in top_models:
    scores = results[name]["scores"]
    deciles = pd.qcut(scores, q=10, duplicates="drop")

    print(f"\n  {name}:")
    print(f"  {'Decile':>30s}  {'N':>7s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}  {'ATE':>9s}  {'Score':>9s}")
    print(f"  {'-' * 80}")

    for decile in sorted(deciles.unique()):
        mask = deciles == decile
        y_sub = y_test_arr[mask]
        t_sub = t_test_arr[mask]
        s_sub = scores[mask]

        tm = t_sub == 1
        vt = y_sub[tm].mean() if tm.sum() > 0 else float("nan")
        vc = y_sub[~tm].mean() if (~tm).sum() > 0 else float("nan")
        ate = vt - vc

        print(f"  {str(decile):>30s}  {mask.sum():>7,}  {vt:>9.4f}  {vc:>9.4f}  {ate:>+9.4f}  {s_sub.mean():>+9.4f}")


# ============================================================
# EFFICIENCY ANALYSIS
# ============================================================
# How much of the total uplift is captured by targeting top K%?

print("\n\n" + "=" * 60)
print("TARGETING EFFICIENCY — % of Total Uplift Captured")
print("=" * 60)

# Total uplift if we target everyone
total_uplift = true_ate * n_test

print(f"\n  Total uplift (target everyone): {total_uplift:.0f} incremental visits")
print(f"\n  {'Model':<25s}", end="")
for pct in [10, 20, 30, 50]:
    print(f"  {'@' + str(pct) + '%':>10s}", end="")
print()
print(f"  {'-' * (25 + 12 * 4)}")

for name in ["X-Learner", "S-Learner", "Standard Classifier", "Random"]:
    scores = results[name]["scores"]
    print(f"  {name:<25s}", end="")

    for pct in [10, 20, 30, 50]:
        k = int(n_test * pct / 100)
        top_k_idx = np.argsort(-scores)[:k]
        y_topk = y_test_arr[top_k_idx]
        t_topk = t_test_arr[top_k_idx]

        tm = t_topk == 1
        if tm.sum() > 0 and (~tm).sum() > 0:
            ate_k = y_topk[tm].mean() - y_topk[~tm].mean()
            uplift_k = ate_k * k
            pct_captured = uplift_k / total_uplift * 100
        else:
            pct_captured = float("nan")

        print(f"  {pct_captured:>9.1f}%", end="")

    print()

print(f"\n  Read as: 'By targeting only X% of users, model captures Y% of")
print(f"  the total uplift you'd get from targeting everyone.'")
print(f"  Higher is better — means the model front-loads persuadables.")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Step 6 — Model Comparison", fontsize=14, fontweight="bold")

plot_models = [
    ("X-Learner", scores_x, "#9C27B0", "-", 2.5),
    ("S-Learner", scores_s, "#2196F3", "-", 2.0),
    ("T-Learner", scores_t, "#FF9800", "--", 1.5),
    ("Std Classifier", scores_std, "#FF6B6B", "--", 1.5),
    ("Random", scores_rand, "#B0BEC5", ":", 1.0),
]

# ---- Plot 1: Uplift Curves ----
ax = axes[0, 0]
for label, scores, color, ls, lw in plot_models:
    x_vals, y_vals = uplift_curve(y_test, scores, t_test)
    x_norm = x_vals / x_vals.max() if x_vals.max() > 0 else x_vals
    ax.plot(x_norm, y_vals, color=color, ls=ls, lw=lw, label=label)
ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("Proportion of Users Targeted")
ax.set_ylabel("Cumulative Uplift")
ax.set_title("Uplift Curves")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Plot 2: Qini Curves ----
ax = axes[0, 1]
for label, scores, color, ls, lw in plot_models:
    x_vals, y_vals = qini_curve(y_test, scores, t_test)
    x_norm = x_vals / x_vals.max() if x_vals.max() > 0 else x_vals
    ax.plot(x_norm, y_vals, color=color, ls=ls, lw=lw, label=label)
ax.axhline(y=0, color="black", lw=0.5)
ax.set_xlabel("Proportion of Users Targeted")
ax.set_ylabel("Qini Score")
ax.set_title("Qini Curves")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Plot 3: AUUC Bar Chart ----
ax = axes[1, 0]
model_names = [m[0] for m in plot_models]
auuc_vals = [results[n]["auuc"] for n in
             ["X-Learner", "S-Learner", "T-Learner", "Standard Classifier", "Random"]]
colors = [m[2] for m in plot_models]
bars = ax.bar(model_names, auuc_vals, color=colors, width=0.5, alpha=0.8)
ax.set_ylabel("AUUC Score")
ax.set_title("Area Under Uplift Curve")
ax.axhline(y=0, color="black", lw=0.5)
for bar, val in zip(bars, auuc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ---- Plot 4: Uplift@K% for top models ----
ax = axes[1, 1]
for name, color, ls in [("X-Learner", "#9C27B0", "-"),
                         ("S-Learner", "#2196F3", "-"),
                         ("Standard Classifier", "#FF6B6B", "--"),
                         ("Random", "#B0BEC5", ":")]:
    scores = results[name]["scores"]
    pcts = list(range(5, 105, 5))
    ates_at_k = []
    for pct in pcts:
        k = int(n_test * pct / 100)
        top_k_idx = np.argsort(-scores)[:k]
        y_topk = y_test_arr[top_k_idx]
        t_topk = t_test_arr[top_k_idx]
        tm = t_topk == 1
        if tm.sum() > 0 and (~tm).sum() > 0:
            ates_at_k.append(y_topk[tm].mean() - y_topk[~tm].mean())
        else:
            ates_at_k.append(float("nan"))
    ax.plot(pcts, ates_at_k, color=color, ls=ls, lw=2, label=name)

ax.axhline(y=true_ate, color="black", lw=1, ls="--", alpha=0.5,
           label=f"Overall ATE ({true_ate:.4f})")
ax.set_xlabel("% of Users Targeted (top K%)")
ax.set_ylabel("ATE in Targeted Group")
ax.set_title("Uplift@K%: Treatment Effect by Targeting Depth")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{BASE}/results/analysis/06_model_comparison.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n\nSaved plot: {plot_path}")


# ============================================================
# FINAL SUMMARY & RECOMMENDATION
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL SUMMARY & RECOMMENDATION")
print("=" * 60)
print(f"""
  OVERALL AUUC RANKING:
""")
sorted_results = sorted(results.items(), key=lambda x: x[1]["auuc"], reverse=True)
for rank, (name, info) in enumerate(sorted_results, 1):
    marker = " ← best" if rank == 1 else ""
    print(f"    {rank}. {name:<25s}  AUUC={info['auuc']:.6f}  Qini={info['qini']:.6f}{marker}")

print(f"""
  RECOMMENDATION:

  For this dataset, the Standard Classifier leads on AUUC because
  propensity and uplift are positively correlated — an unusual property.

  However, the X-LEARNER is the recommended choice because:
    1. It actually models the treatment effect (causal, not accidental)
    2. Its decile table shows meaningful monotonic ATE progression
    3. It generalizes to datasets where propensity ≠ uplift
    4. It produces calibrated uplift scores usable for targeting thresholds

  The S-LEARNER is a strong second choice with better calibration
  at the extremes but slightly lower AUUC than the X-learner.

  PRACTICAL GUIDANCE:
  - Target top 10-20% of users ranked by X-learner uplift score
  - This captures a disproportionate share of total uplift
  - Users below the bottom decile can be safely excluded from targeting
  - Review the uplift@K% table to align targeting depth with budget
""")

tee.close()
print("Done. Results saved to results/models/06_model_comparison.txt")
