"""
Criteo Uplift Modeling: Step 8 — Practical Targeting Analysis
Translate the best uplift model (X-learner) into business decisions:
  1. Cost-benefit analysis: optimal targeting threshold
  2. Budget-constrained targeting: maximize ROI under budget limits
  3. Segment analysis: which feature-defined segments have highest uplift?
  4. Savings analysis: how much budget is saved vs. blanket targeting?
  5. Connection to geo experiments
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

from sklift.metrics import uplift_auc_score, uplift_curve
from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/08_targeting_analysis.txt")
sys.stdout = tee

print("Criteo Uplift Modeling: Step 8 — Practical Targeting Analysis")
print("=" * 60)


# ============================================================
# DATA & MODEL SETUP
# ============================================================

from sklift.datasets import fetch_criteo

print("\nLoading data and building X-learner...")
data = fetch_criteo(percent10=True)

df = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(12)])
df["treatment"] = data.treatment.astype(int)
df["visit"] = data.target.astype(int)

feature_cols = [f"f{i}" for i in range(12)]
X = df[feature_cols]; y = df["visit"]; treatment = df["treatment"]

strat_col = treatment * 2 + y
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X, y, treatment, test_size=0.3, random_state=42, stratify=strat_col
)

treat_mask = t_train == 1
X_tr_t = X_train[treat_mask]; y_tr_t = y_train[treat_mask]
X_tr_c = X_train[~treat_mask]; y_tr_c = y_train[~treat_mask]

lgb_params = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbose": -1,
}
lgb_reg = {**lgb_params, "objective": "regression", "metric": "rmse"}
n_rounds = 300

# Stage 1: outcome models
model_t = lgb.train(lgb_params, lgb.Dataset(X_tr_t[feature_cols], label=y_tr_t),
                    num_boost_round=n_rounds)
model_c = lgb.train(lgb_params, lgb.Dataset(X_tr_c[feature_cols], label=y_tr_c),
                    num_boost_round=n_rounds)

# Stage 2: cross-impute
d_treated = y_tr_t.values - model_c.predict(X_tr_t[feature_cols])
d_control = model_t.predict(X_tr_c[feature_cols]) - y_tr_c.values

# Stage 3: model effects
tau_1 = lgb.train(lgb_reg, lgb.Dataset(X_tr_t[feature_cols], label=d_treated),
                  num_boost_round=n_rounds)
tau_0 = lgb.train(lgb_reg, lgb.Dataset(X_tr_c[feature_cols], label=d_control),
                  num_boost_round=n_rounds)

g = t_train.mean()
uplift_scores = g * tau_0.predict(X_test[feature_cols]) + (1 - g) * tau_1.predict(X_test[feature_cols])

y_arr = y_test.values
t_arr = t_test.values
n_test = len(X_test)
true_ate = y_arr[t_arr == 1].mean() - y_arr[t_arr == 0].mean()

print(f"  X-learner built. Test set: {n_test:,} users.")
print(f"  True ATE: {true_ate:.4f}")
print(f"  Uplift score range: [{uplift_scores.min():.4f}, {uplift_scores.max():.4f}]")


# ============================================================
# 1. COST-BENEFIT ANALYSIS
# ============================================================
# The core business question: given ad costs and conversion values,
# what's the optimal targeting threshold?

print("\n\n" + "=" * 60)
print("1. COST-BENEFIT ANALYSIS")
print("=" * 60)

# Hypothetical economics (realistic digital ad ranges)
AD_COST_PER_USER = 0.05       # $0.05 to show one ad impression
CONVERSION_VALUE = 25.00      # $25 revenue per visit/conversion

print(f"\n  Assumptions:")
print(f"    Cost per ad impression:  ${AD_COST_PER_USER:.2f}")
print(f"    Value per conversion:    ${CONVERSION_VALUE:.2f}")
print(f"    Break-even uplift:       {AD_COST_PER_USER / CONVERSION_VALUE:.4f} ({AD_COST_PER_USER / CONVERSION_VALUE:.2%})")

breakeven_uplift = AD_COST_PER_USER / CONVERSION_VALUE

# Sort users by predicted uplift
sort_idx = np.argsort(-uplift_scores)
sorted_uplift = uplift_scores[sort_idx]
sorted_y = y_arr[sort_idx]
sorted_t = t_arr[sort_idx]

# For each possible targeting depth, compute profit
pcts = np.arange(1, 101)
profits_model = []
profits_blanket = []
ates_at_k = []
revenues_at_k = []
costs_at_k = []

for pct in pcts:
    k = int(n_test * pct / 100)
    y_topk = sorted_y[:k]
    t_topk = sorted_t[:k]

    tm = t_topk == 1
    if tm.sum() > 0 and (~tm).sum() > 0:
        ate_k = y_topk[tm].mean() - y_topk[~tm].mean()
    else:
        ate_k = 0

    incremental_conversions = ate_k * k
    revenue = incremental_conversions * CONVERSION_VALUE
    cost = k * AD_COST_PER_USER
    profit = revenue - cost

    profits_model.append(profit)
    ates_at_k.append(ate_k)
    revenues_at_k.append(revenue)
    costs_at_k.append(cost)

    # Blanket targeting comparison (no model, random order)
    blanket_incremental = true_ate * k
    blanket_revenue = blanket_incremental * CONVERSION_VALUE
    blanket_cost = k * AD_COST_PER_USER
    profits_blanket.append(blanket_revenue - blanket_cost)

optimal_pct = pcts[np.argmax(profits_model)]
max_profit = max(profits_model)
blanket_profit_100 = profits_blanket[-1]

print(f"\n  Model-guided targeting:")
print(f"    Optimal targeting depth: top {optimal_pct}%")
print(f"    Max profit:              ${max_profit:,.0f}")
print(f"    At that depth:")
print(f"      Users targeted:        {int(n_test * optimal_pct / 100):,}")
print(f"      ATE in group:          {ates_at_k[optimal_pct - 1]:.4f}")
print(f"      Revenue:               ${revenues_at_k[optimal_pct - 1]:,.0f}")
print(f"      Ad cost:               ${costs_at_k[optimal_pct - 1]:,.0f}")

print(f"\n  Blanket targeting (show ads to everyone):")
print(f"    Profit:                  ${blanket_profit_100:,.0f}")
print(f"    Revenue:                 ${true_ate * n_test * CONVERSION_VALUE:,.0f}")
print(f"    Ad cost:                 ${n_test * AD_COST_PER_USER:,.0f}")

savings = max_profit - blanket_profit_100
print(f"\n  MODEL ADVANTAGE:")
print(f"    Extra profit vs blanket: ${savings:,.0f} ({savings / abs(blanket_profit_100) * 100 if blanket_profit_100 != 0 else 0:+.1f}%)")


# ============================================================
# 2. TARGETING AT DIFFERENT PRICE POINTS
# ============================================================

print("\n\n" + "=" * 60)
print("2. SENSITIVITY ANALYSIS — Different Ad Costs")
print("=" * 60)

ad_costs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

print(f"\n  Conversion value fixed at ${CONVERSION_VALUE:.2f}")
print(f"\n  {'Ad Cost':>10s}  {'Break-even':>12s}  {'Optimal %':>10s}  {'Profit(Model)':>15s}  {'Profit(Blanket)':>16s}  {'Savings':>12s}")
print(f"  {'-' * 80}")

for ac in ad_costs:
    be = ac / CONVERSION_VALUE
    best_profit = -999999
    best_pct = 100

    for pct in pcts:
        k = int(n_test * pct / 100)
        ate_k = ates_at_k[pct - 1]
        revenue = ate_k * k * CONVERSION_VALUE
        cost = k * ac
        profit = revenue - cost
        if profit > best_profit:
            best_profit = profit
            best_pct = pct

    blanket_p = true_ate * n_test * CONVERSION_VALUE - n_test * ac
    sav = best_profit - blanket_p

    print(f"  ${ac:>8.2f}  {be:>12.4f}  {best_pct:>9d}%  ${best_profit:>14,.0f}  ${blanket_p:>15,.0f}  ${sav:>11,.0f}")


# ============================================================
# 3. SEGMENT ANALYSIS
# ============================================================
# Which feature-defined user segments have the highest predicted uplift?

print("\n\n" + "=" * 60)
print("3. SEGMENT ANALYSIS — Uplift by Feature")
print("=" * 60)

print(f"\n  For each feature, split into quintiles and show average")
print(f"  predicted uplift and actual ATE per quintile.")

# Focus on the most important features from our EDA
key_features = ["f0", "f2", "f8", "f9"]
X_test_df = X_test.copy()
X_test_df["uplift"] = uplift_scores
X_test_df["visit"] = y_arr
X_test_df["treatment"] = t_arr

for col in key_features:
    quintiles = pd.qcut(X_test_df[col], q=5, duplicates="drop")

    print(f"\n  {col}:")
    print(f"  {'Quintile':>25s}  {'N':>7s}  {'Pred Uplift':>12s}  {'Actual ATE':>11s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}")
    print(f"  {'-' * 80}")

    for q in sorted(quintiles.unique()):
        mask = quintiles == q
        sub = X_test_df[mask]
        pred_u = sub["uplift"].mean()

        treat_sub = sub[sub["treatment"] == 1]["visit"]
        ctrl_sub = sub[sub["treatment"] == 0]["visit"]
        vt = treat_sub.mean() if len(treat_sub) > 0 else float("nan")
        vc = ctrl_sub.mean() if len(ctrl_sub) > 0 else float("nan")
        actual_ate = vt - vc

        print(f"  {str(q):>25s}  {mask.sum():>7,}  {pred_u:>+12.4f}  {actual_ate:>+11.4f}  {vt:>9.4f}  {vc:>9.4f}")


# ============================================================
# 4. USER SEGMENT PROFILES
# ============================================================
# Profile the top 10% (target) vs bottom 10% (exclude) vs middle

print("\n\n" + "=" * 60)
print("4. USER SEGMENT PROFILES")
print("=" * 60)

top_10_mask = uplift_scores >= np.percentile(uplift_scores, 90)
bottom_10_mask = uplift_scores <= np.percentile(uplift_scores, 10)
middle_mask = ~top_10_mask & ~bottom_10_mask

segments = [
    ("Top 10% (target these)", top_10_mask),
    ("Middle 80%", middle_mask),
    ("Bottom 10% (skip these)", bottom_10_mask),
]

for seg_name, seg_mask in segments:
    seg_data = X_test_df[seg_mask]
    seg_y = y_arr[seg_mask]
    seg_t = t_arr[seg_mask]
    tm = seg_t == 1

    vt = seg_y[tm].mean()
    vc = seg_y[~tm].mean()
    seg_ate = vt - vc

    print(f"\n  {seg_name} (N={seg_mask.sum():,}):")
    print(f"    Predicted uplift: {uplift_scores[seg_mask].mean():+.4f}")
    print(f"    Actual ATE:       {seg_ate:+.4f}")
    print(f"    Visit rate (T):   {vt:.4f}")
    print(f"    Visit rate (C):   {vc:.4f}")
    print(f"    Feature means:")
    for col in feature_cols:
        overall_mean = X_test_df[col].mean()
        seg_mean = seg_data[col].mean()
        diff = seg_mean - overall_mean
        if abs(diff) > 0.5 * X_test_df[col].std():  # only show notable differences
            direction = "↑" if diff > 0 else "↓"
            print(f"      {col}: {seg_mean:.2f} (overall: {overall_mean:.2f}) {direction}")


# ============================================================
# 5. BUDGET SCENARIO ANALYSIS
# ============================================================

print("\n\n" + "=" * 60)
print("5. BUDGET SCENARIOS")
print("=" * 60)

print(f"\n  'I have $X to spend on ads. How should I allocate it?'")
print(f"  (At ${AD_COST_PER_USER:.2f} per impression, ${CONVERSION_VALUE:.2f} per conversion)")

budgets = [1000, 5000, 10000, 25000, 50000]

print(f"\n  {'Budget':>10s}  {'Users':>10s}  {'% of Total':>10s}  {'Incr. Visits':>13s}  {'Revenue':>10s}  {'Profit':>10s}  {'ROI':>8s}")
print(f"  {'-' * 75}")

for budget in budgets:
    n_targeted = int(budget / AD_COST_PER_USER)
    n_targeted = min(n_targeted, n_test)
    pct_targeted = n_targeted / n_test * 100

    # Use model ranking
    top_idx = np.argsort(-uplift_scores)[:n_targeted]
    y_top = y_arr[top_idx]
    t_top = t_arr[top_idx]
    tm = t_top == 1
    if tm.sum() > 0 and (~tm).sum() > 0:
        ate = y_top[tm].mean() - y_top[~tm].mean()
    else:
        ate = 0

    incr = ate * n_targeted
    rev = incr * CONVERSION_VALUE
    profit = rev - budget
    roi = (rev / budget - 1) * 100 if budget > 0 else 0

    print(f"  ${budget:>9,}  {n_targeted:>10,}  {pct_targeted:>9.1f}%  {incr:>13.0f}  ${rev:>9,.0f}  ${profit:>+9,.0f}  {roi:>+7.0f}%")


# ============================================================
# 6. CONNECTION TO GEO EXPERIMENTS
# ============================================================

print("\n\n" + "=" * 60)
print("6. CONNECTION TO GEO EXPERIMENTS")
print("=" * 60)
print(f"""
  How uplift modeling complements geo experiment work:

  GEO EXPERIMENTS (what you do now):
    - Design: pick treatment/control DMAs
    - Run: spend $70K over 10 weeks
    - Analyze: difference-in-differences → one ATE number
    - Limitation: can't see WHICH users drove the lift
    - Limitation: underpowered for small effects (MDE = 18%)

  UPLIFT MODELING (what you just learned):
    - Requires: one good RCT (could be a past geo experiment!)
    - Produces: per-user uplift scores
    - Enables: targeted ad delivery (show ads only to persuadables)
    - Enables: budget optimization (cut waste on Sure Things)

  HOW THEY WORK TOGETHER:

  1. Run ONE well-powered geo experiment (or user-level RCT)
     to collect ground truth treatment/control data

  2. Build an uplift model on that data to identify which
     user segments respond most to ads

  3. Use the model to make FUTURE campaigns more efficient:
     - Target only high-uplift segments
     - Exclude sleeping dogs
     - Allocate budget to markets/segments with highest CATE

  4. Run a FOLLOW-UP geo experiment to VALIDATE the model's
     predictions: "did targeting persuadables actually improve ROI?"

  This is the virtuous cycle: experiment → model → smarter targeting
  → validate with another experiment → refine model → ...

  The Criteo dataset is Step 1-2. Your geo experiments could feed
  into this same pipeline if you had user-level outcome data.
""")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Step 8 — Practical Targeting Analysis", fontsize=14, fontweight="bold")

# ---- Plot 1: Profit curve ----
ax = axes[0, 0]
ax.plot(pcts, profits_model, color="#9C27B0", lw=2.5, label="Model-guided targeting")
ax.plot(pcts, profits_blanket, color="#B0BEC5", lw=2, ls="--", label="Blanket targeting")
ax.axhline(y=0, color="black", lw=0.5)
ax.axvline(x=optimal_pct, color="#9C27B0", lw=1, ls=":", alpha=0.7,
           label=f"Optimal: {optimal_pct}%")
ax.scatter([optimal_pct], [max_profit], color="#9C27B0", s=100, zorder=5)
ax.set_xlabel("% of Users Targeted")
ax.set_ylabel("Profit ($)")
ax.set_title(f"Profit Curve (ad=${AD_COST_PER_USER}, conv=${CONVERSION_VALUE})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ---- Plot 2: ATE by targeting depth ----
ax = axes[0, 1]
ax.plot(pcts, ates_at_k, color="#9C27B0", lw=2)
ax.axhline(y=true_ate, color="black", lw=1, ls="--",
           label=f"Overall ATE ({true_ate:.4f})")
ax.axhline(y=breakeven_uplift, color="red", lw=1, ls="--",
           label=f"Break-even uplift ({breakeven_uplift:.4f})")
ax.fill_between(pcts, breakeven_uplift, ates_at_k,
                where=[a > breakeven_uplift for a in ates_at_k],
                alpha=0.2, color="#4CAF50", label="Profitable zone")
ax.fill_between(pcts, breakeven_uplift, ates_at_k,
                where=[a <= breakeven_uplift for a in ates_at_k],
                alpha=0.2, color="#FF5252", label="Unprofitable zone")
ax.set_xlabel("% of Users Targeted")
ax.set_ylabel("ATE in Targeted Group")
ax.set_title("Treatment Effect by Targeting Depth")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---- Plot 3: Uplift distribution with threshold ----
ax = axes[1, 0]
ax.hist(uplift_scores, bins=80, color="#9C27B0", alpha=0.7, edgecolor="white", linewidth=0.3)
ax.axvline(x=0, color="red", lw=1.5, ls="--", label="Zero uplift")
ax.axvline(x=breakeven_uplift, color="green", lw=2, ls="-",
           label=f"Break-even ({breakeven_uplift:.4f})")
p90 = np.percentile(uplift_scores, 90)
ax.axvline(x=p90, color="orange", lw=2, ls="-",
           label=f"Top 10% threshold ({p90:.4f})")
ax.set_xlabel("Predicted Uplift Score")
ax.set_ylabel("Count")
ax.set_title("Uplift Distribution with Thresholds")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ---- Plot 4: Segment feature profiles (radar-style as grouped bars) ----
ax = axes[1, 1]
seg_means = {}
for seg_name, seg_mask in segments:
    short_name = seg_name.split("(")[0].strip()
    means = []
    for col in feature_cols:
        # Standardize: (segment_mean - overall_mean) / overall_std
        overall_mean = X_test_df[col].mean()
        overall_std = X_test_df[col].std()
        seg_mean = X_test_df[seg_mask][col].mean()
        means.append((seg_mean - overall_mean) / overall_std if overall_std > 0 else 0)
    seg_means[short_name] = means

x_pos = np.arange(len(feature_cols))
width = 0.25
seg_colors = {"Top 10%": "#9C27B0", "Middle 80%": "#B0BEC5", "Bottom 10%": "#FF6B6B"}

for i, (name, means) in enumerate(seg_means.items()):
    ax.bar(x_pos + i * width, means, width, label=name,
           color=seg_colors.get(name, "#333"), alpha=0.8)

ax.set_xticks(x_pos + width)
ax.set_xticklabels(feature_cols, fontsize=8)
ax.set_ylabel("Standardized Difference from Overall Mean")
ax.set_title("Feature Profiles by Segment")
ax.axhline(y=0, color="black", lw=0.5)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{BASE}/results/analysis/08_targeting_analysis.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved plot: {plot_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL TARGETING SUMMARY")
print("=" * 60)
print(f"""
  OPTIMAL STRATEGY (at ${AD_COST_PER_USER}/impression, ${CONVERSION_VALUE}/conversion):
    Target the top {optimal_pct}% of users by X-learner score
    Expected profit: ${max_profit:,.0f}
    vs. blanket targeting profit: ${blanket_profit_100:,.0f}
    Uplift model advantage: ${savings:,.0f}

  PRACTICAL RULES:
    - Predicted uplift > {breakeven_uplift:.4f}: profitable to target
    - Predicted uplift < 0: sleeping dogs — do NOT show ads
    - Top 10% has {ates_at_k[9]:.1%} ATE (vs {true_ate:.1%} overall = {ates_at_k[9]/true_ate:.1f}x)

  KEY FEATURES DRIVING UPLIFT:
    f8 and f0 are the strongest moderators of treatment effect.
    Users with low f8 and low f0 values respond most to ads.
""")

tee.close()
print("Done. Results saved to results/models/08_targeting_analysis.txt")
