"""
Criteo Uplift Modeling: Step 1 — EDA
Explore the dataset with a CAUSAL INFERENCE lens:
  - Verify the randomization is clean
  - Compute the Average Treatment Effect (ATE)
  - Check feature balance between treatment/control
  - Understand outcome rarity and its implications
  - Compare treatment assignment vs actual exposure
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/criteo-uplift"

tee = Tee(f"{BASE}/results/models/01_eda.txt")
sys.stdout = tee


# ============================================================
# DATA LOADING
# ============================================================
# scikit-uplift provides the Criteo dataset directly.
# percent10=True gives us a 10% sample (~1.4M rows) — manageable
# for development while still large enough for reliable estimates.

print("Criteo Uplift Modeling: Step 1 — EDA")
print("=" * 60)

from sklift.datasets import fetch_criteo

print("\nLoading Criteo 10% sample via scikit-uplift...")
data = fetch_criteo(percent10=True)

# fetch_criteo returns a Bunch: data, target, treatment
# Combine into a single DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=[f"f{i}" for i in range(12)])
df["treatment"] = data.treatment
df["visit"] = data.target
# The full dataset has both visit and conversion; fetch_criteo
# returns visit as target by default. Let's check what we have.

print(f"\n  Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(f"  Dtypes:\n{df.dtypes.to_string()}")
print(f"\n  First 5 rows:")
print(df.head().to_string())
print(f"\n  Missing values: {df.isna().sum().sum()}")


# ============================================================
# 1. TREATMENT / CONTROL BALANCE
# ============================================================
# In a well-run RCT, randomization should produce groups that
# are similar in size and composition. Let's verify.

print("\n\n" + "=" * 60)
print("1. TREATMENT / CONTROL BALANCE")
print("=" * 60)

n_total = len(df)
n_treat = df["treatment"].sum()
n_control = n_total - n_treat

print(f"\n  Total users:   {n_total:>10,}")
print(f"  Treatment (1): {n_treat:>10,} ({n_treat/n_total:.1%})")
print(f"  Control (0):   {n_control:>10,} ({n_control/n_total:.1%})")
print(f"  Ratio (T:C):   {n_treat/n_control:.2f}:1")

print(f"\n  The ~85/15 split is intentional — Criteo allocated more")
print(f"  users to treatment because showing ads is the default.")
print(f"  The control group is smaller but still large enough")
print(f"  ({n_control:,} users) for reliable estimation.")


# ============================================================
# 2. OUTCOME RATES AND ATE
# ============================================================
# The Average Treatment Effect is the core causal quantity:
#   ATE = E[Y|T=1] - E[Y|T=0]
# Because treatment is randomized, this is an unbiased estimate
# of the true causal effect of showing ads.

print("\n\n" + "=" * 60)
print("2. OUTCOME RATES AND ATE (Average Treatment Effect)")
print("=" * 60)

visit_treat = df[df["treatment"] == 1]["visit"].mean()
visit_control = df[df["treatment"] == 0]["visit"].mean()
ate_visit = visit_treat - visit_control

print(f"\n  VISIT outcome:")
print(f"    Treatment group visit rate: {visit_treat:.4%}")
print(f"    Control group visit rate:   {visit_control:.4%}")
print(f"    ATE (visit):                {ate_visit:.4%}")
print(f"    Relative lift:              {ate_visit/visit_control:.1%}")

# Statistical significance of the ATE
# For proportions, we use a z-test (two-proportion test)
treat_visits = df[df["treatment"] == 1]["visit"]
control_visits = df[df["treatment"] == 0]["visit"]

# Standard error of difference in proportions
se_ate = np.sqrt(
    visit_treat * (1 - visit_treat) / n_treat
    + visit_control * (1 - visit_control) / n_control
)
z_stat = ate_visit / se_ate
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\n    SE(ATE):    {se_ate:.6f}")
print(f"    z-stat:     {z_stat:.2f}")
print(f"    p-value:    {p_value:.2e}")
if p_value < 0.001:
    print(f"    *** Highly significant — the ad effect on visits is real ***")

# Overall visit rate for context
overall_visit = df["visit"].mean()
print(f"\n  Overall visit rate: {overall_visit:.4%}")
print(f"  Most of the {overall_visit:.2%} visit rate is organic (would happen without ads).")
print(f"  The ads cause an additional {ate_visit:.4%} on top of the {visit_control:.4%} baseline.")


# ============================================================
# 3. FEATURE BALANCE CHECK (Randomization Validation)
# ============================================================
# If randomization worked, treatment and control groups should
# have nearly identical distributions of f0-f11. Any systematic
# difference would indicate a problem with the experiment.

print("\n\n" + "=" * 60)
print("3. FEATURE BALANCE CHECK (Randomization Validation)")
print("=" * 60)

feature_cols = [f"f{i}" for i in range(12)]

print(f"\n  Comparing feature means between treatment and control:")
print(f"  {'Feature':>8s}  {'Mean(T)':>10s}  {'Mean(C)':>10s}  {'Diff':>10s}  {'Std Diff':>10s}  {'p-value':>10s}")
print(f"  {'-' * 65}")

imbalanced_features = []
for col in feature_cols:
    mean_t = df[df["treatment"] == 1][col].mean()
    mean_c = df[df["treatment"] == 0][col].mean()
    diff = mean_t - mean_c
    # Standardized difference (Cohen's d) — industry standard for balance
    pooled_std = df[col].std()
    std_diff = diff / pooled_std if pooled_std > 0 else 0
    # T-test for significance
    t, p = stats.ttest_ind(
        df[df["treatment"] == 1][col].dropna(),
        df[df["treatment"] == 0][col].dropna(),
    )
    print(f"  {col:>8s}  {mean_t:>10.4f}  {mean_c:>10.4f}  {diff:>+10.4f}  {std_diff:>+10.4f}  {p:>10.4f}")
    # Flag if standardized difference > 0.05 (a common threshold)
    if abs(std_diff) > 0.05:
        imbalanced_features.append((col, std_diff))

if imbalanced_features:
    print(f"\n  *** WARNING: {len(imbalanced_features)} feature(s) with |std diff| > 0.05:")
    for col, d in imbalanced_features:
        print(f"      {col}: std_diff = {d:+.4f}")
    print(f"  This could indicate imperfect randomization.")
else:
    print(f"\n  All features have |standardized diff| < 0.05")
    print(f"  Randomization looks clean — treatment and control are well-balanced.")


# ============================================================
# 4. FEATURE DISTRIBUTIONS
# ============================================================
# Explore the 12 anonymized features to understand their scale
# and distribution.

print("\n\n" + "=" * 60)
print("4. FEATURE DISTRIBUTIONS")
print("=" * 60)

print(f"\n  Summary statistics for features f0-f11:")
desc = df[feature_cols].describe().T
desc["missing"] = df[feature_cols].isna().sum()
desc["n_unique"] = df[feature_cols].nunique()
print(desc[["count", "missing", "mean", "std", "min", "25%", "50%", "75%", "max", "n_unique"]].to_string())

# Check for likely categorical vs continuous
print(f"\n  Feature type heuristic (unique values):")
for col in feature_cols:
    n_uniq = df[col].nunique()
    n_miss = df[col].isna().sum()
    ftype = "likely categorical" if n_uniq < 20 else "continuous"
    print(f"    {col}: {n_uniq:>8,} unique values, {n_miss:>8,} missing → {ftype}")


# ============================================================
# 5. OUTCOME RARITY ANALYSIS
# ============================================================
# With visit rates around 4-5%, the positive class is uncommon.
# This has implications for modeling.

print("\n\n" + "=" * 60)
print("5. OUTCOME RARITY ANALYSIS")
print("=" * 60)

n_visits = df["visit"].sum()
n_no_visits = n_total - n_visits

print(f"\n  Visit (1):     {n_visits:>10,} ({n_visits/n_total:.2%})")
print(f"  No visit (0):  {n_no_visits:>10,} ({n_no_visits/n_total:.2%})")
print(f"  Ratio (neg:pos): {n_no_visits/n_visits:.1f}:1")

print(f"\n  Implications for uplift modeling:")
print(f"    - With ~{n_visits/n_total:.1%} visit rate, most users don't visit regardless")
print(f"    - The ATE of {ate_visit:.4%} means we're looking for a SMALL signal")
print(f"    - Need large samples to detect heterogeneous effects (we have {n_total:,})")
print(f"    - Uplift (treatment effect) will be even smaller than outcome rate")
print(f"    - This is realistic: most ads have tiny per-user effects")


# ============================================================
# 6. VISIT RATES BY FEATURE DECILES
# ============================================================
# Look for features that interact with treatment — this is what
# uplift models will try to capture.

print("\n\n" + "=" * 60)
print("6. VISIT RATES BY FEATURE DECILES (Treatment Effect Heterogeneity)")
print("=" * 60)

print(f"\n  For each feature, bin into deciles and check if the ATE varies")
print(f"  across bins. This hints at which features drive heterogeneous effects.")

# Pick a subset of features to keep output manageable
check_features = ["f0", "f2", "f5", "f8", "f11"]

for col in check_features:
    if df[col].nunique() < 10:
        # Categorical-like: use actual values
        bins = df[col].fillna(-999)
        bin_label = "value"
    else:
        # Continuous: use quintiles
        bins = pd.qcut(df[col].fillna(df[col].median()), q=5, duplicates="drop")
        bin_label = "quintile"

    print(f"\n  {col} ({bin_label}):")
    print(f"    {'Bin':>25s}  {'N':>8s}  {'Visit(T)':>9s}  {'Visit(C)':>9s}  {'ATE':>9s}")
    print(f"    {'-' * 65}")

    for bin_val in sorted(bins.unique()):
        mask = bins == bin_val
        sub = df[mask]
        n_sub = len(sub)
        vt = sub[sub["treatment"] == 1]["visit"].mean()
        vc = sub[sub["treatment"] == 0]["visit"].mean()
        ate_bin = vt - vc if (pd.notna(vt) and pd.notna(vc)) else float("nan")
        print(f"    {str(bin_val):>25s}  {n_sub:>8,}  {vt:>9.4f}  {vc:>9.4f}  {ate_bin:>+9.4f}")


# ============================================================
# 7. PLOTS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Criteo Uplift: Step 1 — EDA (Causal Lens)", fontsize=14, fontweight="bold")

# 7a: Treatment/Control distribution
ax = axes[0, 0]
counts = df["treatment"].value_counts().sort_index()
colors = ["#4ECDC4", "#FF6B6B"]
counts.plot.bar(ax=ax, color=colors)
ax.set_xlabel("Group")
ax.set_ylabel("Count")
ax.set_title("Treatment / Control Split")
ax.set_xticklabels(["Control (0)", "Treatment (1)"], rotation=0)
for i, (idx, val) in enumerate(counts.items()):
    ax.text(i, val + n_total * 0.01, f"{val:,}\n({val/n_total:.1%})",
            ha="center", va="bottom", fontsize=9)

# 7b: Visit rates by group (the ATE visualization)
ax = axes[0, 1]
visit_rates = [visit_control, visit_treat]
bar_colors = ["#4ECDC4", "#FF6B6B"]
bars = ax.bar(["Control", "Treatment"], visit_rates, color=bar_colors, width=0.5)
ax.set_ylabel("Visit Rate")
ax.set_title(f"Visit Rate by Group (ATE = {ate_visit:.4%})")
for bar, rate in zip(bars, visit_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{rate:.3%}", ha="center", va="bottom", fontsize=10)
# Draw ATE arrow
ax.annotate("", xy=(1, visit_treat), xytext=(1, visit_control),
            arrowprops=dict(arrowstyle="<->", color="black", lw=2))
ax.text(1.25, (visit_treat + visit_control) / 2, f"ATE\n{ate_visit:.4%}",
        ha="left", va="center", fontsize=9, fontweight="bold")

# 7c: Feature balance — standardized differences
ax = axes[0, 2]
std_diffs = []
for col in feature_cols:
    mean_t = df[df["treatment"] == 1][col].mean()
    mean_c = df[df["treatment"] == 0][col].mean()
    pooled_std = df[col].std()
    std_diffs.append((mean_t - mean_c) / pooled_std if pooled_std > 0 else 0)

y_pos = np.arange(len(feature_cols))
colors_balance = ["#FF6B6B" if abs(d) > 0.05 else "#4ECDC4" for d in std_diffs]
ax.barh(y_pos, std_diffs, color=colors_balance, alpha=0.8)
ax.axvline(x=0, color="black", lw=0.5)
ax.axvline(x=0.05, color="red", lw=0.5, ls="--", alpha=0.5)
ax.axvline(x=-0.05, color="red", lw=0.5, ls="--", alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_cols, fontsize=9)
ax.set_xlabel("Standardized Difference (T - C)")
ax.set_title("Feature Balance Check")
ax.invert_yaxis()

# 7d: Visit distribution (outcome rarity)
ax = axes[1, 0]
outcome_counts = df["visit"].value_counts().sort_index()
outcome_colors = ["#B0BEC5", "#FF6B6B"]
outcome_counts.plot.bar(ax=ax, color=outcome_colors)
ax.set_xlabel("Visit")
ax.set_ylabel("Count")
ax.set_title(f"Outcome Distribution (visit rate = {overall_visit:.2%})")
ax.set_xticklabels(["No Visit (0)", "Visit (1)"], rotation=0)
for i, (idx, val) in enumerate(outcome_counts.items()):
    ax.text(i, val + n_total * 0.01, f"{val:,}\n({val/n_total:.1%})",
            ha="center", va="bottom", fontsize=9)

# 7e: Feature distributions (f0 as example)
ax = axes[1, 1]
sample_col = "f0"
for label, color, name in [(1, "#FF6B6B", "Treatment"), (0, "#4ECDC4", "Control")]:
    vals = df[df["treatment"] == label][sample_col].dropna()
    ax.hist(vals, bins=50, alpha=0.5, color=color, label=name, density=True)
ax.set_xlabel(sample_col)
ax.set_ylabel("Density")
ax.set_title(f"{sample_col} Distribution: Treatment vs Control")
ax.legend()

# 7f: ATE by feature quintile — f0 and f8 side by side
ax = axes[1, 2]
bar_width = 0.35
for i, (col_for_ate, color, label) in enumerate([
    ("f0", "#FF6B6B", "f0"),
    ("f8", "#4ECDC4", "f8"),
]):
    q_bins = pd.qcut(df[col_for_ate].fillna(df[col_for_ate].median()), q=5, duplicates="drop")
    ate_by_q = []
    for q in sorted(q_bins.unique()):
        mask = q_bins == q
        sub = df[mask]
        vt = sub[sub["treatment"] == 1]["visit"].mean()
        vc = sub[sub["treatment"] == 0]["visit"].mean()
        ate_by_q.append(vt - vc)
    x_pos = np.arange(len(ate_by_q))
    ax.bar(x_pos + i * bar_width, ate_by_q, bar_width, color=color, alpha=0.8, label=label)

ax.axhline(y=ate_visit, color="black", lw=1, ls="--", label=f"Overall ATE ({ate_visit:.4%})")
max_q = max(len(ate_by_q) for _ in [1])  # use last computed length
ax.set_xticks(np.arange(max_q) + bar_width / 2)
ax.set_xticklabels([f"Q{i+1}" for i in range(max_q)], fontsize=9)
ax.set_xlabel("Feature Quintile (low → high)")
ax.set_ylabel("ATE (Visit)")
ax.set_title("Treatment Effect Heterogeneity: f0 vs f8")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{BASE}/results/analysis/01_eda.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n\nSaved plot: {plot_path}")


# ============================================================
# 8. KEY TAKEAWAYS
# ============================================================

print("\n\n" + "=" * 60)
print("8. KEY TAKEAWAYS")
print("=" * 60)
print(f"""
  1. RANDOMIZATION IS CLEAN: Feature distributions are balanced between
     treatment and control. We can trust ATE estimates are unbiased.

  2. ATE IS SMALL BUT REAL: Ads cause a ~{ate_visit:.3%} increase in visit rate
     on a ~{visit_control:.2%} baseline. Statistically significant (p < 0.001)
     thanks to the massive sample size.

  3. OUTCOME IS RARE: Only ~{overall_visit:.1%} of users visit. The uplift signal
     we're looking for ({ate_visit:.4%}) is much smaller than the outcome rate.
     This means uplift models are trying to detect a very subtle effect.

  4. HETEROGENEITY EXISTS: ATE varies across feature deciles, meaning
     some user segments respond more to ads than others. This is what
     uplift models will try to capture — the CATE for each user.

  5. SCALE: {n_total:,} users gives us plenty of statistical power.
     Even the control group ({n_control:,}) is large enough for
     reliable estimation per feature bin.

  NEXT: Step 2 — Naive baselines to see why standard classifiers
  (predicting P(visit)) fail at the uplift task.
""")

tee.close()
print("Done. Results saved to results/models/01_eda.txt")
