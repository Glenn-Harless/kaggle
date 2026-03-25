"""
Marketing Mix Modeling: Step 9 — Channel Attribution & Budget Optimization

Turn the fitted MMM into a decision tool:
  1. ROAS by channel (return on ad spend with uncertainty)
  2. Response curves (where is each channel on the saturation curve?)
  3. Budget reallocation (where should we shift spend?)
  4. Scenario analysis ("what if we cut Sponsorship by 50%?")
  5. Honest caveats
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

import pymc as pm
import arviz as az

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/09_budget_optimization.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 9 — Budget Optimization")
print("=" * 60)


# ============================================================
# 1. REBUILD FITTED MODEL (same as Step 8)
# ============================================================

print("\n\n" + "=" * 60)
print("1. REBUILD FITTED MODEL")
print("=" * 60)

df = pd.read_csv(f"{DATA}/mmm_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

channels = ["spend_sponsorship", "spend_performance", "spend_brand", "spend_sem"]
controls = ["has_special_sale", "trend"]

mmm = MMM(
    date_column="date",
    channel_columns=channels,
    control_columns=controls,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)

X = df[["date"] + channels + controls]
y = df["total_gmv"]

print(f"  Fitting model (same config as Step 8)...")
idata = mmm.fit(X, y, random_seed=42, target_accept=0.95,
                draws=2000, tune=2000, progressbar=False)
print(f"  Done. {int(idata.sample_stats['diverging'].sum().values)} divergences.")

posterior = idata.posterior
channel_names = list(posterior.coords["channel"].values)


# ============================================================
# 2. ROAS BY CHANNEL — RETURN ON AD SPEND
# ============================================================
# ROAS = (revenue attributed to channel) / (spend on channel)
# A ROAS of 3.0x means: for every $1 spent, $3 in sales returned.
# ROAS > 1.0 → profitable (revenue exceeds cost)
# ROAS < 1.0 → unprofitable (spending more than you earn)
#
# We compute ROAS with uncertainty — not just one number.

print("\n\n" + "=" * 60)
print("2. ROAS BY CHANNEL (Return on Ad Spend)")
print("=" * 60)

ch_contrib = posterior["channel_contribution_original_scale"]

# Total contribution per channel across all weeks (in original scale)
# and total spend per channel
print(f"\n  {'Channel':20s}  {'Total Contrib':>15s}  {'Total Spend':>12s}  "
      f"{'ROAS Mean':>10s}  {'ROAS 95% HDI':>20s}")
print(f"  {'-'*85}")

roas_samples = {}
for ch in channel_names:
    # Contribution samples: sum across all weeks per posterior draw
    contrib_samples = ch_contrib.sel(channel=ch).values.reshape(-1, len(df)).sum(axis=1)
    total_spend = df[ch].sum()

    if total_spend > 0:
        roas = contrib_samples / total_spend
    else:
        roas = np.zeros_like(contrib_samples)

    roas_samples[ch] = roas
    hdi = az.hdi(roas, hdi_prob=0.95)
    label = ch.replace("spend_", "")
    print(f"  {label:20s}  {contrib_samples.mean():>15,.0f}  {total_spend:>12.1f}  "
          f"{roas.mean():>10,.0f}  [{hdi[0]:,.0f} — {hdi[1]:,.0f}]")

print(f"\n  NOTE: ROAS values are in original GMV units per spend unit.")
print(f"  The absolute numbers depend on how spend was scaled in the dataset.")
print(f"  What matters is the RELATIVE ranking and whether ROAS > 0.")

# Plot ROAS distributions
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["purple", "green", "steelblue", "orange"]
for ch, color in zip(channel_names, colors):
    roas = roas_samples[ch]
    # Clip extreme values for visualization
    roas_clipped = roas[roas < np.percentile(roas, 99)]
    label = ch.replace("spend_", "").title()
    ax.hist(roas_clipped, bins=50, density=True, alpha=0.4, color=color,
            edgecolor="white", linewidth=0.5,
            label=f"{label}: mean={roas.mean():,.0f}")

ax.set_title("ROAS Distributions by Channel\n(wider = more uncertain)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Return on Ad Spend (GMV per spend unit)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS}/09_roas_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 09_roas_distributions.png")


# ============================================================
# 3. RESPONSE CURVES — WHERE ARE WE ON THE SATURATION CURVE?
# ============================================================
# For each channel, show the saturation curve and mark where
# current spend sits. This tells you:
# - If you're on the steep part → increasing spend will help
# - If you're on the flat part → you're saturated, don't increase

print("\n\n" + "=" * 60)
print("3. RESPONSE CURVES — SATURATION POSITION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, ch, color in zip(axes.flat, channel_names, colors):
    label = ch.replace("spend_", "").title()

    # Get posterior samples of saturation lambda for this channel
    lam_samples = posterior["saturation_lam"].sel(channel=ch).values.flatten()
    beta_samples = posterior["saturation_beta"].sel(channel=ch).values.flatten()

    # Current average spend (after scaling — use normalized)
    current_spend = df[ch].mean()
    max_spend = df[ch].max()

    # Generate response curve using posterior mean
    x_spend = np.linspace(0, max_spend * 2, 200)

    # Plot multiple posterior draws to show uncertainty
    n_draws = 200
    idx = np.random.choice(len(lam_samples), n_draws, replace=False)
    for i in idx:
        lam = lam_samples[i]
        beta = beta_samples[i]
        response = beta * (1 - np.exp(-lam * x_spend))
        ax.plot(x_spend, response, alpha=0.02, color=color)

    # Mean response curve
    lam_mean = lam_samples.mean()
    beta_mean = beta_samples.mean()
    response_mean = beta_mean * (1 - np.exp(-lam_mean * x_spend))
    ax.plot(x_spend, response_mean, color=color, linewidth=2, label="Mean response")

    # Mark current spend level
    current_response = beta_mean * (1 - np.exp(-lam_mean * current_spend))
    ax.axvline(current_spend, color="red", linestyle="--", linewidth=1.5,
               label=f"Current spend: {current_spend:.1f}")
    ax.scatter([current_spend], [current_response], color="red", s=100, zorder=5)

    # How saturated are we?
    ceiling = beta_mean
    saturation_pct = current_response / ceiling * 100 if ceiling > 0 else 0

    ax.set_title(f"{label}\n(~{saturation_pct:.0f}% saturated at current spend)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Weekly spend")
    ax.set_ylabel("Effect on GMV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Response Curves: Where Is Each Channel on the Saturation Curve?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/09_response_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 09_response_curves.png")

print(f"\n  Saturation status per channel:")
for ch, color in zip(channel_names, colors):
    label = ch.replace("spend_", "")
    lam_mean = posterior["saturation_lam"].sel(channel=ch).values.mean()
    beta_mean = posterior["saturation_beta"].sel(channel=ch).values.mean()
    current = df[ch].mean()
    current_effect = beta_mean * (1 - np.exp(-lam_mean * current))
    ceiling = beta_mean
    sat_pct = current_effect / ceiling * 100 if ceiling > 0 else 0
    status = "Room to grow" if sat_pct < 70 else "Mostly saturated" if sat_pct < 90 else "Fully saturated"
    print(f"    {label:20s}  {sat_pct:>5.0f}% saturated  [{status}]")


# ============================================================
# 4. BUDGET REALLOCATION — THE OPTIMIZATION QUESTION
# ============================================================
# Given the same total budget, what's the optimal split?
# This is a simple exercise: shift spend FROM saturated channels
# TO channels with room to grow.

print("\n\n" + "=" * 60)
print("4. BUDGET REALLOCATION")
print("=" * 60)

# Current allocation
current_allocation = {ch: df[ch].sum() for ch in channel_names}
total_budget = sum(current_allocation.values())

print(f"\n  Current weekly budget allocation:")
print(f"  {'Channel':20s}  {'Spend':>10s}  {'% of Budget':>12s}")
print(f"  {'-'*45}")
for ch in channel_names:
    label = ch.replace("spend_", "")
    spend = current_allocation[ch]
    pct = spend / total_budget
    print(f"  {label:20s}  {spend:>10.1f}  {pct:>11.1%}")
print(f"  {'TOTAL':20s}  {total_budget:>10.1f}")

# Simple reallocation: shift 20% from the most saturated to the least
# This is a manual demonstration — in production you'd use BudgetOptimizer
print(f"\n  Simple reallocation scenario:")
print(f"  → Shift 20% from highest-spending channel (Sponsorship)")
print(f"  → To the highest-ROAS channel (Performance)")

shift_amount = current_allocation["spend_sponsorship"] * 0.20
scenario = current_allocation.copy()
scenario["spend_sponsorship"] -= shift_amount
scenario["spend_performance"] += shift_amount

print(f"\n  {'Channel':20s}  {'Current':>10s}  {'Proposed':>10s}  {'Change':>10s}")
print(f"  {'-'*55}")
for ch in channel_names:
    label = ch.replace("spend_", "")
    curr = current_allocation[ch]
    prop = scenario[ch]
    change = prop - curr
    print(f"  {label:20s}  {curr:>10.1f}  {prop:>10.1f}  {change:>+10.1f}")

# Estimate impact using posterior mean response curves
print(f"\n  Estimated impact (using posterior mean response curves):")
current_total_effect = 0
proposed_total_effect = 0
for ch in channel_names:
    lam_mean = posterior["saturation_lam"].sel(channel=ch).values.mean()
    beta_mean = posterior["saturation_beta"].sel(channel=ch).values.mean()
    alpha_mean = posterior["adstock_alpha"].sel(channel=ch).values.mean()

    # Current effect (simplified — using mean weekly spend)
    curr_weekly = current_allocation[ch] / len(df)
    prop_weekly = scenario[ch] / len(df)

    curr_effect = beta_mean * (1 - np.exp(-lam_mean * curr_weekly))
    prop_effect = beta_mean * (1 - np.exp(-lam_mean * prop_weekly))

    current_total_effect += curr_effect
    proposed_total_effect += prop_effect

    label = ch.replace("spend_", "")
    change_pct = (prop_effect - curr_effect) / curr_effect * 100 if curr_effect > 0 else 0
    print(f"    {label:20s}  current={curr_effect:.4f}  proposed={prop_effect:.4f}  change={change_pct:+.1f}%")

total_change = (proposed_total_effect - current_total_effect) / current_total_effect * 100
print(f"\n    Total media effect change: {total_change:+.1f}%")
print(f"    (Same total budget, just reallocated)")

# Plot current vs proposed
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Current allocation pie
ax = axes[0]
labels = [ch.replace("spend_", "").title() for ch in channel_names]
curr_vals = [current_allocation[ch] for ch in channel_names]
ax.pie(curr_vals, labels=labels, colors=colors, autopct="%1.0f%%",
       startangle=90, textprops={"fontsize": 11})
ax.set_title("Current Budget Allocation", fontsize=13, fontweight="bold")

# Proposed allocation pie
ax = axes[1]
prop_vals = [scenario[ch] for ch in channel_names]
ax.pie(prop_vals, labels=labels, colors=colors, autopct="%1.0f%%",
       startangle=90, textprops={"fontsize": 11})
ax.set_title(f"Proposed Reallocation\n(est. {total_change:+.1f}% total media effect)",
             fontsize=13, fontweight="bold")

plt.suptitle("Budget Reallocation: Same Total Budget, Different Split",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/09_budget_reallocation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 09_budget_reallocation.png")


# ============================================================
# 5. SCENARIO ANALYSIS — "WHAT IF" QUESTIONS
# ============================================================

print("\n\n" + "=" * 60)
print("5. SCENARIO ANALYSIS")
print("=" * 60)

scenarios = {
    "Cut Sponsorship 50%": {"spend_sponsorship": 0.5, "spend_performance": 1.0,
                             "spend_brand": 1.0, "spend_sem": 1.0},
    "Double Performance": {"spend_sponsorship": 1.0, "spend_performance": 2.0,
                            "spend_brand": 1.0, "spend_sem": 1.0},
    "Cut all by 30%":     {"spend_sponsorship": 0.7, "spend_performance": 0.7,
                            "spend_brand": 0.7, "spend_sem": 0.7},
    "Double Brand":       {"spend_sponsorship": 1.0, "spend_performance": 1.0,
                            "spend_brand": 2.0, "spend_sem": 1.0},
}

print(f"\n  {'Scenario':30s}  {'Budget Change':>15s}  {'Effect Change':>15s}")
print(f"  {'-'*65}")

for scenario_name, multipliers in scenarios.items():
    # Compute new budget
    new_budget = sum(current_allocation[ch] * multipliers[ch] for ch in channel_names)
    budget_change = (new_budget - total_budget) / total_budget

    # Compute new effect
    new_effect = 0
    for ch in channel_names:
        lam_mean = posterior["saturation_lam"].sel(channel=ch).values.mean()
        beta_mean = posterior["saturation_beta"].sel(channel=ch).values.mean()
        new_weekly = (current_allocation[ch] * multipliers[ch]) / len(df)
        new_effect += beta_mean * (1 - np.exp(-lam_mean * new_weekly))

    effect_change = (new_effect - current_total_effect) / current_total_effect

    print(f"  {scenario_name:30s}  {budget_change:>+14.1%}  {effect_change:>+14.1%}")

print(f"""
  KEY OBSERVATIONS:
  - Cutting Sponsorship 50% saves budget but may lose less effect
    than you'd think (if it's already saturated)
  - Doubling Performance could add significant effect (if not saturated)
  - Cutting all by 30% shows how much budget is "necessary" vs "wasted"
  - These estimates carry uncertainty — use the posterior distributions
    for confidence intervals on each scenario
""")


# ============================================================
# 6. HONEST CAVEATS
# ============================================================

print("=" * 60)
print("6. HONEST CAVEATS — WHAT TO TELL STAKEHOLDERS")
print("=" * 60)

print(f"""
  WHAT THESE RESULTS CAN SUPPORT:
  ─────────────────────────────────────────────────────
  ✓ Relative channel ranking (Performance likely #1)
  ✓ Directional budget shifts (less Sponsorship, more Performance)
  ✓ General saturation assessment (where channels have room to grow)
  ✓ Framework for future, better-data analysis

  WHAT THESE RESULTS CANNOT SUPPORT:
  ─────────────────────────────────────────────────────
  ✗ Precise ROAS numbers for budget justification
  ✗ Exact optimal allocation percentages
  ✗ Causal claims without experimental validation
  ✗ Predictions for untested spend levels (extrapolation)

  TO MAKE THESE RESULTS PRODUCTION-GRADE:
  ─────────────────────────────────────────────────────
  1. Get 2+ years of WEEKLY spend data (from ad platforms directly)
  2. Run geo experiments to calibrate channel priors
     → PyMC-Marketing: mmm.add_lift_test_measurements()
  3. Add more controls (competitor activity, pricing, weather)
  4. Validate with holdout period (last 8-12 weeks)
  5. Re-run quarterly as data accumulates

  FOR YOUR WORK (Google/Facebook/TikTok + CTV):
  ─────────────────────────────────────────────────────
  - You have 80-100 weeks → 2x better than DT Mart
  - You have weekly spend from ad platforms → no monthly distribution hack
  - You have a planned CTV geo experiment → can calibrate CTV prior
  - 3 main channels → fewer params, tighter posteriors
  - This puts you in a MUCH better position than this exercise
""")


# ============================================================
# 7. FINAL SUMMARY — THE COMPLETE MMM DELIVERABLE
# ============================================================

print("=" * 60)
print("7. EXECUTIVE SUMMARY — DT MART MMM")
print("=" * 60)

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║           DT MART — MARKETING MIX MODEL RESULTS             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  CHANNEL EFFECTIVENESS RANKING:                              ║
  ║    #1  Performance (Online + Affiliates)  — 62% of media     ║
  ║    #2  Brand (TV + Digital)               — 15% of media     ║
  ║    #3  Sponsorship                        — 12% of media     ║
  ║    #4  SEM                                — 11% of media     ║
  ║                                                              ║
  ║  CONFIDENCE:                                                 ║
  ║    91% confident Performance > Brand                         ║
  ║    94% confident Performance > SEM                           ║
  ║    Cannot distinguish Brand vs Sponsorship vs SEM            ║
  ║                                                              ║
  ║  RECOMMENDATION:                                             ║
  ║    Consider shifting budget FROM Sponsorship (47% of spend,  ║
  ║    12% of effect) TO Performance (33% of spend, 62% of       ║
  ║    effect). Validate with a controlled test before acting.    ║
  ║                                                              ║
  ║  CAVEATS:                                                    ║
  ║    - 47 weeks of data (below 100+ week minimum)              ║
  ║    - Monthly spend granularity (weekly preferred)             ║
  ║    - Channel correlations limit individual attribution        ║
  ║    - Validate recommendations with geo experiments            ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
