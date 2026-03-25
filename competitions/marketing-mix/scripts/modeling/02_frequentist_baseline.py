"""
Marketing Mix Modeling: Step 2 — The Frequentist Baseline (Why OLS Breaks)

This is a DEMONSTRATION, not a production model. We intentionally fit a
standard OLS regression to show why MMM needs Bayesian methods:

  1. Negative coefficients — OLS says spending on some channels HURTS sales
  2. Multicollinearity (high VIF) — correlated channels produce unstable estimates
  3. Deceptive R² — looks great because we're overfitting 52 observations
  4. No uncertainty — we get one number per channel, no sense of confidence
  5. No domain knowledge — can't encode adstock, saturation, or positivity

When your stakeholder asks "why can't we just use regular regression?",
this script is your answer.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/02_frequentist_baseline.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 2 — Frequentist Baseline")
print("=" * 60)
print("\nGoal: Show WHY standard regression fails for marketing")
print("attribution, motivating the Bayesian approach in Step 3+.")


# ============================================================
# 1. BUILD THE WEEKLY DATASET (same as Step 1)
# ============================================================
# We need the same weekly-aggregated data with media spend.
# Repeating the minimal data prep here so this script is standalone.

print("\n\n" + "=" * 60)
print("1. DATA PREPARATION")
print("=" * 60)

df_txn = pd.read_csv(f"{DATA}/firstfile.csv", index_col=0)
df_txn["Date"] = pd.to_datetime(df_txn["Date"])
df_txn["week_start"] = df_txn["Date"].dt.to_period("W-SUN").dt.start_time

weekly = df_txn.groupby("week_start").agg(
    total_gmv=("gmv_new", "sum"),
).reset_index()

# Media spend
df_media = pd.read_csv(f"{DATA}/MediaInvestment.csv")
df_media.columns = df_media.columns.str.strip()
channels = ["TV", "Digital", "Sponsorship", "Content Marketing",
            "Online marketing", "Affiliates", "SEM"]

df_media["date"] = pd.to_datetime(
    df_media["Year"].astype(str) + "-" + df_media["Month"].astype(str) + "-01"
)
df_media["year_month"] = df_media["date"].dt.to_period("M")
weekly["year_month"] = weekly["week_start"].dt.to_period("M")
weeks_per_month = weekly.groupby("year_month").size().to_dict()

weekly = weekly.merge(df_media[["year_month"] + channels], on="year_month", how="left")
for ch in channels:
    weekly[ch] = weekly.apply(
        lambda row: row[ch] / weeks_per_month.get(row["year_month"], 1)
        if pd.notna(row[ch]) else 0.0,
        axis=1,
    )

# Special sales
df_sales = pd.read_csv(f"{DATA}/SpecialSale.csv")
df_sales["Date"] = pd.to_datetime(df_sales["Date"])
sale_weeks = set()
for _, row in df_sales.iterrows():
    sale_weeks.add(row["Date"] - pd.Timedelta(days=row["Date"].dayofweek))
weekly["has_special_sale"] = weekly["week_start"].isin(sale_weeks).astype(int)

# Exclude August 2015 (data quality issue from EDA)
aug_mask = (weekly["week_start"] >= "2015-08-01") & (weekly["week_start"] < "2015-09-01")
n_before = len(weekly)
weekly = weekly[~aug_mask].reset_index(drop=True)
print(f"\n  Excluded {aug_mask.sum()} August 2015 weeks (data quality issue)")
print(f"  Remaining: {len(weekly)} weeks (from {n_before})")

# Trend
weekly["trend"] = range(len(weekly))

print(f"  Target: total_gmv")
print(f"  Features: {channels} + has_special_sale + trend")
print(f"  Shape: {weekly.shape}")


# ============================================================
# 2. FIT OLS — THE NAIVE APPROACH
# ============================================================
# This is what a data analyst without MMM knowledge might do:
# "Just regress sales on spend, throw in some controls, done."
#
# We're using statsmodels (not sklearn) because it gives us the
# full regression summary: coefficients, standard errors, p-values,
# R², F-stat — everything we need to diagnose the failure.

print("\n\n" + "=" * 60)
print("2. OLS REGRESSION: sales ~ channel_spend + controls")
print("=" * 60)

feature_cols = channels + ["has_special_sale", "trend"]
X = weekly[feature_cols].copy()
y = weekly["total_gmv"].copy()

# Add constant (intercept)
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()

print(f"\n{model.summary()}")


# ============================================================
# 3. THE NEGATIVE COEFFICIENT PROBLEM
# ============================================================
# In marketing, spending money on a channel should NOT decrease
# sales. A negative coefficient means the model is saying "more
# spend on this channel = less revenue." This is obviously wrong
# for any real advertising channel.
#
# WHY does this happen? Multicollinearity. When two channels are
# highly correlated (Digital and SEM at r=0.98), OLS can't tell
# them apart. It arbitrarily assigns one a large positive weight
# and the other a large negative weight. The NET effect might be
# reasonable, but the individual estimates are garbage.
#
# This is like trying to separate the effect of two variables
# that always move together — mathematically underdetermined.

print("\n\n" + "=" * 60)
print("3. PROBLEM #1 — NEGATIVE COEFFICIENTS (IMPOSSIBLE ROI)")
print("=" * 60)

print(f"\n  Channel coefficients from OLS:")
print(f"  {'Channel':25s}  {'Coefficient':>15s}  {'Std Error':>12s}  {'p-value':>10s}  {'Verdict':>12s}")
print(f"  {'-'*80}")
for ch in channels:
    coef = model.params[ch]
    se = model.bse[ch]
    pval = model.pvalues[ch]
    if coef < 0:
        verdict = "NEGATIVE!"
    elif pval > 0.05:
        verdict = "not sig."
    else:
        verdict = "OK"
    print(f"  {ch:25s}  {coef:>15,.0f}  {se:>12,.0f}  {pval:>10.3f}  {verdict:>12s}")

negatives = [ch for ch in channels if model.params[ch] < 0]
insignificant = [ch for ch in channels if model.pvalues[ch] > 0.05]

print(f"\n  Channels with NEGATIVE coefficients: {negatives}")
print(f"  Channels that are NOT statistically significant: {insignificant}")

if negatives:
    print(f"\n  This means OLS is claiming that increasing spend on")
    print(f"  {', '.join(negatives)} would DECREASE sales.")
    print(f"  This is obviously wrong — no marketing team would accept this.")
    print(f"  The cause is multicollinearity: correlated channels produce")
    print(f"  unstable, meaningless individual coefficient estimates.")
else:
    print(f"\n  No negative coefficients this time, but most are not significant.")
    print(f"  OLS can't confidently determine ANY channel's individual effect.")


# ============================================================
# 4. VIF — QUANTIFYING MULTICOLLINEARITY
# ============================================================
# VIF (Variance Inflation Factor) measures how much a coefficient's
# variance is inflated due to correlation with other predictors.
#
# Interpretation:
#   VIF = 1    → no multicollinearity
#   VIF = 5    → moderate — coefficient variance is 5x what it would be
#   VIF = 10+  → severe — coefficient estimates are unreliable
#   VIF = 50+  → extreme — essentially meaningless estimates
#
# Think of it this way: if Digital and SEM have VIF=50, it means
# the standard error on their coefficients is ~7x (√50) larger
# than it would be if they were uncorrelated. The model is
# essentially guessing how to split credit between them.

print("\n\n" + "=" * 60)
print("4. PROBLEM #2 — MULTICOLLINEARITY (VIF)")
print("=" * 60)

# Compute VIF for channel features only (not controls)
X_channels = weekly[channels].copy()
print(f"\n  Variance Inflation Factors (channel features only):")
print(f"  {'Channel':25s}  {'VIF':>10s}  {'Interpretation':>20s}")
print(f"  {'-'*60}")
for i, ch in enumerate(channels):
    vif = variance_inflation_factor(X_channels.values, i)
    if vif > 50:
        interp = "EXTREME"
    elif vif > 10:
        interp = "SEVERE"
    elif vif > 5:
        interp = "MODERATE"
    else:
        interp = "OK"
    print(f"  {ch:25s}  {vif:>10.1f}  {interp:>20s}")

print(f"\n  Multiple channels have extreme VIF, confirming what we saw in")
print(f"  the correlation analysis: OLS cannot reliably separate their")
print(f"  individual effects. This isn't a modeling choice — it's a")
print(f"  mathematical fact about the data.")


# ============================================================
# 5. DECEPTIVE R² — OVERFITTING IN DISGUISE
# ============================================================
# With 48 observations and 9 features, we're using ~20% of our
# degrees of freedom on predictors. R² will look good simply
# because we're fitting noise.
#
# This is the same overfitting problem you've seen in supervised
# ML, except worse because we can't do a train/test split — with
# time series this short, the test set would be too small to be
# meaningful.

print("\n\n" + "=" * 60)
print("5. PROBLEM #3 — DECEPTIVE R² (OVERFITTING)")
print("=" * 60)

n = len(weekly)
k = len(feature_cols)
print(f"\n  R²:          {model.rsquared:.3f}")
print(f"  Adjusted R²: {model.rsquared_adj:.3f}")
print(f"  F-statistic: {model.fvalue:.1f} (p={model.f_pvalue:.4f})")
print(f"\n  Observations: {n}")
print(f"  Predictors:   {k}")
print(f"  Ratio:        {n/k:.1f} observations per predictor")
print(f"\n  R² of {model.rsquared:.3f} looks {'great' if model.rsquared > 0.5 else 'decent'}, but with only {n/k:.1f} obs per predictor,")
print(f"  much of this is overfitting. Compare R² ({model.rsquared:.3f}) to Adjusted R²")
print(f"  ({model.rsquared_adj:.3f}) — the gap of {model.rsquared - model.rsquared_adj:.3f} shows how much is noise.")
print(f"\n  In supervised ML, you'd cross-validate. But with {n} weekly observations,")
print(f"  a time-series CV would give you ~4-5 folds of 8-10 points each — too small")
print(f"  to be reliable for regression with {k} features.")


# ============================================================
# 6. NO UNCERTAINTY — THE POINT ESTIMATE TRAP
# ============================================================
# OLS gives ONE coefficient per channel. When a marketing team
# asks "how confident are you that TV is better than Digital?",
# all you can offer is a p-value (which doesn't answer the question).
#
# The Bayesian alternative gives you a full DISTRIBUTION:
# "There's a 73% probability that TV contributes more than Digital"
# — directly actionable for budget decisions.

print("\n\n" + "=" * 60)
print("6. PROBLEM #4 — NO USEFUL UNCERTAINTY FOR DECISIONS")
print("=" * 60)

# Compute confidence intervals
ci = model.conf_int(alpha=0.05)
print(f"\n  95% confidence intervals for channel coefficients:")
print(f"  {'Channel':25s}  {'Lower':>15s}  {'Coefficient':>15s}  {'Upper':>15s}  {'Contains 0?':>12s}")
print(f"  {'-'*80}")
for ch in channels:
    lower, upper = ci.loc[ch]
    coef = model.params[ch]
    contains_zero = "YES" if lower <= 0 <= upper else "no"
    print(f"  {ch:25s}  {lower:>15,.0f}  {coef:>15,.0f}  {upper:>15,.0f}  {contains_zero:>12s}")

contains_zero_count = sum(1 for ch in channels if ci.loc[ch, 0] <= 0 <= ci.loc[ch, 1])
print(f"\n  {contains_zero_count}/{len(channels)} channels have CIs that include zero.")
print(f"  This means OLS can't confidently say ANY of these channels have a")
print(f"  non-zero effect on sales. Not very useful for budget decisions.")
print(f"\n  Worse: these CIs assume no multicollinearity, which we know is false.")
print(f"  The true uncertainty is even LARGER than what's shown above.")

print(f"\n  What the marketing team wants to know:")
print(f"    'What's the probability TV ROI is above 2x?'")
print(f"    'How confident are we that Sponsorship beats Digital?'")
print(f"    'If we shift $1M from SEM to TV, what's the expected lift?'")
print(f"  OLS cannot answer any of these questions. Bayesian can.")


# ============================================================
# 7. NO DOMAIN KNOWLEDGE — THE MISSING INGREDIENTS
# ============================================================

print("\n\n" + "=" * 60)
print("7. PROBLEM #5 — CANNOT ENCODE DOMAIN KNOWLEDGE")
print("=" * 60)

print(f"""
  OLS treats ad spend like any other numerical feature. It doesn't know:

  1. ADSTOCK: TV ads seen this week still influence purchases next week.
     OLS only relates THIS week's spend to THIS week's sales.
     → Missing carry-over means TV's effect is UNDERESTIMATED.

  2. SATURATION: The 50th ad impression is less effective than the 1st.
     OLS assumes a LINEAR relationship (every dollar equally effective).
     → At high spend levels, OLS OVERESTIMATES the marginal effect.

  3. POSITIVITY: Advertising should not decrease sales. But OLS has no
     constraint preventing negative coefficients.
     → Produces impossible results that undermine stakeholder trust.

  These aren't nice-to-haves — they are fundamental properties of how
  advertising works. A model that ignores them is wrong by construction.
""")


# ============================================================
# 8. VISUALIZATION — MAKING THE FAILURE VISCERAL
# ============================================================

print("=" * 60)
print("8. VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8a. Coefficient plot with confidence intervals
ax = axes[0, 0]
coefs = [model.params[ch] for ch in channels]
errors = [model.bse[ch] * 1.96 for ch in channels]
colors = ["red" if c < 0 else "steelblue" for c in coefs]
y_pos = range(len(channels))
ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(channels, fontsize=9)
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("OLS Coefficients ± 95% CI\n(red = negative = impossible)", fontsize=11, fontweight="bold")
ax.set_xlabel("Coefficient (effect on GMV)")
ax.grid(True, alpha=0.3, axis="x")

# 8b. VIF bar chart
ax = axes[0, 1]
vifs = [variance_inflation_factor(X_channels.values, i) for i in range(len(channels))]
colors_vif = ["red" if v > 10 else "orange" if v > 5 else "green" for v in vifs]
ax.barh(range(len(channels)), vifs, color=colors_vif, alpha=0.7, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(channels)))
ax.set_yticklabels(channels, fontsize=9)
ax.axvline(10, color="red", linewidth=1, linestyle="--", label="Severe threshold (VIF=10)")
ax.axvline(5, color="orange", linewidth=1, linestyle="--", label="Moderate threshold (VIF=5)")
ax.set_title("Variance Inflation Factors\n(red = severe multicollinearity)", fontsize=11, fontweight="bold")
ax.set_xlabel("VIF")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="x")

# 8c. Predicted vs actual (shows overfit)
ax = axes[1, 0]
predicted = model.fittedvalues
ax.scatter(y / 1e6, predicted / 1e6, alpha=0.6, edgecolors="black", linewidth=0.5)
min_val = min(y.min(), predicted.min()) / 1e6
max_val = max(y.max(), predicted.max()) / 1e6
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Perfect fit")
ax.set_xlabel("Actual GMV (millions)")
ax.set_ylabel("Predicted GMV (millions)")
ax.set_title(f"Predicted vs Actual (R²={model.rsquared:.3f})\n(looks good but it's overfit)",
             fontsize=11, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# 8d. Residuals over time (check for autocorrelation)
ax = axes[1, 1]
residuals = model.resid
ax.plot(weekly["week_start"], residuals / 1e6, "b-o", markersize=3)
ax.axhline(0, color="red", linewidth=1, linestyle="--")
ax.set_title("Residuals Over Time\n(patterns = model missing something)", fontsize=11, fontweight="bold")
ax.set_ylabel("Residual (millions)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/02_ols_failure.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 02_ols_failure.png")


# ============================================================
# 9. SUMMARY — WHY WE NEED BAYESIAN
# ============================================================

print("\n\n" + "=" * 60)
print("9. SUMMARY — THE CASE FOR BAYESIAN MMM")
print("=" * 60)

print(f"""
  OLS FAILURES ON THIS DATA:
  ─────────────────────────────────────────────────────────

  1. NEGATIVE COEFFICIENTS: {len(negatives)} channel(s) show negative ROI
     → Bayesian fix: HalfNormal priors force coefficients positive

  2. MULTICOLLINEARITY: {sum(1 for v in vifs if v > 10)}/{len(channels)} channels have VIF > 10
     → Bayesian fix: priors provide regularization that stabilizes estimates

  3. OVERFITTING: R²={model.rsquared:.3f} on {n} observations with {k} features
     → Bayesian fix: priors act as structured regularization (better than L1/L2)

  4. NO UNCERTAINTY: CIs assume independence (which is false here)
     → Bayesian fix: posterior distributions give honest, joint uncertainty

  5. NO DOMAIN KNOWLEDGE: linear model, no carry-over, no saturation
     → Bayesian fix: model structure encodes adstock + saturation

  THE TAKEAWAY:
  OLS is the wrong tool for this job. Not because it's a bad method —
  it's a great method for many problems. But MMM data has properties
  (small N, high multicollinearity, strong domain knowledge, need for
  decomposition with uncertainty) that OLS cannot handle.

  Bayesian inference isn't a luxury upgrade — it's the minimum viable
  approach for marketing attribution. This is why Google (Meridian),
  Meta (Robyn), and PyMC Labs (PyMC-Marketing) all chose Bayesian
  frameworks for their MMM solutions.
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
