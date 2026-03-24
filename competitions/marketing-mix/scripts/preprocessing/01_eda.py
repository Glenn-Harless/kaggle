"""
Marketing Mix Modeling: Step 1 — EDA
Explore the DT Mart dataset with a MARKETING MEASUREMENT lens:
  - Aggregate transaction-level data to weekly granularity
  - Map monthly media spend to weekly observations
  - Assess channel spend variation and correlation (identifiability)
  - Identify seasonality and special sale effects
  - Reality-check whether we have enough data for causal decomposition

This EDA is NOT about finding predictive features — it's about assessing
whether the data can support causal attribution of sales to marketing channels.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/01_eda.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 1 — EDA")
print("=" * 60)


# ============================================================
# 1. RAW DATA INVENTORY
# ============================================================
# Before any analysis, understand what files we have, their size,
# and granularity. In a real MMM engagement, the first week is
# always "data wrangling" — assembling sources that were never
# designed to work together.

print("\n\n" + "=" * 60)
print("1. RAW DATA INVENTORY")
print("=" * 60)

# --- 1a. Transaction data (firstfile.csv) ---
# This is the source of truth for sales. Each row = one item sold.
# We'll aggregate this to weekly to create our target variable (total GMV).
print("\n--- firstfile.csv (transaction-level sales) ---")
df_txn = pd.read_csv(f"{DATA}/firstfile.csv", index_col=0)
df_txn["Date"] = pd.to_datetime(df_txn["Date"])

print(f"  Shape: {df_txn.shape[0]:,} rows x {df_txn.shape[1]} columns")
print(f"  Date range: {df_txn['Date'].min()} to {df_txn['Date'].max()}")
print(f"  Columns: {list(df_txn.columns)}")
print(f"  Missing values:\n{df_txn.isna().sum().to_string()}")
print(f"\n  Product categories ({df_txn['product_category'].nunique()}):")
for cat, count in df_txn["product_category"].value_counts().items():
    print(f"    {cat:30s}  {count:>10,} txns ({count/len(df_txn):.1%})")
print(f"\n  Sales/promotion names ({df_txn['Sales_name'].nunique()}):")
for name, count in df_txn["Sales_name"].value_counts().items():
    print(f"    {name:35s}  {count:>10,} txns ({count/len(df_txn):.1%})")
print(f"\n  GMV (gmv_new) stats:")
print(f"    Total:  {df_txn['gmv_new'].sum():>20,.0f}")
print(f"    Mean:   {df_txn['gmv_new'].mean():>20,.0f}")
print(f"    Median: {df_txn['gmv_new'].median():>20,.0f}")
print(f"    Std:    {df_txn['gmv_new'].std():>20,.0f}")
print(f"    Min:    {df_txn['gmv_new'].min():>20,.0f}")
print(f"    Max:    {df_txn['gmv_new'].max():>20,.0f}")

# --- 1b. Media investment (monthly) ---
# CRITICAL: This is monthly, not weekly. Each month has ONE row of spend
# across 9 channels. When we go to weekly, we'll have to distribute this
# evenly — meaning within any month, every week looks identical in terms
# of media spend. The model literally cannot distinguish week-to-week
# variation within a month. This is a HUGE limitation.
print("\n\n--- MediaInvestment.csv (monthly media spend) ---")
df_media = pd.read_csv(f"{DATA}/MediaInvestment.csv")
# Fix the leading space in " Affiliates"
df_media.columns = df_media.columns.str.strip()
print(f"  Shape: {df_media.shape[0]} rows x {df_media.shape[1]} columns")
print(f"  Months covered: {df_media.shape[0]} (Jul 2015 - Jun 2016)")
channels = ["TV", "Digital", "Sponsorship", "Content Marketing",
            "Online marketing", "Affiliates", "SEM", "Radio", "Other"]
print(f"\n  Channel spend summary (units unclear — likely lakhs or crores):")
for ch in channels:
    non_null = df_media[ch].notna().sum()
    total = df_media[ch].sum()
    print(f"    {ch:25s}  months with data: {non_null:>2}/12  total: {total:>8.1f}")
print(f"\n  Radio and Other only have data for 3/12 months — very sparse.")
print(f"  These channels may need to be dropped or grouped.")

# --- 1c. Special sales (event calendar) ---
print("\n\n--- SpecialSale.csv (Indian market promotions) ---")
df_sales = pd.read_csv(f"{DATA}/SpecialSale.csv")
df_sales["Date"] = pd.to_datetime(df_sales["Date"])
print(f"  Total sale-days: {len(df_sales)}")
print(f"\n  Events:")
for name, group in df_sales.groupby("Sales Name"):
    dates = group["Date"].dt.strftime("%Y-%m-%d").tolist()
    print(f"    {name:30s}  {len(group)} days  ({dates[0]} to {dates[-1]})")

# --- 1d. NPS scores ---
print("\n\n--- MonthlyNPSscore.csv ---")
df_nps = pd.read_csv(f"{DATA}/MonthlyNPSscore.csv")
df_nps["Date"] = pd.to_datetime(df_nps["Date"])
print(f"  Shape: {df_nps.shape}")
print(f"  NPS range: {df_nps['NPS'].min():.1f} to {df_nps['NPS'].max():.1f}")
print(f"  NPS mean:  {df_nps['NPS'].mean():.1f}")
print(f"  NPS std:   {df_nps['NPS'].std():.1f}")


# ============================================================
# 2. AGGREGATE TO WEEKLY — BUILD THE TARGET VARIABLE
# ============================================================
# MMM operates on time-series data, typically weekly. Why weekly?
# - Daily is too noisy (day-of-week effects dominate)
# - Monthly is too few observations (only 12 here!)
# - Weekly is the sweet spot: smooths out daily noise while
#   preserving enough temporal variation for the model to learn from
#
# We aggregate: total GMV, total units, total discount per week.
# We use Monday-start weeks (ISO convention).

print("\n\n" + "=" * 60)
print("2. AGGREGATE TRANSACTIONS TO WEEKLY")
print("=" * 60)

# Create a week start column (Monday of each week)
df_txn["week_start"] = df_txn["Date"].dt.to_period("W-SUN").dt.start_time

weekly = df_txn.groupby("week_start").agg(
    total_gmv=("gmv_new", "sum"),
    total_units=("units", "sum"),
    total_discount=("discount", "sum"),
    total_mrp=("product_mrp", "sum"),
    n_transactions=("gmv_new", "count"),
).reset_index()

# Also aggregate by product category for potential sub-analysis
weekly_by_cat = df_txn.groupby(["week_start", "product_category"]).agg(
    gmv=("gmv_new", "sum"),
    units=("units", "sum"),
).reset_index()

print(f"\n  Weekly aggregation:")
print(f"    Total weeks: {len(weekly)}")
print(f"    Date range:  {weekly['week_start'].min()} to {weekly['week_start'].max()}")
print(f"    Mean weekly GMV:    {weekly['total_gmv'].mean():>15,.0f}")
print(f"    Std weekly GMV:     {weekly['total_gmv'].std():>15,.0f}")
print(f"    CV (std/mean):      {weekly['total_gmv'].std() / weekly['total_gmv'].mean():.2f}")
print(f"    Min weekly GMV:     {weekly['total_gmv'].min():>15,.0f}")
print(f"    Max weekly GMV:     {weekly['total_gmv'].max():>15,.0f}")
print(f"    Mean weekly units:  {weekly['total_units'].mean():>15,.0f}")
print(f"    Mean transactions:  {weekly['n_transactions'].mean():>15,.0f}")

# Check for incomplete weeks at boundaries
print(f"\n  First week ({weekly['week_start'].iloc[0].date()}):")
print(f"    Transactions: {weekly['n_transactions'].iloc[0]:,}")
print(f"    GMV: {weekly['total_gmv'].iloc[0]:,.0f}")
print(f"  Last week ({weekly['week_start'].iloc[-1].date()}):")
print(f"    Transactions: {weekly['n_transactions'].iloc[-1]:,}")
print(f"    GMV: {weekly['total_gmv'].iloc[-1]:,.0f}")
print(f"\n  NOTE: Boundary weeks may be incomplete (partial data).")
print(f"  Watch for anomalous drops at the start/end of the series.")


# ============================================================
# 3. TARGET VARIABLE — GMV OVER TIME
# ============================================================
# Before thinking about channels, understand the thing we're
# trying to explain. What does total weekly GMV look like?
# Are there trends? Spikes? Patterns?

print("\n\n" + "=" * 60)
print("3. TARGET VARIABLE — WEEKLY GMV OVER TIME")
print("=" * 60)

print(f"\n  Weekly GMV distribution:")
print(f"    25th percentile: {weekly['total_gmv'].quantile(0.25):>15,.0f}")
print(f"    50th percentile: {weekly['total_gmv'].quantile(0.50):>15,.0f}")
print(f"    75th percentile: {weekly['total_gmv'].quantile(0.75):>15,.0f}")
print(f"    IQR:             {weekly['total_gmv'].quantile(0.75) - weekly['total_gmv'].quantile(0.25):>15,.0f}")

# Check for obvious outliers
q1 = weekly["total_gmv"].quantile(0.25)
q3 = weekly["total_gmv"].quantile(0.75)
iqr = q3 - q1
outlier_low = q1 - 1.5 * iqr
outlier_high = q3 + 1.5 * iqr
outliers = weekly[(weekly["total_gmv"] < outlier_low) | (weekly["total_gmv"] > outlier_high)]
print(f"\n  Outlier weeks (IQR method): {len(outliers)}")
for _, row in outliers.iterrows():
    direction = "HIGH" if row["total_gmv"] > outlier_high else "LOW"
    print(f"    {row['week_start'].date()}  GMV={row['total_gmv']:>15,.0f}  [{direction}]")

# Plot 1: Weekly GMV time series
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(weekly["week_start"], weekly["total_gmv"] / 1e6, "b-o", markersize=3, linewidth=1)
axes[0].set_title("Weekly GMV Over Time", fontsize=14, fontweight="bold")
axes[0].set_ylabel("GMV (millions)")
axes[0].grid(True, alpha=0.3)

# Mark special sale weeks
for _, row in df_sales.iterrows():
    week_of_sale = row["Date"] - pd.Timedelta(days=row["Date"].dayofweek)
    if week_of_sale in weekly["week_start"].values:
        axes[0].axvline(week_of_sale, color="red", alpha=0.15, linewidth=2)

# Add a legend annotation for the red lines
axes[0].annotate("Red bands = special sale weeks", xy=(0.02, 0.95),
                 xycoords="axes fraction", fontsize=9, color="red", alpha=0.7)

# Plot 2: GMV by product category over time
for cat in weekly_by_cat["product_category"].unique():
    cat_data = weekly_by_cat[weekly_by_cat["product_category"] == cat]
    axes[1].plot(cat_data["week_start"], cat_data["gmv"] / 1e6,
                 label=cat, linewidth=1, alpha=0.8)
axes[1].set_title("Weekly GMV by Product Category", fontsize=14, fontweight="bold")
axes[1].set_ylabel("GMV (millions)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/01_weekly_gmv_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 01_weekly_gmv_timeseries.png")


# ============================================================
# 4. MAP MONTHLY MEDIA SPEND TO WEEKLY
# ============================================================
# This is the most important data engineering step AND the biggest
# limitation. Media spend is monthly — we distribute it evenly
# across weeks within each month.
#
# Why this is a problem:
# - If TV spend in October is 6.1, every week in October gets 6.1/~4.3
# - The model sees identical TV spend in week 1 and week 4 of October
# - If sales spike in week 3 (Diwali), the model can't tell if it's
#   the TV spend or the Diwali sale — both are constant within the month
# - This REDUCES the effective variation the model can learn from
#
# In production MMM, you'd have weekly or even daily spend from ad
# platforms (Google Ads, Facebook Ads Manager, etc.)

print("\n\n" + "=" * 60)
print("4. MAP MONTHLY MEDIA SPEND TO WEEKLY")
print("=" * 60)

# Create a date column for joining
df_media["date"] = pd.to_datetime(
    df_media["Year"].astype(str) + "-" + df_media["Month"].astype(str) + "-01"
)
df_media["year_month"] = df_media["date"].dt.to_period("M")

# Add year_month to weekly for joining
weekly["year_month"] = weekly["week_start"].dt.to_period("M")

# Count weeks per month (for distributing monthly spend)
weeks_per_month = weekly.groupby("year_month").size().to_dict()

# Join and distribute
weekly = weekly.merge(
    df_media[["year_month"] + channels],
    on="year_month",
    how="left",
)

# Distribute monthly spend evenly across weeks within the month
for ch in channels:
    weekly[ch] = weekly.apply(
        lambda row: row[ch] / weeks_per_month.get(row["year_month"], 1)
        if pd.notna(row[ch]) else 0.0,
        axis=1,
    )

print(f"\n  Weeks per month in our data:")
for ym, count in sorted(weeks_per_month.items()):
    print(f"    {ym}: {count} weeks")

print(f"\n  After distributing monthly spend to weekly:")
print(f"    Shape: {weekly.shape}")
print(f"\n  Weekly channel spend summary (after distribution):")
for ch in channels:
    nonzero = (weekly[ch] > 0).sum()
    print(f"    {ch:25s}  mean={weekly[ch].mean():>6.2f}  "
          f"std={weekly[ch].std():>6.2f}  nonzero weeks: {nonzero}/{len(weekly)}")

print(f"\n  CRITICAL LIMITATION: Within any month, all weeks have identical")
print(f"  channel spend values. This means the model's ability to attribute")
print(f"  sales to channels is limited to BETWEEN-MONTH variation.")
print(f"  With only 12 months, that's very little signal.")


# ============================================================
# 5. SPECIAL SALE EFFECTS — CONTROL VARIABLES
# ============================================================
# These promotions are CONFOUNDERS. If Diwali drives both increased
# ad spend AND increased sales, failing to control for it would make
# the model attribute Diwali's sales lift to the ad channels.
#
# This is exactly the omitted variable bias you'd see in a naive
# regression — it's why controls matter in causal inference.

print("\n\n" + "=" * 60)
print("5. SPECIAL SALE EFFECTS (CONTROL VARIABLES)")
print("=" * 60)

# Create binary: was there a special sale active this week?
sale_weeks = set()
for _, row in df_sales.iterrows():
    week_start = row["Date"] - pd.Timedelta(days=row["Date"].dayofweek)
    sale_weeks.add(week_start)

weekly["has_special_sale"] = weekly["week_start"].isin(sale_weeks).astype(int)

# Also create individual event indicators for the major ones
major_events = ["Big Diwali Sale", "Christmas & New Year Sale",
                "Daussera sale", "Eid & Rathayatra sale",
                "Independence Sale", "Republic Day"]
for event in major_events:
    event_dates = df_sales[df_sales["Sales Name"] == event]["Date"]
    event_weeks = set()
    for d in event_dates:
        event_weeks.add(d - pd.Timedelta(days=d.dayofweek))
    col_name = "sale_" + event.lower().replace(" ", "_").replace("&", "and").replace("'", "")
    weekly[col_name] = weekly["week_start"].isin(event_weeks).astype(int)

n_sale_weeks = weekly["has_special_sale"].sum()
print(f"\n  Weeks with any special sale: {n_sale_weeks}/{len(weekly)} ({n_sale_weeks/len(weekly):.0%})")
print(f"\n  GMV comparison:")
sale_gmv = weekly[weekly["has_special_sale"] == 1]["total_gmv"]
nosale_gmv = weekly[weekly["has_special_sale"] == 0]["total_gmv"]
print(f"    Sale weeks mean GMV:      {sale_gmv.mean():>15,.0f}")
print(f"    Non-sale weeks mean GMV:  {nosale_gmv.mean():>15,.0f}")
if nosale_gmv.mean() > 0:
    lift = (sale_gmv.mean() - nosale_gmv.mean()) / nosale_gmv.mean()
    print(f"    Lift during sale weeks:   {lift:>+15.1%}")
print(f"\n  This lift is the effect we need to CONTROL FOR, not attribute")
print(f"  to advertising. If we don't, the model will think the ad spend")
print(f"  in October caused the Diwali sales spike.")


# ============================================================
# 6. CHANNEL SPEND ANALYSIS — IS THERE ENOUGH VARIATION?
# ============================================================
# For MMM to work, each channel needs meaningful variation in spend
# over time. If TV spend is roughly constant, the model can't
# distinguish its effect from the baseline intercept.
#
# Think of it like an A/B test: you need variation (treatment vs
# control) to measure an effect. Channels with flat spend are like
# running an A/B test where everyone gets treatment — you can't
# measure anything.

print("\n\n" + "=" * 60)
print("6. CHANNEL SPEND ANALYSIS — IDENTIFIABILITY")
print("=" * 60)

print(f"\n  For a channel to be identifiable, we need:")
print(f"    1. Sufficient variation in spend (high CV)")
print(f"    2. Variation that's NOT perfectly correlated with other channels")
print(f"    3. Enough non-zero weeks to estimate the effect")

print(f"\n  Channel variation (coefficient of variation = std/mean):")
print(f"  {'Channel':25s}  {'Mean':>8s}  {'Std':>8s}  {'CV':>6s}  {'Zero weeks':>12s}  {'Assessment':>15s}")
print(f"  {'-'*80}")
for ch in channels:
    ch_data = weekly[ch]
    nonzero = ch_data[ch_data > 0]
    if len(nonzero) > 0:
        cv = nonzero.std() / nonzero.mean() if nonzero.mean() > 0 else 0
        assessment = "Good" if cv > 0.3 and len(nonzero) > 20 else "Marginal" if len(nonzero) > 10 else "Sparse"
    else:
        cv = 0
        assessment = "No data"
    zero_wks = (ch_data == 0).sum()
    print(f"  {ch:25s}  {ch_data.mean():>8.2f}  {ch_data.std():>8.2f}  "
          f"{cv:>6.2f}  {zero_wks:>5}/{len(weekly):>3}      {assessment:>15s}")

print(f"\n  Radio and Other have data in only a few months — likely unusable.")
print(f"  Content Marketing has near-zero spend in most months — marginal.")

# Plot 2: Monthly media spend stacked bar (the raw monthly data)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Stacked bar of monthly spend
core_channels = ["TV", "Digital", "Sponsorship", "Content Marketing",
                 "Online marketing", "Affiliates", "SEM"]
month_labels = [f"{row['Year']}-{row['Month']:02d}" for _, row in df_media.iterrows()]
bottom = np.zeros(len(df_media))
colors = plt.cm.Set2(np.linspace(0, 1, len(core_channels)))
for i, ch in enumerate(core_channels):
    vals = df_media[ch].fillna(0).values
    axes[0].bar(month_labels, vals, bottom=bottom, label=ch, color=colors[i])
    bottom += vals
axes[0].set_title("Monthly Media Investment by Channel", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Spend (units from dataset)")
axes[0].legend(fontsize=8, loc="upper left")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True, alpha=0.3, axis="y")

# Time series of weekly spend (after distribution)
for i, ch in enumerate(core_channels):
    axes[1].plot(weekly["week_start"], weekly[ch], label=ch,
                 linewidth=1, alpha=0.8, color=colors[i])
axes[1].set_title("Weekly Media Spend (Monthly Distributed)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Spend per week")
axes[1].legend(fontsize=8, loc="upper left")
axes[1].grid(True, alpha=0.3)
axes[1].annotate("Notice the step-function pattern — identical within each month",
                 xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9,
                 fontstyle="italic", color="gray")

plt.tight_layout()
plt.savefig(f"{RESULTS}/01_media_spend.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 01_media_spend.png")


# ============================================================
# 7. CORRELATION ANALYSIS — THE MULTICOLLINEARITY PROBLEM
# ============================================================
# If two channels always increase/decrease together (e.g., TV and
# Digital both spike during Q4), the model can't tell which one
# caused the sales increase. This is multicollinearity.
#
# Analogy from your ML work: imagine two features that are 0.95
# correlated. LightGBM can handle this (trees split on one or the
# other), but OLS assigns arbitrary weights between them. Bayesian
# regression handles it better (priors provide regularization), but
# highly correlated channels will still have wide posteriors — the
# model is honestly saying "I can't tell these apart."

print("\n\n" + "=" * 60)
print("7. CORRELATION ANALYSIS — MULTICOLLINEARITY")
print("=" * 60)

# Use only channels with sufficient data
usable_channels = [ch for ch in core_channels if (weekly[ch] > 0).sum() > 20]
print(f"\n  Usable channels (>20 non-zero weeks): {usable_channels}")

corr_matrix = weekly[usable_channels].corr()
print(f"\n  Pairwise correlations:")
print(f"  {'':25s}  ", end="")
for ch in usable_channels:
    print(f"{ch[:8]:>9s}", end="")
print()
for ch1 in usable_channels:
    print(f"  {ch1:25s}  ", end="")
    for ch2 in usable_channels:
        r = corr_matrix.loc[ch1, ch2]
        marker = " **" if abs(r) > 0.7 and ch1 != ch2 else ""
        print(f"{r:>8.2f}{marker[0] if marker else ' '}", end="")
    print()

# Flag high correlations
print(f"\n  Highly correlated pairs (|r| > 0.7): **")
flagged = []
for i, ch1 in enumerate(usable_channels):
    for ch2 in usable_channels[i+1:]:
        r = corr_matrix.loc[ch1, ch2]
        if abs(r) > 0.7:
            flagged.append((ch1, ch2, r))
            print(f"    {ch1} <-> {ch2}: r={r:.2f}")

if not flagged:
    print(f"    (none found)")

print(f"\n  WARNING: Monthly-distributed spend inflates correlations because")
print(f"  within-month variation is zero. Channels that happen to vary")
print(f"  similarly across months will appear highly correlated even if")
print(f"  their weekly patterns differ.")

# Also correlate spend with GMV — which channels co-move with sales?
print(f"\n  Channel-GMV correlations (raw — NOT causal):")
for ch in usable_channels:
    r = weekly[ch].corr(weekly["total_gmv"])
    print(f"    {ch:25s}  r={r:>+.2f}")
print(f"  Note: These correlations include confounders (seasonality, promos).")
print(f"  High correlation does NOT mean high causal effect.")

# Plot 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
# Include GMV in the correlation matrix for context
corr_with_gmv = weekly[usable_channels + ["total_gmv"]].corr()
mask = np.triu(np.ones_like(corr_with_gmv, dtype=bool))
sns.heatmap(corr_with_gmv, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
            square=True, linewidths=0.5)
ax.set_title("Channel Spend & GMV Correlations\n(confounded — NOT causal)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/01_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 01_correlation_heatmap.png")


# ============================================================
# 8. SEASONALITY — WHAT PATTERNS EXIST IN GMV?
# ============================================================
# MMM needs to account for seasonality to avoid attributing seasonal
# sales patterns to ad spend. If sales always rise in October (pre-
# Diwali) and ad spend also rises in October, the model needs to
# know that some of that rise is seasonal, not advertising-driven.

print("\n\n" + "=" * 60)
print("8. SEASONALITY ANALYSIS")
print("=" * 60)

weekly["month"] = weekly["week_start"].dt.month
weekly["month_name"] = weekly["week_start"].dt.month_name()

monthly_gmv = weekly.groupby("month").agg(
    mean_gmv=("total_gmv", "mean"),
    std_gmv=("total_gmv", "std"),
    n_weeks=("total_gmv", "count"),
).reset_index()

print(f"\n  Monthly GMV patterns:")
print(f"  {'Month':>12s}  {'Mean GMV':>15s}  {'Std':>15s}  {'Weeks':>6s}")
print(f"  {'-'*55}")
for _, row in monthly_gmv.iterrows():
    print(f"  {row['month']:>12.0f}  {row['mean_gmv']:>15,.0f}  {row['std_gmv']:>15,.0f}  {row['n_weeks']:>6.0f}")

# Check if there's a trend
weekly["week_num"] = range(len(weekly))
trend_corr = weekly["week_num"].corr(weekly["total_gmv"])
print(f"\n  Trend check:")
print(f"    GMV vs. time correlation: r={trend_corr:+.2f}")
if abs(trend_corr) > 0.3:
    direction = "upward" if trend_corr > 0 else "downward"
    print(f"    Moderate {direction} trend detected — include trend as a control variable")
else:
    print(f"    No strong linear trend — may still have non-linear patterns")

# Plot 4: Seasonality
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot by month
monthly_data = [weekly[weekly["month"] == m]["total_gmv"].values / 1e6
                for m in range(1, 13) if m in weekly["month"].values]
month_labels_box = [pd.Timestamp(2015, m, 1).month_name()[:3]
                    for m in range(1, 13) if m in weekly["month"].values]
axes[0].boxplot(monthly_data, labels=month_labels_box)
axes[0].set_title("Weekly GMV Distribution by Month", fontsize=14, fontweight="bold")
axes[0].set_ylabel("GMV (millions)")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True, alpha=0.3, axis="y")

# GMV with sale weeks highlighted
axes[1].plot(weekly["week_start"], weekly["total_gmv"] / 1e6, "b-o", markersize=3)
sale_mask = weekly["has_special_sale"] == 1
axes[1].scatter(weekly.loc[sale_mask, "week_start"],
                weekly.loc[sale_mask, "total_gmv"] / 1e6,
                color="red", s=50, zorder=5, label="Special sale week")
axes[1].set_title("Weekly GMV — Sale Weeks Highlighted", fontsize=14, fontweight="bold")
axes[1].set_ylabel("GMV (millions)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/01_seasonality.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 01_seasonality.png")


# ============================================================
# 9. SAMPLE SIZE REALITY CHECK
# ============================================================
# This is the most important section for an MMM practitioner.
# The question: do we have enough data to reliably estimate
# the model we want to fit?
#
# Rule of thumb for Bayesian MMM:
#   - Minimum ~2 years of weekly data (104 observations)
#   - Ideally 3+ years (156+)
#   - Parameters should be < observations / 10
#
# Our situation:
#   - ~52 weeks of data
#   - 7 channels (7 adstock + 7 saturation + 7 beta = 21 channel params)
#   - Plus intercept, trend, seasonality, sale indicators
#   - Total params could easily exceed 30
#   - 52 observations / 30 parameters = 1.7 obs per parameter
#   - This is TERRIBLE. OLS would be completely unreliable.
#   - Bayesian helps via priors, but results will be heavily prior-driven.

print("\n\n" + "=" * 60)
print("9. SAMPLE SIZE REALITY CHECK")
print("=" * 60)

n_weeks = len(weekly)

# Count potential parameters
n_channels_full = 7  # core channels (excluding Radio, Other)
n_channel_params = n_channels_full * 3  # each: adstock alpha, saturation lambda, beta
n_controls = 3  # trend + has_special_sale + NPS
n_seasonality = 2  # at minimum, yearly fourier terms
n_intercept = 1
n_noise = 1  # sigma
n_total_params = n_channel_params + n_controls + n_seasonality + n_intercept + n_noise

print(f"\n  Observations: {n_weeks} weeks")
print(f"\n  Parameter count (full 7-channel model):")
print(f"    Channel params:   {n_channels_full} channels × 3 (α, λ, β) = {n_channel_params}")
print(f"    Control vars:     {n_controls} (trend, sale indicator, NPS)")
print(f"    Seasonality:      ~{n_seasonality} (fourier terms)")
print(f"    Intercept:        {n_intercept}")
print(f"    Noise (σ):        {n_noise}")
print(f"    {'─'*40}")
print(f"    TOTAL:            ~{n_total_params} parameters")
print(f"\n  Observations per parameter: {n_weeks / n_total_params:.1f}")
print(f"  (Industry minimum guideline: ~5-10 obs/param)")

print(f"\n  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  VERDICT: {n_weeks} weeks and ~{n_total_params} parameters is NOT enough     ║")
print(f"  ║  for a reliable 7-channel model.                        ║")
print(f"  ║                                                         ║")
print(f"  ║  Options:                                               ║")
print(f"  ║  1. Reduce to 3-4 channel groups → ~13-16 params        ║")
print(f"  ║  2. Use strong informative priors (Bayesian helps here) ║")
print(f"  ║  3. Accept wide posteriors as honest uncertainty         ║")
print(f"  ║  4. Supplement with lift test data (PyMC-Marketing)     ║")
print(f"  ╚══════════════════════════════════════════════════════════╝")

# Suggested channel grouping
print(f"\n  Suggested channel grouping for Step 7 (data prep):")
print(f"    Group 1: Sponsorship (largest spender, dominates budget)")
print(f"    Group 2: Online marketing + Affiliates (performance channels)")
print(f"    Group 3: TV + Digital (brand awareness)")
print(f"    Group 4: SEM (search — different intent signal)")
print(f"    Drop:    Radio, Other (too sparse), Content Marketing (near-zero)")
print(f"\n  This reduces to 4 channels × 3 = 12 channel params + 4 controls = ~16 total")
print(f"  52 obs / 16 params = {52/16:.1f} obs/param — marginal but workable with good priors")


# ============================================================
# 10. AUGUST 2015 ANOMALY CHECK
# ============================================================
# I noticed from Secondfile.csv that August 2015 had dramatically
# lower revenue than other months. Let's investigate.

print("\n\n" + "=" * 60)
print("10. DATA QUALITY — ANOMALY INVESTIGATION")
print("=" * 60)

# Check daily transaction counts
daily_counts = df_txn.groupby("Date").agg(
    n_txns=("gmv_new", "count"),
    gmv=("gmv_new", "sum"),
).reset_index()

aug2015 = daily_counts[(daily_counts["Date"] >= "2015-08-01") &
                        (daily_counts["Date"] < "2015-09-01")]
print(f"\n  August 2015 daily transaction counts:")
print(f"    Days with data: {len(aug2015)}")
print(f"    Total txns:     {aug2015['n_txns'].sum():,}")
print(f"    Total GMV:      {aug2015['gmv'].sum():,.0f}")
if len(aug2015) > 0:
    print(f"    Daily txn range: {aug2015['n_txns'].min():,} to {aug2015['n_txns'].max():,}")

# Compare to other months
print(f"\n  Monthly transaction counts:")
monthly_txns = df_txn.groupby(df_txn["Date"].dt.to_period("M")).size()
for period, count in monthly_txns.items():
    marker = " <-- ANOMALY" if count < 1000 else ""
    print(f"    {period}: {count:>10,} transactions{marker}")

print(f"\n  If August 2015 has almost no data, it may be a data collection")
print(f"  issue, not real low sales. Consider excluding it from the model.")


# ============================================================
# 11. SUMMARY — KEY FINDINGS FOR MODEL DESIGN
# ============================================================

print("\n\n" + "=" * 60)
print("11. SUMMARY — KEY FINDINGS FOR MODEL DESIGN")
print("=" * 60)

print(f"""
  DATA:
  - {df_txn.shape[0]:,} transactions → {n_weeks} weekly observations
  - 5 product categories, dominated by EntertainmentSmall
  - {len(df_sales)} special sale days across ~{df_sales['Sales Name'].nunique()} events
  - 9 media channels (monthly granularity only)

  GOOD NEWS:
  - GMV has meaningful variation week-to-week (CV = {weekly['total_gmv'].std() / weekly['total_gmv'].mean():.2f})
  - Special sales create identifiable spikes (useful as controls)
  - Monthly media spend varies substantially across months
  - Clear seasonal pattern gives us something to model

  BAD NEWS:
  - Only {n_weeks} weekly observations — far below the 104+ week minimum
  - Media spend is MONTHLY — within-month variation is zero
  - August 2015 appears to have a data quality issue
  - Multiple channels are likely correlated (co-vary seasonally)
  - With 7+ channels, we have ~{n_total_params} params for {n_weeks} observations

  IMPLICATIONS FOR MODELING:
  - Must reduce channels to 3-4 groups (Step 7)
  - Results will be heavily influenced by priors — be transparent
  - Wide posteriors are EXPECTED and HONEST with this data
  - This dataset is excellent for LEARNING the workflow
  - Would NOT be suitable for real marketing decisions without more data
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
