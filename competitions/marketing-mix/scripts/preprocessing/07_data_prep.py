"""
Marketing Mix Modeling: Step 7 — Data Preparation

Transform DT Mart's raw data into an MMM-ready weekly dataset.
This is the data engineering step — assembling sources that were
never designed to work together.

  1. Aggregate transactions to weekly GMV
  2. Join monthly media spend (distribute to weekly)
  3. Create control variables (special sales, NPS, trend)
  4. Group 9 channels into 3-4 groups (reduce multicollinearity)
  5. Handle August 2015 anomaly
  6. Export final MMM-ready dataset
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/07_data_prep.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 7 — Data Preparation")
print("=" * 60)
print("\nGoal: Build a clean weekly dataset for the Bayesian MMM.")


# ============================================================
# 1. AGGREGATE TRANSACTIONS TO WEEKLY
# ============================================================

print("\n\n" + "=" * 60)
print("1. AGGREGATE TRANSACTIONS TO WEEKLY GMV")
print("=" * 60)

df_txn = pd.read_csv(f"{DATA}/firstfile.csv", index_col=0)
df_txn["Date"] = pd.to_datetime(df_txn["Date"])
df_txn["week_start"] = df_txn["Date"].dt.to_period("W-SUN").dt.start_time

weekly = df_txn.groupby("week_start").agg(
    total_gmv=("gmv_new", "sum"),
    total_units=("units", "sum"),
    n_transactions=("gmv_new", "count"),
).reset_index()

print(f"\n  Raw transactions: {len(df_txn):,}")
print(f"  Weekly observations: {len(weekly)}")
print(f"  Date range: {weekly['week_start'].min().date()} to {weekly['week_start'].max().date()}")


# ============================================================
# 2. HANDLE AUGUST 2015 ANOMALY
# ============================================================
# From EDA (Step 1): August 2015 has only 254 transactions vs
# ~100K+ in other months. This is a data collection failure.

print("\n\n" + "=" * 60)
print("2. HANDLE AUGUST 2015 ANOMALY")
print("=" * 60)

aug_mask = (weekly["week_start"] >= "2015-08-01") & (weekly["week_start"] < "2015-09-01")
aug_weeks = weekly[aug_mask]
print(f"\n  August 2015 weeks:")
for _, row in aug_weeks.iterrows():
    print(f"    {row['week_start'].date()}: {row['n_transactions']:,} txns, GMV={row['total_gmv']:,.0f}")

# Also check first week (may be partial — data starts Jul 1 which is a Wednesday)
first_week = weekly.iloc[0]
print(f"\n  First week ({first_week['week_start'].date()}): {first_week['n_transactions']:,} txns")
print(f"  This is likely partial (data starts mid-week).")

# Remove August 2015 and the partial first week
weekly = weekly[~aug_mask].copy()
if first_week["n_transactions"] < 5000:
    weekly = weekly.iloc[1:].copy()
    print(f"  Removed partial first week.")

weekly = weekly.reset_index(drop=True)
print(f"\n  After cleaning: {len(weekly)} weeks")


# ============================================================
# 3. JOIN MONTHLY MEDIA SPEND
# ============================================================
# Monthly spend distributed evenly across weeks within each month.
# This is a limitation — we discussed it in EDA.

print("\n\n" + "=" * 60)
print("3. JOIN MONTHLY MEDIA SPEND")
print("=" * 60)

df_media = pd.read_csv(f"{DATA}/MediaInvestment.csv")
df_media.columns = df_media.columns.str.strip()

all_channels = ["TV", "Digital", "Sponsorship", "Content Marketing",
                "Online marketing", "Affiliates", "SEM", "Radio", "Other"]

# Fill NaN with 0 for Radio and Other
for ch in all_channels:
    df_media[ch] = df_media[ch].fillna(0)

df_media["date"] = pd.to_datetime(
    df_media["Year"].astype(str) + "-" + df_media["Month"].astype(str) + "-01"
)
df_media["year_month"] = df_media["date"].dt.to_period("M")
weekly["year_month"] = weekly["week_start"].dt.to_period("M")

# Count weeks per month for distribution
weeks_per_month = weekly.groupby("year_month").size().to_dict()

# Join
weekly = weekly.merge(
    df_media[["year_month"] + all_channels],
    on="year_month", how="left",
)

# Distribute monthly spend evenly across weeks
for ch in all_channels:
    weekly[ch] = weekly.apply(
        lambda row: row[ch] / weeks_per_month.get(row["year_month"], 1),
        axis=1,
    )

print(f"\n  Weeks per month:")
for ym, n in sorted(weeks_per_month.items()):
    print(f"    {ym}: {n} weeks")

print(f"\n  Monthly spend distributed to weekly (each week in a month gets equal share).")
print(f"  Limitation: within-month variation is zero.")


# ============================================================
# 4. GROUP CHANNELS — REDUCE FROM 9 TO 4
# ============================================================
# With ~47 weeks, we can't estimate 9 separate channels.
# From EDA:
#   - Digital/SEM/Content Marketing are r>0.9 correlated
#   - Online marketing/Affiliates are r=0.99 correlated
#   - Radio/Other only have 3 months of data
#
# Grouping strategy (driven by correlation + business logic):
#   Group 1: Sponsorship (largest spender, keep separate)
#   Group 2: Online + Affiliates ("Performance" — highest GMV correlation)
#   Group 3: TV + Digital ("Brand/Awareness" — traditional + digital brand)
#   Group 4: SEM (search intent — distinct channel type)
#   Drop: Radio, Other (too sparse), Content Marketing (near-zero spend)

print("\n\n" + "=" * 60)
print("4. GROUP CHANNELS (9 → 4)")
print("=" * 60)

# Create channel groups
weekly["spend_sponsorship"] = weekly["Sponsorship"]
weekly["spend_performance"] = weekly["Online marketing"] + weekly["Affiliates"]
weekly["spend_brand"] = weekly["TV"] + weekly["Digital"]
weekly["spend_sem"] = weekly["SEM"]

channel_groups = ["spend_sponsorship", "spend_performance",
                  "spend_brand", "spend_sem"]

print(f"\n  Channel grouping:")
print(f"    spend_sponsorship:  Sponsorship (largest single channel)")
print(f"    spend_performance:  Online marketing + Affiliates (r=0.99, same intent)")
print(f"    spend_brand:        TV + Digital (brand awareness)")
print(f"    spend_sem:          SEM (search intent, separate behavior)")
print(f"    DROPPED:            Radio, Other (too sparse), Content Marketing (~zero)")

print(f"\n  Grouped channel statistics:")
print(f"  {'Group':25s}  {'Mean':>8s}  {'Std':>8s}  {'CV':>6s}  {'% of total':>12s}")
print(f"  {'-'*65}")
total_spend = sum(weekly[ch].sum() for ch in channel_groups)
for ch in channel_groups:
    mean = weekly[ch].mean()
    std = weekly[ch].std()
    cv = std / mean if mean > 0 else 0
    pct = weekly[ch].sum() / total_spend
    print(f"  {ch:25s}  {mean:>8.2f}  {std:>8.2f}  {cv:>6.2f}  {pct:>11.1%}")

# Check correlation of grouped channels
print(f"\n  Correlation matrix (grouped channels):")
corr = weekly[channel_groups].corr()
for ch in channel_groups:
    print(f"    {ch:25s}", end="")
    for ch2 in channel_groups:
        print(f"  {corr.loc[ch, ch2]:>6.2f}", end="")
    print()

# Flag remaining high correlations
for i, ch1 in enumerate(channel_groups):
    for ch2 in channel_groups[i+1:]:
        r = corr.loc[ch1, ch2]
        if abs(r) > 0.7:
            print(f"\n  WARNING: {ch1} and {ch2} still correlated at r={r:.2f}")


# ============================================================
# 5. CREATE CONTROL VARIABLES
# ============================================================

print("\n\n" + "=" * 60)
print("5. CREATE CONTROL VARIABLES")
print("=" * 60)

# --- 5a. Special sales ---
df_sales = pd.read_csv(f"{DATA}/SpecialSale.csv")
df_sales["Date"] = pd.to_datetime(df_sales["Date"])

sale_weeks = set()
for _, row in df_sales.iterrows():
    week_start = row["Date"] - pd.Timedelta(days=row["Date"].dayofweek)
    sale_weeks.add(week_start)

weekly["has_special_sale"] = weekly["week_start"].isin(sale_weeks).astype(int)
n_sale = weekly["has_special_sale"].sum()
print(f"\n  Special sale weeks: {n_sale}/{len(weekly)} ({n_sale/len(weekly):.0%})")

# --- 5b. NPS ---
df_nps = pd.read_csv(f"{DATA}/MonthlyNPSscore.csv")
df_nps["Date"] = pd.to_datetime(df_nps["Date"])
df_nps["year_month"] = df_nps["Date"].dt.to_period("M")

weekly = weekly.merge(df_nps[["year_month", "NPS"]], on="year_month", how="left")
print(f"  NPS joined: {weekly['NPS'].notna().sum()}/{len(weekly)} weeks have NPS data")
# Fill any NaN with median
weekly["NPS"] = weekly["NPS"].fillna(weekly["NPS"].median())

# --- 5c. Trend ---
weekly["trend"] = range(len(weekly))
print(f"  Trend variable: 0 to {len(weekly)-1}")

# --- 5d. Month indicators for seasonality ---
weekly["month"] = weekly["week_start"].dt.month
print(f"  Month variable: {weekly['month'].min()} to {weekly['month'].max()}")


# ============================================================
# 6. FINAL DATASET
# ============================================================

print("\n\n" + "=" * 60)
print("6. FINAL MMM-READY DATASET")
print("=" * 60)

# Select final columns
final_cols = (["week_start", "total_gmv"]
              + channel_groups
              + ["has_special_sale", "NPS", "trend", "month"])

df_mmm = weekly[final_cols].copy()
df_mmm = df_mmm.rename(columns={"week_start": "date"})

print(f"\n  Final dataset shape: {df_mmm.shape}")
print(f"  Date range: {df_mmm['date'].min().date()} to {df_mmm['date'].max().date()}")
print(f"  Weeks: {len(df_mmm)}")
print(f"\n  Columns:")
for col in df_mmm.columns:
    print(f"    {col:25s}  dtype={df_mmm[col].dtype}  "
          f"mean={df_mmm[col].mean():>12.2f}" if df_mmm[col].dtype != 'datetime64[ns]'
          else f"    {col:25s}  dtype=datetime")

print(f"\n  Sample rows:")
print(df_mmm.head(10).to_string())

# Save
output_path = f"{DATA}/mmm_weekly.csv"
df_mmm.to_csv(output_path, index=False)
print(f"\n  Saved: {output_path}")

# --- Parameter count check ---
n_channels = len(channel_groups)
n_channel_params = n_channels * 3  # alpha, lambda, beta each
n_controls = 3  # has_special_sale, NPS, trend
n_other = 2  # intercept, sigma
n_total = n_channel_params + n_controls + n_other
print(f"\n  Parameter budget:")
print(f"    Channel params: {n_channels} × 3 = {n_channel_params}")
print(f"    Controls:       {n_controls}")
print(f"    Other:          {n_other} (intercept + sigma)")
print(f"    Total:          ~{n_total}")
print(f"    Obs/param:      {len(df_mmm)}/{n_total} = {len(df_mmm)/n_total:.1f}")
print(f"    (guideline: >5, ideal: >10)")

if len(df_mmm) / n_total < 5:
    print(f"\n    Still tight at {len(df_mmm)/n_total:.1f} obs/param.")
    print(f"    Priors will meaningfully influence results.")
    print(f"    This is expected and honest — we'll address it in Step 8.")


# ============================================================
# 7. VISUALIZATION
# ============================================================

print("\n\n" + "=" * 60)
print("7. VISUALIZATION — THE MMM-READY DATA")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Panel 1: Target variable (GMV)
ax = axes[0]
ax.plot(df_mmm["date"], df_mmm["total_gmv"] / 1e6, "b-o", markersize=3)
sale_mask = df_mmm["has_special_sale"] == 1
ax.scatter(df_mmm.loc[sale_mask, "date"],
           df_mmm.loc[sale_mask, "total_gmv"] / 1e6,
           color="red", s=50, zorder=5, label="Special sale week")
ax.set_title("Target: Weekly GMV", fontsize=13, fontweight="bold")
ax.set_ylabel("GMV (millions)")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Channel spend (grouped)
ax = axes[1]
colors = ["purple", "green", "steelblue", "orange"]
for ch, color in zip(channel_groups, colors):
    label = ch.replace("spend_", "").title()
    ax.plot(df_mmm["date"], df_mmm[ch], label=label, color=color, linewidth=1.5)
ax.set_title("Channel Spend (Grouped, Weekly)", fontsize=13, fontweight="bold")
ax.set_ylabel("Spend")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Controls
ax = axes[2]
ax2 = ax.twinx()
ax.bar(df_mmm["date"], df_mmm["has_special_sale"], alpha=0.3,
       color="red", label="Special sale", width=5)
ax2.plot(df_mmm["date"], df_mmm["NPS"], "g-", linewidth=1.5,
         label="NPS", alpha=0.7)
ax.set_title("Control Variables", fontsize=13, fontweight="bold")
ax.set_ylabel("Has special sale (0/1)")
ax2.set_ylabel("NPS Score")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/07_mmm_ready_data.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 07_mmm_ready_data.png")


# ============================================================
# 8. SUMMARY
# ============================================================

print("\n\n" + "=" * 60)
print("8. SUMMARY")
print("=" * 60)

print(f"""
  WHAT WE BUILT:
  - {len(df_mmm)} weekly observations from {len(df_txn):,} raw transactions
  - 4 channel groups (from 9 raw channels)
  - 3 control variables (special sales, NPS, trend)
  - August 2015 anomaly excluded

  CHANNEL GROUPS:
  1. Sponsorship — largest single spender
  2. Performance — Online marketing + Affiliates (r=0.99)
  3. Brand — TV + Digital (brand awareness combo)
  4. SEM — Search intent (distinct behavior)

  LIMITATIONS TO CARRY INTO STEP 8:
  - Media spend is MONTHLY distributed to weekly (within-month variation = 0)
  - Only {len(df_mmm)} observations for ~{n_total} parameters ({len(df_mmm)/n_total:.1f} obs/param)
  - Channel groups are still somewhat correlated
  - Priors will meaningfully influence posteriors

  OUTPUT: {output_path}
  Ready for Step 8 (fitting the Bayesian MMM).
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
