import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# ============================================================
# 1. TITLE EXTRACTION
# ============================================================
print("=" * 60)
print("1. TITLE EXTRACTION FROM NAME")
print("=" * 60)

# Titles sit between ", " and "."
train["Title"] = train["Name"].str.extract(r", (\w+)\.", expand=False)

print("\n--- All titles ---")
title_counts = train.groupby("Title")["Survived"].agg(["mean", "count"]).sort_values("count", ascending=False)
title_counts.columns = ["survival_rate", "count"]
print(title_counts)
print()

# Group rare titles
title_map = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",  # French Miss
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Mme": "Mrs",  # French Mrs
    "Capt": "Rare",
    "Sir": "Rare",
}
train["TitleGrouped"] = train["Title"].map(title_map).fillna("Rare")

print("--- Grouped titles ---")
grouped = train.groupby("TitleGrouped")["Survived"].agg(["mean", "count"]).sort_values("count", ascending=False)
grouped.columns = ["survival_rate", "count"]
print(grouped)
print()

# Title as age imputation helper
print("--- Median age by title (for imputation) ---")
age_by_title = train.groupby("Title")["Age"].median().sort_values()
print(age_by_title[age_by_title.index.isin(["Master", "Miss", "Mr", "Mrs"])])
print()

# How many missing ages could title help with?
missing_age = train[train["Age"].isnull()]
print(f"Missing ages: {len(missing_age)}")
print(f"Missing ages by title:")
print(missing_age["TitleGrouped"].value_counts())
print()

# ============================================================
# 2. DECK EXTRACTION FROM CABIN
# ============================================================
print("=" * 60)
print("2. DECK EXTRACTION FROM CABIN")
print("=" * 60)

train["Deck"] = train["Cabin"].str[0]  # First letter

print("\n--- Survival by deck ---")
deck_survival = train.groupby("Deck")["Survived"].agg(["mean", "count"]).sort_values("count", ascending=False)
deck_survival.columns = ["survival_rate", "count"]
print(deck_survival)
print()

# HasCabin
train["HasCabin"] = train["Cabin"].notna().astype(int)
print("--- HasCabin ---")
cabin_surv = train.groupby("HasCabin")["Survived"].agg(["mean", "count"])
cabin_surv.index = ["No Cabin", "Has Cabin"]
cabin_surv.columns = ["survival_rate", "count"]
print(cabin_surv)
print()

# ============================================================
# 3. TICKET GROUPS
# ============================================================
print("=" * 60)
print("3. TICKET GROUPS (shared tickets)")
print("=" * 60)

ticket_counts = train.groupby("Ticket")["PassengerId"].transform("count")
train["TicketGroupSize"] = ticket_counts

print("\n--- Survival by ticket group size ---")
tg_survival = train.groupby("TicketGroupSize")["Survived"].agg(["mean", "count"])
tg_survival.columns = ["survival_rate", "count"]
print(tg_survival)
print()

# Compare to FamilySize
train["FamilySize"] = train["SibSp"] + train["Parch"]
print("--- Ticket group size vs family size (are they the same?) ---")
same = (train["TicketGroupSize"] == train["FamilySize"] + 1).mean()
print(f"Ticket group size == FamilySize + 1: {same:.1%} of the time")
diff = train[train["TicketGroupSize"] != train["FamilySize"] + 1]
print(f"Rows where they differ: {len(diff)}")
print()

# ============================================================
# 4. FARE PER PERSON
# ============================================================
print("=" * 60)
print("4. FARE PER PERSON")
print("=" * 60)

train["FarePerPerson"] = train["Fare"] / train["TicketGroupSize"]

print("\n--- Survival by FarePerPerson quartile ---")
train["FPP_Q"] = pd.qcut(train["FarePerPerson"], q=4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (priciest)"])
fpp_surv = train.groupby("FPP_Q", observed=True)["Survived"].agg(["mean", "count"])
fpp_surv.columns = ["survival_rate", "count"]
print(fpp_surv)
print()

# Compare to raw Fare quartiles
print("--- Raw Fare quartile survival (for comparison) ---")
train["Fare_Q"] = pd.qcut(train["Fare"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
fare_surv = train.groupby("Fare_Q", observed=True)["Survived"].agg(["mean", "count"])
fare_surv.columns = ["survival_rate", "count"]
print(fare_surv)
print()

# ============================================================
# 5. PERSON TYPE (Age x Sex interaction)
# ============================================================
print("=" * 60)
print("5. PERSON TYPE (child / adult_female / adult_male)")
print("=" * 60)

def person_type(row):
    if row["Age"] <= 12:
        return "child"
    elif row["Sex"] == "female":
        return "adult_female"
    else:
        return "adult_male"

# Only for rows with age
has_age = train[train["Age"].notna()].copy()
has_age["PersonType"] = has_age.apply(person_type, axis=1)

print()
pt_surv = has_age.groupby("PersonType")["Survived"].agg(["mean", "count"])
pt_surv.columns = ["survival_rate", "count"]
print(pt_surv)
print()

# ============================================================
# VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 1. Title grouped
title_data = train.groupby("TitleGrouped")["Survived"].mean().sort_values(ascending=False)
title_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c" for r in title_data.values]
bars = axes[0, 0].bar(title_data.index, title_data.values, color=title_colors)
axes[0, 0].set_title("Survival by Title", fontsize=13, fontweight="bold")
axes[0, 0].set_ylabel("Survival Rate")
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, title_data.values):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

# 2. Deck
deck_data = train[train["Deck"].notna()].groupby("Deck")["Survived"].mean().sort_values(ascending=False)
deck_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c" for r in deck_data.values]
bars = axes[0, 1].bar(deck_data.index, deck_data.values, color=deck_colors)
axes[0, 1].set_title("Survival by Deck (23% of passengers)", fontsize=13, fontweight="bold")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_ylim(0, 1.1)
axes[0, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, deck_data.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

# 3. HasCabin
hc_data = train.groupby("HasCabin")["Survived"].mean()
hc_labels = ["No Cabin", "Has Cabin"]
hc_colors = ["#e74c3c", "#2ecc71"]
bars = axes[0, 2].bar(hc_labels, hc_data.values, color=hc_colors)
axes[0, 2].set_title("Survival by HasCabin", fontsize=13, fontweight="bold")
axes[0, 2].set_ylabel("Survival Rate")
axes[0, 2].set_ylim(0, 1.1)
axes[0, 2].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, hc_data.values):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

# 4. Ticket group size
tg_data = train.groupby("TicketGroupSize")["Survived"].mean()
tg_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c" for r in tg_data.values]
bars = axes[1, 0].bar(tg_data.index.astype(str), tg_data.values, color=tg_colors)
axes[1, 0].set_title("Survival by Ticket Group Size", fontsize=13, fontweight="bold")
axes[1, 0].set_ylabel("Survival Rate")
axes[1, 0].set_xlabel("# passengers sharing ticket")
axes[1, 0].set_ylim(0, 1.1)
axes[1, 0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, tg_data.values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=9, fontweight="bold")

# 5. FarePerPerson quartiles
fpp_data = train.groupby("FPP_Q", observed=True)["Survived"].mean()
fpp_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c" for r in fpp_data.values]
bars = axes[1, 1].bar(fpp_data.index, fpp_data.values, color=fpp_colors)
axes[1, 1].set_title("Survival by Fare Per Person Quartile", fontsize=13, fontweight="bold")
axes[1, 1].set_ylabel("Survival Rate")
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, fpp_data.values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

# 6. PersonType
pt_data = has_age.groupby("PersonType")["Survived"].mean().sort_values(ascending=False)
pt_colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c" for r in pt_data.values]
bars = axes[1, 2].bar(pt_data.index, pt_data.values, color=pt_colors)
axes[1, 2].set_title("Survival by Person Type", fontsize=13, fontweight="bold")
axes[1, 2].set_ylabel("Survival Rate")
axes[1, 2].set_ylim(0, 1.1)
axes[1, 2].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5)
for bar, val in zip(bars, pt_data.values):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/feature_exploration.png", dpi=150)
print("Plot saved to plots/eda/feature_exploration.png")
