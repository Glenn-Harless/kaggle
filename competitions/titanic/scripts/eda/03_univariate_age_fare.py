import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]

# --- Age Stats ---
print("=" * 50)
print("AGE BY SURVIVAL")
print("=" * 50)
print(f"Survived - mean: {survived['Age'].mean():.1f}, median: {survived['Age'].median():.1f}")
print(f"Died     - mean: {died['Age'].mean():.1f}, median: {died['Age'].median():.1f}")
print()

# Age bins
bins = [0, 5, 12, 18, 30, 50, 80]
labels = ["0-5", "6-12", "13-18", "19-30", "31-50", "51-80"]
train["AgeBin"] = pd.cut(train["Age"], bins=bins, labels=labels)
age_survival = train.groupby("AgeBin", observed=True)["Survived"].agg(["mean", "count"])
age_survival.columns = ["survival_rate", "count"]
print("Survival by age group:")
print(age_survival)
print()

# --- Fare Stats ---
print("=" * 50)
print("FARE BY SURVIVAL")
print("=" * 50)
print(f"Survived - mean: ${survived['Fare'].mean():.2f}, median: ${survived['Fare'].median():.2f}")
print(f"Died     - mean: ${died['Fare'].mean():.2f}, median: ${died['Fare'].median():.2f}")
print()

# Fare quartiles
train["FareQ"] = pd.qcut(train["Fare"], q=4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (priciest)"])
fare_survival = train.groupby("FareQ", observed=True)["Survived"].agg(["mean", "count"])
fare_survival.columns = ["survival_rate", "count"]
print("Survival by fare quartile:")
print(fare_survival)
print()

# --- Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution by survival (overlapping histograms)
axes[0, 0].hist(died["Age"].dropna(), bins=30, alpha=0.6, color="#e74c3c", label="Died", edgecolor="white")
axes[0, 0].hist(survived["Age"].dropna(), bins=30, alpha=0.6, color="#2ecc71", label="Survived", edgecolor="white")
axes[0, 0].set_title("Age Distribution by Survival", fontsize=13, fontweight="bold")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# Age bin survival rates
colors = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
          for r in age_survival["survival_rate"]]
bars = axes[0, 1].bar(age_survival.index, age_survival["survival_rate"], color=colors)
axes[0, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[0, 1].set_title("Survival Rate by Age Group", fontsize=13, fontweight="bold")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].legend()
for bar, val, cnt in zip(bars, age_survival["survival_rate"], age_survival["count"]):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

# Fare distribution by survival
axes[1, 0].hist(died["Fare"].dropna(), bins=40, alpha=0.6, color="#e74c3c", label="Died",
                edgecolor="white", range=(0, 200))
axes[1, 0].hist(survived["Fare"].dropna(), bins=40, alpha=0.6, color="#2ecc71", label="Survived",
                edgecolor="white", range=(0, 200))
axes[1, 0].set_title("Fare Distribution by Survival (capped at $200)", fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Fare ($)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

# Fare quartile survival rates
colors_f = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in fare_survival["survival_rate"]]
bars_f = axes[1, 1].bar(fare_survival.index, fare_survival["survival_rate"], color=colors_f)
axes[1, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[1, 1].set_title("Survival Rate by Fare Quartile", fontsize=13, fontweight="bold")
axes[1, 1].set_ylabel("Survival Rate")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
for bar, val, cnt in zip(bars_f, fare_survival["survival_rate"], fare_survival["count"]):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/univariate_age_fare.png", dpi=150)
print("Plot saved to plots/eda/univariate_age_fare.png")
