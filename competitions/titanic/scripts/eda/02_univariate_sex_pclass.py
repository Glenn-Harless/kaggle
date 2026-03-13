import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# --- Sex vs Survival ---
print("=" * 50)
print("SURVIVAL BY SEX")
print("=" * 50)
sex_survival = train.groupby("Sex")["Survived"].agg(["mean", "sum", "count"])
sex_survival.columns = ["survival_rate", "survived", "total"]
sex_survival["died"] = sex_survival["total"] - sex_survival["survived"]
print(sex_survival)
print()

# --- Pclass vs Survival ---
print("=" * 50)
print("SURVIVAL BY PCLASS")
print("=" * 50)
pclass_survival = train.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"])
pclass_survival.columns = ["survival_rate", "survived", "total"]
pclass_survival["died"] = pclass_survival["total"] - pclass_survival["survived"]
print(pclass_survival)
print()

# --- Passenger counts by class ---
print("=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)
print(train["Pclass"].value_counts().sort_index())
print()

# --- Visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sex survival bar chart
sex_rates = train.groupby("Sex")["Survived"].mean()
bars1 = axes[0].bar(sex_rates.index, sex_rates.values, color=["#4a90d9", "#d94a4a"])
axes[0].set_title("Survival Rate by Sex", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)
for bar, val in zip(bars1, sex_rates.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")

# Pclass survival bar chart
pclass_rates = train.groupby("Pclass")["Survived"].mean()
bars2 = axes[1].bar(pclass_rates.index.astype(str), pclass_rates.values,
                     color=["#2ecc71", "#f39c12", "#e74c3c"])
axes[1].set_title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Survival Rate")
axes[1].set_xlabel("Class (1=Upper, 2=Middle, 3=Lower)")
axes[1].set_ylim(0, 1)
for bar, val in zip(bars2, pclass_rates.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/scripts/univariate_sex_pclass.png", dpi=150)
print("Plot saved to scripts/univariate_sex_pclass.png")
