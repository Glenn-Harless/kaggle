import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# --- Sex x Pclass survival ---
print("=" * 60)
print("SURVIVAL BY SEX x PCLASS")
print("=" * 60)
cross = train.groupby(["Sex", "Pclass"])["Survived"].agg(["mean", "sum", "count"])
cross.columns = ["survival_rate", "survived", "total"]
cross["died"] = cross["total"] - cross["survived"]
print(cross)
print()

# Pivot for readability
pivot = train.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
pivot.columns = ["1st", "2nd", "3rd"]
print("Survival rate matrix:")
print(pivot.round(3))
print()

# Count pivot
count_pivot = train.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="count")
count_pivot.columns = ["1st", "2nd", "3rd"]
print("Passenger count matrix:")
print(count_pivot)
print()

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Grouped bar chart
x = np.arange(3)
width = 0.35
classes = [1, 2, 3]

female_rates = [pivot.loc["female", c] for c in ["1st", "2nd", "3rd"]]
male_rates = [pivot.loc["male", c] for c in ["1st", "2nd", "3rd"]]

bars_f = axes[0].bar(x - width/2, female_rates, width, label="Female", color="#4a90d9")
bars_m = axes[0].bar(x + width/2, male_rates, width, label="Male", color="#d94a4a")
axes[0].set_title("Survival Rate by Sex and Class", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Survival Rate")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
axes[0].set_ylim(0, 1.1)
axes[0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5, label="Overall rate (38%)")
axes[0].legend()

for bar, val in zip(bars_f, female_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")
for bar, val in zip(bars_m, male_rates):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")

# Heatmap style
survival_matrix = pivot.values
im = axes[1].imshow(survival_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Female", "Male"])
axes[1].set_title("Survival Rate Heatmap", fontsize=14, fontweight="bold")

for i in range(2):
    for j in range(3):
        count = int(count_pivot.values[i, j])
        axes[1].text(j, i, f"{survival_matrix[i, j]:.0%}\n(n={count})",
                     ha="center", va="center", fontsize=12, fontweight="bold",
                     color="black")

plt.colorbar(im, ax=axes[1], shrink=0.8, label="Survival Rate")
plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/bivariate_sex_pclass.png", dpi=150)
print("Plot saved to plots/eda/bivariate_sex_pclass.png")
