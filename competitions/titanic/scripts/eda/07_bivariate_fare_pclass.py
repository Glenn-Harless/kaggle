import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# --- Fare stats by class ---
print("=" * 60)
print("FARE DISTRIBUTION BY PCLASS")
print("=" * 60)
fare_by_class = train.groupby("Pclass")["Fare"].describe().round(2)
print(fare_by_class)
print()

# Correlation
print("=" * 60)
print("CORRELATION: FARE vs PCLASS")
print("=" * 60)
corr = train["Fare"].corr(train["Pclass"])
print(f"Pearson correlation: {corr:.3f}")
print("(Negative because Pclass 1=upper, 3=lower, while Fare increases with wealth)")
print()

# Does Fare add signal within class?
print("=" * 60)
print("SURVIVAL BY FARE QUARTILE WITHIN EACH CLASS")
print("=" * 60)
for pclass in [1, 2, 3]:
    subset = train[train["Pclass"] == pclass].copy()
    try:
        subset["FareQ"] = pd.qcut(subset["Fare"], q=2, labels=["Lower Half", "Upper Half"])
        result = subset.groupby("FareQ", observed=True)["Survived"].agg(["mean", "count"])
        result.columns = ["survival_rate", "count"]
        print(f"\nClass {pclass}:")
        print(result)
    except ValueError:
        print(f"\nClass {pclass}: Not enough fare variation to split")
print()

# --- Visualizations ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Box plot: Fare by class
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
bp = axes[0].boxplot(
    [train[train["Pclass"] == c]["Fare"].dropna() for c in [1, 2, 3]],
    labels=["1st", "2nd", "3rd"],
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker="o", markersize=3, alpha=0.5)
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title("Fare Distribution by Class", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Fare ($)")
axes[0].set_xlabel("Passenger Class")

# Scatter: Fare vs Survival colored by class
for pclass, color, label in zip([1, 2, 3], colors, ["1st", "2nd", "3rd"]):
    subset = train[train["Pclass"] == pclass]
    # Add jitter to survival for visibility
    jitter = np.random.uniform(-0.1, 0.1, size=len(subset))
    axes[1].scatter(subset["Fare"], subset["Survived"] + jitter,
                     alpha=0.3, color=color, label=label, s=20)
axes[1].set_title("Fare vs Survival by Class", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Fare ($)")
axes[1].set_ylabel("Survived (jittered)")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["Died", "Survived"])
axes[1].legend(title="Class")
axes[1].set_xlim(0, 300)

# Within-class fare effect: survival by fare median split per class
class_labels = []
lower_rates = []
upper_rates = []
for pclass in [1, 2, 3]:
    subset = train[train["Pclass"] == pclass].copy()
    median_fare = subset["Fare"].median()
    low = subset[subset["Fare"] <= median_fare]["Survived"].mean()
    high = subset[subset["Fare"] > median_fare]["Survived"].mean()
    class_labels.append(f"Class {pclass}")
    lower_rates.append(low)
    upper_rates.append(high)

x = np.arange(3)
width = 0.35
axes[2].bar(x - width/2, lower_rates, width, label="Below median fare", color="#e74c3c", alpha=0.8)
axes[2].bar(x + width/2, upper_rates, width, label="Above median fare", color="#2ecc71", alpha=0.8)
axes[2].set_title("Within-Class Fare Effect on Survival", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Survival Rate")
axes[2].set_xticks(x)
axes[2].set_xticklabels(class_labels)
axes[2].set_ylim(0, 1.1)
axes[2].legend()
for i, (lo, hi) in enumerate(zip(lower_rates, upper_rates)):
    axes[2].text(i - width/2, lo + 0.02, f"{lo:.0%}", ha="center", fontsize=10, fontweight="bold")
    axes[2].text(i + width/2, hi + 0.02, f"{hi:.0%}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/bivariate_fare_pclass.png", dpi=150)
print("Plot saved to plots/eda/bivariate_fare_pclass.png")
