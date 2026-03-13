import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# --- SibSp vs Survival ---
print("=" * 50)
print("SURVIVAL BY SIBSP (siblings/spouses aboard)")
print("=" * 50)
sibsp_survival = train.groupby("SibSp")["Survived"].agg(["mean", "count"])
sibsp_survival.columns = ["survival_rate", "count"]
print(sibsp_survival)
print()

# --- Parch vs Survival ---
print("=" * 50)
print("SURVIVAL BY PARCH (parents/children aboard)")
print("=" * 50)
parch_survival = train.groupby("Parch")["Survived"].agg(["mean", "count"])
parch_survival.columns = ["survival_rate", "count"]
print(parch_survival)
print()

# --- Combined family size ---
train["FamilySize"] = train["SibSp"] + train["Parch"]
train["IsAlone"] = (train["FamilySize"] == 0).astype(int)

print("=" * 50)
print("SURVIVAL BY TOTAL FAMILY SIZE (SibSp + Parch)")
print("=" * 50)
family_survival = train.groupby("FamilySize")["Survived"].agg(["mean", "count"])
family_survival.columns = ["survival_rate", "count"]
print(family_survival)
print()

print("=" * 50)
print("ALONE vs WITH FAMILY")
print("=" * 50)
alone_survival = train.groupby("IsAlone")["Survived"].agg(["mean", "count"])
alone_survival.index = ["With Family", "Alone"]
alone_survival.columns = ["survival_rate", "count"]
print(alone_survival)
print()

# --- Visualizations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SibSp survival rates
colors_s = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in sibsp_survival["survival_rate"]]
bars_s = axes[0, 0].bar(sibsp_survival.index.astype(str), sibsp_survival["survival_rate"], color=colors_s)
axes[0, 0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[0, 0].set_title("Survival Rate by SibSp", fontsize=13, fontweight="bold")
axes[0, 0].set_ylabel("Survival Rate")
axes[0, 0].set_xlabel("# of Siblings/Spouses Aboard")
axes[0, 0].set_ylim(0, 1)
axes[0, 0].legend()
for bar, val, cnt in zip(bars_s, sibsp_survival["survival_rate"], sibsp_survival["count"]):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

# Parch survival rates
colors_p = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in parch_survival["survival_rate"]]
bars_p = axes[0, 1].bar(parch_survival.index.astype(str), parch_survival["survival_rate"], color=colors_p)
axes[0, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[0, 1].set_title("Survival Rate by Parch", fontsize=13, fontweight="bold")
axes[0, 1].set_ylabel("Survival Rate")
axes[0, 1].set_xlabel("# of Parents/Children Aboard")
axes[0, 1].set_ylim(0, 1)
axes[0, 1].legend()
for bar, val, cnt in zip(bars_p, parch_survival["survival_rate"], parch_survival["count"]):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

# Family size survival rates
colors_f = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in family_survival["survival_rate"]]
bars_f = axes[1, 0].bar(family_survival.index.astype(str), family_survival["survival_rate"], color=colors_f)
axes[1, 0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[1, 0].set_title("Survival Rate by Family Size", fontsize=13, fontweight="bold")
axes[1, 0].set_ylabel("Survival Rate")
axes[1, 0].set_xlabel("Total Family Members Aboard (SibSp + Parch)")
axes[1, 0].set_ylim(0, 1)
axes[1, 0].legend()
for bar, val, cnt in zip(bars_f, family_survival["survival_rate"], family_survival["count"]):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

# Alone vs With Family
colors_a = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in alone_survival["survival_rate"]]
bars_a = axes[1, 1].bar(alone_survival.index, alone_survival["survival_rate"], color=colors_a)
axes[1, 1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[1, 1].set_title("Survival Rate: Alone vs With Family", fontsize=13, fontweight="bold")
axes[1, 1].set_ylabel("Survival Rate")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
for bar, val, cnt in zip(bars_a, alone_survival["survival_rate"], alone_survival["count"]):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.0%}\n(n={cnt})", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/univariate_sibsp_parch.png", dpi=150)
print("Plot saved to plots/eda/univariate_sibsp_parch.png")
