import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

print("=" * 50)
print("SURVIVAL BY EMBARKED (port of embarkation)")
print("=" * 50)
print("C = Cherbourg, Q = Queenstown, S = Southampton\n")

embarked_survival = train.groupby("Embarked")["Survived"].agg(["mean", "count"])
embarked_survival.columns = ["survival_rate", "count"]
print(embarked_survival)
print()

# What's the class mix at each port?
print("=" * 50)
print("CLASS DISTRIBUTION BY PORT")
print("=" * 50)
port_class = pd.crosstab(train["Embarked"], train["Pclass"], normalize="index").round(3) * 100
port_class.columns = ["1st %", "2nd %", "3rd %"]
print(port_class)
print()

# Sex mix at each port?
print("=" * 50)
print("SEX DISTRIBUTION BY PORT")
print("=" * 50)
port_sex = pd.crosstab(train["Embarked"], train["Sex"], normalize="index").round(3) * 100
print(port_sex)
print()

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Survival rate by port
port_labels = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
colors_e = ["#2ecc71" if r > 0.5 else "#f39c12" if r > 0.38 else "#e74c3c"
            for r in embarked_survival["survival_rate"]]
bars = axes[0].bar([port_labels[p] for p in embarked_survival.index],
                    embarked_survival["survival_rate"], color=colors_e)
axes[0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.7, label="Overall rate (38%)")
axes[0].set_title("Survival Rate by Port", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)
axes[0].legend()
for bar, val, cnt in zip(bars, embarked_survival["survival_rate"], embarked_survival["count"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}\n(n={cnt})", ha="center", fontsize=10, fontweight="bold")

# Class mix by port (stacked bar)
port_class_raw = pd.crosstab(train["Embarked"], train["Pclass"], normalize="index")
ports = port_class_raw.index.tolist()
port_names = [port_labels[p] for p in ports]
bottom1 = [0] * len(ports)
bottom2 = port_class_raw[1].tolist()
bottom3 = [b1 + b2 for b1, b2 in zip(port_class_raw[1].tolist(), port_class_raw[2].tolist())]

axes[1].bar(port_names, port_class_raw[1], label="1st Class", color="#2ecc71")
axes[1].bar(port_names, port_class_raw[2], bottom=port_class_raw[1], label="2nd Class", color="#f39c12")
axes[1].bar(port_names, port_class_raw[3], bottom=[b1+b2 for b1,b2 in zip(port_class_raw[1], port_class_raw[2])],
            label="3rd Class", color="#e74c3c")
axes[1].set_title("Class Mix by Port", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Proportion")
axes[1].legend()

# Sex mix by port
port_sex_raw = pd.crosstab(train["Embarked"], train["Sex"], normalize="index")
axes[2].bar(port_names, port_sex_raw["female"], label="Female", color="#4a90d9")
axes[2].bar(port_names, port_sex_raw["male"], bottom=port_sex_raw["female"], label="Male", color="#d94a4a")
axes[2].set_title("Sex Mix by Port", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Proportion")
axes[2].legend()

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/univariate_embarked.png", dpi=150)
print("Plot saved to plots/eda/univariate_embarked.png")
