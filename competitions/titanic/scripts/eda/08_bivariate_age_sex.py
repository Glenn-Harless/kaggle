import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# --- Age bins by sex and survival ---
bins = [0, 5, 12, 18, 30, 50, 80]
labels = ["0-5", "6-12", "13-18", "19-30", "31-50", "51-80"]
train["AgeBin"] = pd.cut(train["Age"], bins=bins, labels=labels)

print("=" * 60)
print("SURVIVAL BY AGE GROUP x SEX")
print("=" * 60)
cross = train.groupby(["AgeBin", "Sex"], observed=True)["Survived"].agg(["mean", "count"])
cross.columns = ["survival_rate", "count"]
print(cross)
print()

# Pivot for readability
pivot = train.pivot_table(values="Survived", index="AgeBin", columns="Sex", aggfunc=["mean", "count"])
print("Survival rates:")
print(pivot["mean"].round(3))
print("\nCounts:")
print(pivot["count"])
print()

# --- Is "children first" gendered? ---
print("=" * 60)
print("CHILDREN (age <= 12) BY SEX")
print("=" * 60)
children = train[train["Age"] <= 12]
print(children.groupby("Sex")["Survived"].agg(["mean", "count"]))
print()

adults = train[train["Age"] > 12]
print("ADULTS (age > 12) BY SEX")
print(adults.groupby("Sex")["Survived"].agg(["mean", "count"]))
print()

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Age group survival by sex (grouped bar)
age_labels = labels
female_rates = []
male_rates = []
female_counts = []
male_counts = []

for age_bin in age_labels:
    f = train[(train["AgeBin"] == age_bin) & (train["Sex"] == "female")]
    m = train[(train["AgeBin"] == age_bin) & (train["Sex"] == "male")]
    female_rates.append(f["Survived"].mean() if len(f) > 0 else 0)
    male_rates.append(m["Survived"].mean() if len(m) > 0 else 0)
    female_counts.append(len(f))
    male_counts.append(len(m))

x = np.arange(len(age_labels))
width = 0.35

bars_f = axes[0].bar(x - width/2, female_rates, width, label="Female", color="#4a90d9")
bars_m = axes[0].bar(x + width/2, male_rates, width, label="Male", color="#d94a4a")
axes[0].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5, label="Overall (38%)")
axes[0].set_title("Survival Rate by Age Group and Sex", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Survival Rate")
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_labels)
axes[0].set_xlabel("Age Group")
axes[0].set_ylim(0, 1.15)
axes[0].legend()

for bar, val, cnt in zip(bars_f, female_rates, female_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}\n({cnt})", ha="center", fontsize=8, fontweight="bold")
for bar, val, cnt in zip(bars_m, male_rates, male_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}\n({cnt})", ha="center", fontsize=8, fontweight="bold")

# Children vs Adults by sex
categories = ["Girl\n(<=12)", "Boy\n(<=12)", "Woman\n(>12)", "Man\n(>12)"]
girl = children[children["Sex"] == "female"]["Survived"].mean()
boy = children[children["Sex"] == "male"]["Survived"].mean()
woman = adults[adults["Sex"] == "female"]["Survived"].mean()
man = adults[adults["Sex"] == "male"]["Survived"].mean()
rates = [girl, boy, woman, man]
counts = [
    len(children[children["Sex"] == "female"]),
    len(children[children["Sex"] == "male"]),
    len(adults[adults["Sex"] == "female"]),
    len(adults[adults["Sex"] == "male"]),
]
colors = ["#4a90d9", "#d94a4a", "#4a90d9", "#d94a4a"]

bars = axes[1].bar(categories, rates, color=colors, alpha=0.8)
axes[1].axhline(y=0.38, color="gray", linestyle="--", alpha=0.5, label="Overall (38%)")
axes[1].set_title("Children vs Adults by Sex", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1.15)
axes[1].legend()
for bar, val, cnt in zip(bars, rates, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.0%}\n(n={cnt})", ha="center", fontsize=10, fontweight="bold")

# Age distribution overlay: survived vs died, faceted by sex
for sex, color_s, color_d, title in [
    ("female", "#2ecc71", "#e74c3c", "Female Age Distribution"),
    ("male", "#2ecc71", "#e74c3c", "Male Age Distribution"),
]:
    ax = axes[2] if sex == "male" else axes[2]

# Just do male since female would need a 4th panel - combine into one
axes[2].hist(train[(train["Sex"] == "male") & (train["Survived"] == 0)]["Age"].dropna(),
             bins=20, alpha=0.5, color="#e74c3c", label="Male - Died", edgecolor="white")
axes[2].hist(train[(train["Sex"] == "male") & (train["Survived"] == 1)]["Age"].dropna(),
             bins=20, alpha=0.5, color="#2ecc71", label="Male - Survived", edgecolor="white")
axes[2].hist(train[(train["Sex"] == "female") & (train["Survived"] == 0)]["Age"].dropna(),
             bins=20, alpha=0.3, color="#d94a4a", label="Female - Died", edgecolor="white",
             linestyle="--")
axes[2].hist(train[(train["Sex"] == "female") & (train["Survived"] == 1)]["Age"].dropna(),
             bins=20, alpha=0.3, color="#4a90d9", label="Female - Survived", edgecolor="white",
             linestyle="--")
axes[2].set_title("Age Distribution by Sex & Survival", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Count")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/bivariate_age_sex.png", dpi=150)
print("Plot saved to plots/eda/bivariate_age_sex.png")
