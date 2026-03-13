import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")

# Encode Sex as numeric for correlation
train["Sex_encoded"] = (train["Sex"] == "male").astype(int)

# Add engineered features we've been discussing
train["FamilySize"] = train["SibSp"] + train["Parch"]
train["IsAlone"] = (train["FamilySize"] == 0).astype(int)

# Select numeric columns for correlation
cols = ["Survived", "Pclass", "Sex_encoded", "Age", "SibSp", "Parch",
        "Fare", "FamilySize", "IsAlone"]
corr = train[cols].corr().round(3)

print("=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
print(corr.to_string())
print()

# Correlations with Survived specifically
print("=" * 60)
print("CORRELATIONS WITH SURVIVED (sorted by absolute value)")
print("=" * 60)
surv_corr = corr["Survived"].drop("Survived").abs().sort_values(ascending=False)
for feat in surv_corr.index:
    val = corr.loc[feat, "Survived"]
    print(f"  {feat:15s}: {val:+.3f}")
print()

# Notable feature-to-feature correlations
print("=" * 60)
print("NOTABLE FEATURE-FEATURE CORRELATIONS (|r| > 0.4)")
print("=" * 60)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if cols[i] == "Survived" or cols[j] == "Survived":
            continue
        r = corr.iloc[i, j]
        if abs(r) > 0.4:
            print(f"  {cols[i]:15s} x {cols[j]:15s}: {r:+.3f}")

# --- Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Full heatmap
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=axes[0], mask=mask,
            square=True, linewidths=0.5,
            xticklabels=[c.replace("_encoded", "").replace("_", "\n") for c in cols],
            yticklabels=[c.replace("_encoded", "").replace("_", "\n") for c in cols])
axes[0].set_title("Correlation Heatmap", fontsize=14, fontweight="bold")

# Bar chart of correlations with Survived
surv_corr_values = corr["Survived"].drop("Survived").sort_values()
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in surv_corr_values]
bars = axes[1].barh(
    [c.replace("_encoded", "").replace("_", " ") for c in surv_corr_values.index],
    surv_corr_values.values,
    color=colors
)
axes[1].set_title("Correlation with Survival", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Pearson Correlation")
axes[1].axvline(x=0, color="gray", linewidth=0.5)
for bar, val in zip(bars, surv_corr_values.values):
    x_pos = val + 0.02 if val > 0 else val - 0.06
    axes[1].text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/Users/glennharless/dev-brain/kaggle/competitions/titanic/plots/eda/bivariate_correlation.png", dpi=150)
print("\nPlot saved to plots/eda/bivariate_correlation.png")
