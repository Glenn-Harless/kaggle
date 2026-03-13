import pandas as pd

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")
train["FamilySize"] = train["SibSp"] + train["Parch"]

fam3 = train[train["FamilySize"] == 3]

print("=" * 50)
print("FAMILY SIZE 3 BREAKDOWN (n=29)")
print("=" * 50)
print()

# Sex breakdown
print("--- By Sex ---")
sex_breakdown = fam3.groupby("Sex")["Survived"].agg(["mean", "count"])
sex_breakdown.columns = ["survival_rate", "count"]
print(sex_breakdown)
print()

# Who are the ones who died?
print("--- Who died? ---")
died_fam3 = fam3[fam3["Survived"] == 0]
print(f"Total died: {len(died_fam3)}")
print(f"  Male: {len(died_fam3[died_fam3['Sex'] == 'male'])}")
print(f"  Female: {len(died_fam3[died_fam3['Sex'] == 'female'])}")
print()

# Show the actual passengers
print("--- All family size 3 passengers ---")
cols = ["Name", "Sex", "Age", "SibSp", "Parch", "Pclass", "Survived"]
print(fam3[cols].sort_values(["Survived", "Sex"]).to_string())
