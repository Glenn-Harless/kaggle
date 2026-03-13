import pandas as pd

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")
train["Deck"] = train["Cabin"].str[0]
train["Title"] = train["Name"].str.extract(r", (\w+)\.", expand=False)

# ============================================================
# DECK vs PCLASS and FARE
# ============================================================
print("=" * 60)
print("DECK vs PCLASS")
print("=" * 60)
deck_class = pd.crosstab(train["Deck"], train["Pclass"])
deck_class.columns = ["1st", "2nd", "3rd"]
print(deck_class)
print()

# As percentages
deck_class_pct = pd.crosstab(train["Deck"], train["Pclass"], normalize="index").round(3) * 100
deck_class_pct.columns = ["1st %", "2nd %", "3rd %"]
print(deck_class_pct)
print()

print("=" * 60)
print("DECK vs FARE (median)")
print("=" * 60)
deck_fare = train.groupby("Deck")["Fare"].agg(["median", "mean", "count"]).round(2)
print(deck_fare)
print()

# Can we infer deck from Pclass?
print("=" * 60)
print("PCLASS DISTRIBUTION OF MISSING vs NON-MISSING CABIN")
print("=" * 60)
train["HasCabin"] = train["Cabin"].notna().astype(int)
print(pd.crosstab(train["HasCabin"], train["Pclass"], normalize="index").round(3) * 100)
print()

# ============================================================
# MASTER TITLE - AGE RANGE
# ============================================================
print("=" * 60)
print("MASTER TITLE - AGE DISTRIBUTION")
print("=" * 60)
masters = train[train["Title"] == "Master"]
print(f"Count: {len(masters)}")
print(f"Age range: {masters['Age'].min():.1f} - {masters['Age'].max():.1f}")
print(f"Mean: {masters['Age'].mean():.1f}, Median: {masters['Age'].median():.1f}")
print()
print("All Master ages:")
print(masters["Age"].dropna().sort_values().values)
print()

# Compare: what title do males 12-18 have?
print("=" * 60)
print("TITLES OF MALES AGE 12-18")
print("=" * 60)
teen_males = train[(train["Sex"] == "male") & (train["Age"] >= 12) & (train["Age"] <= 18)]
print(teen_males[["Name", "Age", "Title", "Survived"]].sort_values("Age").to_string())
print()

# What about using Title for PersonType instead of age cutoff?
print("=" * 60)
print("TITLE-BASED PERSON TYPE")
print("=" * 60)
def person_type_title(row):
    if row["Title"] == "Master":
        return "child"
    elif row["Sex"] == "female":
        return "adult_female"
    else:
        return "adult_male"

train["PersonType_Title"] = train.apply(person_type_title, axis=1)
pt = train.groupby("PersonType_Title")["Survived"].agg(["mean", "count"])
pt.columns = ["survival_rate", "count"]
print(pt)
print()

# Compare: age-based vs title-based
def person_type_age(row):
    if pd.notna(row["Age"]) and row["Age"] <= 12:
        return "child"
    elif row["Sex"] == "female":
        return "adult_female"
    else:
        return "adult_male"

train["PersonType_Age"] = train.apply(person_type_age, axis=1)
pt2 = train.groupby("PersonType_Age")["Survived"].agg(["mean", "count"])
pt2.columns = ["survival_rate", "count"]
print("For comparison - age-based (<=12, with missing ages falling into adult):")
print(pt2)
