import pandas as pd

train = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/train.csv")
test = pd.read_csv("/Users/glennharless/dev-brain/kaggle/competitions/titanic/data/test.csv")

print("=" * 60)
print("TRAINING SET")
print("=" * 60)
print(f"Shape: {train.shape[0]} rows x {train.shape[1]} columns\n")

print("--- First 5 rows ---")
print(train.head().to_string())

print("\n--- Data Types ---")
print(train.dtypes)

print("\n--- Missing Values ---")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(1)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
print(missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False))
print(f"\nTotal rows with any missing value: {train.isnull().any(axis=1).sum()} / {len(train)}")

print("\n--- Basic Stats ---")
print(train.describe().round(2).to_string())

print("\n\n" + "=" * 60)
print("TEST SET")
print("=" * 60)
print(f"Shape: {test.shape[0]} rows x {test.shape[1]} columns\n")

print("--- Missing Values ---")
missing_test = test.isnull().sum()
missing_test_pct = (missing_test / len(test) * 100).round(1)
missing_test_df = pd.DataFrame({"missing": missing_test, "pct": missing_test_pct})
print(missing_test_df[missing_test_df["missing"] > 0].sort_values("pct", ascending=False))
