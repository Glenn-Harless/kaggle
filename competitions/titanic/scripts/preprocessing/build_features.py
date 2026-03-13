"""
Titanic Feature Engineering Pipeline (v3)

v3 changes (systematic interaction testing, Stages A-C):
  - Replaced ordinal Pclass with one-hot (Pclass_2, Pclass_3; Pclass_1 is reference)
  - Added Sex x Pclass interaction terms (Sex_x_Pclass_2, Sex_x_Pclass_3)
  - Added grouped deck features (Deck_ABC, Deck_DE, Deck_FG from Cabin letter)
  - Rejected: log1p(Fare) — did not improve CV
  - Rejected: SurnameGroupSize, SurnameSurvHint — hurt CV even with OOF encoding

v2 changes (informed by code review + leaderboard analysis):
  - Removed TicketGroupSize (leakage: was computed on train+test combined)
  - Removed FarePerPerson (depended on TicketGroupSize)
  - Removed PersonType (redundant with Sex + Title)
  - Replaced raw Age with IsChild bucket (EDA showed non-linear threshold at ~12)
  - Replaced raw FamilySize with buckets: IsAlone, IsLargeFamily (EDA showed
    non-linear: alone=30%, small=55-72%, large=0-20%)
  - Kept: Pclass, Sex, Fare, HasCabin, Title, Embarked

Input:  data/train.csv, data/test.csv
Output: data/train_processed.csv, data/test_processed.csv
"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/data"


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    return train, test


def extract_title(df):
    """Extract title from Name field."""
    df["Title"] = df["Name"].str.extract(r", (\w+)\.", expand=False)

    title_map = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
        "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
        "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare",
        "Mme": "Mrs", "Capt": "Rare", "Sir": "Rare",
    }
    df["Title"] = df["Title"].map(title_map).fillna("Rare")
    return df


def impute_age(df, age_medians):
    """Impute missing ages using median age by Title group."""
    for title, median_age in age_medians.items():
        mask = (df["Age"].isnull()) & (df["Title"] == title)
        df.loc[mask, "Age"] = median_age
    return df


def impute_embarked(df):
    """Fill missing Embarked with mode (S = Southampton)."""
    df["Embarked"] = df["Embarked"].fillna("S")
    return df


def impute_fare(df, fare_medians):
    """Fill missing Fare using median by Pclass."""
    for pclass, median_fare in fare_medians.items():
        mask = (df["Fare"].isnull()) & (df["Pclass"] == pclass)
        df.loc[mask, "Fare"] = median_fare
    return df


def engineer_features(df):
    """Create engineered features aligned with EDA conclusions.

    v2: uses buckets for non-linear features (Age, FamilySize),
    removes ticket-derived features, removes redundant gender encodings.
    """

    # Family size buckets (EDA: alone=30%, small=55-72%, large=0-20%)
    family_size = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (family_size == 0).astype(int)
    df["IsLargeFamily"] = (family_size >= 4).astype(int)
    # Small family (1-3) is the implicit reference category

    # Cabin feature
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # v3: Grouped deck from Cabin letter
    # ABC = high decks (mostly 1st class), DE = mid, FG = low, U = unknown (reference)
    deck = df["Cabin"].fillna("U").str[0].replace("T", "U")
    df["Deck_ABC"] = deck.isin(["A", "B", "C"]).astype(int)
    df["Deck_DE"] = deck.isin(["D", "E"]).astype(int)
    df["Deck_FG"] = deck.isin(["F", "G"]).astype(int)

    # Age bucket (EDA: child <=12 has ~58% survival regardless of sex,
    # gender "switches on" at ~13)
    df["IsChild"] = (df["Age"] <= 12).astype(int)

    return df


def encode_categoricals(df):
    """One-hot encode categorical features."""
    df["Sex"] = (df["Sex"] == "male").astype(int)

    df = pd.get_dummies(df, columns=["Title"], prefix="Title", dtype=int)
    df = pd.get_dummies(df, columns=["Embarked"], prefix="Emb", dtype=int)

    return df


def add_interactions(df):
    """v3: Replace ordinal Pclass with one-hot + Sex x Pclass interactions.

    Captures the strongest EDA finding: class effect differs by sex.
    1st class women: 97% vs 3rd class women: 50%, but
    1st class men: 37% vs 3rd class men: 15%.
    """
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
    df["Sex_x_Pclass_2"] = df["Sex"] * df["Pclass_2"]
    df["Sex_x_Pclass_3"] = df["Sex"] * df["Pclass_3"]
    df = df.drop(columns=["Pclass"])
    return df


def drop_raw_columns(df):
    """Drop columns replaced by engineered features."""
    drop_cols = [
        "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin",
        "Age",  # v2: replaced by IsChild bucket
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def run_pipeline():
    """Execute the full preprocessing pipeline."""
    train, test = load_data()

    # Store PassengerId for submission file later
    test_ids = test["PassengerId"].copy()

    # Store target before processing
    y_train = train["Survived"].copy()
    train = train.drop(columns=["Survived"])

    # ---- Compute medians from training data only (prevent leakage) ----
    train = extract_title(train)
    test = extract_title(test)

    age_medians = train.groupby("Title")["Age"].median().to_dict()
    fare_medians = train.groupby("Pclass")["Fare"].median().to_dict()

    # ---- Apply identical transformations to both sets ----
    for df in [train, test]:
        impute_age(df, age_medians)
        impute_embarked(df)
        impute_fare(df, fare_medians)

    train = engineer_features(train)
    test = engineer_features(test)

    train = encode_categoricals(train)
    test = encode_categoricals(test)

    # v3: interactions must come after encode_categoricals (Sex is now 0/1)
    train = add_interactions(train)
    test = add_interactions(test)

    train = drop_raw_columns(train)
    test = drop_raw_columns(test)

    # ---- Align columns (in case test is missing a one-hot column) ----
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    for col in train_cols - test_cols:
        test[col] = 0
    for col in test_cols - train_cols:
        train[col] = 0

    # Ensure same column order
    test = test[train.columns]

    # ---- Add target back ----
    train.insert(0, "Survived", y_train)

    # ---- Save ----
    train.to_csv(f"{DATA_DIR}/train_processed.csv", index=False)

    test_out = test.copy()
    test_out.insert(0, "PassengerId", test_ids)
    test_out.to_csv(f"{DATA_DIR}/test_processed.csv", index=False)

    return train, test, test_ids


if __name__ == "__main__":
    train, test, test_ids = run_pipeline()

    print("=" * 60)
    print("PREPROCESSING COMPLETE (v3)")
    print("=" * 60)
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"\nTrain columns ({len(train.columns)}):")
    print(train.columns.tolist())
    print(f"\nMissing values in train:")
    missing = train.isnull().sum()
    if missing.sum() == 0:
        print("  None!")
    else:
        print(missing[missing > 0])
    print(f"\nMissing values in test:")
    missing_test = test.isnull().sum()
    if missing_test.sum() == 0:
        print("  None!")
    else:
        print(missing_test[missing_test > 0])
    print(f"\nTarget distribution:")
    print(train["Survived"].value_counts())
    print(f"\nFirst 5 rows of processed train:")
    print(train.head().to_string())
    print(f"\nSaved to:")
    print(f"  {DATA_DIR}/train_processed.csv")
    print(f"  {DATA_DIR}/test_processed.csv")
