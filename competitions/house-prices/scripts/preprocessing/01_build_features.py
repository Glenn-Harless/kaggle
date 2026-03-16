"""
House Prices: Step 1 — Quick EDA + Preprocessing

Understand the target, handle missing values, encode features.

Inputs:  data/train.csv, data/test.csv
Outputs: data/train_processed.csv, data/test_processed.csv
Log:     results/models/01_preprocessing.txt
"""

import sys
import pandas as pd
import numpy as np

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# ============================================================
# CONFIG: Feature classification
# ============================================================

# Categorical features where NaN means "this feature doesn't exist"
# Example: PoolQC NaN = no pool (not a data collection error)
NA_MEANS_NONE = [
    "Alley", "MasVnrType",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
]

# Numeric features where NaN means zero (feature doesn't exist)
# Example: GarageArea NaN = no garage = 0 sq ft
NA_MEANS_ZERO = [
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]

# Ordinal: quality scale (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)
QUALITY_SCALE = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]

# Other ordinal features with custom scales
ORDINAL_MAPS = {
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0},
    "LotShape": {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0},
    "LandSlope": {"Gtl": 2, "Mod": 1, "Sev": 0},
    "PavedDrive": {"Y": 2, "P": 1, "N": 0},
    "CentralAir": {"Y": 1, "N": 0},
    "Street": {"Pave": 1, "Grvl": 0},
    "Utilities": {"AllPub": 3, "NoSewr": 2, "NoSeWa": 1, "ELO": 0},
}

# Nominal features → one-hot encode (no natural ordering)
# MSSubClass looks numeric but is actually a dwelling type code
NOMINAL_FEATURES = [
    "MSSubClass", "MSZoning", "Alley", "LotConfig", "LandContour",
    "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating", "Electrical", "GarageType", "MiscFeature",
    "SaleType", "SaleCondition",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    train = pd.read_csv(f"{BASE}/data/train.csv")
    test = pd.read_csv(f"{BASE}/data/test.csv")
    return train, test


# ============================================================
# EDA FUNCTIONS
# ============================================================

def examine_target(train):
    """Target distribution analysis."""
    y = train["SalePrice"]

    print("=" * 60)
    print("TARGET: SalePrice Distribution")
    print("=" * 60)
    print(f"  Count:    {len(y)}")
    print(f"  Mean:     ${y.mean():,.0f}")
    print(f"  Median:   ${y.median():,.0f}")
    print(f"  Std:      ${y.std():,.0f}")
    print(f"  Min:      ${y.min():,.0f}")
    print(f"  Max:      ${y.max():,.0f}")
    print(f"  Skewness: {y.skew():.3f}  (>1 = heavy right skew)")
    print(f"  Kurtosis: {y.kurtosis():.3f}  (>3 = heavy tails)")
    print()

    # Log transform preview
    log_y = np.log1p(y)
    print("  After log1p(SalePrice):")
    print(f"    Skewness: {log_y.skew():.3f}  (was {y.skew():.3f})")
    print(f"    Kurtosis: {log_y.kurtosis():.3f}  (was {y.kurtosis():.3f})")
    print(f"    -> log transform makes it nearly normal (skew near 0)")
    print()

    # Price distribution buckets
    bins = [0, 100_000, 150_000, 200_000, 250_000, 300_000, 500_000, 800_000]
    labels = ["<100K", "100-150K", "150-200K", "200-250K", "250-300K", "300-500K", "500K+"]
    buckets = pd.cut(y, bins=bins, labels=labels)
    print("  Price distribution:")
    for label in labels:
        count = (buckets == label).sum()
        pct = count / len(y) * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:>10s}: {count:4d} ({pct:5.1f}%) {bar}")
    print()


def analyze_missing(train, test):
    """Missing value analysis for both datasets."""
    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    for name, df in [("Train", train), ("Test", test)]:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()

        print(f"\n  {name} ({df.shape[0]} rows, {df.shape[1]} cols):")
        print(f"  Total missing: {total_missing} / {total_cells} "
              f"({total_missing / total_cells * 100:.1f}%)")

        if len(missing) > 0:
            for col, count in missing.items():
                pct = count / len(df) * 100
                if col in NA_MEANS_NONE:
                    marker = "<- feature absent"
                elif col in NA_MEANS_ZERO or col == "GarageYrBlt":
                    marker = "<- feature absent (0)"
                else:
                    marker = "<- truly missing"
                print(f"    {col:20s}: {count:4d} ({pct:5.1f}%) {marker}")
    print()


def top_correlations(df, target="SalePrice", n=15):
    """Top numeric feature correlations with target."""
    print("=" * 60)
    print(f"TOP {n} CORRELATIONS WITH {target}")
    print("=" * 60)
    print("  (numeric + ordinal-encoded features, Pearson r)")

    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        print("  [target not in numeric columns, skipping]")
        return

    raw_corr = numeric.corr()[target].drop(target)
    abs_corr = raw_corr.abs().sort_values(ascending=False)

    for col in abs_corr.head(n).index:
        r = raw_corr[col]
        sign = "+" if r > 0 else "-"
        bar = "#" * int(abs(r) * 30)
        print(f"    {sign}{abs(r):.3f} {col:25s} {bar}")
    print()


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def handle_missing(df, train_medians, train_modes):
    """
    Handle missing values with domain-informed rules.

    1. Categorical NaN -> "None" where NA means feature is absent
    2. Numeric NaN -> 0 where NA means feature is absent
    3. GarageYrBlt NaN -> 0 (no garage)
    4. Remaining numeric NaN -> train median
    5. Remaining categorical NaN -> train mode
    """
    df = df.copy()

    # Categorical: NaN = "None" (feature absent)
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Numeric: NaN = 0 (feature absent, e.g., no garage -> 0 sq ft)
    for col in NA_MEANS_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # GarageYrBlt: NaN = no garage -> 0
    # (imperfect: 0 isn't a real year, but captures "no garage" for linear models)
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    # Remaining numeric -> train median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any() and col in train_medians.index:
            df[col] = df[col].fillna(train_medians[col])

    # Remaining categorical -> train mode
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any() and col in train_modes:
            df[col] = df[col].fillna(train_modes[col])

    return df


def encode_ordinals(df):
    """Convert ordinal features to numeric using predefined scales."""
    df = df.copy()

    for col in QUALITY_COLS:
        if col in df.columns:
            original = df[col].copy()
            df[col] = df[col].map(QUALITY_SCALE)
            unmapped = df[col].isnull() & original.notnull()
            if unmapped.any():
                vals = original[unmapped].unique()
                print(f"    WARNING: {col} has unmapped values: {vals}")

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            original = df[col].copy()
            df[col] = df[col].map(mapping)
            unmapped = df[col].isnull() & original.notnull()
            if unmapped.any():
                vals = original[unmapped].unique()
                print(f"    WARNING: {col} has unmapped values: {vals}")

    return df


def encode_nominals(df):
    """One-hot encode nominal features."""
    df = df.copy()

    # MSSubClass: numeric codes that are actually categorical
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    cols_to_encode = [c for c in NOMINAL_FEATURES if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    tee = Tee(f"{BASE}/results/models/01_preprocessing.txt")
    sys.stdout = tee

    print("House Prices: Step 1 — Quick EDA + Preprocessing")
    print("=" * 60)
    print()

    # ---- Load ----
    train, test = load_data()
    print(f"Loaded: train={train.shape}, test={test.shape}")
    print(f"Features: {train.shape[1] - 2} (excluding Id and SalePrice)")
    print()

    # ---- EDA ----
    examine_target(train)
    analyze_missing(train, test)

    # ---- Prepare for processing ----
    target = train["SalePrice"].copy()
    train_ids = train["Id"].copy()
    test_ids = test["Id"].copy()

    train_feat = train.drop(columns=["Id", "SalePrice"])
    test_feat = test.drop(columns=["Id"])

    # Compute imputation values from TRAIN ONLY (no test leakage)
    train_medians = train_feat.select_dtypes(include=[np.number]).median()
    train_modes = {}
    for col in train_feat.select_dtypes(include=["object"]).columns:
        mode = train_feat[col].mode()
        train_modes[col] = mode.iloc[0] if len(mode) > 0 else "None"

    # ---- Process ----
    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    # Combine train+test for consistent column handling during encoding
    # (no information leakage: imputation values come from train only,
    #  one-hot encoding just creates indicator columns, no target info used)
    n_train = len(train_feat)
    combined = pd.concat([train_feat, test_feat], ignore_index=True)

    # Step 1: Handle missing values
    combined = handle_missing(combined, train_medians, train_modes)
    remaining_na = combined.isnull().sum().sum()
    print(f"\n  Missing values handled:")
    print(f"    {len(NA_MEANS_NONE)} categorical cols: NaN -> 'None' (feature absent)")
    print(f"    {len(NA_MEANS_ZERO)} numeric cols: NaN -> 0 (feature absent)")
    print(f"    GarageYrBlt: NaN -> 0 (no garage)")
    print(f"    Remaining numeric: -> train median")
    print(f"    Remaining categorical: -> train mode")
    print(f"    NaN remaining: {remaining_na}")

    # Step 2: Encode ordinals
    combined = encode_ordinals(combined)
    n_ordinal = len(QUALITY_COLS) + len(ORDINAL_MAPS)
    print(f"\n  Ordinal encoding: {n_ordinal} features -> numeric")
    print(f"    Quality scale (10 cols): Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0")
    print(f"    Custom scales (12 cols): BsmtExposure, Functional, Fence, etc.")

    # Show correlations with SalePrice (after ordinal encoding, before one-hot)
    print()
    train_for_corr = combined.iloc[:n_train].copy()
    train_for_corr["SalePrice"] = target.values
    top_correlations(train_for_corr)

    # Step 3: Encode nominals (one-hot)
    pre_cols = combined.shape[1]
    combined = encode_nominals(combined)
    n_dummies = combined.shape[1] - pre_cols + len(NOMINAL_FEATURES)
    print(f"  One-hot encoding:")
    print(f"    {len(NOMINAL_FEATURES)} nominal features -> {n_dummies} dummy columns")
    print(f"    drop_first=False (Ridge/Lasso handle collinearity; keeps all info)")

    # ---- Split back ----
    train_feat = combined.iloc[:n_train].reset_index(drop=True)
    test_feat = combined.iloc[n_train:].reset_index(drop=True)

    # ---- Final checks ----
    final_na_train = train_feat.isnull().sum().sum()
    final_na_test = test_feat.isnull().sum().sum()

    print(f"\n  Final shapes:")
    print(f"    Train features: {train_feat.shape}")
    print(f"    Test features:  {test_feat.shape}")
    print(f"    NaN (train): {final_na_train}")
    print(f"    NaN (test):  {final_na_test}")

    if final_na_train > 0 or final_na_test > 0:
        print(f"\n  *** WARNING: NaN remaining ***")
        for name, df in [("Train", train_feat), ("Test", test_feat)]:
            na_cols = df.isnull().sum()
            na_cols = na_cols[na_cols > 0]
            if len(na_cols) > 0:
                for col, count in na_cols.items():
                    print(f"    {name} - {col}: {count}")

    # ---- Save ----
    train_out = train_feat.copy()
    train_out.insert(0, "Id", train_ids.values)
    train_out["SalePrice"] = target.values

    test_out = test_feat.copy()
    test_out.insert(0, "Id", test_ids.values)

    train_out.to_csv(f"{BASE}/data/train_processed.csv", index=False)
    test_out.to_csv(f"{BASE}/data/test_processed.csv", index=False)

    print(f"\n  Saved:")
    print(f"    data/train_processed.csv {train_out.shape}")
    print(f"    data/test_processed.csv  {test_out.shape}")

    # ---- Feature type summary ----
    print(f"\n{'=' * 60}")
    print("FEATURE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total features: {train_feat.shape[1]}")
    print(f"  Original numeric:  ~35 (LotArea, GrLivArea, YearBuilt, ...)")
    print(f"  Ordinal -> numeric: {n_ordinal} (quality ratings, conditions)")
    print(f"  One-hot dummies:    {n_dummies} (neighborhood, exterior, ...)")
    print()

    tee.close()
    return train_out, test_out


if __name__ == "__main__":
    main()
