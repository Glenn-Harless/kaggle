"""
House Prices: Step 7 — Ordinal Encoding Deep Dive

Test three encoding strategies for the 10 quality features
(ExterQual, KitchenQual, BsmtQual, etc.):

  1. One-hot: each level is an independent dummy column
  2. Ordinal: Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0 (current)
  3. Target-mean (OOF): replace with mean log(SalePrice) per level,
     computed out-of-fold to prevent leakage

Same model (ElasticNet), same outlier removal. Only the quality
feature encoding changes.

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/07_encoding_experiment.txt
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold

from shared.evaluate import Tee, rmsle, regression_cv, report_regression_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"

# The 10 quality features that share the Ex/Gd/TA/Fa/Po scale
QUALITY_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test = pd.read_csv(f"{BASE}/data/test_processed.csv")
    y = train["SalePrice"]
    X_train = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])
    test_ids = test["Id"]

    # Remove outliers (from Step 6)
    mask = X_train['GrLivArea'] <= 4000
    X_train = X_train[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return X_train, y, X_test, test_ids


def make_model():
    """Best model from Step 4."""
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )


def make_onehot(X, cols):
    """Replace ordinal quality columns with one-hot dummies."""
    X = X.copy()
    for col in cols:
        X[col] = X[col].astype(int).astype(str)
    X = pd.get_dummies(X, columns=cols, dtype=int)
    return X


def cv_with_target_encoding(model, X, y, encode_cols,
                             n_splits=5, n_repeats=5, random_state=42):
    """
    CV with proper out-of-fold target encoding.

    For each fold:
      1. Compute mean log1p(SalePrice) per quality level from TRAINING fold
      2. Map those means onto VALIDATION fold
      3. Fit model on encoded training, evaluate on encoded validation

    This prevents any target leakage — the validation fold never sees
    its own target values during encoding.
    """
    y_log = np.log1p(y)
    scores = []

    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True,
                    random_state=random_state + repeat)

        for train_idx, val_idx in kf.split(X):
            X_tr = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()

            # Target-encode each quality column
            for col in encode_cols:
                # Mean log target per level, from training fold only
                means = y_log.iloc[train_idx].groupby(X_tr[col]).mean()
                X_tr[col] = X_tr[col].map(means).astype(float)
                X_val[col] = X_val[col].map(means).astype(float)

                # Unseen levels in validation -> training global mean
                global_mean = y_log.iloc[train_idx].mean()
                X_tr[col] = X_tr[col].fillna(global_mean)
                X_val[col] = X_val[col].fillna(global_mean)

            m = clone(model)
            m.fit(X_tr, y.iloc[train_idx])
            preds = np.clip(m.predict(X_val), 0, None)
            scores.append(rmsle(y.iloc[val_idx], preds))

    return np.array(scores)


def show_spacing(X, y, cols):
    """
    Show actual mean SalePrice per quality level.
    This reveals whether equal spacing (ordinal) is appropriate.
    """
    y_log = np.log1p(y)
    level_names = {0: "None", 1: "Po", 2: "Fa", 3: "TA", 4: "Gd", 5: "Ex"}

    print(f"\n  Actual mean prices per quality level:")
    print(f"  (if spacing is unequal, target encoding captures this)")
    print(f"\n  {'Feature':15s}", end="")
    for level in range(6):
        print(f"  {level_names[level]:>8s}", end="")
    print()
    print(f"  {'-' * 70}")

    for col in cols:
        print(f"  {col:15s}", end="")
        for level in range(6):
            mask = X[col] == level
            if mask.sum() > 0:
                mean_price = y[mask].mean()
                print(f"  ${mean_price / 1000:>6.0f}K", end="")
            else:
                print(f"  {'---':>8s}", end="")
        print()

    # Show the same in log space (what the model actually sees)
    print(f"\n  {'Feature':15s}", end="")
    for level in range(6):
        print(f"  {level_names[level]:>8s}", end="")
    print("   Equal?")
    print(f"  {'-' * 78}")

    for col in cols:
        print(f"  {col:15s}", end="")
        values = []
        for level in range(6):
            mask = X[col] == level
            if mask.sum() > 0:
                mean_log = y_log[mask].mean()
                values.append(mean_log)
                print(f"  {mean_log:>8.3f}", end="")
            else:
                values.append(None)
                print(f"  {'---':>8s}", end="")

        # Check if spacing is roughly equal
        diffs = []
        for i in range(len(values) - 1):
            if values[i] is not None and values[i + 1] is not None:
                diffs.append(values[i + 1] - values[i])
        if len(diffs) >= 2:
            spacing_ratio = max(diffs) / (min(diffs) + 1e-10)
            equal = "~yes" if spacing_ratio < 2.0 else f"NO ({spacing_ratio:.1f}x)"
            print(f"   {equal}")
        else:
            print()


def main():
    tee = Tee(f"{BASE}/results/models/07_encoding_experiment.txt")
    sys.stdout = tee

    print("House Prices: Step 7 — Ordinal Encoding Deep Dive")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    print(f"Train: {X_train.shape} (outliers removed)")
    print(f"Quality features to test: {len(QUALITY_COLS)}")
    print(f"Model: ElasticNet (alpha=0.02848, l1_ratio=0.1, log target)")
    print()

    # ============================================================
    # SPACING ANALYSIS
    # ============================================================
    print("=" * 60)
    print("SPACING ANALYSIS: Is equal spacing (ordinal) appropriate?")
    print("=" * 60)
    show_spacing(X_train, y, QUALITY_COLS)
    print()

    # ============================================================
    # ENCODING 1: Ordinal (current)
    # ============================================================
    print("=" * 60)
    print("ENCODING 1: Ordinal (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)")
    print("  Assumes equal spacing. 10 features stay as 10 numeric columns.")
    print("=" * 60)

    model = make_model()
    scores_ordinal = regression_cv(model, X_train, y)
    report_regression_cv(scores_ordinal, "Ordinal")
    print(f"  Feature count: {X_train.shape[1]}")
    print()

    # ============================================================
    # ENCODING 2: One-hot
    # ============================================================
    print("=" * 60)
    print("ENCODING 2: One-hot (each level is an independent dummy)")
    print("  Makes no ordering assumption. 10 features -> ~40 dummies.")
    print("=" * 60)

    X_onehot = make_onehot(X_train, QUALITY_COLS)
    model = make_model()
    scores_onehot = regression_cv(model, X_onehot, y)
    report_regression_cv(scores_onehot, "One-hot")
    print(f"  Feature count: {X_onehot.shape[1]} "
          f"(was {X_train.shape[1]}, +{X_onehot.shape[1] - X_train.shape[1]})")
    print()

    # ============================================================
    # ENCODING 3: Target-mean (OOF)
    # ============================================================
    print("=" * 60)
    print("ENCODING 3: Target-mean (OOF)")
    print("  Replace each level with mean log(SalePrice) for that level.")
    print("  Computed out-of-fold to prevent target leakage.")
    print("  Captures non-equal spacing automatically.")
    print("=" * 60)

    model = make_model()
    scores_target = cv_with_target_encoding(model, X_train, y, QUALITY_COLS)
    report_regression_cv(scores_target, "Target-mean")
    print(f"  Feature count: {X_train.shape[1]} (same as ordinal)")
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    all_results = {
        "Ordinal (current)": scores_ordinal,
        "One-hot": scores_onehot,
        "Target-mean (OOF)": scores_target,
    }

    print(f"\n  {'Encoding':25s} {'Mean RMSLE':>12s} {'Std':>10s} {'Features':>10s}")
    print(f"  {'-' * 60}")
    feature_counts = {
        "Ordinal (current)": X_train.shape[1],
        "One-hot": X_onehot.shape[1],
        "Target-mean (OOF)": X_train.shape[1],
    }
    for name, scores in sorted(all_results.items(), key=lambda x: x[1].mean()):
        marker = " <--" if scores.mean() == min(s.mean() for s in all_results.values()) else ""
        print(f"  {name:25s} {scores.mean():12.5f} {scores.std():10.5f} "
              f"{feature_counts[name]:>10d}{marker}")

    best_name = min(all_results, key=lambda k: all_results[k].mean())
    worst_name = max(all_results, key=lambda k: all_results[k].mean())
    spread = all_results[worst_name].mean() - all_results[best_name].mean()
    print(f"\n  Spread (best to worst): {spread:.5f} RMSLE")
    print(f"  Best: {best_name}")
    print()

    tee.close()


if __name__ == "__main__":
    main()
