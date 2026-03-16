"""
House Prices: Step 2 — Baselines

Three baselines to establish reference points:
1. Median baseline: predict median price for everything (floor score)
2. 2-feature model: LinearRegression on GrLivArea + OverallQual
3. All-features Ridge: Ridge on all processed features

Each gets: 5-fold repeated CV RMSLE and Kaggle submission.

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/02_baselines.txt
         submissions/median_baseline.csv
         submissions/two_feature.csv
         submissions/ridge_baseline.csv
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import BaseEstimator, RegressorMixin

from shared.evaluate import Tee, rmsle, regression_cv, report_regression_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predict the training set median for every input."""
    def fit(self, X, y):
        self.median_ = np.median(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test = pd.read_csv(f"{BASE}/data/test_processed.csv")

    y = train["SalePrice"]
    train_ids = train["Id"]
    test_ids = test["Id"]

    X_train = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])

    return X_train, y, X_test, train_ids, test_ids


def submit(preds, test_ids, filename):
    """Create a Kaggle submission CSV."""
    preds = np.clip(preds, 0, None)  # No negative prices
    sub = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    sub.to_csv(f"{BASE}/submissions/{filename}", index=False)
    print(f"  Saved: submissions/{filename}")


def main():
    tee = Tee(f"{BASE}/results/models/02_baselines.txt")
    sys.stdout = tee

    print("House Prices: Step 2 — Baselines")
    print("=" * 60)
    print()

    X_train, y, X_test, train_ids, test_ids = load_processed()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target: ${y.min():,.0f} - ${y.max():,.0f} "
          f"(median ${y.median():,.0f})")
    print()

    results = {}

    # ============================================================
    # BASELINE 1: Median
    # ============================================================
    print("=" * 60)
    print("BASELINE 1: Median")
    print("  Predict training median for every house.")
    print("  This is the absolute floor — any real model must beat this.")
    print("=" * 60)

    model = MedianBaseline()
    scores = regression_cv(model, X_train, y)
    report_regression_cv(scores, "Median")
    results["Median"] = scores

    model.fit(X_train, y)
    print(f"\n  Training median: ${model.median_:,.0f}")

    preds = model.predict(X_test)
    submit(preds, test_ids, "median_baseline.csv")
    print()

    # ============================================================
    # BASELINE 2: 2-Feature Linear Regression
    # ============================================================
    print("=" * 60)
    print("BASELINE 2: LinearRegression (GrLivArea + OverallQual)")
    print("  The two features most correlated with SalePrice.")
    print("  No regularization — just OLS on 2 features.")
    print("=" * 60)

    feature_cols = ["GrLivArea", "OverallQual"]
    X_train_2f = X_train[feature_cols]
    X_test_2f = X_test[feature_cols]

    model = LinearRegression()
    scores = regression_cv(model, X_train_2f, y)
    report_regression_cv(scores, "2-Feature")
    results["2-Feature LR"] = scores

    # Show coefficients for interpretability
    model.fit(X_train_2f, y)
    print(f"\n  Coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"    {feat:20s}: ${coef:,.1f}")
    print(f"    {'Intercept':20s}: ${model.intercept_:,.1f}")

    preds = model.predict(X_test_2f)
    n_negative = (preds < 0).sum()
    if n_negative > 0:
        print(f"\n  WARNING: {n_negative} negative predictions (clipped to 0)")
    submit(np.clip(preds, 0, None), test_ids, "two_feature.csv")
    print()

    # ============================================================
    # BASELINE 3: Ridge (all features)
    # ============================================================
    print("=" * 60)
    print("BASELINE 3: Ridge Regression (all features, alpha=1.0)")
    print("  All 238 processed features with L2 regularization.")
    print("  alpha=1.0 is a default — Step 4 will tune it properly.")
    print("=" * 60)

    model = Ridge(alpha=1.0)
    scores = regression_cv(model, X_train, y)
    report_regression_cv(scores, "Ridge")
    results["Ridge (all)"] = scores

    # Show top coefficients
    model.fit(X_train, y)
    coef_df = pd.Series(model.coef_, index=X_train.columns)
    top_pos = coef_df.nlargest(10)
    top_neg = coef_df.nsmallest(5)
    print(f"\n  Top 10 positive coefficients:")
    for feat, coef in top_pos.items():
        print(f"    {feat:35s}: ${coef:>10,.1f}")
    print(f"\n  Top 5 negative coefficients:")
    for feat, coef in top_neg.items():
        print(f"    {feat:35s}: ${coef:>10,.1f}")

    preds = model.predict(X_test)
    n_negative = (preds < 0).sum()
    if n_negative > 0:
        print(f"\n  WARNING: {n_negative} negative predictions (clipped to 0)")
    submit(np.clip(preds, 0, None), test_ids, "ridge_baseline.csv")
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON (lower RMSLE is better)")
    print("=" * 60)
    print(f"  {'Model':25s} {'Mean RMSLE':>12s} {'Std':>10s}")
    print(f"  {'-' * 50}")
    for name, scores in results.items():
        print(f"  {name:25s} {scores.mean():12.5f} {scores.std():10.5f}")
    print()

    # Improvement over median
    median_rmsle = results["Median"].mean()
    for name, scores in results.items():
        if name == "Median":
            continue
        improvement = (median_rmsle - scores.mean()) / median_rmsle * 100
        print(f"  {name} improves over Median by {improvement:.1f}%")
    print()

    tee.close()


if __name__ == "__main__":
    main()
