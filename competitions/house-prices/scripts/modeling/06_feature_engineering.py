"""
House Prices: Step 6 — Residual-Guided Feature Engineering

Each feature is motivated by a specific residual pattern from Step 5:
  1. TotalSF — model struggles with extreme sizes; combined SF gives full picture
  2. HouseAge — age-related residual patterns
  3. RemodelAge — recently remodeled houses had higher errors
  4. TotalBath — 4 bathroom columns → 1 combined feature
  5. QualxSF — OverallQual=10 disaster suggests quality*size interaction
  6. Outlier removal — 2 extreme houses distort the model

Method: test each individually, then combine winners.

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/06_feature_engineering.txt
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

from shared.evaluate import Tee, rmsle, regression_cv, report_regression_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test = pd.read_csv(f"{BASE}/data/test_processed.csv")
    y = train["SalePrice"]
    X_train = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])
    test_ids = test["Id"]
    return X_train, y, X_test, test_ids


def add_features(X, features):
    """
    Add engineered features to the dataframe.
    Each feature is motivated by a residual pattern from Step 5.
    """
    X = X.copy()

    if 'TotalSF' in features:
        # Total living space: basement + above ground
        # Residual pattern: model errors grow at extremes of individual
        # size features. A combined feature captures total usable space.
        X['TotalSF'] = X['TotalBsmtSF'] + X['GrLivArea']

    if 'HouseAge' in features:
        # Age at time of sale
        # Residual pattern: errors vary by YearBuilt quintile,
        # but age relative to sale year is more meaningful
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']

    if 'RemodelAge' in features:
        # Years since last remodel
        # Residual pattern: recently remodeled houses (age 0-3) had
        # 13.7% error vs 6.8% for age 3-10
        X['RemodelAge'] = X['YrSold'] - X['YearRemodAdd']

    if 'TotalBath' in features:
        # Combined bathroom count (half baths count as 0.5)
        # Currently split across 4 columns — harder for linear model
        # to learn "more bathrooms = higher price" from separate features
        X['TotalBath'] = (X['FullBath'] + 0.5 * X['HalfBath'] +
                          X['BsmtFullBath'] + 0.5 * X['BsmtHalfBath'])

    if 'QualxSF' in features:
        # Quality x Size interaction
        # Residual pattern: OverallQual=10 houses had 113.7% error.
        # The model treats quality and size independently, but a
        # high-quality large house is worth more than their sum suggests.
        X['QualxSF'] = X['OverallQual'] * X['GrLivArea']

    return X


def make_model():
    """Best model from Step 4: ElasticNet with tuned hyperparameters."""
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )


def run_experiment(name, X_train, y, description=""):
    """Run repeated CV and return scores."""
    model = make_model()
    scores = regression_cv(model, X_train, y)
    return scores


def main():
    tee = Tee(f"{BASE}/results/models/06_feature_engineering.txt")
    sys.stdout = tee

    print("House Prices: Step 6 — Residual-Guided Feature Engineering")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    print(f"Train: {X_train.shape}")
    print(f"Model: ElasticNet (alpha=0.02848, l1_ratio=0.1, log target)")
    print(f"CV: 5x5 repeated KFold")
    print()

    results = {}

    # ============================================================
    # BASELINE
    # ============================================================
    print("=" * 60)
    print("BASELINE: No changes (Step 4 best)")
    print("=" * 60)
    scores = run_experiment("Baseline", X_train, y)
    report_regression_cv(scores, "Baseline")
    results["Baseline"] = scores.mean()
    baseline_rmsle = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 1: Remove outliers
    # ============================================================
    print("=" * 60)
    print("EXP 1: Remove outliers (GrLivArea > 4000)")
    print("  Two houses with 4,676 and 5,642 sq ft were massively")
    print("  overpredicted. Both are Partial sales in Edwards.")
    print("=" * 60)
    outlier_mask = X_train['GrLivArea'] <= 4000
    n_removed = (~outlier_mask).sum()
    print(f"  Removing {n_removed} houses")
    scores = run_experiment("No outliers", X_train[outlier_mask], y[outlier_mask])
    report_regression_cv(scores, "No outliers")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["Remove outliers"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 2: +TotalSF
    # ============================================================
    print("=" * 60)
    print("EXP 2: +TotalSF (TotalBsmtSF + GrLivArea)")
    print("  Combined living space captures total usable area.")
    print("=" * 60)
    X_exp = add_features(X_train, ['TotalSF'])
    scores = run_experiment("+TotalSF", X_exp, y)
    report_regression_cv(scores, "+TotalSF")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["+TotalSF"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 3: +HouseAge + RemodelAge
    # ============================================================
    print("=" * 60)
    print("EXP 3: +HouseAge + RemodelAge")
    print("  Age at sale and years since remodel.")
    print("=" * 60)
    X_exp = add_features(X_train, ['HouseAge', 'RemodelAge'])
    scores = run_experiment("+Age features", X_exp, y)
    report_regression_cv(scores, "+Age features")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["+HouseAge+RemodelAge"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 4: +TotalBath
    # ============================================================
    print("=" * 60)
    print("EXP 4: +TotalBath")
    print("  Combined bathroom count (half baths = 0.5).")
    print("=" * 60)
    X_exp = add_features(X_train, ['TotalBath'])
    scores = run_experiment("+TotalBath", X_exp, y)
    report_regression_cv(scores, "+TotalBath")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["+TotalBath"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 5: +QualxSF
    # ============================================================
    print("=" * 60)
    print("EXP 5: +QualxSF (OverallQual * GrLivArea)")
    print("  Quality-size interaction. A quality-10 mansion is worth")
    print("  more than quality and size independently suggest.")
    print("=" * 60)
    X_exp = add_features(X_train, ['QualxSF'])
    scores = run_experiment("+QualxSF", X_exp, y)
    report_regression_cv(scores, "+QualxSF")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["+QualxSF"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 6: All features combined
    # ============================================================
    print("=" * 60)
    print("EXP 6: All engineered features combined")
    print("=" * 60)
    all_features = ['TotalSF', 'HouseAge', 'RemodelAge', 'TotalBath', 'QualxSF']
    X_exp = add_features(X_train, all_features)
    print(f"  Features: {X_train.shape[1]} -> {X_exp.shape[1]} "
          f"(+{X_exp.shape[1] - X_train.shape[1]})")
    scores = run_experiment("All features", X_exp, y)
    report_regression_cv(scores, "All features")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["All engineered"] = scores.mean()
    print()

    # ============================================================
    # EXPERIMENT 7: All features + outlier removal
    # ============================================================
    print("=" * 60)
    print("EXP 7: All features + outlier removal")
    print("=" * 60)
    X_exp = add_features(X_train, all_features)
    X_exp_clean = X_exp[outlier_mask]
    y_clean = y[outlier_mask]
    print(f"  Features: {X_exp.shape[1]}, Samples: {len(y_clean)} "
          f"(removed {n_removed})")
    scores = run_experiment("All + no outliers", X_exp_clean, y_clean)
    report_regression_cv(scores, "All + no outliers")
    delta = (scores.mean() - baseline_rmsle) / baseline_rmsle * 100
    print(f"  Delta vs baseline: {delta:+.2f}%")
    results["All + no outliers"] = scores.mean()
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON (lower RMSLE is better)")
    print("=" * 60)
    print(f"\n  {'Experiment':30s} {'RMSLE':>10s} {'Delta':>10s}")
    print(f"  {'-' * 55}")
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        delta = (score - baseline_rmsle) / baseline_rmsle * 100
        marker = " <-- best" if score == min(results.values()) else ""
        delta_str = f"{delta:+.2f}%" if name != "Baseline" else "---"
        print(f"  {name:30s} {score:10.5f} {delta_str:>10s}{marker}")
    print()

    best_name = min(results, key=results.get)
    best_score = results[best_name]
    total_improvement = (baseline_rmsle - best_score) / baseline_rmsle * 100
    print(f"  Best: {best_name}")
    print(f"  Improvement over baseline: {total_improvement:.2f}%")
    print()

    tee.close()


if __name__ == "__main__":
    main()
