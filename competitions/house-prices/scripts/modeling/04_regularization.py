"""
House Prices: Step 4 — Regularization Sweep

Ridge vs Lasso vs ElasticNet on log1p(SalePrice) with StandardScaler.
Sweep alpha values for each, compare CV scores and feature selection.

All models use the same setup from Step 3:
  StandardScaler → model → log1p target

Key questions:
  1. What's the optimal alpha for each?
  2. How many features does Lasso kill?
  3. Which features survive Lasso's selection?

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/04_regularization.txt
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold

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


def alpha_sweep(model_class, X, y_log, alphas, n_splits=5, random_state=42,
                **model_kwargs):
    """
    Sweep alpha values with KFold CV.
    Trains in log space, evaluates RMSE in log space (= RMSLE).
    Also counts non-zero coefficients per alpha (for Lasso).
    """
    results = []

    for alpha in alphas:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_scores = []
        fold_nonzero = []

        for train_idx, val_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X.iloc[train_idx])
            X_val = scaler.transform(X.iloc[val_idx])

            model = model_class(alpha=alpha, max_iter=10000, **model_kwargs)
            model.fit(X_tr, y_log.iloc[train_idx])

            log_preds = model.predict(X_val)
            rmse_log = np.sqrt(np.mean(
                (log_preds - y_log.iloc[val_idx].values) ** 2
            ))
            fold_scores.append(rmse_log)
            fold_nonzero.append(np.sum(np.abs(model.coef_) > 1e-10))

        results.append({
            'alpha': alpha,
            'mean_rmsle': np.mean(fold_scores),
            'std_rmsle': np.std(fold_scores),
            'mean_nonzero': np.mean(fold_nonzero),
        })

    return pd.DataFrame(results)


def elasticnet_sweep(X, y_log, alphas, l1_ratios, n_splits=5, random_state=42):
    """Sweep alpha x l1_ratio grid for ElasticNet."""
    results = []

    for l1_ratio in l1_ratios:
        for alpha in alphas:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            fold_scores = []
            fold_nonzero = []

            for train_idx, val_idx in kf.split(X):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X.iloc[train_idx])
                X_val = scaler.transform(X.iloc[val_idx])

                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                                   max_iter=10000)
                model.fit(X_tr, y_log.iloc[train_idx])

                log_preds = model.predict(X_val)
                rmse_log = np.sqrt(np.mean(
                    (log_preds - y_log.iloc[val_idx].values) ** 2
                ))
                fold_scores.append(rmse_log)
                fold_nonzero.append(np.sum(np.abs(model.coef_) > 1e-10))

            results.append({
                'alpha': alpha,
                'l1_ratio': l1_ratio,
                'mean_rmsle': np.mean(fold_scores),
                'std_rmsle': np.std(fold_scores),
                'mean_nonzero': np.mean(fold_nonzero),
            })

    return pd.DataFrame(results)


def feature_analysis(model, feature_names, label, top_n=20):
    """Analyze which features survived regularization."""
    coefs = pd.Series(model.coef_, index=feature_names)
    nonzero = coefs[coefs.abs() > 1e-10].sort_values(key=abs, ascending=False)
    zero = coefs[coefs.abs() <= 1e-10]

    print(f"\n  Feature Analysis — {label}:")
    print(f"    Non-zero: {len(nonzero)} / {len(coefs)}  "
          f"({len(zero)} zeroed out)")

    print(f"\n    Top {min(top_n, len(nonzero))} features by |coefficient|:")
    print(f"    {'Feature':40s} {'Coef':>10s}")
    print(f"    {'-' * 55}")
    for feat, coef in nonzero.head(top_n).items():
        print(f"    {feat:40s} {coef:>+10.5f}")

    return nonzero, zero


def main():
    tee = Tee(f"{BASE}/results/models/04_regularization.txt")
    sys.stdout = tee

    print("House Prices: Step 4 — Regularization Sweep")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    y_log = np.log1p(y)
    n_features = X_train.shape[1]
    print(f"Train: {X_train.shape}")
    print(f"All models: StandardScaler + log1p target")
    print(f"Alpha sweep: 5-fold CV, RMSE in log space (= RMSLE)")
    print()

    # ============================================================
    # RIDGE: Alpha Sweep
    # ============================================================
    print("=" * 60)
    print("RIDGE (L2): Alpha Sweep")
    print("  Shrinks all coefficients toward zero, never to exactly zero.")
    print("  Expect: all 238 features survive at every alpha.")
    print("=" * 60)

    ridge_alphas = np.logspace(-1, 3, 20)
    ridge_results = alpha_sweep(Ridge, X_train, y_log, ridge_alphas)

    print(f"\n  {'Alpha':>12s} {'Mean RMSLE':>12s} {'Std':>10s} {'Non-zero':>10s}")
    print(f"  {'-' * 48}")
    best_ridge_rmsle = ridge_results['mean_rmsle'].min()
    for _, row in ridge_results.iterrows():
        marker = " <--" if row['mean_rmsle'] == best_ridge_rmsle else ""
        print(f"  {row['alpha']:>12.3f} {row['mean_rmsle']:>12.5f} "
              f"{row['std_rmsle']:>10.5f} {int(row['mean_nonzero']):>10d}{marker}")

    best_ridge = ridge_results.loc[ridge_results['mean_rmsle'].idxmin()]
    print(f"\n  Best: alpha={best_ridge['alpha']:.3f}, "
          f"RMSLE={best_ridge['mean_rmsle']:.5f}")
    print()

    # ============================================================
    # LASSO: Alpha Sweep
    # ============================================================
    print("=" * 60)
    print("LASSO (L1): Alpha Sweep")
    print("  Drives useless coefficients to exactly zero.")
    print("  Watch the non-zero count drop as alpha increases.")
    print("=" * 60)

    lasso_alphas = np.logspace(-5, -1, 20)
    lasso_results = alpha_sweep(Lasso, X_train, y_log, lasso_alphas)

    print(f"\n  {'Alpha':>12s} {'Mean RMSLE':>12s} {'Std':>10s} {'Non-zero':>10s}")
    print(f"  {'-' * 48}")
    best_lasso_rmsle = lasso_results['mean_rmsle'].min()
    for _, row in lasso_results.iterrows():
        marker = " <--" if row['mean_rmsle'] == best_lasso_rmsle else ""
        print(f"  {row['alpha']:>12.7f} {row['mean_rmsle']:>12.5f} "
              f"{row['std_rmsle']:>10.5f} {int(row['mean_nonzero']):>10d}{marker}")

    best_lasso = lasso_results.loc[lasso_results['mean_rmsle'].idxmin()]
    print(f"\n  Best: alpha={best_lasso['alpha']:.7f}, "
          f"RMSLE={best_lasso['mean_rmsle']:.5f}, "
          f"non-zero={int(best_lasso['mean_nonzero'])}/{n_features}")
    print()

    # ============================================================
    # ELASTICNET: Alpha x l1_ratio Sweep
    # ============================================================
    print("=" * 60)
    print("ELASTICNET (L1+L2): Alpha x l1_ratio Sweep")
    print("  l1_ratio=1.0 is pure Lasso, l1_ratio=0 is pure Ridge.")
    print("  Handles correlated features better than Lasso alone.")
    print("=" * 60)

    en_alphas = np.logspace(-4, -1, 12)
    en_l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    en_results = elasticnet_sweep(X_train, y_log, en_alphas, en_l1_ratios)

    print(f"\n  Best alpha per l1_ratio:")
    print(f"  {'l1_ratio':>10s} {'Alpha':>12s} {'Mean RMSLE':>12s} "
          f"{'Std':>10s} {'Non-zero':>10s}")
    print(f"  {'-' * 58}")
    for ratio in en_l1_ratios:
        subset = en_results[en_results['l1_ratio'] == ratio]
        best = subset.loc[subset['mean_rmsle'].idxmin()]
        print(f"  {ratio:>10.1f} {best['alpha']:>12.6f} "
              f"{best['mean_rmsle']:>12.5f} {best['std_rmsle']:>10.5f} "
              f"{int(best['mean_nonzero']):>10d}")

    best_en = en_results.loc[en_results['mean_rmsle'].idxmin()]
    print(f"\n  Best overall: alpha={best_en['alpha']:.6f}, "
          f"l1_ratio={best_en['l1_ratio']:.1f}, "
          f"RMSLE={best_en['mean_rmsle']:.5f}, "
          f"non-zero={int(best_en['mean_nonzero'])}/{n_features}")
    print()

    # ============================================================
    # FEATURE ANALYSIS
    # ============================================================
    print("=" * 60)
    print("FEATURE ANALYSIS: What does each regularizer keep?")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Ridge — keeps everything
    ridge_model = Ridge(alpha=best_ridge['alpha'])
    ridge_model.fit(X_scaled, y_log)
    ridge_nonzero, _ = feature_analysis(
        ridge_model, X_train.columns,
        f"Ridge (alpha={best_ridge['alpha']:.3f})")

    # Lasso — feature selection
    lasso_model = Lasso(alpha=best_lasso['alpha'], max_iter=10000)
    lasso_model.fit(X_scaled, y_log)
    lasso_nonzero, lasso_zero = feature_analysis(
        lasso_model, X_train.columns,
        f"Lasso (alpha={best_lasso['alpha']:.7f})")

    if len(lasso_zero) > 0:
        print(f"\n    Lasso killed {len(lasso_zero)} features. Sample:")
        killed = sorted(lasso_zero.index.tolist())
        # Show first 40
        for i in range(0, min(len(killed), 40), 4):
            chunk = killed[i:i+4]
            print(f"      {', '.join(chunk)}")
        if len(killed) > 40:
            print(f"      ... and {len(killed) - 40} more")

    # ElasticNet
    en_model = ElasticNet(alpha=best_en['alpha'],
                          l1_ratio=best_en['l1_ratio'],
                          max_iter=10000)
    en_model.fit(X_scaled, y_log)
    en_nonzero, en_zero = feature_analysis(
        en_model, X_train.columns,
        f"ElasticNet (alpha={best_en['alpha']:.6f}, "
        f"l1_ratio={best_en['l1_ratio']:.1f})")

    # Compare: features all three agree on
    if len(lasso_nonzero) > 0 and len(en_nonzero) > 0:
        lasso_set = set(lasso_nonzero.index)
        en_set = set(en_nonzero.index)
        both = lasso_set & en_set
        lasso_only = lasso_set - en_set
        en_only = en_set - lasso_set

        print(f"\n  Feature overlap:")
        print(f"    Both Lasso & ElasticNet keep: {len(both)}")
        print(f"    Only Lasso keeps:             {len(lasso_only)}")
        print(f"    Only ElasticNet keeps:        {len(en_only)}")
    print()

    # ============================================================
    # FINAL COMPARISON (repeated CV)
    # ============================================================
    print("=" * 60)
    print("FINAL COMPARISON (5x5 repeated CV on best alphas)")
    print("=" * 60)

    models = {
        f"Ridge (a={best_ridge['alpha']:.1f})": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=best_ridge['alpha']))
            ]),
            func=np.log1p, inverse_func=np.expm1,
        ),
        f"Lasso (a={best_lasso['alpha']:.5f})": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=best_lasso['alpha'], max_iter=10000))
            ]),
            func=np.log1p, inverse_func=np.expm1,
        ),
        f"ElasticNet (a={best_en['alpha']:.4f})": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(alpha=best_en['alpha'],
                                     l1_ratio=best_en['l1_ratio'],
                                     max_iter=10000))
            ]),
            func=np.log1p, inverse_func=np.expm1,
        ),
    }

    print(f"\n  {'Model':35s} {'Mean RMSLE':>12s} {'Std':>10s}")
    print(f"  {'-' * 60}")
    print(f"  {'Step 3 baseline (Ridge a=1.0)':35s} {'0.15104':>12s} {'0.03396':>10s}")

    best_name, best_score = None, 999
    for name, model in models.items():
        scores = regression_cv(model, X_train, y)
        report_regression_cv(scores, name)
        print(f"  {name:35s} {scores.mean():12.5f} {scores.std():10.5f}")
        if scores.mean() < best_score:
            best_score = scores.mean()
            best_name = name

    print(f"\n  Best: {best_name} (RMSLE={best_score:.5f})")
    print()

    tee.close()


if __name__ == "__main__":
    main()
