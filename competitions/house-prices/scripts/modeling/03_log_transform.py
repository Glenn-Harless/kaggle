"""
House Prices: Step 3 — The Log Transform Experiment

Compare two identical pipelines that differ ONLY in the target:
  Model A: StandardScaler + Ridge → raw SalePrice
  Model B: StandardScaler + Ridge → log1p(SalePrice), expm1 back

Both use the same features (238) and alpha (1.0).
After CV comparison, analyze residuals to see WHY one is better.

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/03_log_transform.txt
         submissions/ridge_log_target.csv
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
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


def oof_predictions(model, X, y, n_splits=5, random_state=42):
    """
    Single pass of KFold CV, returning per-fold RMSLE scores and
    out-of-fold predictions (each sample predicted when it was in
    the validation set, not the training set).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.full(len(y), np.nan)
    fold_scores = []

    for train_idx, val_idx in kf.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        preds = np.clip(preds, 0, None)
        oof_preds[val_idx] = preds
        fold_scores.append(rmsle(y.iloc[val_idx], preds))

    return np.array(fold_scores), oof_preds


def residual_analysis(y_true, y_pred, label):
    """
    Analyze residual patterns, focusing on heteroscedasticity.

    Key diagnostic: if absolute errors grow with price but percentage
    errors stay flat, the model has heteroscedastic residuals.
    Log-transforming the target should fix this.
    """
    residuals = y_true.values - y_pred
    abs_residuals = np.abs(residuals)
    pct_errors = abs_residuals / y_true.values * 100

    print(f"\n  Residual Analysis — {label}:")
    print(f"    Mean residual:       ${residuals.mean():>10,.0f}  (bias)")
    print(f"    Std of residuals:    ${residuals.std():>10,.0f}")
    print(f"    Mean |error|:        ${abs_residuals.mean():>10,.0f}")
    print(f"    Mean % error:        {pct_errors.mean():>9.1f}%")
    print(f"    Residual skewness:   {pd.Series(residuals).skew():>9.3f}")

    # Heteroscedasticity check: errors by price quintile
    # If |errors| grow with price → heteroscedastic (bad for RMSLE)
    # If % errors are flat across quintiles → homoscedastic in log space (good)
    quintiles = pd.qcut(y_true, 5, labels=[
        "Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"
    ])
    print(f"\n    Heteroscedasticity check (errors by price quintile):")
    print(f"    {'Quintile':15s} {'Price Range':>20s} {'Mean |Error|':>15s} {'Mean % Error':>13s}")
    print(f"    {'-' * 68}")
    for q in ["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 priciest"]:
        mask = quintiles == q
        q_prices = y_true[mask]
        mae = abs_residuals[mask].mean()
        mpe = pct_errors[mask].mean()
        price_range = f"${q_prices.min():,.0f}-${q_prices.max():,.0f}"
        print(f"    {q:15s} {price_range:>20s} ${mae:>13,.0f} {mpe:>12.1f}%")

    # Summary: is it heteroscedastic?
    q1_mask = quintiles == "Q1 cheapest"
    q5_mask = quintiles == "Q5 priciest"
    q1_mae = abs_residuals[q1_mask].mean()
    q5_mae = abs_residuals[q5_mask].mean()
    q1_mpe = pct_errors[q1_mask].mean()
    q5_mpe = pct_errors[q5_mask].mean()

    print(f"\n    Absolute error ratio (Q5/Q1):   {q5_mae / q1_mae:.1f}x")
    print(f"    Percentage error ratio (Q5/Q1): {q5_mpe / q1_mpe:.1f}x")
    if q5_mae / q1_mae > 2.0:
        print(f"    → Heteroscedastic: expensive houses have much larger absolute errors")
    else:
        print(f"    → Relatively homoscedastic: errors are proportional to price")

    # Worst predictions
    worst_idx = np.argsort(abs_residuals)[-5:][::-1]
    print(f"\n    5 worst predictions:")
    print(f"    {'Actual':>12s} {'Predicted':>12s} {'Error':>12s} {'% Off':>8s}")
    print(f"    {'-' * 48}")
    for idx in worst_idx:
        actual = y_true.iloc[idx]
        pred = y_pred[idx]
        err = residuals[idx]
        pct = abs(err) / actual * 100
        print(f"    ${actual:>10,.0f} ${pred:>10,.0f} ${err:>+10,.0f} {pct:>7.1f}%")


def main():
    tee = Tee(f"{BASE}/results/models/03_log_transform.txt")
    sys.stdout = tee

    print("House Prices: Step 3 — The Log Transform Experiment")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    print(f"Train: {X_train.shape}")
    print(f"Target: ${y.min():,.0f} - ${y.max():,.0f} "
          f"(median ${y.median():,.0f}, skew {y.skew():.3f})")
    print(f"Log target: {np.log1p(y).min():.3f} - {np.log1p(y).max():.3f} "
          f"(skew {np.log1p(y).skew():.3f})")
    print()

    # ============================================================
    # MODEL A: Raw SalePrice (with StandardScaler)
    # ============================================================
    print("=" * 60)
    print("MODEL A: StandardScaler + Ridge → raw SalePrice")
    print("  Loss: minimize (predicted_price - actual_price)^2")
    print("  This optimizes RMSE on dollar amounts, NOT RMSLE.")
    print("=" * 60)

    model_a = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

    # Full repeated CV for robust RMSLE estimate
    scores_a = regression_cv(model_a, X_train, y)
    report_regression_cv(scores_a, "Model A")

    # Single-pass OOF predictions for residual analysis
    _, oof_a = oof_predictions(model_a, X_train, y)
    residual_analysis(y, oof_a, "Model A (OOF predictions)")
    print()

    # ============================================================
    # MODEL B: log1p(SalePrice) (with StandardScaler)
    # ============================================================
    print("=" * 60)
    print("MODEL B: StandardScaler + Ridge → log1p(SalePrice)")
    print("  Loss: minimize (log(predicted) - log(actual))^2")
    print("  This directly optimizes RMSLE (the competition metric).")
    print("=" * 60)

    # TransformedTargetRegressor handles log1p on fit, expm1 on predict
    model_b = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    scores_b = regression_cv(model_b, X_train, y)
    report_regression_cv(scores_b, "Model B")

    # Single-pass OOF predictions for residual analysis
    _, oof_b = oof_predictions(model_b, X_train, y)
    residual_analysis(y, oof_b, "Model B (OOF predictions)")
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n  {'':25s} {'Mean RMSLE':>12s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-' * 70}")

    # Include Step 2 Ridge baseline for reference
    print(f"  {'Step 2 Ridge (no scale)':25s} {'0.23643':>12s} {'0.16625':>10s} "
          f"{'0.13814':>10s} {'0.68719':>10s}")
    print(f"  {'Model A (scaled, raw y)':25s} {scores_a.mean():12.5f} {scores_a.std():10.5f} "
          f"{scores_a.min():10.5f} {scores_a.max():10.5f}")
    print(f"  {'Model B (scaled, log y)':25s} {scores_b.mean():12.5f} {scores_b.std():10.5f} "
          f"{scores_b.min():10.5f} {scores_b.max():10.5f}")

    # Improvements
    step2_rmsle = 0.23643
    print(f"\n  Scaling effect (Step 2 → Model A): "
          f"{(step2_rmsle - scores_a.mean()) / step2_rmsle * 100:+.1f}%")
    print(f"  Log transform effect (A → B):      "
          f"{(scores_a.mean() - scores_b.mean()) / scores_a.mean() * 100:+.1f}%")
    print(f"  Combined effect (Step 2 → Model B): "
          f"{(step2_rmsle - scores_b.mean()) / step2_rmsle * 100:+.1f}%")

    # Stability comparison
    print(f"\n  CV stability (lower std = more stable):")
    print(f"    Step 2: std = 0.16625 (range 0.138-0.687)")
    print(f"    Model A: std = {scores_a.std():.5f} "
          f"(range {scores_a.min():.3f}-{scores_a.max():.3f})")
    print(f"    Model B: std = {scores_b.std():.5f} "
          f"(range {scores_b.min():.3f}-{scores_b.max():.3f})")

    # Verify RMSE in log space = RMSLE
    print(f"\n  Verification: RMSE(log space) = RMSLE(original scale)")
    model_b.fit(X_train, y)
    log_preds = model_b.regressor_.predict(X_train)
    log_actual = np.log1p(y).values
    rmse_log = np.sqrt(np.mean((log_preds - log_actual) ** 2))
    rmsle_orig = rmsle(y, model_b.predict(X_train))
    print(f"    RMSE(log_pred, log_actual) = {rmse_log:.6f}")
    print(f"    RMSLE(pred, actual)        = {rmsle_orig:.6f}")
    print()

    # ============================================================
    # SUBMIT MODEL B
    # ============================================================
    print("=" * 60)
    print("SUBMISSION: Model B (scaled Ridge, log target)")
    print("=" * 60)

    preds_test = model_b.predict(X_test)
    preds_test = np.clip(preds_test, 0, None)

    n_nonpositive = (preds_test <= 0).sum()
    if n_nonpositive > 0:
        print(f"  WARNING: {n_nonpositive} non-positive predictions clipped")
    else:
        print(f"  All predictions positive (log transform prevents negatives)")

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": preds_test})
    sub.to_csv(f"{BASE}/submissions/ridge_log_target.csv", index=False)
    print(f"  Saved: submissions/ridge_log_target.csv")
    print(f"  Prediction range: ${preds_test.min():,.0f} - ${preds_test.max():,.0f}")
    print()

    tee.close()


if __name__ == "__main__":
    main()
