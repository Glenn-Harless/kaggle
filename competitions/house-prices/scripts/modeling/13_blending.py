"""
House Prices: Step 13 — OOF Blending

Blend ElasticNet and XGBoost using out-of-fold predictions.
Weights are fit on OOF predictions only — no Kaggle feedback.

Validation protocol:
  - Same KFold(5, shuffle=True, random_state=42) as Steps 10-11
  - OOF predictions: each sample predicted by a model that never
    saw it during training
  - Blend weights: constrained (non-negative, sum to 1) via
    scipy.optimize.minimize on OOF RMSLE

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/13_blending.txt
         submissions/blend_final.csv
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

from shared.evaluate import Tee, rmsle

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"

# Canonical CV splitter — same folds as Steps 10-11
CV_SPLITS = 5
CV_SEED = 42


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test = pd.read_csv(f"{BASE}/data/test_processed.csv")
    y = train["SalePrice"]
    X_train = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])
    test_ids = test["Id"]
    return X_train, y, X_test, test_ids


def collect_oof_elasticnet(X, y):
    """Collect OOF predictions from ElasticNet (Phase 2-4 best)."""
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.full(len(y), np.nan)

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = np.clip(m.predict(X.iloc[val_idx]), 0, None)

    return oof


def collect_oof_xgboost(X, y):
    """Collect OOF predictions from tuned XGBoost (Step 11 best)."""
    y_log = np.log1p(y)
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    oof = np.full(len(y), np.nan)

    # Best hyperparameters from Step 11
    params = {
        'max_depth': 3,
        'learning_rate': 0.08,
        'subsample': 0.7,
        'colsample_bytree': 0.5,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'early_stopping_rounds': 50,
        'n_estimators': 3000,
        'random_state': CV_SEED,
    }

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_val = X.iloc[val_idx]

        # Inner split for early stopping (deterministic per fold)
        fit_idx, es_idx = train_test_split(
            np.arange(len(X_tr)), test_size=0.2, random_state=fold_i
        )

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr.iloc[fit_idx], y_tr.iloc[fit_idx],
            eval_set=[(X_tr.iloc[es_idx], y_tr.iloc[es_idx])],
            verbose=False,
        )

        log_preds = model.predict(X_val)
        oof[val_idx] = np.clip(np.expm1(log_preds), 0, None)

    return oof


def find_optimal_weights(oof_dict, y):
    """
    Find blend weights that minimize RMSLE on OOF predictions.

    Constraints:
      - All weights >= 0 (no short-selling a model)
      - Weights sum to 1 (convex combination)
    """
    names = list(oof_dict.keys())
    oof_matrix = np.column_stack([oof_dict[n] for n in names])
    n_models = len(names)

    def objective(weights):
        blended = oof_matrix @ weights
        return rmsle(y, blended)

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    # Bounds: each weight >= 0
    bounds = [(0.0, 1.0)] * n_models
    # Start: equal weights
    x0 = np.ones(n_models) / n_models

    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    weights = dict(zip(names, result.x))
    blend_rmsle = result.fun

    return weights, blend_rmsle


def train_full_and_predict(X_train, y, X_test):
    """Train both models on full training data, return test predictions."""
    # ElasticNet
    en_model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )
    en_model.fit(X_train, y)
    en_preds = np.clip(en_model.predict(X_test), 0, None)

    # XGBoost (train on full data with early stopping on held-out portion)
    y_log = np.log1p(y)
    fit_idx, es_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.2, random_state=CV_SEED
    )
    xgb_model = xgb.XGBRegressor(
        max_depth=3, learning_rate=0.08, subsample=0.7,
        colsample_bytree=0.5, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=2.0,
        n_estimators=3000, early_stopping_rounds=50,
        random_state=CV_SEED,
    )
    xgb_model.fit(
        X_train.iloc[fit_idx], y_log.iloc[fit_idx],
        eval_set=[(X_train.iloc[es_idx], y_log.iloc[es_idx])],
        verbose=False,
    )
    xgb_preds = np.clip(np.expm1(xgb_model.predict(X_test)), 0, None)

    return en_preds, xgb_preds


def main():
    tee = Tee(f"{BASE}/results/models/13_blending.txt")
    sys.stdout = tee

    print("House Prices: Step 13 — OOF Blending")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    print(f"Train: {X_train.shape} (full data)")
    print(f"Test:  {X_test.shape}")
    print(f"CV: KFold({CV_SPLITS}, shuffle=True, random_state={CV_SEED})")
    print()

    # ============================================================
    # COLLECT OOF PREDICTIONS
    # ============================================================
    print("=" * 60)
    print("COLLECTING OOF PREDICTIONS")
    print("=" * 60)

    print("\n  ElasticNet (alpha=0.02848, l1_ratio=0.1)...")
    oof_en = collect_oof_elasticnet(X_train, y)
    en_rmsle = rmsle(y, oof_en)
    print(f"    OOF RMSLE: {en_rmsle:.5f}")

    print("\n  XGBoost (tuned, Step 11 best)...")
    oof_xgb = collect_oof_xgboost(X_train, y)
    xgb_rmsle = rmsle(y, oof_xgb)
    print(f"    OOF RMSLE: {xgb_rmsle:.5f}")
    print()

    # ============================================================
    # ERROR CORRELATION ANALYSIS
    # ============================================================
    print("=" * 60)
    print("ERROR CORRELATION (do models fail on different houses?)")
    print("=" * 60)

    en_errors = np.abs(y.values - oof_en)
    xgb_errors = np.abs(y.values - oof_xgb)
    error_corr = np.corrcoef(en_errors, xgb_errors)[0, 1]
    print(f"\n  Correlation of absolute errors: {error_corr:.3f}")
    print(f"  (1.0 = identical errors, 0.0 = uncorrelated, <0.5 = good for blending)")

    # How often does each model win?
    en_wins = (en_errors < xgb_errors).sum()
    xgb_wins = (xgb_errors < en_errors).sum()
    ties = (en_errors == xgb_errors).sum()
    print(f"\n  Per-sample wins:")
    print(f"    ElasticNet better: {en_wins} ({en_wins/len(y)*100:.1f}%)")
    print(f"    XGBoost better:    {xgb_wins} ({xgb_wins/len(y)*100:.1f}%)")
    print(f"    Tied:              {ties}")

    # Where does ElasticNet help most?
    en_advantage = xgb_errors - en_errors  # positive = EN is better
    top_en_idx = np.argsort(en_advantage)[-5:][::-1]
    print(f"\n  Houses where ElasticNet is most helpful:")
    print(f"  {'Actual':>10s} {'EN Pred':>10s} {'XGB Pred':>10s} "
          f"{'EN Err':>8s} {'XGB Err':>8s}")
    print(f"  {'-' * 50}")
    for idx in top_en_idx:
        print(f"  ${y.iloc[idx]:>9,.0f} ${oof_en[idx]:>9,.0f} "
              f"${oof_xgb[idx]:>9,.0f} "
              f"${en_errors[idx]:>7,.0f} ${xgb_errors[idx]:>7,.0f}")
    print()

    # ============================================================
    # BLEND EXPERIMENTS
    # ============================================================
    print("=" * 60)
    print("BLEND EXPERIMENTS")
    print("=" * 60)

    oof_dict = {"ElasticNet": oof_en, "XGBoost": oof_xgb}
    blend_results = {}

    # 1. Equal average
    blend_50 = 0.5 * oof_en + 0.5 * oof_xgb
    score_50 = rmsle(y, blend_50)
    blend_results["50/50 average"] = score_50
    print(f"\n  50/50 average:     RMSLE = {score_50:.5f}")

    # 2. Sweep fixed weights
    print(f"\n  Weight sweep (ElasticNet weight):")
    print(f"  {'EN Weight':>10s} {'XGB Weight':>10s} {'RMSLE':>10s}")
    print(f"  {'-' * 35}")
    sweep_scores = {}
    for en_w in np.arange(0.0, 1.05, 0.1):
        xgb_w = 1.0 - en_w
        blend = en_w * oof_en + xgb_w * oof_xgb
        score = rmsle(y, blend)
        sweep_scores[en_w] = score
        marker = " <--" if score == min(sweep_scores.values()) else ""
        print(f"  {en_w:>10.1f} {xgb_w:>10.1f} {score:>10.5f}{marker}")

    # 3. Constrained optimization
    opt_weights, opt_rmsle = find_optimal_weights(oof_dict, y)
    blend_results["Optimized"] = opt_rmsle
    print(f"\n  Constrained optimization:")
    for name, weight in opt_weights.items():
        print(f"    {name:15s}: {weight:.4f}")
    print(f"    RMSLE:           {opt_rmsle:.5f}")
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\n  {'Model':30s} {'OOF RMSLE':>12s}")
    print(f"  {'-' * 45}")
    print(f"  {'ElasticNet (alone)':30s} {en_rmsle:12.5f}")
    print(f"  {'XGBoost tuned (alone)':30s} {xgb_rmsle:12.5f}")
    print(f"  {'50/50 average':30s} {score_50:12.5f}")
    print(f"  {'Optimized blend':30s} {opt_rmsle:12.5f}")

    best_solo = min(en_rmsle, xgb_rmsle)
    best_blend = min(score_50, opt_rmsle)
    if best_blend < best_solo:
        improvement = (best_solo - best_blend) / best_solo * 100
        print(f"\n  Blend beats best solo by {improvement:.2f}%")
    else:
        print(f"\n  Blend does NOT beat best solo model")
    print()

    # ============================================================
    # GENERATE SUBMISSION
    # ============================================================
    print("=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)

    en_preds, xgb_preds = train_full_and_predict(X_train, y, X_test)

    # Use the better strategy (optimized or solo XGBoost)
    if best_blend < best_solo:
        en_w = opt_weights["ElasticNet"]
        xgb_w = opt_weights["XGBoost"]
        final_preds = en_w * en_preds + xgb_w * xgb_preds
        strategy = f"Optimized blend (EN={en_w:.3f}, XGB={xgb_w:.3f})"
    else:
        final_preds = xgb_preds
        strategy = "XGBoost solo (blend didn't help)"

    print(f"\n  Strategy: {strategy}")
    print(f"  Prediction range: ${final_preds.min():,.0f} - ${final_preds.max():,.0f}")

    sub = pd.DataFrame({"Id": test_ids, "SalePrice": final_preds})
    sub.to_csv(f"{BASE}/submissions/blend_final.csv", index=False)
    print(f"  Saved: submissions/blend_final.csv")

    # Also save XGBoost-only for comparison
    sub_xgb = pd.DataFrame({"Id": test_ids, "SalePrice": xgb_preds})
    sub_xgb.to_csv(f"{BASE}/submissions/xgboost_tuned.csv", index=False)
    print(f"  Saved: submissions/xgboost_tuned.csv")
    print()

    tee.close()


if __name__ == "__main__":
    main()
