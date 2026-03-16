"""
House Prices: Step 10 — Tree Model Baselines (XGBoost + LightGBM)

Compare XGBoost and LightGBM against ElasticNet on the FULL dataset.
No outlier removal — trees handle outliers natively.

Validation protocol:
  - Outer: KFold(5, shuffle=True, random_state=42) — same folds for all models
  - Inner: 80/20 split within each training fold for early stopping
  - Early-stopping split uses random_state=fold_index for reproducibility

All models train on log1p(SalePrice).

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/10_tree_baselines.txt
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
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
import lightgbm as lgb

from shared.evaluate import Tee, rmsle

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test = pd.read_csv(f"{BASE}/data/test_processed.csv")
    y = train["SalePrice"]
    X_train = train.drop(columns=["Id", "SalePrice"])
    X_test = test.drop(columns=["Id"])
    test_ids = test["Id"]
    return X_train, y, X_test, test_ids


def cv_elasticnet(X, y, n_splits=5, random_state=42):
    """
    ElasticNet CV on full data with same folds as tree models.
    Uses TransformedTargetRegressor for log1p target.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    fold_scores = []

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
        preds = np.clip(m.predict(X.iloc[val_idx]), 0, None)
        oof[val_idx] = preds

        score = rmsle(y.iloc[val_idx], preds)
        fold_scores.append(score)
        print(f"    Fold {fold_i + 1}: RMSLE = {score:.5f}")

    return np.array(fold_scores), oof


def cv_xgboost(X, y, n_splits=5, random_state=42):
    """
    XGBoost CV with early stopping inside each fold.

    Early stopping uses an 80/20 split of the TRAINING fold.
    The validation fold is for scoring only.
    """
    y_log = np.log1p(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    n_rounds_used = []

    params = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 2000,       # high ceiling, early stopping will cut
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'early_stopping_rounds': 50,
    }

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y_log.iloc[val_idx]

        # Inner split: 80% fit, 20% early-stopping monitor
        # Deterministic per fold (random_state=fold_i)
        fit_idx, es_idx = train_test_split(
            np.arange(len(X_tr)), test_size=0.2, random_state=fold_i
        )

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr.iloc[fit_idx], y_tr.iloc[fit_idx],
            eval_set=[(X_tr.iloc[es_idx], y_tr.iloc[es_idx])],
            verbose=False,
        )

        # Predict on validation fold (in log space, convert back)
        log_preds = model.predict(X_val)
        preds = np.expm1(log_preds)
        preds = np.clip(preds, 0, None)
        oof[val_idx] = preds

        score = rmsle(y.iloc[val_idx], preds)
        fold_scores.append(score)
        try:
            best_round = model.best_iteration
        except AttributeError:
            best_round = params['n_estimators']
        n_rounds_used.append(best_round)
        print(f"    Fold {fold_i + 1}: RMSLE = {score:.5f}  "
              f"(stopped at round {best_round})")

    print(f"    Avg rounds: {np.mean(n_rounds_used):.0f}")
    return np.array(fold_scores), oof


def cv_lightgbm(X, y, n_splits=5, random_state=42):
    """
    LightGBM CV with early stopping inside each fold.
    Same protocol as XGBoost.
    """
    y_log = np.log1p(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    n_rounds_used = []

    params = {
        'num_leaves': 31,
        'max_depth': -1,            # LightGBM controls complexity via num_leaves
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': random_state,
        'verbosity': -1,
    }

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y_log.iloc[val_idx]

        # Inner split for early stopping (deterministic per fold)
        fit_idx, es_idx = train_test_split(
            np.arange(len(X_tr)), test_size=0.2, random_state=fold_i
        )

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr.iloc[fit_idx], y_tr.iloc[fit_idx],
            eval_set=[(X_tr.iloc[es_idx], y_tr.iloc[es_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        log_preds = model.predict(X_val)
        preds = np.expm1(log_preds)
        preds = np.clip(preds, 0, None)
        oof[val_idx] = preds

        score = rmsle(y.iloc[val_idx], preds)
        fold_scores.append(score)
        best_round = model.best_iteration_ if model.best_iteration_ else params['n_estimators']
        n_rounds_used.append(best_round)
        print(f"    Fold {fold_i + 1}: RMSLE = {score:.5f}  "
              f"(stopped at round {best_round})")

    print(f"    Avg rounds: {np.mean(n_rounds_used):.0f}")
    return np.array(fold_scores), oof


def report_scores(scores, label):
    """Print CV summary."""
    print(f"  [{label}] Mean RMSLE: {scores.mean():.5f}")
    print(f"  [{label}] Std:        {scores.std():.5f}")
    print(f"  [{label}] Per fold:   {', '.join(f'{s:.5f}' for s in scores)}")


def main():
    tee = Tee(f"{BASE}/results/models/10_tree_baselines.txt")
    sys.stdout = tee

    print("House Prices: Step 10 — Tree Model Baselines")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids = load_processed()
    print(f"Train: {X_train.shape} (FULL data, no outlier removal)")
    print(f"Test:  {X_test.shape}")
    print(f"Target: log1p(SalePrice) for all models")
    print(f"CV: KFold(5, shuffle=True, random_state=42) — same folds for all")
    print()

    all_results = {}

    # ============================================================
    # ELASTICNET (reference)
    # ============================================================
    print("=" * 60)
    print("ELASTICNET (Phase 2-4 best, full data reference)")
    print("  StandardScaler + ElasticNet(a=0.02848, l1=0.1)")
    print("=" * 60)
    scores_en, oof_en = cv_elasticnet(X_train, y)
    report_scores(scores_en, "ElasticNet")
    all_results["ElasticNet"] = scores_en
    print()

    # ============================================================
    # XGBOOST
    # ============================================================
    print("=" * 60)
    print("XGBOOST (conservative defaults)")
    print("  max_depth=3, lr=0.05, subsample=0.8, colsample=0.8")
    print("  Early stopping: 80/20 inner split, patience=default")
    print("=" * 60)
    scores_xgb, oof_xgb = cv_xgboost(X_train, y)
    report_scores(scores_xgb, "XGBoost")
    all_results["XGBoost"] = scores_xgb
    print()

    # ============================================================
    # LIGHTGBM
    # ============================================================
    print("=" * 60)
    print("LIGHTGBM (conservative defaults)")
    print("  num_leaves=31, lr=0.05, subsample=0.8, colsample=0.8")
    print("  Early stopping: 80/20 inner split, patience=50 rounds")
    print("=" * 60)
    scores_lgb, oof_lgb = cv_lightgbm(X_train, y)
    report_scores(scores_lgb, "LightGBM")
    all_results["LightGBM"] = scores_lgb
    print()

    # ============================================================
    # COMPARISON
    # ============================================================
    print("=" * 60)
    print("COMPARISON (same 5 folds, full data, lower is better)")
    print("=" * 60)

    print(f"\n  {'Model':20s} {'Mean RMSLE':>12s} {'Std':>10s} "
          f"{'Best Fold':>10s} {'Worst Fold':>11s}")
    print(f"  {'-' * 68}")
    for name, scores in sorted(all_results.items(), key=lambda x: x[1].mean()):
        marker = " <--" if scores.mean() == min(s.mean() for s in all_results.values()) else ""
        print(f"  {name:20s} {scores.mean():12.5f} {scores.std():10.5f} "
              f"{scores.min():10.5f} {scores.max():11.5f}{marker}")

    # Head-to-head per fold
    print(f"\n  Per-fold head-to-head:")
    print(f"  {'Fold':>6s} {'ElasticNet':>12s} {'XGBoost':>12s} {'LightGBM':>12s} {'Winner':>12s}")
    print(f"  {'-' * 58}")
    for i in range(len(scores_en)):
        en_s = scores_en[i]
        xgb_s = scores_xgb[i]
        lgb_s = scores_lgb[i]
        best = min(en_s, xgb_s, lgb_s)
        if best == en_s:
            winner = "ElasticNet"
        elif best == xgb_s:
            winner = "XGBoost"
        else:
            winner = "LightGBM"
        print(f"  {i + 1:>6d} {en_s:12.5f} {xgb_s:12.5f} {lgb_s:12.5f} {winner:>12s}")

    # OOF RMSLE (overall, not per-fold average)
    print(f"\n  Overall OOF RMSLE (computed on all 1460 predictions):")
    print(f"    ElasticNet: {rmsle(y, oof_en):.5f}")
    print(f"    XGBoost:    {rmsle(y, oof_xgb):.5f}")
    print(f"    LightGBM:   {rmsle(y, oof_lgb):.5f}")

    # Quick blend preview
    blend_simple = 0.5 * oof_en + 0.5 * oof_xgb
    blend_3way = (oof_en + oof_xgb + oof_lgb) / 3
    print(f"\n  Quick blend preview (OOF, for Step 13):")
    print(f"    50/50 ElasticNet+XGBoost:    {rmsle(y, blend_simple):.5f}")
    print(f"    Equal 3-way:                 {rmsle(y, blend_3way):.5f}")
    print()

    tee.close()


if __name__ == "__main__":
    main()
