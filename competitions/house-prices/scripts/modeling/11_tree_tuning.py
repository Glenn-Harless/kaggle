"""
House Prices: Step 11 — XGBoost Hyperparameter Tuning

Randomized search over key XGBoost hyperparameters.
Uses the same validation protocol as Step 10:
  - Outer: KFold(5, shuffle=True, random_state=42)
  - Inner: 80/20 split within training fold for early stopping
  - Early-stopping split is deterministic per fold (random_state=fold_index)

Inputs:  data/train_processed.csv
Outputs: results/models/11_tree_tuning.txt
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

from shared.evaluate import Tee, rmsle

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


def load_processed():
    train = pd.read_csv(f"{BASE}/data/train_processed.csv")
    y = train["SalePrice"]
    X_train = train.drop(columns=["Id", "SalePrice"])
    return X_train, y


def cv_xgboost(X, y, params, n_splits=5, random_state=42):
    """
    XGBoost CV with early stopping per the validation protocol.
    Returns mean RMSLE, std, per-fold scores, and avg rounds.
    """
    y_log = np.log1p(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    n_rounds_used = []

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y_log.iloc[val_idx]

        # Inner split for early stopping (deterministic per fold)
        fit_idx, es_idx = train_test_split(
            np.arange(len(X_tr)), test_size=0.2, random_state=fold_i
        )

        model = xgb.XGBRegressor(
            **params,
            n_estimators=3000,          # high ceiling
            early_stopping_rounds=50,
            random_state=random_state,
        )
        model.fit(
            X_tr.iloc[fit_idx], y_tr.iloc[fit_idx],
            eval_set=[(X_tr.iloc[es_idx], y_tr.iloc[es_idx])],
            verbose=False,
        )

        log_preds = model.predict(X_val)
        preds = np.clip(np.expm1(log_preds), 0, None)

        score = rmsle(y.iloc[val_idx], preds)
        fold_scores.append(score)

        try:
            best_round = model.best_iteration
        except AttributeError:
            best_round = 3000
        n_rounds_used.append(best_round)

    scores = np.array(fold_scores)
    return {
        'mean_rmsle': scores.mean(),
        'std_rmsle': scores.std(),
        'fold_scores': scores,
        'avg_rounds': np.mean(n_rounds_used),
    }


def random_search(X, y, param_distributions, n_iter=60, random_state=42):
    """
    Randomized hyperparameter search.

    Samples n_iter random combinations from param_distributions,
    evaluates each with 5-fold CV, returns sorted results.
    """
    rng = np.random.RandomState(random_state)
    results = []

    for trial in range(n_iter):
        # Sample random hyperparameters
        params = {}
        for name, values in param_distributions.items():
            params[name] = rng.choice(values)

        # Evaluate
        cv_result = cv_xgboost(X, y, params)

        result = {**params, **cv_result}
        results.append(result)

        # Progress
        print(f"    Trial {trial + 1:3d}/{n_iter}: "
              f"RMSLE={cv_result['mean_rmsle']:.5f}  "
              f"depth={params['max_depth']}, "
              f"lr={params['learning_rate']:.3f}, "
              f"sub={params['subsample']:.1f}, "
              f"col={params['colsample_bytree']:.1f}, "
              f"mcw={params['min_child_weight']}, "
              f"rounds={cv_result['avg_rounds']:.0f}")

    return pd.DataFrame(results).sort_values('mean_rmsle')


def main():
    tee = Tee(f"{BASE}/results/models/11_tree_tuning.txt")
    sys.stdout = tee

    print("House Prices: Step 11 — XGBoost Hyperparameter Tuning")
    print("=" * 60)
    print()

    X_train, y = load_processed()
    print(f"Train: {X_train.shape} (full data)")
    print(f"Method: Randomized search, 60 trials")
    print(f"CV: KFold(5, shuffle=True, random_state=42)")
    print(f"Early stopping: 80/20 inner split, patience=50")
    print()

    # ============================================================
    # SEARCH SPACE
    # ============================================================
    print("=" * 60)
    print("SEARCH SPACE")
    print("=" * 60)

    param_distributions = {
        'max_depth':        [2, 3, 4, 5, 6, 7, 8],
        'learning_rate':    [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 5, 7, 10],
        'reg_alpha':        [0, 0.001, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda':       [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

    for name, values in param_distributions.items():
        print(f"  {name:25s}: {values}")
    n_combos = 1
    for v in param_distributions.values():
        n_combos *= len(v)
    print(f"\n  Total possible combinations: {n_combos:,}")
    print(f"  Sampling: 60 random trials")
    print()

    # ============================================================
    # RUN SEARCH
    # ============================================================
    print("=" * 60)
    print("RUNNING RANDOMIZED SEARCH")
    print("=" * 60)

    results = random_search(X_train, y, param_distributions, n_iter=60)
    print()

    # ============================================================
    # TOP 10 RESULTS
    # ============================================================
    print("=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)

    display_cols = ['mean_rmsle', 'std_rmsle', 'max_depth', 'learning_rate',
                    'subsample', 'colsample_bytree', 'min_child_weight',
                    'reg_alpha', 'reg_lambda', 'avg_rounds']
    print(f"\n  {'Rank':>4s} {'RMSLE':>8s} {'Std':>7s} {'Depth':>5s} "
          f"{'LR':>5s} {'Sub':>4s} {'Col':>4s} {'MCW':>4s} "
          f"{'L1':>5s} {'L2':>5s} {'Rnds':>5s}")
    print(f"  {'-' * 70}")
    for rank, (_, row) in enumerate(results.head(10).iterrows(), 1):
        marker = " <--" if rank == 1 else ""
        print(f"  {rank:>4d} {row['mean_rmsle']:>8.5f} {row['std_rmsle']:>7.5f} "
              f"{int(row['max_depth']):>5d} {row['learning_rate']:>5.3f} "
              f"{row['subsample']:>4.1f} {row['colsample_bytree']:>4.1f} "
              f"{int(row['min_child_weight']):>4d} "
              f"{row['reg_alpha']:>5.3f} {row['reg_lambda']:>5.1f} "
              f"{row['avg_rounds']:>5.0f}{marker}")

    # ============================================================
    # BEST VS STEP 10 DEFAULTS
    # ============================================================
    best = results.iloc[0]
    print(f"\n{'=' * 60}")
    print("BEST VS STEP 10 DEFAULTS")
    print("=" * 60)

    step10_rmsle = 0.12971
    print(f"\n  Step 10 defaults:  RMSLE = {step10_rmsle:.5f}")
    print(f"  Best tuned:        RMSLE = {best['mean_rmsle']:.5f}")
    improvement = (step10_rmsle - best['mean_rmsle']) / step10_rmsle * 100
    print(f"  Improvement:       {improvement:+.2f}%")

    print(f"\n  Best hyperparameters:")
    for param in param_distributions:
        print(f"    {param:25s}: {best[param]}")
    print(f"    {'avg_rounds':25s}: {best['avg_rounds']:.0f}")

    # Per-fold comparison
    print(f"\n  Per-fold scores (best tuned):")
    for i, s in enumerate(best['fold_scores']):
        print(f"    Fold {i + 1}: {s:.5f}")
    print()

    # ============================================================
    # PARAMETER SENSITIVITY
    # ============================================================
    print("=" * 60)
    print("PARAMETER SENSITIVITY (which parameters matter most?)")
    print("=" * 60)
    print("  Mean RMSLE by parameter value (averaged across all trials):")

    for param, values in param_distributions.items():
        print(f"\n  {param}:")
        for val in sorted(values):
            mask = results[param] == val
            if mask.sum() > 0:
                mean_score = results.loc[mask, 'mean_rmsle'].mean()
                n_trials = mask.sum()
                bar = "#" * max(1, int((0.15 - mean_score) * 500))
                print(f"    {str(val):>8s}: {mean_score:.5f} "
                      f"(n={n_trials:2d}) {bar}")

    print()
    tee.close()


if __name__ == "__main__":
    main()
