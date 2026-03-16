"""
House Prices: Step 8 — Final Model + Error Audit

Fit the best model, analyze remaining errors, final submission.

Best model: ElasticNet (alpha=0.02848, l1_ratio=0.1)
  + StandardScaler + log1p target + outlier removal (GrLivArea > 4000)
  CV RMSLE: 0.112, Kaggle RMSLE: 0.136

Inputs:  data/train_processed.csv, data/test_processed.csv
Outputs: results/models/08_error_audit.txt
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold

from shared.evaluate import Tee, rmsle

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


def load_data():
    train_proc = pd.read_csv(f"{BASE}/data/train_processed.csv")
    test_proc = pd.read_csv(f"{BASE}/data/test_processed.csv")
    train_raw = pd.read_csv(f"{BASE}/data/train.csv")

    y = train_proc["SalePrice"]
    X_train = train_proc.drop(columns=["Id", "SalePrice"])
    X_test = test_proc.drop(columns=["Id"])
    test_ids = test_proc["Id"]

    return X_train, y, X_test, test_ids, train_raw


def oof_predictions(model, X, y, n_splits=5, random_state=42):
    """Collect out-of-fold predictions."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in kf.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = np.clip(m.predict(X.iloc[val_idx]), 0, None)
    return oof


def main():
    tee = Tee(f"{BASE}/results/models/08_error_audit.txt")
    sys.stdout = tee

    print("House Prices: Step 8 — Final Model + Error Audit")
    print("=" * 60)
    print()

    X_train, y, X_test, test_ids, raw = load_data()

    # Remove outliers
    mask = X_train['GrLivArea'] <= 4000
    X_clean = X_train[mask].reset_index(drop=True)
    y_clean = y[mask].reset_index(drop=True)
    raw_clean = raw[mask].reset_index(drop=True)
    n_removed = (~mask).sum()

    print(f"Train: {X_clean.shape} ({n_removed} outliers removed)")
    print(f"Test: {X_test.shape}")
    print()

    # ============================================================
    # FINAL MODEL
    # ============================================================
    print("=" * 60)
    print("FINAL MODEL")
    print("=" * 60)
    print(f"  ElasticNet (alpha=0.02848, l1_ratio=0.1)")
    print(f"  StandardScaler + log1p target")
    print(f"  Outlier removal: GrLivArea > 4000 ({n_removed} houses)")
    print()

    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )

    # OOF predictions for error analysis
    oof = oof_predictions(model, X_clean, y_clean)
    oof_rmsle = rmsle(y_clean, oof)
    print(f"  OOF RMSLE: {oof_rmsle:.5f}")

    residuals = y_clean.values - oof
    pct_errors = np.abs(residuals) / y_clean.values * 100

    # ============================================================
    # ERROR DISTRIBUTION
    # ============================================================
    print(f"\n{'=' * 60}")
    print("ERROR DISTRIBUTION")
    print("=" * 60)

    print(f"\n  Overall:")
    print(f"    Mean |error|:     ${np.abs(residuals).mean():>10,.0f}")
    print(f"    Median |error|:   ${np.median(np.abs(residuals)):>10,.0f}")
    print(f"    Mean % error:     {pct_errors.mean():>9.1f}%")
    print(f"    Median % error:   {np.median(pct_errors):>9.1f}%")

    # Error buckets
    buckets = [0, 5, 10, 15, 20, 30, 50, 100, 1000]
    labels = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50-100%", ">100%"]
    binned = pd.cut(pct_errors, bins=buckets, labels=labels)
    print(f"\n  Error distribution:")
    cumulative = 0
    for label in labels:
        count = (binned == label).sum()
        pct = count / len(pct_errors) * 100
        cumulative += pct
        bar = "#" * int(pct / 2)
        print(f"    {label:>8s}: {count:4d} ({pct:5.1f}%) cum {cumulative:5.1f}%  {bar}")

    # ============================================================
    # ERROR BY CATEGORY
    # ============================================================
    print(f"\n{'=' * 60}")
    print("ERRORS BY PROPERTY TYPE")
    print("=" * 60)

    # By building type
    print(f"\n  By BldgType:")
    print(f"  {'Type':>12s} {'N':>6s} {'Mean %Err':>10s} {'Median %Err':>12s}")
    print(f"  {'-' * 45}")
    for btype in sorted(raw_clean['BldgType'].unique()):
        bm = raw_clean['BldgType'] == btype
        print(f"  {btype:>12s} {bm.sum():>6d} {pct_errors[bm].mean():>9.1f}% "
              f"{np.median(pct_errors[bm]):>11.1f}%")

    # By price tier
    print(f"\n  By price tier:")
    tiers = pd.qcut(y_clean, 4, labels=["Budget", "Moderate", "Upper", "Premium"])
    print(f"  {'Tier':>12s} {'N':>6s} {'Price Range':>22s} {'Mean %Err':>10s}")
    print(f"  {'-' * 55}")
    for tier in ["Budget", "Moderate", "Upper", "Premium"]:
        tm = tiers == tier
        prices = y_clean[tm]
        print(f"  {tier:>12s} {tm.sum():>6d} "
              f"${prices.min():>8,.0f}-${prices.max():>8,.0f} "
              f"{pct_errors[tm].mean():>9.1f}%")

    # By neighborhood (top errors)
    print(f"\n  Neighborhoods with highest error:")
    nbhd_errors = {}
    for nbhd in raw_clean['Neighborhood'].unique():
        nm = raw_clean['Neighborhood'] == nbhd
        if nm.sum() >= 10:
            nbhd_errors[nbhd] = pct_errors[nm].mean()
    nbhd_sorted = sorted(nbhd_errors.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Neighborhood':>12s} {'N':>6s} {'Mean %Err':>10s}")
    print(f"  {'-' * 32}")
    for nbhd, err in nbhd_sorted[:8]:
        nm = raw_clean['Neighborhood'] == nbhd
        print(f"  {nbhd:>12s} {nm.sum():>6d} {err:>9.1f}%")

    # ============================================================
    # REMAINING HARD CASES
    # ============================================================
    print(f"\n{'=' * 60}")
    print("REMAINING HARD CASES (>20% error)")
    print("=" * 60)

    hard_mask = pct_errors > 20
    n_hard = hard_mask.sum()
    print(f"\n  {n_hard} houses with >20% error ({n_hard/len(y_clean)*100:.1f}% of training)")

    if n_hard > 0:
        hard_idx = np.where(hard_mask)[0]
        hard_rows = raw_clean.iloc[hard_idx]

        # Common patterns
        print(f"\n  SaleCondition breakdown:")
        sc = hard_rows['SaleCondition'].value_counts()
        for cond, count in sc.items():
            total = (raw_clean['SaleCondition'] == cond).sum()
            print(f"    {cond:>10s}: {count:3d} / {total:4d} "
                  f"({count/total*100:.0f}% of {cond} houses are hard)")

        print(f"\n  OverallQual distribution:")
        print(f"    Hard cases mean:  {hard_rows['OverallQual'].mean():.1f}")
        print(f"    All houses mean:  {raw_clean['OverallQual'].mean():.1f}")

        print(f"\n  Direction of errors:")
        hard_residuals = residuals[hard_mask]
        over = (hard_residuals > 0).sum()
        under = (hard_residuals < 0).sum()
        print(f"    Model overpredicts (actual > pred): {over}")
        print(f"    Model underpredicts (pred > actual): {under}")

    # ============================================================
    # WHAT'S FIXABLE VS NOT
    # ============================================================
    print(f"\n{'=' * 60}")
    print("WHAT'S FIXABLE VS WHAT'S NOISE")
    print("=" * 60)

    # Categorize hard cases
    if n_hard > 0:
        hard_df = pd.DataFrame({
            'actual': y_clean.values[hard_mask],
            'predicted': oof[hard_mask],
            'pct_error': pct_errors[hard_mask],
            'sale_cond': raw_clean.iloc[hard_idx]['SaleCondition'].values,
            'qual': raw_clean.iloc[hard_idx]['OverallQual'].values,
            'nbhd': raw_clean.iloc[hard_idx]['Neighborhood'].values,
        })

        non_normal = hard_df['sale_cond'] != 'Normal'
        print(f"\n  Hard cases from non-normal sales: "
              f"{non_normal.sum()} / {n_hard} ({non_normal.sum()/n_hard*100:.0f}%)")
        print(f"  Hard cases from normal sales:     "
              f"{(~non_normal).sum()} / {n_hard} ({(~non_normal).sum()/n_hard*100:.0f}%)")
        print(f"\n  Non-normal sales are inherently hard to predict because")
        print(f"  the price reflects sale circumstances, not house features.")
        print(f"  These are likely NOT fixable with better features.")

        normal_hard = hard_df[~non_normal]
        if len(normal_hard) > 0:
            print(f"\n  Normal-sale hard cases ({len(normal_hard)}) — potentially fixable:")
            for _, row in normal_hard.sort_values('pct_error', ascending=False).head(10).iterrows():
                print(f"    ${row['actual']:>9,.0f} predicted ${row['predicted']:>9,.0f} "
                      f"({row['pct_error']:.0f}% off) "
                      f"Q={int(row['qual'])} {row['nbhd']}")

    # ============================================================
    # SCORE PROGRESSION
    # ============================================================
    print(f"\n{'=' * 60}")
    print("SCORE PROGRESSION: Full Journey")
    print("=" * 60)
    print(f"""
  Step  What changed                      CV RMSLE  Kaggle
  -------------------------------------------------------
  2     Ridge (raw, unscaled)               0.236    0.184
  3     + StandardScaler + log target       0.151      —
  4     + Tuned alpha (ElasticNet)          0.138      —
  6     + Outlier removal (4 houses)        0.112    0.136
  7     Encoding experiment                 ~0.112     —
  8     Final model (same as Step 6)        {oof_rmsle:.3f}      —

  Key drivers of improvement:
  1. Log transform:    0.236 -> 0.151  (36% better)
  2. Alpha tuning:     0.151 -> 0.138  ( 9% better)
  3. Outlier removal:  0.138 -> 0.112  (19% better)
  4. Feature eng:      no improvement  ( 0%)
  5. Encoding choice:  no improvement  ( 0%)
""")

    # ============================================================
    # FINAL PLOT
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Step 8: Final Model Error Audit', fontsize=14, fontweight='bold')

    # Predicted vs Actual
    ax = axes[0]
    ax.scatter(oof, y_clean.values, alpha=0.3, s=10, c='steelblue')
    lims = [0, max(y_clean.max(), oof.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
    ax.set_xlabel('Predicted ($)')
    ax.set_ylabel('Actual ($)')
    ax.set_title('Predicted vs Actual (after outlier removal)')
    ax.legend()

    # Error histogram
    ax = axes[1]
    ax.hist(pct_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=pct_errors.mean(), color='red', linestyle='--',
               label=f'Mean: {pct_errors.mean():.1f}%')
    ax.axvline(x=np.median(pct_errors), color='orange', linestyle='--',
               label=f'Median: {np.median(pct_errors):.1f}%')
    ax.set_xlabel('Absolute % Error')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = f"{BASE}/results/analysis/08_error_audit.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")
    print()

    tee.close()


if __name__ == "__main__":
    main()
