"""
House Prices: Step 5 — Residual Analysis

Analyze where the best model (ElasticNet) is still wrong.
Goal: identify systematic patterns in errors that suggest
missing features or relationships.

Uses OOF (out-of-fold) predictions so residuals reflect
generalization error, not training fit.

Inputs:  data/train_processed.csv
Outputs: results/models/05_residual_analysis.txt
         results/analysis/05_residual_plots.png
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shared.evaluate import Tee, rmsle

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/house-prices"


def load_data():
    """Load processed features and raw data (for interpretable columns)."""
    train_proc = pd.read_csv(f"{BASE}/data/train_processed.csv")
    train_raw = pd.read_csv(f"{BASE}/data/train.csv")

    y = train_proc["SalePrice"]
    X = train_proc.drop(columns=["Id", "SalePrice"])
    return X, y, train_raw


def oof_predictions(model, X, y, n_splits=5, random_state=42):
    """Collect out-of-fold predictions (each sample predicted when held out)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)

    for train_idx, val_idx in kf.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        oof[val_idx] = np.clip(preds, 0, None)

    return oof


def residuals_vs_predicted(y_true, y_pred):
    """Check if residual patterns depend on predicted value."""
    residuals = y_true - y_pred
    pct_errors = np.abs(residuals) / y_true * 100

    quintiles = pd.qcut(y_pred, 5, labels=[
        "Q1 low pred", "Q2", "Q3", "Q4", "Q5 high pred"
    ])

    print("  Residuals vs Predicted Value:")
    print(f"  {'Quintile':15s} {'Pred Range':>22s} {'Mean Error':>13s} "
          f"{'Mean |Error|':>14s} {'Mean %Err':>10s}")
    print(f"  {'-' * 78}")
    for q in ["Q1 low pred", "Q2", "Q3", "Q4", "Q5 high pred"]:
        mask = quintiles == q
        preds_q = y_pred[mask]
        mean_err = residuals[mask].mean()
        mae = np.abs(residuals[mask]).mean()
        mpe = pct_errors[mask].mean()
        pred_range = f"${preds_q.min():,.0f}-${preds_q.max():,.0f}"
        print(f"  {q:15s} {pred_range:>22s} ${mean_err:>+11,.0f} "
              f"${mae:>12,.0f} {mpe:>9.1f}%")


def residuals_vs_feature(y_true, y_pred, feature_values, feature_name,
                         n_bins=5, is_categorical=False):
    """Check if residuals correlate with a specific feature."""
    residuals = y_true - y_pred
    pct_errors = np.abs(residuals) / y_true * 100

    if is_categorical:
        groups = feature_values
        unique = sorted(feature_values.unique())
    else:
        try:
            groups = pd.qcut(feature_values, n_bins, duplicates='drop')
            unique = sorted(groups.unique())
        except ValueError:
            groups = pd.cut(feature_values, n_bins, duplicates='drop')
            unique = sorted(groups.unique())

    print(f"\n  Residuals vs {feature_name}:")
    print(f"  {'Group':>25s} {'N':>6s} {'Mean Error':>13s} "
          f"{'Mean |Error|':>14s} {'Mean %Err':>10s}")
    print(f"  {'-' * 72}")
    for g in unique:
        mask = groups == g
        n = mask.sum()
        if n == 0:
            continue
        mean_err = residuals[mask].mean()
        mae = np.abs(residuals[mask]).mean()
        mpe = pct_errors[mask].mean()
        label = str(g)[:25]
        print(f"  {label:>25s} {n:>6d} ${mean_err:>+11,.0f} "
              f"${mae:>12,.0f} {mpe:>9.1f}%")


def worst_predictions(y_true, y_pred, raw_df, n=15):
    """Examine the worst predictions to find common patterns."""
    residuals = y_true.values - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = abs_errors / y_true.values * 100

    worst_idx = np.argsort(pct_errors)[-n:][::-1]

    print(f"\n  Top {n} worst predictions (by % error):")
    print(f"  {'Actual':>10s} {'Pred':>10s} {'%Err':>7s} "
          f"{'Qual':>5s} {'SF':>6s} {'YrBlt':>6s} {'Nbhd':>10s} "
          f"{'SaleCond':>10s} {'BldgType':>10s}")
    print(f"  {'-' * 85}")
    for idx in worst_idx:
        row = raw_df.iloc[idx]
        actual = y_true.iloc[idx]
        pred = y_pred[idx]
        pct = pct_errors[idx]
        print(f"  ${actual:>9,.0f} ${pred:>9,.0f} {pct:>6.1f}% "
              f"{int(row['OverallQual']):>5d} {int(row['GrLivArea']):>6d} "
              f"{int(row['YearBuilt']):>6d} {row['Neighborhood']:>10s} "
              f"{row['SaleCondition']:>10s} {row['BldgType']:>10s}")

    # Look for patterns in worst predictions
    worst_rows = raw_df.iloc[worst_idx]
    print(f"\n  Patterns in worst predictions:")

    # Sale condition
    sale_cond = worst_rows['SaleCondition'].value_counts()
    print(f"    SaleCondition: {dict(sale_cond)}")

    # Overvalued vs undervalued
    over = (residuals[worst_idx] > 0).sum()
    under = (residuals[worst_idx] < 0).sum()
    print(f"    Overvalued (actual > pred): {over}")
    print(f"    Undervalued (actual < pred): {under}")

    # Quality distribution
    print(f"    OverallQual mean: {worst_rows['OverallQual'].mean():.1f} "
          f"(dataset mean: {raw_df['OverallQual'].mean():.1f})")

    # Age
    print(f"    YearBuilt mean: {worst_rows['YearBuilt'].mean():.0f} "
          f"(dataset mean: {raw_df['YearBuilt'].mean():.0f})")

    return worst_idx


def correlation_with_residuals(y_true, y_pred, X, top_n=15):
    """Find features most correlated with residual magnitude."""
    residuals = y_true.values - y_pred
    abs_residuals = np.abs(residuals)

    # Correlate each feature with absolute residual
    corrs = {}
    for col in X.columns:
        if X[col].nunique() > 1:
            corrs[col] = np.corrcoef(X[col].values, abs_residuals)[0, 1]

    corr_series = pd.Series(corrs).sort_values(key=abs, ascending=False)

    print(f"\n  Features most correlated with |residual|:")
    print(f"  (high correlation = model error is systematic, not random)")
    print(f"  {'Feature':40s} {'Corr with |error|':>18s}")
    print(f"  {'-' * 62}")
    for feat, corr in corr_series.head(top_n).items():
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:40s} {sign}{abs(corr):.3f}")


def main():
    tee = Tee(f"{BASE}/results/models/05_residual_analysis.txt")
    sys.stdout = tee

    print("House Prices: Step 5 — Residual Analysis")
    print("=" * 60)
    print()

    X, y, raw = load_data()

    # Best model from Step 4: ElasticNet (alpha=0.0285, l1_ratio=0.1)
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ('scaler', StandardScaler()),
            ('en', ElasticNet(alpha=0.02848, l1_ratio=0.1, max_iter=10000))
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )

    print(f"Model: ElasticNet (alpha=0.02848, l1_ratio=0.1)")
    print(f"Train: {X.shape}")
    print(f"Collecting OOF predictions (5-fold)...")

    oof = oof_predictions(model, X, y)
    overall_rmsle = rmsle(y, oof)
    print(f"OOF RMSLE: {overall_rmsle:.5f}")
    print()

    # ============================================================
    # 1. RESIDUALS VS PREDICTED VALUE
    # ============================================================
    print("=" * 60)
    print("1. RESIDUALS VS PREDICTED VALUE")
    print("   Should be random scatter. Patterns = model is biased.")
    print("=" * 60)
    residuals_vs_predicted(y, oof)
    print()

    # ============================================================
    # 2. RESIDUALS VS KEY FEATURES
    # ============================================================
    print("=" * 60)
    print("2. RESIDUALS VS KEY FEATURES")
    print("   Patterns here suggest the model is missing a relationship.")
    print("=" * 60)

    # Size
    residuals_vs_feature(y, oof, raw['GrLivArea'], 'GrLivArea')

    # Quality
    residuals_vs_feature(y, oof, raw['OverallQual'], 'OverallQual',
                         is_categorical=True)

    # Age
    residuals_vs_feature(y, oof, raw['YearBuilt'], 'YearBuilt')

    # Neighborhood (top neighborhoods by count)
    top_nbhd = raw['Neighborhood'].value_counts().head(8).index
    nbhd_mask = raw['Neighborhood'].isin(top_nbhd)
    residuals_vs_feature(
        y[nbhd_mask], oof[nbhd_mask],
        raw.loc[nbhd_mask, 'Neighborhood'], 'Neighborhood (top 8)',
        is_categorical=True)

    # Total SF (basement + above ground)
    total_sf = raw['TotalBsmtSF'].fillna(0) + raw['GrLivArea']
    residuals_vs_feature(y, oof, total_sf, 'TotalSF (Bsmt+AboveGrd)')

    # Sale condition
    residuals_vs_feature(y, oof, raw['SaleCondition'], 'SaleCondition',
                         is_categorical=True)

    # Remodel recency
    remodel_age = raw['YrSold'] - raw['YearRemodAdd']
    residuals_vs_feature(y, oof, remodel_age, 'RemodelAge (YrSold-YearRemodAdd)')
    print()

    # ============================================================
    # 3. FEATURES CORRELATED WITH ERROR MAGNITUDE
    # ============================================================
    print("=" * 60)
    print("3. FEATURES CORRELATED WITH ERROR MAGNITUDE")
    print("   These features predict WHERE the model goes wrong.")
    print("=" * 60)
    correlation_with_residuals(y, oof, X)
    print()

    # ============================================================
    # 4. WORST PREDICTIONS
    # ============================================================
    print("=" * 60)
    print("4. WORST PREDICTIONS")
    print("   What makes these houses hard to predict?")
    print("=" * 60)
    worst_idx = worst_predictions(y, oof, raw, n=15)
    print()

    # ============================================================
    # 5. OUTLIER CHECK
    # ============================================================
    print("=" * 60)
    print("5. OUTLIER CHECK")
    print("   Houses where |error| > 50% — are they data problems")
    print("   or genuinely unusual properties?")
    print("=" * 60)
    pct_errors = np.abs(y.values - oof) / y.values * 100
    outlier_mask = pct_errors > 50
    n_outliers = outlier_mask.sum()
    print(f"\n  Houses with >50% prediction error: {n_outliers} / {len(y)}")

    if n_outliers > 0:
        outlier_idx = np.where(outlier_mask)[0]
        for idx in outlier_idx:
            row = raw.iloc[idx]
            actual = y.iloc[idx]
            pred = oof[idx]
            err_pct = pct_errors[idx]
            direction = "OVER" if pred > actual else "UNDER"
            print(f"\n  Id={int(row['Id'])} — {direction}predicted by {err_pct:.0f}%")
            print(f"    Actual: ${actual:,.0f}  Predicted: ${pred:,.0f}")
            print(f"    OverallQual={int(row['OverallQual'])}, "
                  f"GrLivArea={int(row['GrLivArea'])}, "
                  f"YearBuilt={int(row['YearBuilt'])}")
            print(f"    Neighborhood={row['Neighborhood']}, "
                  f"SaleCondition={row['SaleCondition']}, "
                  f"BldgType={row['BldgType']}")
            print(f"    TotalBsmtSF={int(row['TotalBsmtSF']) if pd.notna(row['TotalBsmtSF']) else 'NA'}, "
                  f"GarageCars={int(row['GarageCars']) if pd.notna(row['GarageCars']) else 'NA'}, "
                  f"LotArea={int(row['LotArea'])}")
    print()

    # ============================================================
    # 6. RESIDUAL PLOTS
    # ============================================================
    print("=" * 60)
    print("6. RESIDUAL PLOTS")
    print("=" * 60)

    residuals = y.values - oof
    pct_errors = np.abs(residuals) / y.values * 100
    total_sf = raw['TotalBsmtSF'].fillna(0) + raw['GrLivArea']
    house_age = raw['YrSold'] - raw['YearBuilt']

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('Step 5: Residual Analysis — ElasticNet (OOF predictions)',
                 fontsize=14, fontweight='bold')

    # Panel 1: Predicted vs Actual
    ax = axes[0, 0]
    ax.scatter(oof, y.values, alpha=0.3, s=10, c='steelblue')
    lims = [0, max(y.max(), oof.max()) * 1.05]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Predicted Price ($)')
    ax.set_ylabel('Actual Price ($)')
    ax.set_title('Predicted vs Actual')
    ax.legend()

    # Panel 2: Residuals vs Predicted (heteroscedasticity check)
    ax = axes[0, 1]
    ax.scatter(oof, residuals, alpha=0.3, s=10, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted Price ($)')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residuals vs Predicted (funnel = heteroscedastic)')

    # Panel 3: Residuals vs GrLivArea
    ax = axes[1, 0]
    ax.scatter(raw['GrLivArea'], residuals, alpha=0.3, s=10, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('GrLivArea (sq ft)')
    ax.set_ylabel('Residual ($)')
    ax.set_title('Residuals vs Living Area')

    # Panel 4: Residuals vs OverallQual
    ax = axes[1, 1]
    qual_groups = raw['OverallQual'].values
    for q in sorted(raw['OverallQual'].unique()):
        mask = qual_groups == q
        ax.scatter(np.full(mask.sum(), q) + np.random.normal(0, 0.1, mask.sum()),
                   residuals[mask], alpha=0.3, s=10, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('OverallQual')
    ax.set_ylabel('Residual ($)')
    ax.set_title('Residuals vs Overall Quality')

    # Panel 5: % Error by SaleCondition
    ax = axes[2, 0]
    conditions = raw['SaleCondition'].values
    unique_conds = sorted(raw['SaleCondition'].unique())
    cond_data = [pct_errors[conditions == c] for c in unique_conds]
    bp = ax.boxplot(cond_data, labels=unique_conds, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.5)
    ax.set_ylabel('Absolute % Error')
    ax.set_title('Error Distribution by SaleCondition')
    ax.tick_params(axis='x', rotation=30)

    # Panel 6: Residuals vs House Age
    ax = axes[2, 1]
    ax.scatter(house_age, residuals, alpha=0.3, s=10, c='steelblue')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('House Age at Sale (years)')
    ax.set_ylabel('Residual ($)')
    ax.set_title('Residuals vs House Age')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = f"{BASE}/results/analysis/05_residual_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {plot_path}")
    print()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("=" * 60)
    print("SUMMARY: Patterns to address in Step 6")
    print("=" * 60)
    print("""
  Review the residual patterns above to identify:

  1. Systematic bias by price range (Section 1)
     - Does the model consistently over/underpredict for certain price ranges?

  2. Missing relationships (Section 2)
     - Are there features where errors aren't random?
     - E.g., if old houses have large positive residuals, maybe we need
       an age-related feature (YrSold - YearBuilt)

  3. Error-correlated features (Section 3)
     - Features that predict error magnitude suggest the model is
       missing an interaction or nonlinearity involving that feature

  4. Hard-to-predict houses (Section 4)
     - Abnormal sales, partial sales, unusual properties
     - Some may be true outliers we should consider removing

  5. True outliers (Section 5)
     - >50% error houses may be data problems or genuinely unusual
     - Consider whether removing them from training would help
""")

    tee.close()


if __name__ == "__main__":
    main()
