"""
Disaster Tweets: Step 4 — Feature Engineering + Model Comparison

Combine TF-IDF text features with hand-engineered structural features,
then compare Logistic Regression, SVM, and XGBoost.

Structural features capture HOW a tweet is written (length, URLs, caps)
rather than WHAT words it contains (which TF-IDF already handles).
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from shared.evaluate import Tee, classification_cv, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/04_feature_engineering.txt")
sys.stdout = tee

print("Disaster Tweets: Step 4 — Feature Engineering + Model Comparison")
print("=" * 60)


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
y = train["target"]

print(f"\n  Train: {len(train)} tweets")
print(f"  Current best (TF-IDF + LogReg, raw text): F1 = 0.75478")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)
print("""
  These features capture structural patterns that TF-IDF misses:
  - TF-IDF knows WHICH words appear
  - Engineered features know HOW the tweet is written
""")


def engineer_features(df):
    """Extract structural features from tweet text and metadata."""
    feat = pd.DataFrame(index=df.index)

    text = df["text"]

    # Length features
    feat["char_len"] = text.str.len()
    feat["word_count"] = text.str.split().str.len()
    feat["avg_word_len"] = feat["char_len"] / feat["word_count"].clip(lower=1)

    # Structural features
    feat["n_urls"] = text.str.count(r"https?://\S+")
    feat["n_hashtags"] = text.str.count(r"#\w+")
    feat["n_mentions"] = text.str.count(r"@\w+")
    feat["has_url"] = (feat["n_urls"] > 0).astype(int)
    feat["has_hashtag"] = (feat["n_hashtags"] > 0).astype(int)
    feat["has_mention"] = (feat["n_mentions"] > 0).astype(int)

    # Punctuation features
    feat["n_exclamation"] = text.str.count(r"!")
    feat["n_question"] = text.str.count(r"\?")
    feat["n_periods"] = text.str.count(r"\.")

    # Capitalization features
    feat["n_caps_words"] = text.apply(
        lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1)
    )
    feat["caps_ratio"] = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    # Number presence (addresses, death counts, dates)
    feat["has_number"] = text.str.contains(r"\d", regex=True).astype(int)
    feat["n_numbers"] = text.str.count(r"\d+")

    # Keyword feature
    feat["has_keyword"] = df["keyword"].notna().astype(int)

    return feat


train_feat = engineer_features(train)
test_feat = engineer_features(test)

print(f"  Engineered {train_feat.shape[1]} structural features:")
for col in train_feat.columns:
    print(f"    {col:20s}  mean={train_feat[col].mean():.3f}  "
          f"std={train_feat[col].std():.3f}")


# ============================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================

print("\n\n" + "=" * 60)
print("FEATURE CORRELATION WITH TARGET")
print("=" * 60)

correlations = train_feat.corrwith(y).sort_values(ascending=False)
print(f"\n  {'Feature':20s} {'Correlation':>12s}")
print(f"  {'-' * 35}")
for feat_name, corr in correlations.items():
    marker = " ***" if abs(corr) > 0.10 else ""
    print(f"  {feat_name:20s} {corr:>12.4f}{marker}")


# ============================================================
# EXPERIMENT 1: TF-IDF ONLY (baseline)
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 1: TF-IDF ONLY (baseline)")
print("=" * 60)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We need manual CV loop because we're combining sparse TF-IDF with dense features
tfidf = TfidfVectorizer(max_features=10000)

# Baseline: TF-IDF only
baseline_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    X_tfidf_tr = tfidf.fit_transform(train["text"].iloc[train_idx])
    X_tfidf_val = tfidf.transform(train["text"].iloc[val_idx])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf_tr, y.iloc[train_idx])
    preds = model.predict(X_tfidf_val)
    baseline_scores.append(f1_score(y.iloc[val_idx], preds))

baseline_scores = np.array(baseline_scores)
report_f1_cv(baseline_scores, "TF-IDF only")


# ============================================================
# EXPERIMENT 2: ENGINEERED FEATURES ONLY
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 2: ENGINEERED FEATURES ONLY (no text content)")
print("=" * 60)

feat_only_scores = []
scaler = StandardScaler()
for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    X_tr = scaler.fit_transform(train_feat.iloc[train_idx])
    X_val = scaler.transform(train_feat.iloc[val_idx])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    feat_only_scores.append(f1_score(y.iloc[val_idx], preds))

feat_only_scores = np.array(feat_only_scores)
report_f1_cv(feat_only_scores, "Features only")


# ============================================================
# EXPERIMENT 3: TF-IDF + ENGINEERED FEATURES
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 3: TF-IDF + ENGINEERED FEATURES (combined)")
print("=" * 60)
print("""
  We horizontally stack the sparse TF-IDF matrix (10,000 columns)
  with the dense engineered features (17 columns).
  The model sees both WHAT words appear AND HOW the tweet is structured.
""")


def combined_cv(model_class, model_params, label=""):
    """Run CV with combined TF-IDF + engineered features."""
    scores = []
    for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
        # TF-IDF features (sparse)
        X_tfidf_tr = tfidf.fit_transform(train["text"].iloc[train_idx])
        X_tfidf_val = tfidf.transform(train["text"].iloc[val_idx])

        # Engineered features (dense → sparse for hstack)
        sc = StandardScaler()
        feat_tr = sc.fit_transform(train_feat.iloc[train_idx])
        feat_val = sc.transform(train_feat.iloc[val_idx])

        # Combine: [TF-IDF columns | engineered columns]
        X_tr = hstack([X_tfidf_tr, csr_matrix(feat_tr)])
        X_val = hstack([X_tfidf_val, csr_matrix(feat_val)])

        model = model_class(**model_params)
        model.fit(X_tr, y.iloc[train_idx])
        preds = model.predict(X_val)
        scores.append(f1_score(y.iloc[val_idx], preds))

    scores = np.array(scores)
    report_f1_cv(scores, label)
    return scores


# 3a: Logistic Regression
print("\n  --- Logistic Regression ---")
lr_combined_scores = combined_cv(
    LogisticRegression,
    {"max_iter": 1000, "random_state": 42},
    "TF-IDF + Features + LogReg"
)

# 3b: LinearSVC (SVM)
print("\n  --- Linear SVM ---")
svm_combined_scores = combined_cv(
    LinearSVC,
    {"max_iter": 2000, "random_state": 42},
    "TF-IDF + Features + SVM"
)

# 3c: XGBoost
print("\n  --- XGBoost ---")
xgb_combined_scores = combined_cv(
    XGBClassifier,
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
     "random_state": 42, "eval_metric": "logloss", "verbosity": 0},
    "TF-IDF + Features + XGBoost"
)


# ============================================================
# EXPERIMENT 4: KEYWORD AS FEATURE
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 4: TF-IDF + FEATURES + KEYWORD (target-encoded)")
print("=" * 60)
print("""
  Target encoding: replace each keyword with its training-fold disaster
  rate. This encodes the keyword's predictive power as a single number.
  Must be computed per-fold to avoid leakage.
""")

train["keyword_clean"] = train["keyword"].fillna("__MISSING__").str.replace(
    "%20", " ", regex=False
)

kw_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    # Target-encode keyword from training fold only
    fold_train = train.iloc[train_idx]
    kw_rates = fold_train.groupby("keyword_clean")["target"].mean()
    global_rate = fold_train["target"].mean()

    kw_tr = train["keyword_clean"].iloc[train_idx].map(kw_rates).fillna(global_rate)
    kw_val = train["keyword_clean"].iloc[val_idx].map(kw_rates).fillna(global_rate)

    # TF-IDF
    X_tfidf_tr = tfidf.fit_transform(train["text"].iloc[train_idx])
    X_tfidf_val = tfidf.transform(train["text"].iloc[val_idx])

    # Engineered features + keyword rate
    sc = StandardScaler()
    feat_plus_kw_tr = train_feat.iloc[train_idx].copy()
    feat_plus_kw_tr["kw_target_rate"] = kw_tr.values
    feat_plus_kw_val = train_feat.iloc[val_idx].copy()
    feat_plus_kw_val["kw_target_rate"] = kw_val.values

    feat_tr = sc.fit_transform(feat_plus_kw_tr)
    feat_val = sc.transform(feat_plus_kw_val)

    # Combine
    X_tr = hstack([X_tfidf_tr, csr_matrix(feat_tr)])
    X_val = hstack([X_tfidf_val, csr_matrix(feat_val)])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    kw_scores.append(f1_score(y.iloc[val_idx], preds))

kw_scores = np.array(kw_scores)
report_f1_cv(kw_scores, "TF-IDF + Features + Keyword + LogReg")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL COMPARISON — ALL EXPERIMENTS")
print("=" * 60)

all_results = {
    "TF-IDF only (baseline)": baseline_scores,
    "Features only (no text)": feat_only_scores,
    "TF-IDF + Features + LogReg": lr_combined_scores,
    "TF-IDF + Features + SVM": svm_combined_scores,
    "TF-IDF + Features + XGBoost": xgb_combined_scores,
    "TF-IDF + Feat + KW + LogReg": kw_scores,
}

print(f"\n  {'Model':35s} {'Mean F1':>10s} {'Std':>10s} {'Delta':>10s}")
print(f"  {'-' * 68}")
for name, scores in all_results.items():
    delta = scores.mean() - baseline_scores.mean()
    print(f"  {name:35s} {scores.mean():>10.5f} {scores.std():>10.5f} "
          f"{delta:>+10.5f}")

best_name = max(all_results, key=lambda k: all_results[k].mean())
best_score = all_results[best_name].mean()
print(f"\n  Best: {best_name} (F1 = {best_score:.5f})")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("\n\n" + "=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print("""
  1. ENGINEERED FEATURES ALONE are much weaker than TF-IDF — knowing HOW
     a tweet is written matters less than knowing WHAT words it contains.

  2. COMBINING TF-IDF + engineered features: do the structural features
     add signal beyond what TF-IDF already captures?

  3. SVM vs LogReg vs XGBoost: SVMs often excel on high-dimensional sparse
     text features (large margin in high-D space). XGBoost works differently
     — it builds trees on individual features, which is awkward for sparse
     TF-IDF where most features are zero.

  4. KEYWORD TARGET ENCODING adds the keyword's predictive power as a single
     dense feature. Since we know keywords carry real signal (F1=0.673 alone),
     encoding them properly should help.

  5. NEXT STEP: We've now exhausted what bag-of-words can do. Steps 5-6
     will replace word counts with word EMBEDDINGS — dense vectors that
     capture meaning. "fire" and "blaze" will be nearby in vector space,
     letting the model generalize beyond exact word matches.
""")

tee.close()
print("Done. Results saved to results/models/04_feature_engineering.txt")
