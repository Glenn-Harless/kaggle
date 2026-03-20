"""
Disaster Tweets: Step 2 — Bag-of-Words Baselines

Three baselines using the simplest text representations:
1. Keyword-only — how far can metadata get us?
2. CountVectorizer + Logistic Regression — classic bag-of-words
3. TF-IDF + Logistic Regression — weighted bag-of-words

Each gets: 5-fold stratified CV F1 score.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from shared.evaluate import Tee, classification_cv, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/02_bow_baselines.txt")
sys.stdout = tee

print("Disaster Tweets: Step 2 — Bag-of-Words Baselines")
print("=" * 60)


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")

y = train["target"]
print(f"\n  Train: {len(train)} tweets, {y.sum()} disaster ({y.mean():.1%})")
print(f"  Test:  {len(test)} tweets")

# Shared CV splitter — same folds for all models
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# BASELINE 1: KEYWORD-ONLY
# ============================================================
#
# Predict disaster using ONLY the keyword column — ignoring text entirely.
# This is the "how far can metadata get us?" test.
#
# Approach: for each keyword, compute the training-fold disaster rate.
# If rate > 0.5, predict disaster. Tweets without a keyword get the
# overall training-fold base rate.
# ============================================================

print("\n\n" + "=" * 60)
print("BASELINE 1: KEYWORD-ONLY")
print("=" * 60)
print("""
  Strategy: For each keyword, compute its disaster rate from the training
  fold. If rate > 0.5, predict disaster. Missing keywords get base rate.

  This is a 'lookup table' classifier — no ML model at all. It tells us
  how much signal lives in the keyword column vs the text content.
""")

train["keyword_clean"] = train["keyword"].fillna("__MISSING__").str.replace(
    "%20", " ", regex=False
)
test["keyword_clean"] = test["keyword"].fillna("__MISSING__").str.replace(
    "%20", " ", regex=False
)

kw_f1_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    # Compute per-keyword disaster rate from training fold
    fold_train = train.iloc[train_idx]
    kw_rates = fold_train.groupby("keyword_clean")["target"].mean()
    base_rate = fold_train["target"].mean()

    # Predict on validation fold
    fold_val = train.iloc[val_idx]
    val_rates = fold_val["keyword_clean"].map(kw_rates).fillna(base_rate)
    val_preds = (val_rates > 0.5).astype(int)

    fold_f1 = f1_score(y.iloc[val_idx], val_preds)
    kw_f1_scores.append(fold_f1)
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

kw_f1_scores = np.array(kw_f1_scores)
print()
report_f1_cv(kw_f1_scores, "Keyword-only")


# ============================================================
# BASELINE 2: CountVectorizer + LOGISTIC REGRESSION
# ============================================================
#
# The classic bag-of-words pipeline:
# 1. Tokenize: split text into individual words
# 2. Build vocabulary: assign each unique word an index (column)
# 3. Count: for each tweet, count how many times each word appears
# 4. Result: a sparse matrix where each row is a tweet and each
#    column is a word. Most entries are 0 (sparse).
#
# Example: vocabulary = [fire, the, earthquake, love, ...]
#   "Forest fire near La Ronge" -> [1, 0, 0, 0, ...] (sparse vector)
#
# Logistic Regression then learns a weight for each word:
#   P(disaster) = sigmoid(w_fire * count_fire + w_the * count_the + ...)
#   Words like "earthquake" get positive weights.
#   Words like "love" get negative weights.
#
# This is called "bag of words" because word ORDER is completely lost.
# "Man bites dog" = "Dog bites man" = {man:1, bites:1, dog:1}
# ============================================================

print("\n\n" + "=" * 60)
print("BASELINE 2: CountVectorizer + LOGISTIC REGRESSION")
print("=" * 60)
print("""
  How it works:
  1. Tokenize each tweet into words
  2. Count word occurrences → sparse feature matrix (N_tweets x N_vocab)
  3. Logistic Regression learns a weight per word

  Key hyperparameters:
  - max_features: limit vocabulary size (reduces noise from rare words)
  - C=1.0: regularization strength (default, we'll tune later)
""")

count_pipe = Pipeline([
    ("vectorizer", CountVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

count_scores = classification_cv(count_pipe, train["text"], y)
report_f1_cv(count_scores, "CountVec + LogReg")

# Show what the vectorizer learned
count_pipe.fit(train["text"], y)
vocab = count_pipe.named_steps["vectorizer"].get_feature_names_out()
coefs = count_pipe.named_steps["clf"].coef_[0]

top_disaster_words = pd.Series(coefs, index=vocab).nlargest(15)
top_nondisaster_words = pd.Series(coefs, index=vocab).nsmallest(15)

print(f"\n  Vocabulary size: {len(vocab):,}")
print(f"\n  Top 15 words with highest DISASTER weight (positive coefficients):")
print(f"  {'Word':20s} {'Coefficient':>12s}")
print(f"  {'-' * 35}")
for word, coef in top_disaster_words.items():
    print(f"  {word:20s} {coef:>12.4f}")

print(f"\n  Top 15 words with highest NON-DISASTER weight (negative coefficients):")
print(f"  {'Word':20s} {'Coefficient':>12s}")
print(f"  {'-' * 35}")
for word, coef in top_nondisaster_words.items():
    print(f"  {word:20s} {coef:>12.4f}")


# ============================================================
# BASELINE 3: TF-IDF + LOGISTIC REGRESSION
# ============================================================
#
# TF-IDF = Term Frequency * Inverse Document Frequency
#
# The problem with raw counts: common words like "the," "is," "a"
# dominate the feature matrix but carry zero information about whether
# a tweet is about a disaster. They appear in almost every tweet.
#
# TF-IDF fixes this with a two-part weighting:
#
# TF (Term Frequency) = count of word in this tweet
#   → Same as CountVectorizer
#
# IDF (Inverse Document Frequency) = log(N / n_docs_containing_word)
#   → "earthquake" appears in 50/7613 tweets → IDF = log(7613/50) = 5.03
#   → "the" appears in 6000/7613 tweets → IDF = log(7613/6000) = 0.24
#
# TF-IDF = TF * IDF
#   → "earthquake" gets 1 * 5.03 = 5.03 (high weight — rare, distinctive)
#   → "the" gets 1 * 0.24 = 0.24 (low weight — common, uninformative)
#
# The intuition: TF-IDF automatically discovers which words are
# "interesting" (appear in some tweets but not most) vs "boring"
# (appear in almost every tweet). This is like automatic feature
# selection for words.
#
# In practice, TF-IDF almost always beats raw counts because it
# effectively removes the noise of stopwords without explicitly
# listing them.
# ============================================================

print("\n\n" + "=" * 60)
print("BASELINE 3: TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)
print("""
  TF-IDF = Term Frequency * Inverse Document Frequency

  Why better than counts:
  - "earthquake" in 50 tweets → high IDF → high weight (distinctive)
  - "the" in 6000 tweets → low IDF → low weight (uninformative)

  TF-IDF automatically downweights common words — like built-in
  feature selection for free.
""")

tfidf_pipe = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

tfidf_scores = classification_cv(tfidf_pipe, train["text"], y)
report_f1_cv(tfidf_scores, "TF-IDF + LogReg")

# Show what the vectorizer learned
tfidf_pipe.fit(train["text"], y)
vocab_tfidf = tfidf_pipe.named_steps["vectorizer"].get_feature_names_out()
coefs_tfidf = tfidf_pipe.named_steps["clf"].coef_[0]

top_disaster_tfidf = pd.Series(coefs_tfidf, index=vocab_tfidf).nlargest(15)
top_nondisaster_tfidf = pd.Series(coefs_tfidf, index=vocab_tfidf).nsmallest(15)

print(f"\n  Top 15 TF-IDF disaster words:")
print(f"  {'Word':20s} {'Coefficient':>12s}")
print(f"  {'-' * 35}")
for word, coef in top_disaster_tfidf.items():
    print(f"  {word:20s} {coef:>12.4f}")

print(f"\n  Top 15 TF-IDF non-disaster words:")
print(f"  {'Word':20s} {'Coefficient':>12s}")
print(f"  {'-' * 35}")
for word, coef in top_nondisaster_tfidf.items():
    print(f"  {word:20s} {coef:>12.4f}")


# ============================================================
# COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("COMPARISON — ALL BASELINES")
print("=" * 60)

results = {
    "Keyword-only": kw_f1_scores,
    "CountVec + LogReg": count_scores,
    "TF-IDF + LogReg": tfidf_scores,
}

print(f"\n  {'Model':25s} {'Mean F1':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print(f"  {'-' * 68}")
for name, scores in results.items():
    print(f"  {name:25s} {scores.mean():>10.5f} {scores.std():>10.5f} "
          f"{scores.min():>10.5f} {scores.max():>10.5f}")

# Which is best?
best_name = max(results, key=lambda k: results[k].mean())
best_score = results[best_name].mean()
print(f"\n  Best baseline: {best_name} (F1 = {best_score:.5f})")

# Delta from keyword-only to best text model
kw_mean = kw_f1_scores.mean()
text_best = max(count_scores.mean(), tfidf_scores.mean())
print(f"\n  Text model improvement over keyword-only: "
      f"{text_best - kw_mean:+.5f} F1 ({kw_mean:.5f} → {text_best:.5f})")

tfidf_delta = tfidf_scores.mean() - count_scores.mean()
print(f"  TF-IDF improvement over raw counts: {tfidf_delta:+.5f} F1")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("\n\n" + "=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print("""
  1. KEYWORD-ONLY baseline provides a useful floor. The keyword column
     carries real signal, but can't capture the nuance in tweet text.

  2. CountVectorizer + LogReg: each tweet becomes a sparse vector of word
     counts. LogReg learns a weight per word. Simple but effective because
     many words are strongly associated with disaster/non-disaster.

  3. TF-IDF weighting should improve over raw counts by downweighting
     ubiquitous words ("the", "is") and upweighting distinctive ones
     ("earthquake", "wildfire").

  4. All three methods treat text as an unordered set of words.
     "Man bites dog" = "Dog bites man". Later phases will add:
     - Text cleaning (remove URLs, punctuation) in Step 3
     - Engineered features (text length, hashtags) in Step 4
     - Word order via embeddings in Step 5
     - Context-dependent meaning via transformers in Step 7
""")

tee.close()
print("Done. Results saved to results/models/02_bow_baselines.txt")
