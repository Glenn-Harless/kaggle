"""
Disaster Tweets: Step 5 — Pre-trained Word Embeddings (GloVe)

Replace bag-of-words with dense vectors from GloVe (Twitter 27B, 200d).
Each word → a 200-dimensional vector pre-trained on 27 billion tweet tokens.
Each tweet → average of its word vectors → one 200-dim vector → classifier.

This is a fundamentally different representation from TF-IDF:
- TF-IDF: 10,000 sparse features, each is "does this exact word appear?"
- GloVe:  200 dense features, each captures some aspect of meaning

Key advantage: generalization. GloVe knows "wildfire" and "bushfire" are
similar even if the training data only contains one of them.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from shared.evaluate import Tee, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"
GLOVE_PATH = "/Users/glennharless/dev-brain/kaggle/data/glove.twitter.27B.200d.txt"

tee = Tee(f"{BASE}/results/models/05_glove_embeddings.txt")
sys.stdout = tee

print("Disaster Tweets: Step 5 — Pre-trained Word Embeddings (GloVe)")
print("=" * 60)


# ============================================================
# LOAD GLOVE VECTORS
# ============================================================

print("\n\nLoading GloVe vectors (Twitter 27B, 200d)...")

embeddings = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        vec = np.array(parts[1:], dtype=np.float32)
        embeddings[word] = vec

print(f"  Loaded {len(embeddings):,} word vectors")
print(f"  Embedding dimension: {len(next(iter(embeddings.values())))}")

# Quick sanity check: similar words should have high cosine similarity
def cosine_sim(w1, w2):
    if w1 not in embeddings or w2 not in embeddings:
        return float("nan")
    v1, v2 = embeddings[w1], embeddings[w2]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\n  Similarity sanity check:")
pairs = [
    ("fire", "blaze"), ("fire", "earthquake"), ("fire", "ebay"),
    ("disaster", "catastrophe"), ("disaster", "happy"),
    ("flood", "flooding"), ("killed", "dead"),
]
for w1, w2 in pairs:
    sim = cosine_sim(w1, w2)
    print(f"    sim({w1}, {w2}) = {sim:.3f}")


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
y = train["target"]

print(f"\n  Train: {len(train)} tweets")
print(f"  Current best (TF-IDF + LogReg): F1 = 0.755")


# ============================================================
# TWEET → VECTOR CONVERSION
# ============================================================

print("\n\n" + "=" * 60)
print("TWEET TO VECTOR CONVERSION")
print("=" * 60)
print("""
  Strategy: represent each tweet as the AVERAGE of its word vectors.

  "Forest fire near La Ronge"
  → vec("forest") + vec("fire") + vec("near") + vec("la") + vec("ronge")
  → divide by 5
  → one 200-dimensional vector representing the whole tweet

  Words not in GloVe vocabulary → skipped (out-of-vocabulary / OOV).
""")


def tokenize(text):
    """Simple tokenizer: lowercase, extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tweet_to_vec(text, dim=200):
    """Convert a tweet to a vector by averaging its word embeddings."""
    tokens = tokenize(text)
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


# Convert all tweets
X_glove_train = np.array([tweet_to_vec(t) for t in train["text"]])
X_glove_test = np.array([tweet_to_vec(t) for t in test["text"]])

print(f"  Train matrix shape: {X_glove_train.shape}")
print(f"  Test matrix shape:  {X_glove_test.shape}")

# OOV analysis
total_tokens = 0
oov_tokens = 0
oov_examples = set()
for text in train["text"]:
    tokens = tokenize(text)
    for t in tokens:
        total_tokens += 1
        if t not in embeddings:
            oov_tokens += 1
            if len(oov_examples) < 20:
                oov_examples.add(t)

print(f"\n  Total tokens: {total_tokens:,}")
print(f"  Out-of-vocabulary: {oov_tokens:,} ({oov_tokens/total_tokens:.1%})")
print(f"  OOV examples: {sorted(oov_examples)[:15]}")


# ============================================================
# EXPERIMENT 1: GLOVE AVERAGE + LOGISTIC REGRESSION
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 1: GloVe Average + LogReg")
print("=" * 60)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

glove_scores = []
scaler = StandardScaler()
for fold, (train_idx, val_idx) in enumerate(CV.split(X_glove_train, y)):
    X_tr = scaler.fit_transform(X_glove_train[train_idx])
    X_val = scaler.transform(X_glove_train[val_idx])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    fold_f1 = f1_score(y.iloc[val_idx], preds)
    glove_scores.append(fold_f1)
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

glove_scores = np.array(glove_scores)
print()
report_f1_cv(glove_scores, "GloVe Avg + LogReg")


# ============================================================
# EXPERIMENT 2: TF-IDF-WEIGHTED GLOVE AVERAGE
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 2: TF-IDF-Weighted GloVe Average")
print("=" * 60)
print("""
  Instead of simple averaging, weight each word vector by its TF-IDF
  score. Important words (high TF-IDF) contribute more to the tweet
  vector than common words (low TF-IDF).
""")

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit TF-IDF to get per-word weights
tfidf = TfidfVectorizer(max_features=50000)
tfidf.fit(train["text"])
tfidf_vocab = tfidf.vocabulary_
idf_values = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))


def tweet_to_vec_weighted(text, dim=200):
    """Average word vectors weighted by IDF scores."""
    tokens = tokenize(text)
    vecs = []
    weights = []
    for w in tokens:
        if w in embeddings:
            idf = idf_values.get(w, 1.0)
            vecs.append(embeddings[w] * idf)
            weights.append(idf)
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.sum(vecs, axis=0) / sum(weights)


X_glove_weighted_train = np.array([tweet_to_vec_weighted(t) for t in train["text"]])

weighted_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(X_glove_weighted_train, y)):
    X_tr = scaler.fit_transform(X_glove_weighted_train[train_idx])
    X_val = scaler.transform(X_glove_weighted_train[val_idx])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    fold_f1 = f1_score(y.iloc[val_idx], preds)
    weighted_scores.append(fold_f1)
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

weighted_scores = np.array(weighted_scores)
print()
report_f1_cv(weighted_scores, "GloVe TF-IDF-Weighted + LogReg")


# ============================================================
# EXPERIMENT 3: GLOVE + TF-IDF CONCATENATION
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 3: GloVe + TF-IDF Concatenated")
print("=" * 60)
print("""
  Best of both worlds? Concatenate the 200-dim GloVe vector with the
  10,000-dim TF-IDF vector. The model can use both word meaning (GloVe)
  and exact word presence (TF-IDF).
""")

from scipy.sparse import hstack, csr_matrix

tfidf_10k = TfidfVectorizer(max_features=10000)

concat_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(X_glove_train, y)):
    # TF-IDF part
    X_tfidf_tr = tfidf_10k.fit_transform(train["text"].iloc[train_idx])
    X_tfidf_val = tfidf_10k.transform(train["text"].iloc[val_idx])

    # GloVe part (scaled)
    glove_tr = scaler.fit_transform(X_glove_train[train_idx])
    glove_val = scaler.transform(X_glove_train[val_idx])

    # Concatenate
    X_tr = hstack([X_tfidf_tr, csr_matrix(glove_tr)])
    X_val = hstack([X_tfidf_val, csr_matrix(glove_val)])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    fold_f1 = f1_score(y.iloc[val_idx], preds)
    concat_scores.append(fold_f1)
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

concat_scores = np.array(concat_scores)
print()
report_f1_cv(concat_scores, "GloVe + TF-IDF concat + LogReg")


# ============================================================
# EXPERIMENT 4: GLOVE + SVM
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 4: GloVe Average + SVM")
print("=" * 60)

from sklearn.svm import LinearSVC

svm_scores = []
for fold, (train_idx, val_idx) in enumerate(CV.split(X_glove_train, y)):
    X_tr = scaler.fit_transform(X_glove_train[train_idx])
    X_val = scaler.transform(X_glove_train[val_idx])

    model = LinearSVC(max_iter=2000, random_state=42)
    model.fit(X_tr, y.iloc[train_idx])
    preds = model.predict(X_val)
    fold_f1 = f1_score(y.iloc[val_idx], preds)
    svm_scores.append(fold_f1)
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

svm_scores = np.array(svm_scores)
print()
report_f1_cv(svm_scores, "GloVe Avg + SVM")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

tfidf_baseline = 0.75478  # from Step 2

all_results = {
    "TF-IDF + LogReg (Step 2)": (tfidf_baseline, 0.0),
    "GloVe Avg + LogReg": (glove_scores.mean(), glove_scores.mean() - tfidf_baseline),
    "GloVe TF-IDF-Weighted": (weighted_scores.mean(), weighted_scores.mean() - tfidf_baseline),
    "GloVe + TF-IDF concat": (concat_scores.mean(), concat_scores.mean() - tfidf_baseline),
    "GloVe Avg + SVM": (svm_scores.mean(), svm_scores.mean() - tfidf_baseline),
}

print(f"\n  {'Model':35s} {'Mean F1':>10s} {'vs TF-IDF':>10s}")
print(f"  {'-' * 58}")
for name, (score, delta) in all_results.items():
    print(f"  {name:35s} {score:>10.5f} {delta:>+10.5f}")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("\n\n" + "=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print("""
  1. GloVe embeddings represent words as dense vectors where similar
     words are nearby. Unlike TF-IDF, the model can generalize:
     "bushfire" is close to "wildfire" even if only one appears in
     training data.

  2. AVERAGING word vectors to represent a tweet is simple but lossy —
     it collapses the entire tweet into one point in 200-D space.
     Word order, emphasis, and context are all lost (like bag-of-words,
     but in embedding space).

  3. TF-IDF WEIGHTING helps by emphasizing important words in the
     average — "earthquake" contributes more than "the" to the tweet
     vector.

  4. CONCATENATING GloVe with TF-IDF can capture both:
     - Meaning similarity (GloVe: fire ≈ blaze)
     - Exact word presence (TF-IDF: does "earthquake" appear?)

  5. The key limitation of averaged embeddings: every tweet is a single
     point. Two very different tweets can average to similar vectors.
     "Firefighters battle wildfire" and "I got fired from my job" both
     contain "fire"-like vectors.

  6. NEXT: Transformers (Step 7) will fix this by producing a DIFFERENT
     vector for "fire" depending on context — the representation changes
     based on surrounding words.
""")

tee.close()
print("Done. Results saved to results/models/05_glove_embeddings.txt")
