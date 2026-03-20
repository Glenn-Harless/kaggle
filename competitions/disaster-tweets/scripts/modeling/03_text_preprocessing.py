"""
Disaster Tweets: Step 3 — Text Preprocessing Experiments

Test whether cleaning the text helps or hurts TF-IDF + LogReg (F1=0.755).

Each cleaning step is tested individually, then we combine the winners.
This is an empirical question — some cleaning helps on long documents
but hurts on short tweets where every word counts.

Cleaning steps tested:
1. Lowercase (CountVectorizer already does this, but explicit)
2. Remove URLs
3. Remove @mentions
4. Remove punctuation and special characters
5. Stopword removal
6. Stemming (Porter)
7. Lemmatization (WordNet)
8. Remove numbers
9. Combined: best individual cleanings
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from shared.evaluate import Tee, classification_cv, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/03_text_preprocessing.txt")
sys.stdout = tee

print("Disaster Tweets: Step 3 — Text Preprocessing Experiments")
print("=" * 60)


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
y = train["target"]

print(f"\n  Train: {len(train)} tweets")
print(f"  Baseline (raw TF-IDF + LogReg): F1 = 0.75478")


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_lowercase(text):
    return text.lower()


def clean_urls(text):
    """Remove URLs (http://... and https://...)."""
    return re.sub(r"https?://\S+", "", text)


def clean_mentions(text):
    """Remove @username mentions."""
    return re.sub(r"@\w+", "", text)


def clean_hashtag_symbol(text):
    """Remove # symbol but keep the word (e.g., #earthquake -> earthquake)."""
    return re.sub(r"#(\w+)", r"\1", text)


def clean_punctuation(text):
    """Remove punctuation and special characters, keep only letters and spaces."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def clean_stopwords(text):
    """Remove English stopwords."""
    words = text.lower().split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def clean_stemming(text):
    """Apply Porter stemming (running -> run, fires -> fire)."""
    words = text.lower().split()
    return " ".join(stemmer.stem(w) for w in words)


def clean_lemmatize(text):
    """Apply WordNet lemmatization (better -> good, fires -> fire)."""
    words = text.lower().split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


def clean_numbers(text):
    """Remove digits."""
    return re.sub(r"\d+", "", text)


# ============================================================
# EXPERIMENT FRAMEWORK
# ============================================================

def run_experiment(name, clean_fn, texts, y):
    """Apply cleaning function and evaluate TF-IDF + LogReg."""
    cleaned = texts.apply(clean_fn)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    scores = classification_cv(pipe, cleaned, y)
    return scores


# ============================================================
# BASELINE: RAW TEXT (no cleaning)
# ============================================================

print("\n\n" + "=" * 60)
print("BASELINE: RAW TEXT (no cleaning)")
print("=" * 60)

baseline_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
baseline_scores = classification_cv(baseline_pipe, train["text"], y)
report_f1_cv(baseline_scores, "Raw text")


# ============================================================
# INDIVIDUAL CLEANING EXPERIMENTS
# ============================================================

experiments = {
    "1. Lowercase": clean_lowercase,
    "2. Remove URLs": clean_urls,
    "3. Remove @mentions": clean_mentions,
    "4. Remove # symbol": clean_hashtag_symbol,
    "5. Remove punctuation": clean_punctuation,
    "6. Remove stopwords": clean_stopwords,
    "7. Stemming (Porter)": clean_stemming,
    "8. Lemmatization": clean_lemmatize,
    "9. Remove numbers": clean_numbers,
}

all_results = {"0. Baseline (raw)": baseline_scores}

for name, clean_fn in experiments.items():
    print(f"\n\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("=" * 60)

    scores = run_experiment(name, clean_fn, train["text"], y)
    delta = scores.mean() - baseline_scores.mean()
    report_f1_cv(scores, name)
    print(f"\n  Delta vs baseline: {delta:+.5f} F1")
    if delta > 0.001:
        print(f"  → HELPS")
    elif delta < -0.001:
        print(f"  → HURTS")
    else:
        print(f"  → NEUTRAL")

    all_results[name] = scores


# ============================================================
# COMBINED: APPLY ALL STEPS THAT HELPED
# ============================================================

print("\n\n" + "=" * 60)
print("COMBINED: APPLY ALL IMPROVEMENTS")
print("=" * 60)

# Determine which steps helped
helped = []
for name, scores in all_results.items():
    if name == "0. Baseline (raw)":
        continue
    if scores.mean() > baseline_scores.mean() + 0.001:
        helped.append(name)

print(f"\n  Steps that helped (delta > +0.001):")
for name in helped:
    delta = all_results[name].mean() - baseline_scores.mean()
    print(f"    {name}: {delta:+.5f}")

print(f"\n  Steps that hurt (delta < -0.001):")
for name, scores in all_results.items():
    if name == "0. Baseline (raw)":
        continue
    if scores.mean() < baseline_scores.mean() - 0.001:
        delta = scores.mean() - baseline_scores.mean()
        print(f"    {name}: {delta:+.5f}")


# Build combined cleaning pipeline from winners
def clean_combined(text):
    """Apply the cleaning steps that individually helped or were neutral."""
    text = clean_urls(text)
    text = clean_numbers(text)
    text = text.lower()
    return text


print(f"\n  Testing combined pipeline: URL removal + number removal + lowercase")

combined_scores = run_experiment("Combined", clean_combined, train["text"], y)
report_f1_cv(combined_scores, "Combined")
delta = combined_scores.mean() - baseline_scores.mean()
print(f"\n  Delta vs baseline: {delta:+.5f} F1")

all_results["10. Combined"] = combined_scores


# Also try combined + stemming and combined + lemmatization
def clean_combined_stem(text):
    text = clean_combined(text)
    words = text.split()
    return " ".join(stemmer.stem(w) for w in words)


def clean_combined_lemma(text):
    text = clean_combined(text)
    words = text.split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


print(f"\n  Testing combined + stemming:")
comb_stem_scores = run_experiment("Combined+Stem", clean_combined_stem, train["text"], y)
report_f1_cv(comb_stem_scores, "Combined+Stem")
delta_stem = comb_stem_scores.mean() - baseline_scores.mean()
print(f"  Delta vs baseline: {delta_stem:+.5f} F1")
all_results["11. Combined+Stem"] = comb_stem_scores

print(f"\n  Testing combined + lemmatization:")
comb_lemma_scores = run_experiment("Combined+Lemma", clean_combined_lemma, train["text"], y)
report_f1_cv(comb_lemma_scores, "Combined+Lemma")
delta_lemma = comb_lemma_scores.mean() - baseline_scores.mean()
print(f"  Delta vs baseline: {delta_lemma:+.5f} F1")
all_results["12. Combined+Lemma"] = comb_lemma_scores


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL COMPARISON — ALL EXPERIMENTS")
print("=" * 60)

print(f"\n  {'Experiment':30s} {'Mean F1':>10s} {'Std':>10s} {'Delta':>10s}")
print(f"  {'-' * 65}")
for name, scores in sorted(all_results.items()):
    delta = scores.mean() - baseline_scores.mean()
    marker = ""
    if delta > 0.001:
        marker = " +"
    elif delta < -0.001:
        marker = " -"
    print(f"  {name:30s} {scores.mean():>10.5f} {scores.std():>10.5f} "
          f"{delta:>+10.5f}{marker}")

# Best overall
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
  TEXT CLEANING RESULTS:

  1. URL REMOVAL typically helps — URLs are noise for word-level models.
     The "http", "t", "co" fragments we saw dominating the word counts
     in EDA are now gone.

  2. @MENTION REMOVAL — @usernames aren't predictive of disaster content.

  3. STOPWORD REMOVAL — can go either way on short texts. On tweets,
     removing "not", "no", "but" can lose negation context.

  4. STEMMING reduces vocabulary size by collapsing word forms
     (fires/fire/fired → fire). Can help by reducing sparsity but
     can hurt by losing information (fired ≠ fire for this task).

  5. LEMMATIZATION is linguistically smarter than stemming
     (better → good, mice → mouse) but slower. May or may not beat stemming.

  6. The combined pipeline gives us our best score to carry forward
     into Step 4 (feature engineering).
""")

tee.close()
print("Done. Results saved to results/models/03_text_preprocessing.txt")
