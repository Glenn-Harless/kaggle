# Disaster Tweets: Plan of Attack

**Competition:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
**Goal:** Learning-focused — understand how NLP evolved from word counting to transformers
**Approach:** Historical progression — build up from bag-of-words to BERT, understanding why each step matters
**Date:** 2026-03-16

---

## The Dataset

- 7,613 training tweets, 3,263 test tweets
- Binary classification: real disaster (43%) vs not (57%)
- Columns: `id`, `keyword` (221 unique, 61 missing), `location` (messy, 33% missing), `text`, `target`
- Tweets are short (mean 101 chars, max 157 — Twitter's old 140-char limit + URLs)
- Metric: **F1 score** (harmonic mean of precision and recall)

## Key Differences from Previous Projects

| | Titanic / House Prices | Disaster Tweets |
|---|---|---|
| Input | Structured columns (Age, GrLivArea) | Free-form text |
| Feature engineering | Manual (combine columns, create ratios) | Representation learning (how to turn text into numbers) |
| Model families | Linear models, tree models | NLP-specific (TF-IDF→embeddings→transformers) |
| Transfer learning | Not applicable | Critical (pre-trained word vectors, fine-tuning BERT) |
| Key challenge | Feature selection, regularization | Representing meaning — "fire" in "forest fire" vs "you're fired" |

## The Big Idea: Representations

The central question in NLP is: **how do you represent text as numbers that a model can use?**

Each phase of this project uses a more sophisticated representation:

1. **Bag of words** — count which words appear. Loses all word order.
   "Dog bites man" = "Man bites dog" = {dog:1, bites:1, man:1}
2. **TF-IDF** — weight words by distinctiveness. Common words (the, a, is) get low weight.
   More informative than raw counts but still no word order.
3. **Word embeddings** — represent each word as a dense vector trained on huge corpora.
   "Fire" and "blaze" are nearby in vector space. Captures meaning but still averages over the sentence.
4. **Transformers (BERT)** — process the entire sentence at once using attention.
   "Fire" gets a *different* vector in "forest fire" vs "you're fired." Context-dependent meaning.

Each step is a genuine conceptual leap, not just a better algorithm on the same features.

---

## Steps

### Phase 1: Baseline (Steps 1-2)

#### Step 1: EDA — Understanding Text Data

Explore the dataset with an NLP lens:

- Target distribution and class balance
- Keyword analysis: which keywords are most predictive? (e.g., "earthquake" is almost always real)
- Text length: do real disaster tweets tend to be longer/shorter?
- Common words in disaster vs non-disaster tweets
- URL, hashtag, and @mention patterns
- Location field: too messy to use? (3,341 unique values, 33% missing)

This is different from numeric EDA — we're looking at word frequencies and patterns, not distributions and correlations.

- Script: `scripts/preprocessing/01_eda.py`

**What you learn:** How to explore text data, what makes NLP EDA different from tabular EDA.

#### Step 2: Bag-of-Words Baselines

Three baselines using the simplest text representations:

1. **Keyword-only baseline:** predict using just the keyword column (ignoring text entirely). Some keywords like "earthquake" are almost always real disasters. This is the "how far can metadata get us" test.
2. **CountVectorizer + Logistic Regression:** count word frequencies, feed to logistic regression. The classic bag-of-words approach.
3. **TF-IDF + Logistic Regression:** weight word counts by document frequency. Should outperform raw counts because it downweights common words like "the."

Each gets: 5-fold CV F1 score.

- Script: `scripts/modeling/02_bow_baselines.py`

**Primer — Bag of Words and TF-IDF:**

Bag of words converts text to a vector by counting word occurrences. The tweet "Forest fire near La Ronge" becomes a sparse vector with 1s at positions for "forest," "fire," "near," "la," "ronge" and 0s everywhere else. The vocabulary might be 10,000+ words wide.

Problem: common words like "the," "is," "a" dominate the counts but carry no meaning. TF-IDF (Term Frequency–Inverse Document Frequency) fixes this by multiplying each count by how rare the word is across all documents. "Earthquake" appears in few tweets → high IDF → high weight. "The" appears in almost every tweet → low IDF → low weight.

Both representations completely ignore word order. "Man bites dog" and "Dog bites man" are identical vectors. This is a fundamental limitation we'll fix in later phases.

**What you learn:** How text becomes a feature matrix, what TF-IDF does and why it's better than raw counts, and how good a "dumb" text model can be.

### Phase 2: Traditional NLP (Steps 3-4)

#### Step 3: Text Preprocessing Experiments

Test whether cleaning the text helps or hurts:

1. **Lowercasing** — should help (Fire = fire)
2. **Remove URLs** — they're not meaningful text, just links
3. **Remove @mentions** — usernames aren't predictive
4. **Remove punctuation and special characters**
5. **Stopword removal** — remove "the," "is," "and," etc.
6. **Stemming** — reduce "running," "runs," "ran" to "run"
7. **Lemmatization** — like stemming but linguistically proper ("better" → "good")

Test each cleaning step individually, then combine the ones that help. Measure with TF-IDF + LogReg CV.

- Script: `scripts/modeling/03_text_preprocessing.py`

**What you learn:** Which text cleaning steps matter and which are noise. This varies by dataset — stopword removal helps on long documents but sometimes hurts on short tweets where every word counts.

#### Step 4: Feature Engineering + Model Comparison

Engineer text-level features that bag-of-words misses:

- Text length (chars and words)
- Number of hashtags, @mentions, URLs
- Number of exclamation marks, question marks, ALL CAPS words
- Keyword column (one-hot or target-encoded)
- Ratio of uppercase characters
- Presence of numbers (addresses, death counts)

Combine TF-IDF features with these engineered features. Test with:
- Logistic Regression
- SVM (often strong for text classification)
- XGBoost (to see if trees help on text features)

- Script: `scripts/modeling/04_feature_engineering.py`

**What you learn:** What structural text features add beyond word content, which classifiers work best for text.

### Phase 3: Word Embeddings (Steps 5-6)

#### Step 5: Pre-trained Word Embeddings (GloVe)

Replace bag-of-words with dense vectors from GloVe (Global Vectors for Word Representation):

- Each word → a 100-300 dimensional vector, pre-trained on billions of words
- Words with similar meanings are nearby: cosine_similarity("fire", "blaze") ≈ 0.7
- Represent each tweet as the average of its word vectors
- Feed the resulting 300-dim vector to a classifier

Compare: GloVe average vectors + LogReg vs TF-IDF + LogReg.

- Script: `scripts/modeling/05_glove_embeddings.py`

**Primer — Word Embeddings:**

A word embedding maps each word to a dense vector (e.g., 300 dimensions) where geometric relationships encode meaning:

- vector("king") - vector("man") + vector("woman") ≈ vector("queen")
- vector("fire") is close to vector("blaze"), vector("flames"), vector("inferno")
- vector("earthquake") is close to vector("tremor"), vector("seismic")

GloVe was trained on 6 billion tokens from Wikipedia and web text. It learned these relationships by observing which words appear near each other in context. We download the pre-trained vectors and use them directly — no training required on our end.

The limitation: each word gets ONE vector regardless of context. "Fire" has the same vector in "forest fire" and "you're fired." This is what transformers will fix.

**What you learn:** How meaning is encoded as geometry, why pre-trained representations beat learning from scratch on small datasets.

#### Step 6: Embedding Experiments

Test variations on the embedding approach:

1. **Average vs TF-IDF-weighted average** — weight important words more
2. **Different embedding dimensions** (50d, 100d, 200d, 300d)
3. **Handle out-of-vocabulary words** — what about words GloVe never saw?
4. **Concatenate embeddings with TF-IDF features** — best of both?

- Script: `scripts/modeling/06_embedding_experiments.py`

**What you learn:** How to combine different representations, the tradeoff between embedding dimension and overfitting.

### Phase 4: Transformers (Steps 7-8)

**Prerequisite:** Install `torch` and `transformers` packages.

#### Step 7: Fine-tune DistilBERT

The modern approach: take a pre-trained transformer model and fine-tune it on our specific task.

- **DistilBERT** — a smaller, faster version of BERT (66M parameters vs 110M)
- Pre-trained on Wikipedia + BookCorpus — already "understands" English
- Fine-tuning: add a classification head on top, train on our 7,613 tweets
- The model produces context-dependent embeddings: "fire" gets different vectors in "forest fire" vs "you're fired"

- Script: `scripts/modeling/07_distilbert.py`

**Primer — How Transformers Work (the intuition):**

A transformer processes all words in a sentence simultaneously and uses "attention" to figure out which words relate to each other:

For "The forest fire destroyed 200 homes":
- "fire" attends to "forest" (what kind of fire?) and "destroyed" (what happened?) → high attention weights
- "200" attends to "homes" (200 what?) → high attention weight
- "The" attends to everything weakly → low attention weights

Each word ends up with a representation that incorporates its full context. This is why BERT understands that "fire" means different things in different sentences — it literally computes a different vector for each occurrence based on surrounding words.

Fine-tuning means: BERT already knows English grammar, word meanings, and context from pre-training. We just teach it the narrow task of "is this tweet about a real disaster?" by training the final classification layer (and slightly adjusting the deeper layers) on our labeled data.

**What you learn:** How fine-tuning works, why pre-trained transformers are so powerful, the practical mechanics of training a neural network.

#### Step 8: Transformer Experiments

Compare transformer variants and approaches:

1. **DistilBERT vs BERT-base** — does the full model justify the extra compute?
2. **Learning rate and epoch tuning** — transformers are sensitive to learning rate
3. **Different pre-trained models** — RoBERTa, ALBERT, etc.
4. **Freeze vs unfreeze layers** — fine-tune everything or just the classification head?

- Script: `scripts/modeling/08_transformer_experiments.py`

**What you learn:** How to select and tune transformer models, what affects fine-tuning performance.

### Phase 5: Wrap-up (Steps 9-10)

#### Step 9: Ensemble + Final Submission

- Blend TF-IDF model with transformer model (different error patterns)
- Final Kaggle submission
- OOF predictions, constrained blend weights (same protocol as House Prices)

- Script: `scripts/modeling/09_ensemble.py`

#### Step 10: Retrospective

- Which representation leap mattered most?
- TF-IDF vs embeddings vs transformers — quantify the gap
- What surprised you?
- How does NLP model development differ from tabular?

- File: `RETROSPECTIVE.md`

---

## Validation Protocol

- **CV:** StratifiedKFold(5, shuffle=True, random_state=42) — same folds for all models
- **Metric:** F1 score (competition metric)
- **Leaderboard:** Max 1-2 submissions per phase. Require local CV improvement before submitting.
- **Transformer training:** use a held-out portion of training fold for early stopping (same principle as XGBoost in House Prices)

## Principles (carried forward)

1. **Every change gets validated** — CV score before and after
2. **Understand before optimizing** — the point is learning representations, not leaderboard
3. **Explain deeply** — every step includes the *why*, not just the *what*
4. **Clean code, clear output** — numbered scripts, Tee logging
5. **No Kaggle writeups** — methodology research and documentation are fine

## File Structure

```
competitions/disaster-tweets/
├── data/                         # Raw data (gitignored)
├── docs/
│   └── PLAN.md                   # This file
├── scripts/
│   ├── preprocessing/
│   │   └── 01_eda.py
│   └── modeling/
│       ├── 02_bow_baselines.py
│       ├── 03_text_preprocessing.py
│       ├── 04_feature_engineering.py
│       ├── 05_glove_embeddings.py
│       ├── 06_embedding_experiments.py
│       ├── 07_distilbert.py
│       ├── 08_transformer_experiments.py
│       └── 09_ensemble.py
├── results/models/               # CV scores, logs
├── results/analysis/             # Plots
├── submissions/                  # Kaggle submission CSVs
└── RETROSPECTIVE.md              # Final reflection
```
