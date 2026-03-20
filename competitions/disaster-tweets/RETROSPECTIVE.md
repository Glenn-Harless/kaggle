# Disaster Tweets: Retrospective

**Competition:** Natural Language Processing with Disaster Tweets
**Final Score:** 0.831 F1 (public leaderboard)
**Submissions Used:** 1
**Date:** 2026-03-20

---

## The Progression

| Step | Model | CV F1 | What It Captures |
|---|---|---|---|
| 2 | Keyword-only | 0.673 | Metadata lookup — no ML at all |
| 2 | CountVectorizer + LogReg | 0.751 | Which words appear (raw counts) |
| 2 | TF-IDF + LogReg | 0.755 | Which words appear (rarity-weighted) |
| 3 | TF-IDF + Lemmatization | 0.756 | Cleaned text (marginal gain) |
| 4 | TF-IDF + Features + KW | 0.755 | Structural features (no gain) |
| 5 | GloVe average | 0.756 | Word meaning (without exact matching) |
| 5 | GloVe + TF-IDF concat | 0.773 | Word meaning + exact word presence |
| 7 | DistilBERT (lr=2e-5) | 0.806 | Context-dependent meaning |
| 8 | DistilBERT (lr=1e-5) | 0.809 | Tuned transformer |
| 9 | Ensemble blend | 0.806 | All three perspectives combined |
| — | **Leaderboard** | **0.831** | — |

## Which Representation Leap Mattered Most?

```
Keyword → TF-IDF:     +0.082  ← biggest jump (text content >> metadata)
TF-IDF → GloVe+TFIDF: +0.018  ← word meaning helps
GloVe → DistilBERT:   +0.033  ← contextual meaning helps more
DistilBERT → Ensemble: +0.001  ← barely anything
```

**The single biggest gain was going from metadata to text content (+0.082).** Simply counting which words appear in a tweet was worth more than all the sophisticated representation learning that followed. This makes sense — most of the signal is in obvious keywords like "earthquake", "wildfire", "flood."

**The second biggest gain was the transformer (+0.033 over GloVe).** Context-dependent meaning genuinely matters for this task because the same words ("fire", "crash", "body") appear in both disaster and non-disaster contexts.

## What Surprised Me

### 1. Text cleaning mostly hurt
Stopword removal dropped F1 by 0.011 — the single worst "improvement" we tried. On short tweets, stopwords like "you", "my", "I" are among the most predictive features because they signal personal/conversational language (non-disaster). TF-IDF already handles the downweighting that stopword removal tries to accomplish, but more gracefully.

### 2. Engineered features added nothing
URL counts, hashtag counts, text length, capitalization — none of these moved the needle when combined with TF-IDF. The reason: TF-IDF already implicitly encodes structural patterns. The presence of "http", "t", "co" tokens encodes "has URL." Punctuation tokens encode style. The engineered features were redundant.

### 3. BERT-base lost to DistilBERT
110M parameters performed slightly worse than 66M on 7,613 short tweets. More capacity became a liability on a small dataset — more parameters to overfit with. DistilBERT's smaller size acted as implicit regularization.

### 4. The ensemble barely helped
When one model (BERT) is dramatically better than the others and the predictions are highly correlated (0.83-0.87), blending adds almost nothing. The simpler models rarely knew something BERT didn't. Compare this to House Prices where linear + tree models captured genuinely different patterns.

### 5. Leaderboard score exceeded CV
0.831 vs 0.806 — unusual. The ensemble likely generalized well, and the test set may have a slightly different distribution of ambiguous tweets.

## What I Learned About NLP

### The Central Problem
NLP is fundamentally about **representation** — how do you turn text into numbers that preserve meaning? Every technique we used was a different answer to this question:

- **Bag-of-words:** text = set of words (loses order and meaning)
- **TF-IDF:** text = set of weighted words (common words matter less)
- **GloVe:** text = average point in meaning-space (but one meaning per word)
- **Transformers:** text = contextual understanding (different meaning per usage)

### Key Concepts Learned
1. **Tokenization** — splitting text into processable units (words, subwords)
2. **TF-IDF** — rarity = importance. Inverse Document Frequency downweights common words
3. **Word embeddings** — words as points in space where similar words are nearby
4. **Static vs contextual** — GloVe gives "fire" one vector; BERT gives it a different vector per sentence
5. **Attention** — the mechanism that lets transformers connect any word to any other word directly
6. **Fine-tuning** — adapting a pre-trained model to a specific task with a small learning rate
7. **WordPiece tokenization** — subword splitting that handles any word, even misspellings

### NLP vs Tabular ML
| Aspect | Tabular (Titanic, House Prices) | NLP (Disaster Tweets) |
|---|---|---|
| Features | Already numbers or easily encoded | Must convert text → numbers |
| Feature engineering | Major lever (creates new signal) | Minimal impact (text already contains it) |
| Best models | Trees + linear blends | Transformers dominate |
| Preprocessing | Almost always helps | Often hurts (especially on short text) |
| Transfer learning | Not applicable | Critical (pre-trained models) |
| Ensembling | Big gains (diverse model types) | Small gains (BERT subsumes simpler models) |

## What I'd Do Differently

1. **Skip Steps 3-4 on a second pass.** Text cleaning and feature engineering added virtually nothing. The signal is in the representation, not the preprocessing.

2. **Try RoBERTa or DeBERTa instead of DistilBERT.** These are stronger pre-trained models that would likely push 0.84+ with minimal extra effort.

3. **Spend more time on the transformer.** Learning rate had more impact than any other hyperparameter. A finer sweep (1e-5, 1.5e-5, 2e-5) might find a better optimum.

4. **Pseudo-labeling** — use confident predictions on the test set as additional training data. Common technique for squeezing out the last few points.

## File Structure

```
competitions/disaster-tweets/
├── data/                           # Raw data
├── docs/
│   └── PLAN.md                     # Original plan
├── scripts/
│   ├── preprocessing/
│   │   └── 01_eda.py               # EDA + plots
│   └── modeling/
│       ├── 02_bow_baselines.py     # Keyword, CountVec, TF-IDF
│       ├── 03_text_preprocessing.py # Cleaning experiments
│       ├── 04_feature_engineering.py # Structural features + model comparison
│       ├── 05_glove_embeddings.py  # GloVe word embeddings
│       ├── 07_distilbert.py        # Fine-tune DistilBERT
│       ├── 08_transformer_experiments.py # LR sweep, BERT, freezing
│       ├── 09_ensemble.py          # Blend + submission
│       └── 09b_disagreement_analysis.py # Model agreement analysis
├── results/
│   ├── models/                     # All step logs (.txt)
│   └── analysis/
│       └── 01_eda.png              # EDA plots
├── submissions/
│   ├── ensemble_v1.csv             # Final submission (0.831)
│   ├── tfidf_v1.csv
│   ├── glove_v1.csv
│   └── bert_v1.csv
└── RETROSPECTIVE.md                # This file
```
