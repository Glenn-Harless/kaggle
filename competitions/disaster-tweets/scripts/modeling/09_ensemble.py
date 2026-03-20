"""
Disaster Tweets: Step 9 — Ensemble + Final Submission

Blend three models with different error patterns:
1. TF-IDF + LogReg (bag-of-words baseline)
2. GloVe + TF-IDF concat + LogReg (embedding model)
3. DistilBERT (transformer)

Steps:
1. Generate OOF probability predictions from each model
2. Find optimal blend weights via grid search on OOF F1
3. Train each model on full training set, predict test
4. Apply blend weights to test predictions
5. Create Kaggle submission
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()

from shared.evaluate import Tee, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"
GLOVE_PATH = "/Users/glennharless/dev-brain/kaggle/data/glove.twitter.27B.200d.txt"

tee = Tee(f"{BASE}/results/models/09_ensemble.txt")
sys.stdout = tee

print("Disaster Tweets: Step 9 — Ensemble + Final Submission")
print("=" * 60)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"\n  Device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")
y = train["target"].values

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"  Train: {len(train)} tweets")
print(f"  Test:  {len(test)} tweets")


# ============================================================
# MODEL 1: TF-IDF + LOGISTIC REGRESSION (OOF + TEST)
# ============================================================

print("\n\n" + "=" * 60)
print("MODEL 1: TF-IDF + LogReg")
print("=" * 60)

tfidf = TfidfVectorizer(max_features=10000)
oof_tfidf = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    X_tr = tfidf.fit_transform(train["text"].iloc[train_idx])
    X_val = tfidf.transform(train["text"].iloc[val_idx])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y[train_idx])
    oof_tfidf[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_f1 = f1_score(y[val_idx], (oof_tfidf[val_idx] > 0.5).astype(int))
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

oof_tfidf_f1 = f1_score(y, (oof_tfidf > 0.5).astype(int))
print(f"  OOF F1: {oof_tfidf_f1:.5f}")

# Full training set → test predictions
X_train_tfidf = tfidf.fit_transform(train["text"])
X_test_tfidf = tfidf.transform(test["text"])
model_tfidf = LogisticRegression(max_iter=1000, random_state=42)
model_tfidf.fit(X_train_tfidf, y)
test_tfidf = model_tfidf.predict_proba(X_test_tfidf)[:, 1]
print(f"  Test predictions generated.")


# ============================================================
# MODEL 2: GloVe + TF-IDF CONCAT + LogReg (OOF + TEST)
# ============================================================

print("\n\n" + "=" * 60)
print("MODEL 2: GloVe + TF-IDF Concat + LogReg")
print("=" * 60)

# Load GloVe
print("  Loading GloVe vectors...")
embeddings = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        embeddings[word] = np.array(parts[1:], dtype=np.float32)
print(f"  Loaded {len(embeddings):,} vectors")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def tweet_to_vec(text, dim=200):
    tokens = tokenize(text)
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if len(vecs) == 0:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


X_glove_train = np.array([tweet_to_vec(t) for t in train["text"]])
X_glove_test = np.array([tweet_to_vec(t) for t in test["text"]])

oof_glove = np.zeros(len(train))
scaler = StandardScaler()
tfidf_g = TfidfVectorizer(max_features=10000)

for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    # TF-IDF part
    X_tfidf_tr = tfidf_g.fit_transform(train["text"].iloc[train_idx])
    X_tfidf_val = tfidf_g.transform(train["text"].iloc[val_idx])

    # GloVe part
    glove_tr = scaler.fit_transform(X_glove_train[train_idx])
    glove_val = scaler.transform(X_glove_train[val_idx])

    # Concat
    X_tr = hstack([X_tfidf_tr, csr_matrix(glove_tr)])
    X_val = hstack([X_tfidf_val, csr_matrix(glove_val)])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr, y[train_idx])
    oof_glove[val_idx] = model.predict_proba(X_val)[:, 1]

    fold_f1 = f1_score(y[val_idx], (oof_glove[val_idx] > 0.5).astype(int))
    print(f"  Fold {fold+1}: F1 = {fold_f1:.4f}")

oof_glove_f1 = f1_score(y, (oof_glove > 0.5).astype(int))
print(f"  OOF F1: {oof_glove_f1:.5f}")

# Full training → test
X_train_tfidf_full = tfidf_g.fit_transform(train["text"])
X_test_tfidf_full = tfidf_g.transform(test["text"])
glove_tr_full = scaler.fit_transform(X_glove_train)
glove_test_full = scaler.transform(X_glove_test)
X_train_concat = hstack([X_train_tfidf_full, csr_matrix(glove_tr_full)])
X_test_concat = hstack([X_test_tfidf_full, csr_matrix(glove_test_full)])
model_glove = LogisticRegression(max_iter=1000, random_state=42)
model_glove.fit(X_train_concat, y)
test_glove = model_glove.predict_proba(X_test_concat)[:, 1]
print(f"  Test predictions generated.")


# ============================================================
# MODEL 3: DistilBERT (OOF + TEST)
# ============================================================

print("\n\n" + "=" * 60)
print("MODEL 3: DistilBERT (lr=1e-5, 3 epochs)")
print("=" * 60)

MAX_LEN = 128
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 3

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.labels = labels
        self.encodings = tokenizer(
            list(texts), max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_bert(train_texts, train_labels, predict_texts):
    """Train DistilBERT and return probabilities for predict_texts."""
    train_dataset = TweetDataset(train_texts, train_labels)
    predict_dataset = TweetDataset(predict_texts)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    predict_loader = DataLoader(predict_dataset, batch_size=BATCH_SIZE)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    # Predict probabilities
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in predict_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(all_probs)


# OOF predictions
oof_bert = np.zeros(len(train))
for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    print(f"  Fold {fold+1}/5...")
    probs = train_bert(
        train["text"].iloc[train_idx].values,
        y[train_idx],
        train["text"].iloc[val_idx].values,
    )
    oof_bert[val_idx] = probs
    fold_f1 = f1_score(y[val_idx], (probs > 0.5).astype(int))
    print(f"    F1 = {fold_f1:.4f}")

oof_bert_f1 = f1_score(y, (oof_bert > 0.5).astype(int))
print(f"  OOF F1: {oof_bert_f1:.5f}")

# Full training → test predictions
print(f"  Training on full dataset for test predictions...")
test_bert = train_bert(
    train["text"].values,
    y,
    test["text"].values,
)
print(f"  Test predictions generated.")


# ============================================================
# BLEND WEIGHT OPTIMIZATION
# ============================================================

print("\n\n" + "=" * 60)
print("BLEND WEIGHT OPTIMIZATION")
print("=" * 60)
print("""
  Grid search over weights (w1, w2, w3) that sum to 1.
  Blended probability = w1*TF-IDF + w2*GloVe + w3*BERT
  Optimize for F1 on OOF predictions.
""")

best_f1 = 0
best_weights = (1/3, 1/3, 1/3)
best_threshold = 0.5

# Grid search over weights
step = 0.05
for w1 in np.arange(0, 1.01, step):
    for w2 in np.arange(0, 1.01 - w1, step):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue

        blended = w1 * oof_tfidf + w2 * oof_glove + w3 * oof_bert

        # Also search threshold
        for thresh in [0.45, 0.475, 0.5, 0.525, 0.55]:
            preds = (blended > thresh).astype(int)
            f1 = f1_score(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = (w1, w2, w3)
                best_threshold = thresh

w1, w2, w3 = best_weights
print(f"  Best weights:")
print(f"    TF-IDF + LogReg:         {w1:.2f}")
print(f"    GloVe + TF-IDF concat:   {w2:.2f}")
print(f"    DistilBERT:              {w3:.2f}")
print(f"  Best threshold: {best_threshold}")
print(f"  Best blend OOF F1: {best_f1:.5f}")


# ============================================================
# INDIVIDUAL VS BLEND COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("INDIVIDUAL VS BLEND COMPARISON (OOF)")
print("=" * 60)

results = {
    "TF-IDF + LogReg": oof_tfidf_f1,
    "GloVe + TF-IDF concat": oof_glove_f1,
    "DistilBERT": oof_bert_f1,
    "Blend (optimized)": best_f1,
}

print(f"\n  {'Model':30s} {'OOF F1':>10s} {'Delta vs BERT':>14s}")
print(f"  {'-' * 57}")
for name, score in results.items():
    delta = score - oof_bert_f1
    print(f"  {name:30s} {score:>10.5f} {delta:>+14.5f}")


# ============================================================
# AGREEMENT ANALYSIS
# ============================================================

print("\n\n" + "=" * 60)
print("MODEL AGREEMENT ANALYSIS")
print("=" * 60)

pred_tfidf = (oof_tfidf > 0.5).astype(int)
pred_glove = (oof_glove > 0.5).astype(int)
pred_bert = (oof_bert > 0.5).astype(int)

all_agree = ((pred_tfidf == pred_glove) & (pred_glove == pred_bert)).sum()
any_disagree = len(train) - all_agree
print(f"\n  All 3 models agree: {all_agree} tweets ({all_agree/len(train):.1%})")
print(f"  At least 1 disagrees: {any_disagree} tweets ({any_disagree/len(train):.1%})")

# When all agree, what's the accuracy?
agree_mask = (pred_tfidf == pred_glove) & (pred_glove == pred_bert)
agree_f1 = f1_score(y[agree_mask], pred_bert[agree_mask])
print(f"  F1 when all agree: {agree_f1:.4f}")

# When they disagree, what's each model's accuracy?
disagree_mask = ~agree_mask
if disagree_mask.sum() > 0:
    print(f"\n  On the {disagree_mask.sum()} disagreement tweets:")
    for name, preds in [("TF-IDF", pred_tfidf), ("GloVe", pred_glove), ("BERT", pred_bert)]:
        f1_d = f1_score(y[disagree_mask], preds[disagree_mask])
        print(f"    {name:10s} F1 = {f1_d:.4f}")
    blend_disagree = (w1 * oof_tfidf + w2 * oof_glove + w3 * oof_bert > best_threshold).astype(int)
    f1_blend_d = f1_score(y[disagree_mask], blend_disagree[disagree_mask])
    print(f"    {'Blend':10s} F1 = {f1_blend_d:.4f}")


# ============================================================
# CORRELATION BETWEEN MODEL PREDICTIONS
# ============================================================

print("\n\n" + "=" * 60)
print("PREDICTION CORRELATION")
print("=" * 60)

corr_df = pd.DataFrame({
    "TF-IDF": oof_tfidf,
    "GloVe": oof_glove,
    "BERT": oof_bert,
})
corr_matrix = corr_df.corr()
print(f"\n  Probability correlation matrix:")
print(f"  {'':10s} {'TF-IDF':>10s} {'GloVe':>10s} {'BERT':>10s}")
print(f"  {'-' * 43}")
for name in ["TF-IDF", "GloVe", "BERT"]:
    print(f"  {name:10s} {corr_matrix.loc[name, 'TF-IDF']:>10.3f} "
          f"{corr_matrix.loc[name, 'GloVe']:>10.3f} "
          f"{corr_matrix.loc[name, 'BERT']:>10.3f}")


# ============================================================
# CREATE SUBMISSION
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL SUBMISSION")
print("=" * 60)

# Apply blend to test predictions
test_blended = w1 * test_tfidf + w2 * test_glove + w3 * test_bert
test_preds = (test_blended > best_threshold).astype(int)

print(f"\n  Test predictions distribution:")
print(f"    Not disaster: {(test_preds == 0).sum()} ({(test_preds == 0).mean():.1%})")
print(f"    Disaster:     {(test_preds == 1).sum()} ({(test_preds == 1).mean():.1%})")

# Sanity check: compare to training distribution
print(f"\n  Training distribution for reference:")
print(f"    Not disaster: {(y == 0).sum()} ({(y == 0).mean():.1%})")
print(f"    Disaster:     {(y == 1).sum()} ({(y == 1).mean():.1%})")

# Save submission
sub = pd.DataFrame({"id": test["id"], "target": test_preds})
sub_path = f"{BASE}/submissions/ensemble_v1.csv"
sub.to_csv(sub_path, index=False)
print(f"\n  Saved: {sub_path}")
print(f"  Shape: {sub.shape}")

# Also save individual model submissions for comparison
for name, preds in [("tfidf", test_tfidf), ("glove", test_glove), ("bert", test_bert)]:
    sub_ind = pd.DataFrame({"id": test["id"], "target": (preds > 0.5).astype(int)})
    path = f"{BASE}/submissions/{name}_v1.csv"
    sub_ind.to_csv(path, index=False)

print(f"  Also saved individual model submissions.")


# ============================================================
# FULL PROJECT SCOREBOARD
# ============================================================

print("\n\n" + "=" * 60)
print("FULL PROJECT SCOREBOARD")
print("=" * 60)

print(f"""
  {'Step':6s} {'Model':35s} {'F1':>10s}
  {'-' * 55}
  {"2":6s} {"Keyword-only":35s} {"0.673":>10s}
  {"2":6s} {"TF-IDF + LogReg":35s} {oof_tfidf_f1:>10.5f}
  {"5":6s} {"GloVe + TF-IDF concat":35s} {oof_glove_f1:>10.5f}
  {"7":6s} {"DistilBERT (lr=1e-5, 3ep)":35s} {oof_bert_f1:>10.5f}
  {"9":6s} {"Ensemble (blend)":35s} {best_f1:>10.5f}
""")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print("""
  1. ENSEMBLING combines models with different error patterns.
     TF-IDF catches keyword signals, GloVe captures meaning similarity,
     BERT understands context. Together they cover each other's blind spots.

  2. BLEND WEIGHTS reveal which model the ensemble trusts most.
     Higher BERT weight = the transformer dominates, but the traditional
     models still contribute on cases where BERT is uncertain.

  3. AGREEMENT ANALYSIS shows where the ensemble adds value — it's on
     the disagreement tweets (where models see different things) that
     blending helps most.

  4. PREDICTION CORRELATION: lower correlation between models = more
     ensemble benefit. If all models made the same errors, blending
     wouldn't help. We want diverse models.

  5. This completes our NLP progression:
     Word counting → weighted word counting → word meaning →
     contextual meaning → blending all three perspectives.
""")

tee.close()
print("Done. Results saved to results/models/09_ensemble.txt")
