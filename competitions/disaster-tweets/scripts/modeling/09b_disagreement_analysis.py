"""Quick analysis: what do the disagreement tweets look like?"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup, logging as hf_logging,
)
hf_logging.set_verbosity_error()

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"
GLOVE_PATH = "/Users/glennharless/dev-brain/kaggle/data/glove.twitter.27B.200d.txt"

train = pd.read_csv(f"{BASE}/data/train.csv")
y = train["target"].values
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Rebuild OOF predictions (TF-IDF) ---
tfidf = TfidfVectorizer(max_features=10000)
oof_tfidf = np.zeros(len(train))
for fold, (ti, vi) in enumerate(CV.split(train, y)):
    X_tr = tfidf.fit_transform(train["text"].iloc[ti])
    X_val = tfidf.transform(train["text"].iloc[vi])
    m = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y[ti])
    oof_tfidf[vi] = m.predict_proba(X_val)[:, 1]

# --- Rebuild OOF predictions (GloVe + TF-IDF) ---
embeddings = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)

def tweet_to_vec(text, dim=200):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)

X_glove = np.array([tweet_to_vec(t) for t in train["text"]])
oof_glove = np.zeros(len(train))
sc = StandardScaler()
tfidf_g = TfidfVectorizer(max_features=10000)
for fold, (ti, vi) in enumerate(CV.split(train, y)):
    Xt = tfidf_g.fit_transform(train["text"].iloc[ti])
    Xv = tfidf_g.transform(train["text"].iloc[vi])
    gt = sc.fit_transform(X_glove[ti])
    gv = sc.transform(X_glove[vi])
    X_tr = hstack([Xt, csr_matrix(gt)])
    X_val = hstack([Xv, csr_matrix(gv)])
    m = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y[ti])
    oof_glove[vi] = m.predict_proba(X_val)[:, 1]

# --- Rebuild OOF predictions (BERT) ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class DS(Dataset):
    def __init__(self, texts, labels=None):
        self.labels = labels
        self.enc = tok(list(texts), max_length=128, padding="max_length",
                       truncation=True, return_tensors="pt")
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i):
        item = {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i]}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

oof_bert = np.zeros(len(train))
for fold, (ti, vi) in enumerate(CV.split(train, y)):
    print(f"BERT fold {fold+1}/5...")
    tds = DS(train["text"].iloc[ti].values, y[ti])
    vds = DS(train["text"].iloc[vi].values)
    tl = DataLoader(tds, batch_size=32, shuffle=True)
    vl = DataLoader(vds, batch_size=32)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1*len(tl)*3), len(tl)*3)
    for ep in range(3):
        model.train()
        for b in tl:
            opt.zero_grad()
            out = model(input_ids=b["input_ids"].to(DEVICE),
                        attention_mask=b["attention_mask"].to(DEVICE),
                        labels=b["labels"].to(DEVICE))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for b in vl:
            out = model(input_ids=b["input_ids"].to(DEVICE),
                        attention_mask=b["attention_mask"].to(DEVICE))
            probs.extend(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy())
    oof_bert[vi] = probs
    del model; torch.mps.empty_cache()

# --- Analysis ---
pred_t = (oof_tfidf > 0.5).astype(int)
pred_g = (oof_glove > 0.5).astype(int)
pred_b = (oof_bert > 0.5).astype(int)

disagree = (pred_t != pred_g) | (pred_g != pred_b) | (pred_t != pred_b)

df = train.copy()
df["true"] = y
df["p_tfidf"] = oof_tfidf
df["p_glove"] = oof_glove
df["p_bert"] = oof_bert
df["pred_tfidf"] = pred_t
df["pred_glove"] = pred_g
df["pred_bert"] = pred_b
df["disagree"] = disagree

dis = df[df["disagree"]].copy()

print("\n" + "=" * 70)
print("BERT RIGHT, TF-IDF WRONG (BERT understands context)")
print("=" * 70)
mask = (dis["pred_bert"] == dis["true"]) & (dis["pred_tfidf"] != dis["true"])
subset = dis[mask].sample(min(10, mask.sum()), random_state=42)
for _, r in subset.iterrows():
    label = "DISASTER" if r["true"] == 1 else "NOT DISASTER"
    kw = r["keyword"] if pd.notna(r["keyword"]) else "—"
    print(f"\n  [{label}] keyword={kw}")
    print(f"  \"{r['text'][:120]}\"")
    print(f"  TF-IDF={r['p_tfidf']:.2f}→{'D' if r['pred_tfidf'] else 'N'}  "
          f"GloVe={r['p_glove']:.2f}→{'D' if r['pred_glove'] else 'N'}  "
          f"BERT={r['p_bert']:.2f}→{'D' if r['pred_bert'] else 'N'}")

print("\n\n" + "=" * 70)
print("TF-IDF RIGHT, BERT WRONG (keyword signal BERT missed)")
print("=" * 70)
mask2 = (dis["pred_tfidf"] == dis["true"]) & (dis["pred_bert"] != dis["true"])
subset2 = dis[mask2].sample(min(10, mask2.sum()), random_state=42)
for _, r in subset2.iterrows():
    label = "DISASTER" if r["true"] == 1 else "NOT DISASTER"
    kw = r["keyword"] if pd.notna(r["keyword"]) else "—"
    print(f"\n  [{label}] keyword={kw}")
    print(f"  \"{r['text'][:120]}\"")
    print(f"  TF-IDF={r['p_tfidf']:.2f}→{'D' if r['pred_tfidf'] else 'N'}  "
          f"GloVe={r['p_glove']:.2f}→{'D' if r['pred_glove'] else 'N'}  "
          f"BERT={r['p_bert']:.2f}→{'D' if r['pred_bert'] else 'N'}")

print("\n\n" + "=" * 70)
print("ALL MODELS WRONG (genuinely hard cases)")
print("=" * 70)
mask3 = (dis["pred_tfidf"] != dis["true"]) & (dis["pred_glove"] != dis["true"]) & (dis["pred_bert"] != dis["true"])
subset3 = dis[mask3].sample(min(10, mask3.sum()), random_state=42)
for _, r in subset3.iterrows():
    label = "DISASTER" if r["true"] == 1 else "NOT DISASTER"
    kw = r["keyword"] if pd.notna(r["keyword"]) else "—"
    print(f"\n  [{label}] keyword={kw}")
    print(f"  \"{r['text'][:120]}\"")
    print(f"  TF-IDF={r['p_tfidf']:.2f}→{'D' if r['pred_tfidf'] else 'N'}  "
          f"GloVe={r['p_glove']:.2f}→{'D' if r['pred_glove'] else 'N'}  "
          f"BERT={r['p_bert']:.2f}→{'D' if r['pred_bert'] else 'N'}")

print(f"\n\nCounts:")
print(f"  BERT right, TF-IDF wrong: {((dis['pred_bert'] == dis['true']) & (dis['pred_tfidf'] != dis['true'])).sum()}")
print(f"  TF-IDF right, BERT wrong: {((dis['pred_tfidf'] == dis['true']) & (dis['pred_bert'] != dis['true'])).sum()}")
print(f"  All 3 wrong:              {mask3.sum()}")
