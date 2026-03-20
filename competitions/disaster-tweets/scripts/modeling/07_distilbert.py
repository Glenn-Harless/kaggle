"""
Disaster Tweets: Step 7 — Fine-tune DistilBERT

The modern approach: take a pre-trained transformer and fine-tune it
on our specific classification task.

DistilBERT:
- 66M parameters (vs BERT's 110M) — 40% smaller, 60% faster
- Pre-trained on Wikipedia + BookCorpus — already "understands" English
- We add a classification head on top and train on our 7,613 tweets
- The model produces CONTEXT-DEPENDENT embeddings:
  "fire" gets different vectors in "forest fire" vs "you're fired"

Architecture:
  Tweet text → DistilBERT tokenizer → token IDs
  → DistilBERT encoder (6 transformer layers)
  → [CLS] token embedding (768-dim, represents whole sentence)
  → Classification head (768 → 2)
  → Softmax → P(disaster)
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from shared.evaluate import Tee, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/07_distilbert.txt")
sys.stdout = tee

print("Disaster Tweets: Step 7 — Fine-tune DistilBERT")
print("=" * 60)

# Device selection
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

print(f"  Train: {len(train)} tweets")
print(f"  Test:  {len(test)} tweets")
print(f"  Current best (GloVe + TF-IDF concat): F1 = 0.773")


# ============================================================
# TOKENIZATION
# ============================================================

print("\n\n" + "=" * 60)
print("TOKENIZATION")
print("=" * 60)
print("""
  DistilBERT uses WordPiece tokenization — different from our simple
  whitespace splitting:

  Simple:    "earthquake" → ["earthquake"]     (1 token)
  WordPiece: "earthquake" → ["earth", "##quake"] (2 tokens)

  The ## prefix means "continuation of previous word." This lets the
  model handle ANY word by breaking it into known subword pieces.
  Even misspellings and slang get tokenized, solving the OOV problem
  that GloVe had (9.4% unknown tokens).

  Max sequence length: 128 tokens (tweets are short, rarely hit this).
""")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Show tokenization examples
examples = [
    "Forest fire near La Ronge Sask. Canada",
    "OMG that mixtape is fire 🔥🔥🔥",
    "earthquake measuring 7.1 magnitude hits Nepal",
]
for text in examples:
    tokens = tokenizer.tokenize(text)
    print(f"  \"{text}\"")
    print(f"    → {tokens}")
    print()


# ============================================================
# DATASET CLASS
# ============================================================

MAX_LEN = 128


class TweetDataset(Dataset):
    """PyTorch Dataset for tweet classification."""

    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            list(texts),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# TRAINING FUNCTION
# ============================================================

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5


def train_fold(train_texts, train_labels, val_texts, val_labels, fold_num):
    """Train DistilBERT on one fold, return val F1 and predictions."""

    # Create datasets
    train_dataset = TweetDataset(train_texts, train_labels)
    val_dataset = TweetDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load fresh pre-trained model for each fold
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    model.to(DEVICE)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)

        val_f1 = f1_score(val_labels, val_preds)
        print(f"    Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}, val_F1={val_f1:.4f}")

    return val_f1, np.array(val_preds), model


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)
print(f"""
  Hyperparameters:
    Epochs:        {EPOCHS}
    Batch size:    {BATCH_SIZE}
    Learning rate: {LEARNING_RATE}
    Max seq len:   {MAX_LEN}
    Warmup:        10% of total steps
    Weight decay:  0.01
""")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
oof_preds = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
    print(f"\n  --- Fold {fold+1}/5 ---")

    train_texts = train["text"].iloc[train_idx].values
    val_texts = train["text"].iloc[val_idx].values
    train_labels = y[train_idx]
    val_labels = y[val_idx]

    val_f1, val_preds, model = train_fold(
        train_texts, train_labels, val_texts, val_labels, fold + 1
    )
    fold_scores.append(val_f1)
    oof_preds[val_idx] = val_preds

    print(f"    Fold {fold+1} final F1: {val_f1:.4f}")

    # Free GPU memory between folds
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

fold_scores = np.array(fold_scores)

print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
report_f1_cv(fold_scores, "DistilBERT")

overall_f1 = f1_score(y, oof_preds)
print(f"\n  Overall OOF F1: {overall_f1:.5f}")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("COMPARISON — ALL PHASES")
print("=" * 60)

print(f"""
  {'Model':35s} {'Mean F1':>10s} {'Delta':>10s}
  {'-' * 58}
  {"Keyword-only (Step 2)":35s} {"0.67270":>10s} {"—":>10s}
  {"TF-IDF + LogReg (Step 2)":35s} {"0.75478":>10s} {"baseline":>10s}
  {"GloVe + TF-IDF concat (Step 5)":35s} {"0.77273":>10s} {"+0.018":>10s}
  {"DistilBERT (Step 7)":35s} {fold_scores.mean():>10.5f} {fold_scores.mean() - 0.75478:>+10.5f}
""")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print("""
  1. DistilBERT produces CONTEXT-DEPENDENT embeddings. Unlike GloVe,
     "fire" gets a different vector in "forest fire" vs "you're fired."
     This is the fundamental advantage of transformers.

  2. FINE-TUNING means: DistilBERT already knows English from pre-training.
     We taught it the narrow task "is this tweet about a real disaster?"
     by training a small classification head on top.

  3. WORDPIECE TOKENIZATION solves the OOV problem: unknown words get split
     into known subword pieces. No more 9.4% unknown tokens.

  4. The training loop: forward pass → compute loss → backward pass →
     update weights. Same as any neural network, just with a pre-trained
     starting point instead of random initialization.

  5. LEARNING RATE is critical for transformers: 2e-5 is tiny compared to
     training from scratch. We're making small adjustments to weights that
     already encode rich language understanding.
""")

tee.close()
print("Done. Results saved to results/models/07_distilbert.txt")
