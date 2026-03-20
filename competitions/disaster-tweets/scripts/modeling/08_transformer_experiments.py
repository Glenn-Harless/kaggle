"""
Disaster Tweets: Step 8 — Transformer Experiments

Compare transformer variants and tuning approaches:
1. Learning rate sweep (1e-5, 2e-5, 3e-5, 5e-5)
2. DistilBERT vs BERT-base (66M vs 110M parameters)
3. More epochs (3 vs 4 vs 5) — does more training help or overfit?
4. Freeze bottom layers — fine-tune only the top transformer layers

Each experiment uses 5-fold CV with the same splits for fair comparison.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from shared.evaluate import Tee, report_f1_cv

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/08_transformer_experiments.txt")
sys.stdout = tee

print("Disaster Tweets: Step 8 — Transformer Experiments")
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

print(f"  Train: {len(train)} tweets")
print(f"  Current best (DistilBERT, lr=2e-5, 3 epochs): F1 = 0.806")


# ============================================================
# SHARED TRAINING INFRASTRUCTURE
# ============================================================

MAX_LEN = 128
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.labels = labels
        self.encodings = tokenizer(
            list(texts), max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_and_eval(model_name, lr, epochs, batch_size=32, freeze_layers=0):
    """Train a transformer model with given hyperparameters, return 5-fold CV F1."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(CV.split(train, y)):
        train_texts = train["text"].iloc[train_idx].values
        val_texts = train["text"].iloc[val_idx].values
        train_labels = y[train_idx]
        val_labels = y[val_idx]

        train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
        val_dataset = TweetDataset(val_texts, val_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Load fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

        # Freeze bottom N transformer layers if requested
        if freeze_layers > 0 and hasattr(model, "distilbert"):
            for i in range(freeze_layers):
                for param in model.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
        elif freeze_layers > 0 and hasattr(model, "bert"):
            for i in range(freeze_layers):
                for param in model.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.01,
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_f1 = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
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
                total_loss += outputs.loss.item()

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                    )
                    val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

            epoch_f1 = f1_score(val_labels, val_preds)
            best_f1 = max(best_f1, epoch_f1)

        fold_scores.append(best_f1)
        print(f"    Fold {fold+1}: F1 = {best_f1:.4f}")

        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    fold_scores = np.array(fold_scores)
    return fold_scores


# ============================================================
# EXPERIMENT 1: LEARNING RATE SWEEP
# ============================================================

print("\n\n" + "=" * 60)
print("EXPERIMENT 1: LEARNING RATE SWEEP (DistilBERT, 3 epochs)")
print("=" * 60)
print("""
  Transformers are very sensitive to learning rate. Too high = destroy
  pre-trained knowledge. Too low = don't learn the task.
  Testing: 1e-5, 2e-5, 3e-5, 5e-5
""")

lr_results = {}
for lr in [1e-5, 2e-5, 3e-5, 5e-5]:
    print(f"\n  --- lr = {lr} ---")
    scores = train_and_eval("distilbert-base-uncased", lr=lr, epochs=3)
    report_f1_cv(scores, f"lr={lr}")
    lr_results[lr] = scores

best_lr = max(lr_results, key=lambda k: lr_results[k].mean())
print(f"\n  Best learning rate: {best_lr} (F1 = {lr_results[best_lr].mean():.5f})")


# ============================================================
# EXPERIMENT 2: EPOCH COUNT
# ============================================================

print("\n\n" + "=" * 60)
print(f"EXPERIMENT 2: EPOCH COUNT (DistilBERT, lr={best_lr})")
print("=" * 60)
print("""
  More epochs = more passes through data. Risk of overfitting on
  small datasets. BERT paper recommends 2-4 epochs.
""")

epoch_results = {}
for n_epochs in [2, 3, 4, 5]:
    print(f"\n  --- {n_epochs} epochs ---")
    scores = train_and_eval("distilbert-base-uncased", lr=best_lr, epochs=n_epochs)
    report_f1_cv(scores, f"{n_epochs} epochs")
    epoch_results[n_epochs] = scores

best_epochs = max(epoch_results, key=lambda k: epoch_results[k].mean())
print(f"\n  Best epoch count: {best_epochs} (F1 = {epoch_results[best_epochs].mean():.5f})")


# ============================================================
# EXPERIMENT 3: DistilBERT vs BERT-base
# ============================================================

print("\n\n" + "=" * 60)
print(f"EXPERIMENT 3: DistilBERT vs BERT-base (lr={best_lr}, {best_epochs} epochs)")
print("=" * 60)
print("""
  DistilBERT: 6 layers, 66M params — smaller, faster
  BERT-base:  12 layers, 110M params — more capacity, slower

  Does the extra capacity help on 7,613 short tweets?
""")

print(f"\n  --- DistilBERT (already computed) ---")
distilbert_scores = epoch_results[best_epochs]
report_f1_cv(distilbert_scores, "DistilBERT")

print(f"\n  --- BERT-base ---")
bert_scores = train_and_eval("bert-base-uncased", lr=best_lr, epochs=best_epochs)
report_f1_cv(bert_scores, "BERT-base")

delta = bert_scores.mean() - distilbert_scores.mean()
print(f"\n  BERT-base vs DistilBERT delta: {delta:+.5f} F1")


# ============================================================
# EXPERIMENT 4: FREEZE BOTTOM LAYERS
# ============================================================

print("\n\n" + "=" * 60)
print(f"EXPERIMENT 4: FREEZE BOTTOM LAYERS (DistilBERT, lr={best_lr}, {best_epochs} epochs)")
print("=" * 60)
print("""
  Bottom layers capture general language patterns (grammar, word types).
  Top layers capture task-specific patterns.
  Freezing bottom layers = train only the top layers.
  Faster training, less risk of overfitting, but less capacity to adapt.
""")

print(f"\n  --- Freeze 0 layers (full fine-tune, already computed) ---")
report_f1_cv(distilbert_scores, "Freeze 0")

print(f"\n  --- Freeze 3 of 6 layers (bottom half) ---")
freeze3_scores = train_and_eval(
    "distilbert-base-uncased", lr=best_lr, epochs=best_epochs, freeze_layers=3
)
report_f1_cv(freeze3_scores, "Freeze 3")

delta_freeze = freeze3_scores.mean() - distilbert_scores.mean()
print(f"\n  Freeze 3 vs full fine-tune delta: {delta_freeze:+.5f} F1")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 60)
print("FINAL COMPARISON — ALL TRANSFORMER EXPERIMENTS")
print("=" * 60)

all_results = {}
for lr, scores in lr_results.items():
    all_results[f"DistilBERT lr={lr} 3ep"] = scores
for ep, scores in epoch_results.items():
    all_results[f"DistilBERT lr={best_lr} {ep}ep"] = scores
all_results["BERT-base"] = bert_scores
all_results["DistilBERT freeze3"] = freeze3_scores

# Deduplicate the baseline entry
baseline_key = f"DistilBERT lr={best_lr} {best_epochs}ep"

print(f"\n  {'Model':40s} {'Mean F1':>10s} {'Std':>10s}")
print(f"  {'-' * 63}")
for name, scores in sorted(all_results.items(), key=lambda x: x[1].mean(), reverse=True):
    marker = " <<<" if name == baseline_key else ""
    print(f"  {name:40s} {scores.mean():>10.5f} {scores.std():>10.5f}{marker}")

best_name = max(all_results, key=lambda k: all_results[k].mean())
best_score = all_results[best_name].mean()
print(f"\n  Best overall: {best_name} (F1 = {best_score:.5f})")


# ============================================================
# WHAT WE LEARNED
# ============================================================

print("\n\n" + "=" * 60)
print("WHAT WE LEARNED")
print("=" * 60)
print(f"""
  1. LEARNING RATE: Transformers are highly sensitive to LR. The difference
     between best and worst LR is often larger than the gap between model
     architectures. Always sweep LR first.

  2. EPOCHS: 3-4 is usually the sweet spot for BERT fine-tuning on small
     datasets. More epochs risk overfitting — the model starts memorizing
     training tweets rather than learning generalizable patterns.

  3. BERT-base vs DistilBERT: Does 110M params beat 66M params on 7,613
     tweets? More capacity isn't always better on small datasets — bigger
     models can overfit more easily.

  4. FREEZING LAYERS: On small datasets, freezing bottom layers can prevent
     overfitting. But it also limits how much the model can adapt to the
     task. The tradeoff depends on dataset size and task similarity to
     pre-training.

  5. The best transformer configuration goes forward to Step 9 (ensemble).
""")

tee.close()
print("Done. Results saved to results/models/08_transformer_experiments.txt")
