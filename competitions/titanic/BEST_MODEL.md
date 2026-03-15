# Best Model: v2 Logistic Regression + Ticket-Mate Rules

**Kaggle score: 0.7751** (324/418 correct)
**Submission file:** `submissions/logreg_18_18b_hybrid_2mate.csv`
**Script:** `scripts/modeling/18_ticket_survival_propagation.py`

---

## How It Works

The model has two layers: a logistic regression base model, and a rule override layer.

### Layer 1: Logistic Regression (v2 features)

A standard logistic regression (`sklearn.linear_model.LogisticRegression`) with:
- **C=0.01** (strong L2 regularization)
- **StandardScaler** preprocessing
- **15 features** (the "v2" feature set)

The model takes a passenger's features and outputs a probability of survival. If P(survived) >= 0.5, predict survived; otherwise predict died.

#### The 15 Features

Built from raw Titanic data by `scripts/preprocessing/build_features.py`. The processed data is saved as `data/train_processed.csv` (v3 format with 21 columns), but the v2 model uses only 15 of those via `shared/evaluate.py:reconstruct_v2_features()`.

| Feature | Type | What it means |
|---------|------|---------------|
| **Sex** | Binary | 1 = male, 0 = female |
| **Pclass** | Ordinal (1/2/3) | Ticket class (1st, 2nd, 3rd) |
| **Fare** | Continuous | Ticket price in pounds |
| **IsChild** | Binary | Age <= 12 |
| **IsAlone** | Binary | No siblings, spouse, parents, or children aboard |
| **IsLargeFamily** | Binary | 4+ family members aboard |
| **HasCabin** | Binary | Cabin number recorded (proxy for wealth/status) |
| **Title_Mr** | Binary | Passenger's title is Mr (adult men) |
| **Title_Mrs** | Binary | Passenger's title is Mrs (married women) |
| **Title_Miss** | Binary | Passenger's title is Miss (unmarried women) |
| **Title_Master** | Binary | Passenger's title is "Master" (boys) |
| **Title_Rare** | Binary | Uncommon title: Dr, Rev, Col, Major, etc. |
| **Emb_C** | Binary | Embarked at Cherbourg |
| **Emb_Q** | Binary | Embarked at Queenstown |
| **Emb_S** | Binary | Embarked at Southampton |

#### What the Coefficients Say

Sorted by importance (absolute coefficient after standardization):

| Feature | Coefficient | Direction |
|---------|------------|-----------|
| Title_Mr | -0.403 | Mr = less likely to survive |
| Sex (male) | -0.384 | Male = less likely to survive |
| IsLargeFamily | -0.321 | Big families died together |
| Pclass | -0.295 | Higher class number = lower survival |
| Title_Mrs | +0.254 | Mrs = more likely to survive |
| HasCabin | +0.222 | Recorded cabin = higher status |
| Title_Miss | +0.210 | Miss = more likely to survive |
| Title_Master | +0.177 | Boys = more likely to survive |
| Fare | +0.126 | Higher fare = higher survival |
| IsChild | +0.101 | Children = more likely to survive |
| Emb_C | +0.073 | Cherbourg embarkation = slight positive |
| Title_Rare | -0.073 | Military/professional titles = slight negative |
| Emb_S | -0.071 | Southampton = slight negative |
| Emb_Q | +0.011 | Negligible |
| IsAlone | -0.009 | Negligible |

**In plain English:** The model primarily predicts based on sex (women survive) modified by class (lower class reduces survival) and family structure (large families are bad, having a cabin is good, being a child helps).

### Layer 2: Ticket-Mate Rule Overrides

After the logistic regression makes its prediction, a rule layer checks whether the passenger has **ticket-mates** in the training data — other passengers who shared the same ticket number (i.e., traveled together, likely family or companions).

**The rule:** If a test passenger has 2 or more ticket-mates in training, AND those mates have a **unanimous** survival outcome (all survived or all died), override the logistic prediction with the group outcome.

- All mates survived → predict survived (probability set to 0.95)
- All mates died → predict died (probability set to 0.05)
- Mixed outcomes or only 1 mate → keep the logistic prediction

**Why it works:** Families and travel parties tended to survive or die together. This is historical fact — lifeboats were loaded by groups, and families stayed together. If three members of a ticket group all died in training, the fourth member (in test) very likely died too.

**Why it's a rule and not a feature:** Adding the ticket-mate survival rate as a continuous feature was tried (step 18a) and was catastrophic (-3.2% CV). The logistic model under C=0.01 regularization couldn't learn to trust it — it suppressed the signal along with everything else. As a hard rule, the signal bypasses regularization entirely.

#### Rule Statistics on Test Set

- 48 test passengers have 2+ ticket-mates in training
- Of those, 48 trigger a rule (all have unanimous mates at 2+ count)
- 13 of those rules actually change the logistic prediction
- 7 flipped to survived (mostly men with surviving families)
- 6 flipped to died (mostly Pclass 3 from families that perished)

---

## How to Reproduce

```bash
# Step 1: Preprocess raw data → v3 features
uv run python competitions/titanic/scripts/preprocessing/build_features.py

# Step 2: Run the model (includes v2 reconstruction, ticket rules, and submission)
uv run python competitions/titanic/scripts/modeling/18_ticket_survival_propagation.py
```

The submission file is `submissions/logreg_18_18b_hybrid_2mate.csv`.

---

## File Map

### Active (current best model pipeline)

```
scripts/preprocessing/build_features.py   — raw CSV → v3 processed features
shared/evaluate.py                        — evaluation harness + reconstruct_v2_features()
scripts/modeling/18_ticket_survival_propagation.py — best model (v2 logistic + rules)
scripts/modeling/10_logreg_v2.py          — original v2 baseline (no rules)
scripts/modeling/13_v2_rebaseline.py      — verified v2 baseline reproduction
results/models/13_v2_baseline_cv_scores.npy — frozen baseline CV scores
```

### Analysis (error audit, retrospective)

```
scripts/modeling/22_error_audit.py        — error structure of the final model
scripts/modeling/22b_error_distribution.py — detailed error distribution
scripts/modeling/19_surname_survival_propagation.py — surname extension (neutral result, reference)
RETROSPECTIVE.md                          — full project retrospective + experiment registry
BEST_MODEL.md                             — this file
```

### Submissions (active)

```
submissions/gender_only.csv               — floor baseline (Kaggle 0.7655)
submissions/logreg_v2.csv                 — v2 logistic, no rules (Kaggle 0.7727)
submissions/logreg_18_18b_hybrid_2mate.csv — BEST: v2 logistic + rules (Kaggle 0.7751)
submissions/logreg_19_19b_ticket_surname.csv — surname extension (Kaggle 0.7751, tied)
```

### Archived (explored and rejected)

Everything in `scripts/modeling/archive/`, `results/models/archive/`, and `submissions/archive/` is a completed experiment that is no longer on the active path. See `RETROSPECTIVE.md` for what each one tried and why it was rejected.
