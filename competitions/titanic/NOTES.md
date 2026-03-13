# Titanic - Machine Learning from Disaster

## Goal
Predict which passengers survived the Titanic shipwreck (binary classification).

## Baseline
- Gender-only model (all women survive, all men die) = 78.7% accuracy. Must beat this.

## Univariate EDA Findings

### Sex (strongest signal)
- Women: 74.2% survived, Men: 18.9% — 55 point gap
- Single most predictive feature

### Pclass
- 1st: 63%, 2nd: 47%, 3rd: 24% — clean monotonic drop
- 55% of passengers were 3rd class (bottom-heavy dataset)

### Age
- Children (0-5): 70% survival — "children first" is real
- All other age groups: 34-43%, roughly flat
- Not a strong linear predictor; really a "child vs. not-child" split
- 20% missing — needs thoughtful imputation

### Fare
- Cheapest quartile: 20%, priciest: 58% — monotonic increase
- Heavily right-skewed ($0-$512, median $14)
- Fare adds within-class signal (22-23 point gap within 1st/2nd class)

### SibSp / Parch / Family Size
- Solo travelers: 30% survival (n=537, majority of passengers)
- Small families (1-3): 55-72% — the sweet spot
- Large families (4+): 0-20% — almost all died
- Non-linear: best modeled as buckets (alone / small / large)
- Family size 3 deep-dive: 72% survival explained by women + young boys surviving while adult fathers died

### Embarked
- Cherbourg: 55%, Queenstown: 39%, Southampton: 34%
- Confounded: Cherbourg was 51% 1st class, Queenstown 94% 3rd class
- Likely no independent predictive power — just reflects class/sex mix at each port
- 2 missing values — fill with "S" (mode)

## Bivariate EDA Findings

### Sex x Pclass (most important interaction)
- 1st class women: 97%, 2nd class women: 92%, 3rd class women: 50% (coin flip)
- 1st class men: 37%, 2nd/3rd class men: ~15%
- Class effect is much stronger for women than men

### Fare vs Pclass
- Correlation: -0.55 (related but not redundant)
- Fare adds within-class signal: 22-23 point gap between below/above median fare in 1st and 2nd class

### Age x Sex
- Children (<=12): ~58% survival regardless of sex — gender-neutral protection
- Gender "switches on" around age 13: girls 75% vs boys 9% in 13-18 bracket
- Adult women: 78%, adult men: 17%
- ANOMALY: Girls age 6-12 survived at only 27% (n=11) vs boys 43% (n=14). Small sample, likely noise.

### Deck (from Cabin)
- Decks A/B/C = 100% first class. D/E mostly 1st. F = 2nd/3rd. G = all 3rd.
- 86% of passengers with cabin data are 1st class — heavy bias
- Not enough 3rd class deck data to impute missing. Stick with HasCabin.

### Title (from Name)
- Mrs: 79%, Miss: 70%, Master: 57%, Mr: 16%
- "Master" perfectly identifies boys age 0-12 (never missing, unlike Age)
- At age 14, boys become "Mr" and survival collapses
- Title enables smarter age imputation: Master median 3.5, Miss 21, Mr 30, Mrs 35

### Ticket Group Size
- Same sweet-spot pattern as FamilySize (groups of 2-3 best)
- Only matches FamilySize 68% of the time — captures 288 passengers in non-family groups
- Genuinely new information beyond SibSp/Parch

## Final Feature Plan

### Features to include in model:
| Feature | Source | Type | Notes |
|---------|--------|------|-------|
| Pclass | raw | ordinal | Keep as-is |
| Sex | raw | binary | Encode male=1, female=0 |
| Age | raw (imputed) | continuous | Impute missing using median age by Title group |
| Fare | raw | continuous | 1 missing in test set — fill with median by Pclass |
| Embarked | raw | categorical | 2 missing — fill with "S" (mode). One-hot encode. |
| Title | extracted from Name | categorical | Group into Mr/Miss/Mrs/Master/Rare. One-hot encode. |
| FamilySize | SibSp + Parch | ordinal | Replaces raw SibSp and Parch |
| IsAlone | FamilySize == 0 | binary | Captures the solo traveler penalty |
| HasCabin | Cabin is not null | binary | 67% vs 30% survival gap |
| TicketGroupSize | count of shared Ticket | ordinal | New info beyond FamilySize for 288 passengers |
| FarePerPerson | Fare / TicketGroupSize | continuous | Slight improvement over raw Fare |
| PersonType | Title + Age + Sex | categorical | Hybrid: "Master" → child, Age<=12 girls → child, else adult_female/adult_male |

### Features to drop:
| Feature | Reason |
|---------|--------|
| PassengerId | Just an index |
| Name | Replaced by Title |
| SibSp | Redundant with FamilySize |
| Parch | Redundant with FamilySize |
| Ticket | Replaced by TicketGroupSize |
| Cabin | Replaced by HasCabin |

### Missing data strategy:
| Feature | Missing | Strategy |
|---------|---------|----------|
| Age | 20% (177 rows) | Median age by Title group (Master: 3.5, Miss: 21, Mr: 30, Mrs: 35) |
| Cabin | 77% (687 rows) | Don't impute — use HasCabin binary flag instead |
| Embarked | 0.2% (2 rows) | Fill with "S" (mode) |
| Fare | 1 row in test | Fill with median Fare for that passenger's Pclass |

### Preprocessing by model type:
| Model Type | One-Hot Encoding | Scaling | Notes |
|------------|-----------------|---------|-------|
| Tree-based (XGBoost, LightGBM, RF) | Optional (included) | Not needed | Trees split on thresholds, invariant to scale |
| Linear (Logistic Regression) | Required | Required | StandardScaler (mean=0, std=1) |
| Neural Network | Required | Required | Add scaling step before NN training |

Current pipeline: one-hot encoding included, scaling NOT included. Add scaling when we try neural networks.

## Key Decisions
- Use Title-based approach for PersonType (handles missing ages)
- Keep both Fare and Pclass (Fare adds within-class signal)
- Drop Deck — too little data for 3rd class to impute reliably
- Hybrid child detection: "Master" title for boys, Age<=12 for girls

## Observations
- The core survival mechanism is "women and children first" + wealth
- Sex is dominant, Pclass second, Fare third
- Non-linear features (Age, FamilySize) need thresholds/buckets, not raw values
- Pearson correlation underestimates non-linear features (Age=-0.077 despite strong child effect)

---

## Pipeline Changelog

### v1 → v2 (2026-03-12)

**Trigger:** Code review + leaderboard analysis. CV scores (83-85%) did not translate
to Kaggle leaderboard (75-77%). Best leaderboard score was simplest model (baseline
LogReg = 77.27%). Tree models / ensembles scored BELOW gender-only baseline (76.55%).

**Root causes identified:**
1. TicketGroupSize computed on combined train+test = data leakage
2. Feature space heavily duplicated (Sex, Title_*, PT_* = same signal 3 ways)
3. EDA concluded non-linear effects but pipeline fed raw continuous values
4. Tree models overtrusted noisy class/fare/ticket signals on 3rd-class women

**v2 Feature set (12 features, down from 20):**
| Feature | Type | Source |
|---------|------|--------|
| Pclass | ordinal | raw |
| Sex | binary | raw (male=1) |
| Fare | continuous | raw |
| HasCabin | binary | Cabin not null |
| IsAlone | binary | FamilySize == 0 |
| IsLargeFamily | binary | FamilySize >= 4 |
| IsChild | binary | Age <= 12 |
| Title_Mr | binary | one-hot from Name |
| Title_Miss | binary | one-hot from Name |
| Title_Mrs | binary | one-hot from Name |
| Title_Master | binary | one-hot from Name |
| Title_Rare | binary | one-hot from Name |
| Emb_C | binary | one-hot from Embarked |
| Emb_Q | binary | one-hot from Embarked |
| Emb_S | binary | one-hot from Embarked |

**Removed from v1:**
| Feature | Reason |
|---------|--------|
| TicketGroupSize | Data leakage (computed on train+test combined) |
| FarePerPerson | Depended on leaky TicketGroupSize |
| PersonType (PT_*) | Redundant with Sex + Title + IsChild |
| raw Age | Replaced by IsChild bucket (non-linear: threshold at ~12) |
| raw FamilySize | Replaced by IsAlone + IsLargeFamily buckets (non-linear) |

**Leaderboard results (v1):**
| Model | CV | Kaggle | Notes |
|-------|-----|--------|-------|
| Baseline LogReg (20 feat) | 83.39% | 77.27% | Best LB score |
| Gender-only | 78.68% | 76.55% | True baseline |
| XGBoost tuned | 85.52% | 75.84% | Below gender-only |
| Ensemble (hard vote) | 85.63% | 75.60% | Worst LB score |

**Key lesson:** Higher CV ≠ better leaderboard. With 891 samples, model complexity
and feature redundancy cause overfitting that CV doesn't catch. Simpler models with
deduplicated, EDA-aligned features generalize better.

### v2 → v3 (2026-03-12)

**Trigger:** v2 model is purely additive — misses the strong Sex × Pclass interaction
that EDA clearly shows (1st class women 97% vs 3rd class 50%). Systematic testing of
interactions, fare transforms, deck features, and surname features.

**Approach:** Added one idea at a time in isolated scripts (11a/11b/11c), measured
each independently, then combined only what proved useful into the final model (11).

**Stage A — Interactions (11a):**
- Replaced ordinal Pclass with one-hot (Pclass_2, Pclass_3; Pclass_1 is reference)
- Added Sex × Pclass_2 and Sex × Pclass_3 interaction terms
- Result: CV 0.8294 (flat vs v2's 0.8305), but tighter std and lower gap
- Coefficients show different sex penalty per class (structurally correct)
- Zero test prediction changes vs v2 — interactions alone don't move the boundary
- **Kept:** structural correctness with no downside

**Stage B — Fare & Deck (11b):**
- B1: log1p(Fare) hurt CV (0.8271) — StandardScaler already handles skew
- B1-alt: Both raw + log1p(Fare) tied at 0.8294 but with 2x std — not worth it
- B2: Grouped deck features (Deck_ABC, Deck_DE, Deck_FG) restored CV to 0.8305
- Deck_DE had strongest coefficient (+0.17); Deck_FG negligible
- 4 passengers flipped vs v2: all male 1st class with known D/E deck cabins
- **Kept:** raw Fare + deck groups. **Rejected:** log1p(Fare)

**Stage C — Surname Features (11c):**
- Used custom CV loop (not cross_val_score) for strict out-of-fold encoding
- SurnameGroupSize: 0.8294 (slightly worse) — redundant with IsAlone/IsLargeFamily
- SurnameSurvHint (OOF): 0.8092 (much worse, 2x std) — most surnames appear once
- Combined: 0.8137 — still below baseline
- Shuffle ablation: both real and shuffled hurt CV, confirming features add noise
- **Rejected:** all surname features

**v3 Feature set (21 features, up from 15):**
| Feature | Type | Source | New in v3 |
|---------|------|--------|-----------|
| Sex | binary | raw (male=1) | |
| Fare | continuous | raw | |
| HasCabin | binary | Cabin not null | |
| IsAlone | binary | FamilySize == 0 | |
| IsLargeFamily | binary | FamilySize >= 4 | |
| IsChild | binary | Age <= 12 | |
| Title_Mr | binary | one-hot from Name | |
| Title_Miss | binary | one-hot from Name | |
| Title_Mrs | binary | one-hot from Name | |
| Title_Master | binary | one-hot from Name | |
| Title_Rare | binary | one-hot from Name | |
| Emb_C | binary | one-hot from Embarked | |
| Emb_Q | binary | one-hot from Embarked | |
| Emb_S | binary | one-hot from Embarked | |
| Pclass_2 | binary | one-hot from Pclass | ✓ |
| Pclass_3 | binary | one-hot from Pclass | ✓ |
| Sex_x_Pclass_2 | binary | Sex × Pclass_2 | ✓ |
| Sex_x_Pclass_3 | binary | Sex × Pclass_3 | ✓ |
| Deck_ABC | binary | Cabin first letter ∈ {A,B,C} | ✓ |
| Deck_DE | binary | Cabin first letter ∈ {D,E} | ✓ |
| Deck_FG | binary | Cabin first letter ∈ {F,G} | ✓ |

**Removed from v2:**
| Feature | Reason |
|---------|--------|
| Pclass (ordinal) | Replaced by Pclass_2 + Pclass_3 one-hot |

**v3 model results (C=0.01, 5-fold StratifiedKFold):**
- CV: 83.05% ± 0.66% (same mean as v2, tighter std)
- Train-CV gap: 0.79% (healthy)
- Test survival rate: 0.414 (v2 was 0.404)
- Female predicted dead: 5 (unchanged)
- 4 passenger changes vs v2 (all male 1st class with known cabins → survived)

**Leaderboard results (v3):**
| Model | CV | Kaggle | Notes |
|-------|-----|--------|-------|
| v3 LogReg (21 feat) | 83.05% | TBD | Interactions + deck |
| v2 LogReg (15 feat) | 83.05% | 77.27% | Additive only |
| v1 LogReg (20 feat) | 83.39% | 77.27% | Had leakage |
| Gender-only | 78.68% | 76.55% | True baseline |

**Key lesson:** Sex × Pclass interactions add structural correctness but don't
dramatically change predictions under strong regularization (C=0.01). Deck features
provide the actual test-time signal for a small subgroup (1st class men with known
cabins). Surname features — even with careful OOF encoding — add noise on a dataset
this small.
