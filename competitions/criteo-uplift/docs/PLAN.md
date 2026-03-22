# Criteo Uplift Modeling: Plan of Attack

**Dataset:** [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
**Goal:** Learning-focused — understand causal inference and uplift modeling, directly applicable to geo experiment work
**Approach:** Progression from naive baselines to sophisticated meta-learners, building causal inference intuition at each step
**Date:** 2026-03-22

---

## The Dataset

- **14M users** from a real Criteo randomized controlled trial (RCT)
- Users were randomly assigned: ~85% saw ads (treatment), ~15% didn't (control)
- Two outcomes: `visit` (4.7% rate) and `conversion` (0.29% rate)
- 12 anonymized features (`f0`-`f11`), float-valued
- No official train/test split — we create our own

### Key Columns

| Column | Type | Description |
|---|---|---|
| `f0`-`f11` | float | 12 anonymized features (randomly projected) |
| `treatment` | 0/1 | Randomized intent-to-treat assignment |
| `exposure` | 0/1 | Whether user actually saw the ad (use `treatment` for clean causal estimation) |
| `visit` | 0/1 | Did user visit? (~4.7% rate) |
| `conversion` | 0/1 | Did user convert? (~0.29% rate) |

### Important Nuance: `treatment` vs `exposure`

`treatment` = randomized assignment (clean, no bias). `exposure` = actually saw the ad (biased — only users who browse certain pages get exposed). **Always use `treatment` as the treatment column** for unbiased causal estimation. The `exposure` column is useful for understanding noncompliance but introduces selection bias if used naively.

## Key Differences from Previous Projects

| | Titanic / House Prices / Disaster Tweets | Criteo Uplift |
|---|---|---|
| Goal | Predict Y (survival, price, disaster) | Predict the EFFECT of treatment on Y |
| Label | Directly observed | Never directly observed (fundamental problem of causal inference) |
| Metric | Accuracy, RMSLE, F1 | AUUC, Qini coefficient (measures quality of uplift ranking) |
| Model output | P(Y=1) | τ(x) = P(Y=1|treated) - P(Y=1|control) — the individual treatment effect |
| Key challenge | Feature engineering, representation | You can never observe both outcomes for the same person |

## The Big Idea: The Fundamental Problem of Causal Inference

For any individual user, you can observe only ONE outcome:
- If treated: you see Y(1) but never Y(0)
- If control: you see Y(0) but never Y(1)

The individual treatment effect τᵢ = Y(1) - Y(0) is **never directly observable**. This is fundamentally different from classification or regression where you can always check your prediction against the truth.

Uplift modeling works around this by estimating CONDITIONAL AVERAGE treatment effects — the average effect for users with similar features. The RCT gives us unbiased group-level effects; the models learn to predict which subgroups respond most.

### The Four Quadrants

Understanding these user segments is the whole point:

```
                        Would convert WITHOUT ad
                        No                  Yes
                    ┌──────────────────┬──────────────────┐
  Converts WITH ad  │ PERSUADABLES     │ SURE THINGS      │
  Yes               │ τ > 0            │ τ ≈ 0            │
                    │ ← target these!  │ waste of ad spend │
                    ├──────────────────┼──────────────────┤
  No                │ LOST CAUSES      │ SLEEPING DOGS    │
                    │ τ ≈ 0            │ τ < 0            │
                    │ don't bother     │ ads hurt! avoid  │
                    └──────────────────┴──────────────────┘
```

Most ad platforms waste budget on Sure Things (high-propensity users who would have converted anyway). Uplift modeling identifies Persuadables — users who convert BECAUSE of the ad.

---

## Connection to Your Geo Experiment Work

```
Geo experiments (what you do):        Uplift modeling (what you'll learn):
─────────────────────────────         ────────────────────────────────────
Treatment: Dallas sees ads            Treatment: user sees ad
Control: Houston doesn't              Control: user doesn't see ad
Measure: aggregate lift in region     Measure: individual-level τ(x)
Question: "Did the campaign work?"    Question: "WHO did it work for?"
Limitation: can't see which users     Strength: estimates per-user effect
Output: one number (total lift)       Output: ranked users by responsiveness
```

The Criteo dataset is effectively a user-level version of your geo experiment — same fundamental question, finer granularity. The causal inference vocabulary (ATE, CATE, counterfactual, potential outcomes) applies to both.

---

## Steps

### Phase 1: Foundation (Steps 1-3)

#### Step 1: EDA — Understanding Experimental Data

Explore the dataset with a causal inference lens (different from predictive EDA):

- Treatment/control balance and outcome rates per group
- ATE calculation: raw difference in conversion rates (treatment vs control)
- Feature distributions: any differences between treatment and control? (should be balanced if randomized)
- Outcome rarity: conversion is 0.29% — implications for modeling
- `treatment` vs `exposure` comparison — understand the noncompliance gap
- Sample size considerations: with 14M rows, we'll work with a 10% sample for development

This is different from our previous EDA — we're not looking for predictive features, we're verifying the experiment is clean and understanding the treatment effect.

- Script: `scripts/preprocessing/01_eda.py`

**What you learn:** How to validate a randomized experiment, what ATE means, why conversion rarity matters.

#### Step 2: Naive Baselines — What NOT to Do

Build intentionally wrong approaches to understand why uplift modeling exists:

1. **Standard classifier:** Train LogReg to predict P(conversion). Rank by predicted probability. This targets SURE THINGS, not persuadables.
2. **Separate classifiers:** Train one model on treatment data, one on control. Subtract predictions. This is a T-learner but without understanding why.
3. **Random targeting:** Predict uplift = 0 for everyone (random targeting baseline).

Compare all three using uplift curves and AUUC. The standard classifier should perform WORSE than random for uplift — this is the key "aha" moment.

- Script: `scripts/modeling/02_naive_baselines.py`

**What you learn:** Why predicting propensity ≠ predicting uplift. High-propensity users are often Sure Things, not Persuadables.

#### Step 3: S-Learner — The Simplest Uplift Model

The S-learner (Single model) is the simplest real uplift approach:

1. Train ONE model: Y ~ f(X, W) where W is the treatment indicator
2. Predict τ(x) = f(x, W=1) - f(x, W=0)

Test with LightGBM as the base learner. Compare to naive baselines.

- Script: `scripts/modeling/03_s_learner.py`

**Primer — S-Learner:**

The treatment indicator is just another feature column. The model learns how treatment interacts with other features to affect the outcome. The uplift prediction is the difference between the model's prediction with treatment=1 vs treatment=0.

Problem: regularization can shrink the treatment effect toward zero. If the treatment signal is weak relative to other features, the model may effectively ignore it. This is the main limitation S-learner and why T-learner exists.

**What you learn:** How to frame uplift as a prediction problem, AUUC/Qini evaluation, S-learner's strengths and limitations.

### Phase 2: Meta-Learners (Steps 4-6)

#### Step 4: T-Learner — Two Separate Models

The T-learner (Two models) trains separate models for treatment and control:

1. Model₁: predict Y using only treatment group data
2. Model₀: predict Y using only control group data
3. τ(x) = Model₁(x) - Model₀(x)

Compare T-learner vs S-learner on AUUC/Qini.

- Script: `scripts/modeling/04_t_learner.py`

**Primer — T-Learner:**

By training separate models, we force the treatment effect to be captured — it can't be regularized away like in S-learner. But the downside: with 85% treatment and 15% control, the control model is trained on far less data. Different model complexities between the two models can create spurious heterogeneity.

**What you learn:** The bias-variance tradeoff specific to uplift modeling, how treatment group imbalance affects estimation.

#### Step 5: X-Learner — Designed for Imbalanced Treatment

The X-learner (Cross-learner) is specifically designed for imbalanced treatment groups like Criteo's 85/15 split:

1. Stage 1: Fit T-learner models (μ₁, μ₀)
2. Stage 2: Impute treatment effects:
   - For treated users: D₁ = Y₁ - μ₀(X₁) (actual outcome minus predicted control outcome)
   - For control users: D₀ = μ₁(X₀) - Y₀ (predicted treatment outcome minus actual outcome)
3. Stage 3: Fit models on imputed effects: τ₁(x), τ₀(x)
4. Final: τ(x) = g(x)·τ₀(x) + (1-g(x))·τ₁(x) weighted by propensity score g(x)

- Script: `scripts/modeling/05_x_learner.py`

**What you learn:** Why treatment group imbalance matters, how propensity scores help, cross-estimation for variance reduction.

#### Step 6: Model Comparison + Uplift Curves

Compare all approaches head-to-head:

- S-learner, T-learner, X-learner
- Uplift curves (cumulative gain)
- Qini curves
- AUUC scores
- Uplift@10%, @20%, @30% (practical: "if we can only target X% of users...")
- Feature importance: which features drive heterogeneous treatment effects?

- Script: `scripts/modeling/06_model_comparison.py`

**What you learn:** How to evaluate and compare uplift models, practical targeting decisions, which approach wins and why.

### Phase 3: Advanced Topics (Steps 7-8)

#### Step 7: DR-Learner + Causal Forest

More sophisticated approaches:

1. **DR-Learner (Doubly Robust):** Consistent if EITHER propensity or outcome model is correct. Uses three-fold cross-fitting. More robust than S/T/X-learners.
2. **Causal Forest (EconML):** Nonparametric with valid confidence intervals. Can quantify uncertainty in treatment effect estimates.

- Script: `scripts/modeling/07_advanced_models.py`

**What you learn:** Doubly robust estimation, confidence intervals for causal effects, when to use each approach.

#### Step 8: Practical Targeting Analysis

Apply the best model to answer real business questions:

1. **Targeting curve:** How much budget can we save by only targeting persuadables?
2. **Cost-benefit analysis:** If each ad costs $X and each conversion is worth $Y, what's the optimal targeting threshold?
3. **Segment analysis:** Which feature-defined segments have the highest uplift?
4. **Connection to geo experiments:** How would you validate these user-level predictions with a geo experiment?

- Script: `scripts/modeling/08_targeting_analysis.py`

**What you learn:** How to translate uplift model outputs into business decisions — the bridge between modeling and marketing strategy.

### Phase 4: Wrap-up (Step 9)

#### Step 9: Retrospective

- Which meta-learner worked best and why?
- How does uplift modeling complement geo experiments?
- What surprised you about causal inference vs predictive modeling?
- How would you apply this at work?
- What would you explain differently to stakeholders?

- File: `RETROSPECTIVE.md`

---

## Validation Protocol

- **Train/test split:** 70/30 stratified by treatment AND conversion (or use 10% sample for development)
- **Metric:** AUUC (Area Under Uplift Curve) — primary. Qini coefficient — secondary.
- **Cross-validation:** StratifiedKFold(5) with stratification on treatment assignment
- **Important:** Evaluation requires both treatment and control users in every fold

## Libraries

- `scikit-uplift` — data loading, S/T-learner wrappers, AUUC/Qini metrics
- `causalml` (Uber) — X-learner, DR-learner, uplift trees
- `econml` (Microsoft) — Causal Forest, confidence intervals
- `lightgbm` — base learner for all meta-learners

## Principles (carried forward)

1. **Every change gets validated** — AUUC/Qini before and after
2. **Understand before optimizing** — the point is learning causal inference, not leaderboard
3. **Explain deeply** — every step includes the *why*, not just the *what*
4. **Clean code, clear output** — numbered scripts, Tee logging
5. **Connect to real work** — relate every concept back to geo experiments and marketing

## File Structure

```
competitions/criteo-uplift/
├── data/                         # Raw data (gitignored)
├── docs/
│   └── PLAN.md                   # This file
├── scripts/
│   ├── preprocessing/
│   │   └── 01_eda.py
│   └── modeling/
│       ├── 02_naive_baselines.py
│       ├── 03_s_learner.py
│       ├── 04_t_learner.py
│       ├── 05_x_learner.py
│       ├── 06_model_comparison.py
│       ├── 07_advanced_models.py
│       └── 08_targeting_analysis.py
├── results/models/               # AUUC scores, logs
├── results/analysis/             # Uplift curves, plots
└── RETROSPECTIVE.md              # Final reflection
```

## Key References

- [Criteo dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Extended paper (2021)](https://arxiv.org/abs/2111.10106)
- [Meta-learners paper (Kunzel 2017)](https://arxiv.org/abs/1706.03461)
- [scikit-uplift docs](https://www.uplift-modeling.com/en/latest/)
- [CausalML docs](https://causalml.readthedocs.io/en/latest/)
- [Causal Inference for the Brave and True — Ch. 21](https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html)
- [Google geo experiments paper](https://research.google/pubs/pub45950/)
