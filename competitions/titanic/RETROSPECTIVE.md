# Titanic Project Retrospective

Last updated: 2026-03-15

## The Result

Best Kaggle score: **0.7751** (324/418 correct)
Gender-only baseline: 0.7655 (320/418 correct)
Net improvement: ~4 passengers

The project's real output is not those 4 passengers. It's the understanding of why those are the only 4.

---

## Experiment Registry

### Submitted to Kaggle

| Step | Model | Hypothesis | CV Mean | Kaggle | Flips | Verdict |
|------|-------|-----------|---------|--------|-------|---------|
| — | Gender-only | Women survive, men die | 0.7870 | 0.7655 | — | Floor baseline |
| 1 | Logistic regression (v1) | Standard features + ticket group size | 0.8316 | 0.7727 | — | Decent baseline |
| 2 | Random forest | Trees capture interactions | 0.8316 | 0.7584 | — | Rejected: overfit |
| 3 | XGBoost | Gradient boosting captures nonlinearity | 0.8552 | 0.7584 | — | Rejected: overfit |
| 9 | Hard-vote ensemble | Combining models reduces variance | 0.8563 | 0.7560 | — | Rejected: worst submission |
| 10 | Logistic v2 (C=0.01) | Clean features + strong regularization | 0.8305 | 0.7727 | — | **New working baseline** |
| 12a | Segmented by sex | Separate models capture sex×feature interactions | 0.8384 | 0.7608 | large | Rejected: overcorrected female Pc3 |
| 16 | RBF SVM | Nonlinear boundaries capture hidden structure | 0.8296 | 0.7655 | 35 | Rejected: same as gender-only |
| **18b** | **v2 + ticket rules (2+ mates)** | **Unanimous ticket-mate outcomes predict survival** | **0.8228** | **0.7751** | **13** | **Best submission** |
| 19b | v2 + ticket + surname rules | Surname+Pclass identifies additional families | 0.8227 | 0.7751 | 21 | Neutral: tied 18b |
| 21 | v3 + ticket rules | Interaction features improve the base model | 0.8203 | 0.7703 | 17 | Rejected: v3 worse than v2 |

### Not submitted (local evaluation only)

| Step | Model | Hypothesis | CV Mean | Key Finding |
|------|-------|-----------|---------|-------------|
| 4 | LightGBM | Another boosting variant | ~0.85 | Same overfit pattern as XGB |
| 5 | Neural network | Deep learning captures complexity | — | Not competitive |
| 6 | XGBoost tuned | Better hyperparameters fix overfit | 0.8552 | Tuning made overfit worse |
| 7 | Random forest tuned | Better hyperparameters fix overfit | 0.8316 | Minimal change |
| 8 | Logistic tuned | Better hyperparameters | 0.8316 | Minimal change |
| 11 | Logistic v3 | One-hot Pclass + interactions | 0.8305 | Matched v2, changed 4 passengers |
| 11a-c | v3 variants | Fare/deck/surname features | ~0.83 | None improved over v3 base |
| 12b | Segment + FareRankInClass | Fare rank within class by sex | — | No improvement over 12a |
| 12c | Segment + threshold tuning | Optimize decision boundary by sex | — | Did not rescue 12a |
| 13 | v2 rebaseline | Verify v2 is reproducible | 0.8301 | Confirmed baseline |
| 14 | Male Pclass 1 interactions | Deck_DE for wealthy men | ~0.83 | Tiny positive signal, not compelling |
| 15 | Ticket group features | Ticket group size as model input | 0.8306 | +0.0007 delta, 0 test flips — ignored by model |
| 17 | Full-manifest groups + C sweep | Full train+test group size; higher C | 0.8302 | C=0.01 confirmed optimal across all feature sets |
| 18a | v2 + TicketSurvRate feature | Continuous survival rate as input | 0.8069 | **Catastrophic** −3.2% CV |
| 18b-loose | v2 + ticket rules (1+ mate) | Looser threshold fires more often | 0.7964 | Too aggressive: −3.7% CV, 128 overrides |
| 19a | Surname-only rules | Surname+Pclass without ticket rules | 0.8247 | Best CV of step 19, but not submitted |
| 19c | Connected components | Union-find merging ticket + surname | 0.8223 | False merges from common names (Johnson: 8 people) |
| 20 | Group-aware CV | Grouped and LOO cross-validation | — | LOO has structural bias; local eval of rules is unreliable |

---

## What Worked

### 1. Strong regularization on a small dataset

C=0.01 was selected early and confirmed optimal across 3 feature sets × 7 C values (step 17). On 891 samples with noisy features, the model benefits from heavy shrinkage. Every attempt to relax regularization performed worse.

### 2. Rules over features for group-level signal

The defining insight of this project. Ticket-mate survival rate as a continuous feature was catastrophic (−3.2% CV) because the logistic model under C=0.01 couldn't learn to trust it — regularization suppressed it alongside everything else. The same information as a hard rule override bypassed regularization entirely and produced the only Kaggle improvement in the project.

This generalizes: when you have a high-confidence domain signal that your model's regularization would suppress, applying it as a post-prediction rule can outperform encoding it as a feature.

### 3. Conservative thresholds

The 2+ unanimous mates threshold was the sweet spot. Requiring unanimity (100% survived or 100% died) kept precision high. Requiring 2+ mates avoided single-witness noise. The 1+ mate variant fired 128 times and collapsed CV by −3.7%. More coverage at lower precision was strictly worse.

### 4. Keeping changes small and interpretable

Every submission that changed many test passengers did worse on Kaggle. The winning approach changed only 13. The segmented model (12a) made a large, concentrated correction to female Pclass 3 and lost 12 points. The RBF SVM flipped 35 passengers and matched gender-only. The pattern was consistent: on this dataset, aggressive corrections overfit training noise.

### 5. The evaluation harness

Building `shared/evaluate.py` with repeated CV, paired comparison, and flip analysis imposed discipline on every experiment. Without it, the project would have submitted many more false positives to Kaggle. The harness didn't prevent all mistakes (it couldn't foresee that CV underestimates group signals), but it prevented the obvious ones.

---

## What Didn't Work

### 1. Complex model classes

Random forest, XGBoost, LightGBM, neural networks, and RBF SVM all tied or underperformed logistic regression on Kaggle despite higher CV scores. The dataset is too small and too noisy for models that capture complex interactions — they find patterns in training that don't exist in test.

### 2. Feature engineering beyond the basics

v3 features (one-hot Pclass, Sex×Pclass interactions, grouped deck) did not improve over v2's simpler ordinal Pclass. Surname features, fare rank, non-family-companion indicators — none moved the needle. The useful features (sex, class, age, family size, title) were all found in the first pass. After that, the signal-to-noise ratio of new features was too low for 891 samples.

### 3. Segmented models

Fitting separate models by sex sounded principled (the survival patterns are genuinely different) but produced an overconfident correction in female Pclass 3 that lost 12 Kaggle points. The female-only model had too few samples to reliably learn the within-female patterns.

### 4. Expanding group coverage with noisy groups

Surname+Pclass groups added 24 new test passengers not covered by ticket rules, but common names (Smith, White, Hansen) created false matches that cancelled out genuine family signal. Connected components were worse — union-find merged unrelated people through shared surnames. Group identification quality matters more than group coverage.

### 5. TicketSurvRate as a feature

The single worst experiment. Adding the continuous ticket-mate survival rate as a model input dropped CV by 3.2%. OOF encoding made the feature noisy during cross-validation, and the model over-relied on it despite regularization. The same information worked brilliantly as a rule but terribly as a feature.

---

## Which Metrics Were Trustworthy?

### Repeated stratified CV: reliable for base models, pessimistic for rules

For comparing logistic regression variants (v2 vs v3, different C values, different features), repeated 5-fold CV with 10 repeats was the most reliable local signal. It correctly ranked v2 > v3, C=0.01 > higher C, and identified catastrophic changes (18a, 18b-loose, segmented models).

It was systematically pessimistic for group-based rules because random fold splits break ticket groups across folds, reducing the rule's coverage during CV. The hybrid model scored 0.8228 locally but 0.7751 on Kaggle — the rule fires for more passengers when all 891 training samples are available.

### Paired comparison: useful for relative ranking

Win/loss/tie counts across 50 folds gave a more nuanced signal than mean CV alone. A model with a tiny mean improvement but 30 losses out of 50 folds was correctly flagged as unreliable.

### Flip analysis: the best risk signal

Counting how many test passengers changed, and in which subgroups, was the most actionable diagnostic. The pattern was clear: submissions with >15 flips and/or concentrated flips in female Pclass 3 always performed worse on Kaggle. The winning submission had 13 flips spread across subgroups.

### Kaggle leaderboard: the only arbiter for rules

For the rule layer specifically, no local metric was reliable. Grouped CV, LOO grouped CV, and standard CV all had structural limitations for evaluating propagation rules on small grouped data. The Kaggle leaderboard was the only way to validate whether the rules generalized.

---

## Where CV and Kaggle Disagreed

| Experiment | CV said | Kaggle said | Why they disagreed |
|-----------|---------|-------------|-------------------|
| XGBoost tuned | 0.8552 (best in repo) | 0.7584 (second worst) | Overfit: complex model found training noise |
| Ensemble | 0.8563 (highest CV ever) | 0.7560 (worst submission) | Ensembling overfit models amplifies overfit |
| Segmented 12a | 0.8384 (big improvement) | 0.7608 (big loss) | Overcorrected female Pc3 — real CV pattern, fake generalization |
| 18b hybrid rules | 0.8228 (below baseline) | 0.7751 (best ever) | CV underestimates group signals due to fold splitting |

The first three are the standard small-dataset lesson: higher local score ≠ better generalization. The fourth is more interesting — CV was *wrong in the other direction*. Group-based rules look worse locally than they actually are, because CV artificially reduces the rule's coverage.

---

## The Error Structure

The error audit (step 22) revealed where the ~94 test errors come from:

| Subgroup | Test N | Est. Errors | % of Total | Why they're hard |
|----------|--------|-------------|-----------|------------------|
| female Pc3 | 72 | ~25 | 33% | 50% survival rate = structural coin flip |
| male Pc1 | 57 | ~21 | 28% | 37% survival; wealth ≠ survival |
| male Pc3 | 146 | ~17 | 23% | Rare survivors have no distinguishing features |
| male Pc2 | 63 | ~8 | 11% | Small group of exceptions |
| female Pc2 | 30 | ~3 | 4% | Near-perfect survival, few exceptions |
| female Pc1 | 50 | ~2 | 2% | Near-perfect survival, rare deaths (e.g., Allisons) |

The model's 34 disagreements with gender-only tell a specific story:
- 26 males predicted to survive: mostly young boys (Masters) and Pclass 1 men with surviving ticket-mates
- 8 females predicted to die: Pclass 3 women from families that all perished (Sages, Goodwins, Johnstons)

The remaining errors are largely irreducible with available features. Male Pclass 3 survivors (Daly, Jussila, Lang) have nothing in the data to distinguish them from the men who died next to them. Female Pclass 3 deaths (Cacic, Oreskovic, Vestrom) look identical to the women who survived. These are individual outcomes determined by circumstances — deck location, proximity to lifeboats, panic, heroism — that aren't encoded in age, fare, class, or ticket number.

---

## What Generalizes Beyond Titanic

### 1. On small datasets, simplicity wins

891 samples is not enough to support 21 features (v3), let alone tree ensembles with hundreds of parameters. The winning model has 15 features under C=0.01 — roughly 2-3 effective parameters given the regularization strength. Every step toward complexity was a step toward overfit.

### 2. Local validation has limits that matter

Standard CV was reliable for model selection but systematically wrong for rule evaluation. The project needed both local metrics (for base model decisions) and the external leaderboard (for rule decisions). Knowing *which* local metric to trust for *which* kind of change is a non-trivial skill.

### 3. Domain knowledge encoded as rules can outperform features

"Families survived or died together" is historical fact. Encoding it as a continuous feature gave the regularized model permission to ignore it. Encoding it as a hard rule gave it no choice. The lesson isn't "always use rules" — it's that the encoding of domain knowledge matters as much as the knowledge itself.

### 4. The ceiling is usually in the data, not the model

Every model class tested (logistic, RF, XGBoost, LightGBM, SVM, neural net) achieved roughly the same effective performance. The problem wasn't that logistic regression was too simple — it was that the features couldn't separate the cases the model got wrong. Switching to a more powerful model on the same features just finds more creative ways to overfit.

### 5. Aggressive corrections are high-risk on noisy targets

Three separate experiments confirmed that aggressively reclassifying female Pclass 3 passengers hurts the leaderboard, despite being locally motivated. When the true base rate is near 50%, any correction is nearly as likely to be wrong as right. The safe strategy is to stay with the prior (predict survived for all women) and accept the structural error floor.

### 6. Group identification quality > group coverage

Adding surname-based groups increased coverage from 48 to 65 test passengers with rule overrides. But the additional groups were noisier (common names → false matches), and the 17 new overrides netted to zero on Kaggle. Precision matters more than recall for rule-based systems.

---

## Honest Ceiling Estimate

- Gender-only: **0.7655** (320/418)
- Current best: **0.7751** (324/418)
- Theoretical with perfect group rules: **~0.79** (330/418)
- Beyond that: requires features not in this dataset

The gap between 0.7751 and the theoretical ceiling is ~6 passengers. They would need to come from higher-precision group identification (filtered surname rules, cabin-sharing groups) or information not available in the standard feature set (deck assignment, lifeboat proximity, boarding order).

---

## Project Timeline

| Date | Steps | Key event |
|------|-------|-----------|
| 2026-03-12 | 1-10 | Initial exploration through v2 baseline (0.7727) |
| 2026-03-13 | 11-16 | Feature engineering, segmentation, SVM — all dead ends |
| 2026-03-14 | 17-20 | C sweep, ticket rules breakthrough (0.7751), surname extension, grouped CV |
| 2026-03-15 | 21-22 | v3+rules (rejected), error audit, this retrospective |

Three days. 22 numbered experiments. 11 Kaggle submissions. One Kaggle improvement of 4 passengers.

The project started as an attempt to beat a simple baseline. It ended as a study in why that baseline was hard to beat — and why that's a more valuable finding than a higher number on a leaderboard.
