# Criteo Uplift Modeling: Retrospective

**Date:** 2026-03-24
**Dataset:** Criteo Uplift Prediction (10% sample, 1.4M users)
**Goal:** Learn causal inference and uplift modeling — first causal project

---

## What I Built

Progression from naive baselines to meta-learners for estimating individual treatment effects:

1. **EDA** — Verified randomization, computed ATE (+1.03%), identified treatment effect heterogeneity across feature deciles
2. **Naive Baselines** — Standard classifier (P(visit)) and two-classifier LogReg. Showed that ranking by propensity isn't the same as ranking by uplift
3. **S-Learner** — Single LightGBM with treatment as a feature. Good calibration at extremes but treatment feature ranked 10/13 by importance (regularization bias)
4. **T-Learner** — Two separate LightGBMs. Forced treatment effect capture but noisy due to 85/15 imbalance (control model only had 147K rows)
5. **X-Learner** — Three-stage cross-imputation. Best uplift model (AUUC 0.0294). Fixed T-learner noise via propensity weighting
6. **Model Comparison** — Head-to-head with Uplift@K% analysis. X-learner captures 54% of total uplift from just 10% of users
7. **Targeting Analysis** — Cost-benefit analysis, budget scenarios, segment profiles

## Final Model Rankings (AUUC)

| Model | AUUC | Notes |
|---|---|---|
| Standard Classifier (LogReg) | 0.0309 | Highest AUUC but accidental — propensity ≈ uplift in this dataset |
| X-Learner (LightGBM) | 0.0294 | Best uplift model — models causal effect directly |
| Two Classifiers (LogReg) | 0.0290 | Naive T-learner baseline |
| S-Learner (LightGBM) | 0.0289 | Best calibration, but regularization bias |
| T-Learner (LightGBM) | 0.0272 | Hurt by 85/15 imbalance |
| Random | -0.0018 | Baseline |

## What Surprised Me

1. **The standard classifier beat all uplift models on AUUC.** In this dataset, high-propensity users also happen to be high-uplift users (positive correlation). The classifier "accidentally" targets the right people. In most real-world datasets this wouldn't hold — Sure Things would dominate the top of the propensity ranking.

2. **The treatment effect is tiny.** ATE of 1.03% on a 4.7% visit rate. Uplift models are trying to find a signal that's 5x smaller than the outcome itself. This is realistic for digital advertising.

3. **The S-learner's regularization bias was visible in feature importance.** Treatment ranked 10th out of 13 features. LightGBM correctly determined treatment is one of the least useful features for predicting visits — but that's the wrong optimization target for uplift.

4. **The T-learner was the worst uplift model.** I expected the 85/15 imbalance to hurt but not this much. 33% of users predicted as Sleeping Dogs when reality showed almost no negative effects.

5. **Uplift modeling's value scales with ad cost.** At $0.05/impression, targeting everyone is optimal. At $0.50/impression, the model saves $141K vs blanket targeting. The more expensive the ads, the more uplift modeling matters.

## Key Concepts I Learned

- **Prediction vs Causation:** P(visit) ≠ P(visit|ad) - P(visit|no ad). The first targets Sure Things, the second targets Persuadables.
- **The Fundamental Problem of Causal Inference:** Can never observe both outcomes for the same individual. Randomization solves this at the group level.
- **ATE vs CATE:** ATE is one average number. CATE is per-user/segment. Uplift modeling estimates CATE.
- **The Four Quadrants:** Persuadables (target), Sure Things (skip), Lost Causes (skip), Sleeping Dogs (avoid).
- **Decile tables are the ground truth check** for any uplift model — does predicted uplift match actual ATE?
- **AUUC measures ranking quality, decile tables measure calibration** — a model can rank well but over/under-predict.

## Connection to My Work

- Geo experiments produce one ATE number. Uplift modeling answers "who did it work for?"
- MDE is a design-stage tool (can we detect the effect?). ATE is a measurement (what was the effect?).
- DMA-level heterogeneity analysis is feasible with 10+ DMAs — same concept as feature-level CATE, just coarser
- The virtuous cycle: experiment → uplift model → smarter targeting → validate → refine
- When client conversion volume is low (MDE = 18% at 60% power), the experiment may not be worth running — document the power analysis

## What I'd Do Differently

- Would try tuning LightGBM hyperparameters — we used the same params for all models
- The X-learner still overpredicts at the top decile (7.2% predicted vs 5.6% actual) — could explore calibration techniques
- Didn't try the DR-learner or Causal Forest — these could provide confidence intervals on CATE estimates
- Could explore the `exposure` column (actual ad viewing vs intent-to-treat) for compliance analysis
