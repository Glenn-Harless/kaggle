# Titanic Handoff And Next Steps

Last updated: 2026-03-14

## Where We Are

The project broke through its plateau. After exhausting individual-level features and model classes (steps 14-17), **group survival propagation** (step 18) produced the first leaderboard improvement:

- the best confirmed Kaggle score is now **`0.7751`** (up from 0.7727)
- the winning approach is a hybrid: v2 logistic regression + ticket-mate survival rules
- the direction is validated and has room for refinement

## Best Known Submission

Current best confirmed Kaggle submission:

- `submissions/logreg_18_18b_hybrid_2mate.csv` — Kaggle **0.7751**

Architecture: v2 LogReg (C=0.01) + rule override when 2+ ticket-mates have unanimous survival outcome.

Supporting files:

- `scripts/modeling/18_ticket_survival_propagation.py`
- `scripts/modeling/10_logreg_v2.py` (base model)

Previous best (for comparison):

- `submissions/logreg_v2.csv` — Kaggle 0.7727

## What We Tried Recently

### v3 line

Files:

- `11_logreg_v3.py`
- `11a_logreg_v3_interactions.py`
- `11b_logreg_v3_fare_deck.py`
- `11c_logreg_v3_surname.py`

What we learned:

- systematic interactions and grouped deck features were locally coherent
- surname features did not help
- v3 was a meaningful local refinement, but it was never separately validated on Kaggle

### Segmented-by-sex line

Files:

- `12a_segment_by_sex.py`
- `12b_segment_fare_rank.py`
- `12c_segment_threshold.py`

What we learned:

- segmentation improved local CV materially
- it made a large, concentrated correction in `female Pclass=3`
- that correction hurt the leaderboard badly
- threshold tuning did not rescue it

This branch is informative, but not a production direction.

### Evaluation harness + narrow interaction probe

Files:

- `shared/evaluate.py`
- `scripts/modeling/13_v2_rebaseline.py`
- `scripts/modeling/14_male_pclass1_interactions.py`

What we learned:

- the harness successfully reproduced the real `v2` baseline
- narrow `male Pclass=1` interactions were evaluated cleanly
- none of the tested interactions made a compelling case for adoption
- the strongest candidate (`Deck_DE`) had only a tiny positive local signal and no convincing net advantage

This was a successful process improvement even though it did not yield a new submission.

### Ticket group features (step 15, 2026-03-13)

Files:

- `scripts/modeling/15_ticket_group_experiment.py`

What we learned:

- ticket group size signal is real in training data (solo 30%, small group 57-63%, large 0-20%)
- but computing from train only is too lossy: 64% of test passengers have tickets unseen in train
- three candidates tested (raw count, buckets, non-family-companion indicator) — all flat or negative
- the best candidate (raw count) had delta +0.0007 with zero test flips — the model effectively ignored it
- this was the #1 priority direction and it did not pan out

### RBF SVM nonlinear probe (step 16, 2026-03-13)

Files:

- `scripts/modeling/16_rbf_svm_probe.py`
- `results/models/16_rbf_svm_probe.txt`
- `submissions/rbf_svm_16.csv`

What we learned:

- grid search selected C=100, gamma=0.1; inner CV 0.8406, harness CV 0.8296
- paired delta -0.0005 vs baseline, W/L/T: 21/22/7 — dead even on CV
- 35 test flips, dominated by 18 female_Pclass3 survived→died
- Kaggle score: **0.7655** — same as gender-only baseline, below v2's 0.7727
- instability 6x worse than logistic (0.054 vs 0.009), overfitting rate 10x worse (0.19 vs 0.02)
- **confirms the ceiling is in the features, not the model class**
- **confirms (for the 3rd time) that aggressively flipping female_Pclass3 hurts the leaderboard**

### Full-manifest groups + relaxed regularization (step 17, 2026-03-14)

Files:

- `scripts/modeling/17_full_manifest_groups.py`

What we learned:

- C=0.01 is genuinely optimal — confirmed with repeated CV across 3 feature sets × 7 C values
- the original 5-fold selection of C=0.01 was NOT noisy; higher C consistently worse
- full-manifest TicketGroupSize still doesn't help as a feature (delta +0.0001)
- the features are noisy enough that strong regularization genuinely helps

### Ticket survival propagation (step 18, 2026-03-14) — BREAKTHROUGH

Files:

- `scripts/modeling/18_ticket_survival_propagation.py`
- `submissions/logreg_18_18b_hybrid_2mate.csv`

What we learned:

- **adding TicketSurvRate as a continuous feature is catastrophic** (18a: CV -3.2%). OOF encoding makes it noisy during CV, and the model over-relies on it
- **hybrid rules work** (18b): override logistic when 2+ ticket-mates have unanimous outcomes
  - Kaggle **0.7751** — first leaderboard improvement in the project
  - 13 test flips: 7 died→survived (men whose families survived), 6 survived→died (Pclass3 whose families died)
- **1+ mate threshold is too aggressive** (18b-loose: CV -3.7%, 128 overrides)
- **CV underestimates group signals** because random splits break ticket groups across folds. The rule fires for more passengers at test time (all 891 available) than during any CV fold
- **rules bypass regularization** — C=0.01 can't suppress a hard override, which is why rules work where features don't

Key paradigm shift: the problem is not individual-level prediction. Survival was a group phenomenon, and group survival context is the signal source that breaks through the individual-feature ceiling.

### Surname-based survival propagation (step 19, 2026-03-14) — NEUTRAL

Files:

- `scripts/modeling/19_surname_survival_propagation.py`
- `results/models/19_surname_propagation.txt`
- `submissions/logreg_19_19b_ticket_surname.csv`

What we learned:

- **surname+Pclass groups add 24 NEW test passengers** not covered by ticket rules (of 418 total). Of those, 16 have unanimous training mates.
- **19a (surname rules only) had the best CV** at 0.8247, W/L/T 21/12/17 vs 18b. Surname groups may be slightly better than ticket groups for some families.
- **19b (ticket priority + surname fallback) tied 18b on Kaggle: 0.7751.** The 8 new surname-driven flips netted to zero — some were correct family matches, others were false positives from common names.
- **19c (connected components) was worst** (CV 0.8223). Union-find merged unrelated people via common names: Johnson component had 8 members from different families connected by "Johnson" surname + shared "LINE" ticket. Andersson component had 9 members from different families.
- **Common-name false positives are the bottleneck.** Smith_Pc1, White_Pc1, Hansen_Pc3 are plausible false matches that canceled out genuine family signal from Ford, Cacic, Oreskovic.
- **17 surname overrides on test exceeded the 15-override risk threshold.**

Key takeaway: the surname signal exists but is noisy at the current filtering level. Stricter surname filtering (3+ mates, common-name exclusion) might extract the genuine signal, but the marginal value is likely small.

## What The Analysis Reports Say

Relevant files:

- `results/analysis/error_analysis_report.md`
- `results/analysis/stability_analysis_report.md`
- `results/analysis/historical_research_report.md`
- `results/analysis/validation_strategy_report.md`

Main conclusions:

1. The dominant uncertainty zone is `female, Pclass=3`
2. Broad corrections in that zone are high-risk and have already backfired
3. `male, Pclass=1` was the cleanest narrow opportunity, but the first pass did not produce a convincing feature
4. The model is already very stable overall; only a small set of passengers are truly unstable
5. The current linear / strongly regularized logistic setup may be close to its ceiling on the current feature representation

## What We Know About The Error Structure

- Many remaining errors are concentrated in `female Pclass=3` and `male Pclass=1`
- `female Pclass=3` contains real interaction structure
- the baseline model already contains most of the relevant raw signals for that subgroup
- adding aggressive interactions there causes collateral damage
- adult `male Pclass=3` survivors are mostly unresolved by the observable feature set

In short:

- the project is not missing one obvious linear feature
- the next gains, if any, will probably come from a qualitatively different approach

## What Not To Do Next

Avoid these unless new evidence appears:

- more tree tuning
- more ensemble work
- broad segmented-model corrections
- more threshold tuning on the current segmented branch
- more tiny linear feature tweaks in the same style as the rejected `male Pclass=1` interactions
- surname survival hints without a compelling new reason

## Priority Next Steps

### ~~1. Leakage-safe family / ticket-group experiment~~ — DONE, rejected

Tested 2026-03-13 (step 15). Three candidates, all flat or negative. The leakage-safe constraint makes the feature too lossy to be useful. See step 15 writeup above.

### ~~2. One bounded nonlinear single-model probe~~ — DONE, rejected

Tested 2026-03-13 (step 16). RBF SVM ties logistic regression on CV but flips 35 passengers aggressively. Kaggle score 0.7655, below baseline. Confirms the problem is linear on these features. See step 16 writeup above.

### ~~3. One tiny hybrid rules-plus-model baseline~~ — DONE, **success**

Tested 2026-03-14 (step 18b). Hybrid ticket-mate survival rules + v2 logistic regression scored **0.7751** on Kaggle — first leaderboard improvement. The approach is validated and ready for refinement.

### 4. Keep the harness as the default gate

Any future experiment should report:

- repeated CV mean and std (but note: CV underestimates group-based signals)
- paired delta vs baseline
- total test flips
- flips by subgroup
- changed PassengerIds

For group-based rules, also consider leave-group-out CV for more honest estimates.

### 5. Extend group survival propagation

The hybrid rules approach is validated. Surname extension tested but neutral.

**~~5a. Surname-based survival propagation~~** — DONE, neutral (step 19)

- Tested 2026-03-14. Surname+Pclass rules add 24 new test passengers but common-name false positives cancel genuine signal.
- 19b (ticket + surname) scored 0.7751 on Kaggle — tied with 18b, not an improvement.
- Stricter filtering (3+ mates, name rarity) might help but diminishing returns likely.

**~~5b. Combined ticket + surname groups~~** — DONE, rejected (step 19c)

- Tested 2026-03-14 via connected components (union-find). False merges from common names made it worse than ticket-only.
- Johnson component: 8 unrelated people. Andersson component: 9 members from different families.

**5c. Relaxed unanimity threshold**

- Currently requires 100% agreement (all survived or all died)
- Test majority rules: e.g., 3 out of 4 mates survived → predict survived
- Higher risk of noise, so test carefully through the harness

**~~5d. Group-aware cross-validation~~** — DONE, completed with caveats (step 20)

Two validators added to `shared/evaluate.py`:
- `repeated_grouped_cv()` / `grouped_cv_splits()`: for base model / feature leakage evaluation. Confirmed no ticket-group leakage in the v2 logistic model (0.8295 grouped vs 0.8301 stratified).
- `leave_one_out_grouped_cv()`: for propagation rule evaluation. **Structural limitation discovered:** LOO produces only trivially-correct fires (unanimous groups) and structurally-wrong fires (dissenter removal from mixed groups). Cannot meaningfully estimate absolute rule accuracy — use for comparative diagnostics (fire rate, coverage) and rely on Kaggle for the final rule evaluation.

**5e. Refined surname filtering**

- Exclude common English/Scandinavian surnames, or require 3+ training mates
- Use name rarity (frequency in the dataset) as a confidence weight
- Goal: keep genuine family matches (Ford, Oreskovic) while dropping false positives (Smith, Johnson)

### 6. What NOT to try

- TicketSurvRate as a continuous feature (step 18a: catastrophic -3.2% CV)
- 1+ mate threshold (step 18b-loose: too aggressive, -3.7% CV)
- Relaxed regularization (step 17: C=0.01 confirmed optimal)
- More nonlinear models on current features (step 16: confirmed linear ceiling)
- Naive surname+Pclass propagation without name filtering (step 19b: net zero on Kaggle)
- Connected components merging ticket + surname groups (step 19c: false merges from common names)

## Recommended Immediate Order

When work resumes:

1. try refined surname filtering (step 5e) — extract genuine family signal while dropping common-name false positives. Use LOO grouped CV for comparative fire rate / coverage diagnostics, Kaggle for final verdict.
2. try relaxed unanimity threshold (step 5c) — e.g., majority rules for groups of 3+. Highest risk of noise.
3. explore cabin-based groups — passengers sharing cabin numbers may be family/travel party (low priority due to many missing values)

Note on evaluation: local CV (stratified or grouped) is authoritative for the base logistic model. For propagation rules, local evaluation is useful for comparative diagnostics but not absolute accuracy. Kaggle is the final arbiter for rule-based changes.

## Practical Baseline To Resume From

Start from:

- `scripts/modeling/18_ticket_survival_propagation.py` (current best, Kaggle 0.7751)
- `scripts/modeling/19_surname_survival_propagation.py` (surname extension reference — neutral result but useful template)
- `submissions/logreg_18_18b_hybrid_2mate.csv` (Kaggle 0.7751)
- `scripts/modeling/13_v2_rebaseline.py` + `shared/evaluate.py:reconstruct_v2_features()` (v2 baseline reconstruction — do NOT re-run `10_logreg_v2.py`, which would consume v3 features)
- `shared/evaluate.py` (evaluation harness)
- `results/models/13_v2_baseline_cv_scores.npy` (frozen baseline CV scores for paired comparison)

Treat `18_ticket_survival_propagation.py` as the reference for group-based rules. Use step 19's results as evidence that naive surname matching is insufficient — any future surname work should filter for name rarity or require stricter mate counts.

## What We Now Know (2026-03-13 through 2026-03-14)

### Individual-level findings (steps 14-17):

1. **The ceiling is in the features, not the model class.** RBF SVM ties logistic regression on CV and does worse on Kaggle.
2. **C=0.01 is genuinely optimal** for logistic regression on these features. Confirmed with repeated CV across 21 combinations.
3. **Ticket group features as model inputs don't help** — whether computed train-only (step 15) or full-manifest (step 17). Regularization suppresses them.
4. **Any model that aggressively reclassifies female_Pclass3 hurts the leaderboard.** Confirmed 3 times.

### Group-level breakthrough (step 18):

5. **Group survival propagation works.** Using ticket-mate outcomes as hard rules (not features) bypasses regularization and captures domain-legitimate signal.
6. **Rules > features for group context.** TicketSurvRate as a feature: -3.2% CV. As a rule: +0.24% Kaggle. The model can't learn to trust the feature under C=0.01, but a rule doesn't need the model's permission.
7. **Standard CV underestimates group signals.** Random splits break groups across folds, reducing the rule's coverage during evaluation. This means CV is pessimistic for group-based approaches.
8. **Conservative rules (2+ unanimous mates) generalize.** Strict thresholds prevent overfitting. The 1+ mate variant was too aggressive.

### Surname extension (step 19):

9. **Surname+Pclass groups add coverage but common names add noise.** 24 new test passengers covered, but Smith/White/Hansen false positives cancel out Ford/Cacic/Oreskovic genuine families.
10. **Naive surname propagation nets to zero on Kaggle.** 19b tied 18b at 0.7751 — the 8 new flips split roughly evenly between correct and incorrect.
11. **Connected components are too aggressive.** Union-find merging ticket + surname creates mega-components from common names (Johnson: 8 members, Andersson: 9). Worse than ticket-only.
12. **Group identification quality matters more than group coverage.** Extending coverage with noisy groups doesn't help — you need high-precision group membership to make the rule pattern work.

## Bottom Line

The project has broken through its plateau by shifting from individual-level prediction to group survival propagation. The key insight: survival was a group phenomenon, and group context provides signal that no individual-level feature can capture.

The naive extension to surname groups (step 19) showed that **group identification quality matters more than coverage** — adding noisy groups doesn't help. The path to 80%+ likely requires either higher-precision group identification (filtered surnames, cabin groups) or a qualitatively different approach (relaxed unanimity, leave-group-out CV for better evaluation).
