# Titanic State Of The World

Last updated: 2026-03-14

## Current Read

The project has found its first leaderboard improvement. After extensive exploration of individual-level features and model classes (all of which failed to beat the v2 baseline), a **hybrid rules-plus-model approach** broke through by using ticket-mate survival propagation.

The best confirmed Kaggle score is now **`0.7751`**, achieved by:

- `submissions/logreg_18_18b_hybrid_2mate.csv` — v2 logistic regression + unanimous ticket-mate survival rules

This beat the previous best of `0.7727` (plain logistic regression). The key insight: **group survival context** (what happened to your travel party) provides signal that no individual-level feature can capture. Families and travel parties tended to survive or die together — this is domain knowledge, not a statistical trick.

Higher local CV has **not** translated into better leaderboard performance for individual-level models. But group-based rules, even with slightly worse CV (due to group splitting across folds), DO generalize.

## What We Know

### Confirmed leaderboard ordering

Sorted by Kaggle score:

| Submission | Local CV | Kaggle | Notes |
| --- | ---: | ---: | --- |
| `logreg_18_18b_hybrid_2mate.csv` | 0.8228 | **0.7751** | **BEST** — hybrid rules + logistic, 13 test flips |
| `logreg_19_19b_ticket_surname.csv` | 0.8227 | 0.7751 | ticket + surname rules, 8 new surname flips netted to zero |
| `logreg_v2.csv` | 0.8305 | 0.7727 | v2 logistic regression, 15 features |
| `logistic_regression.csv` | 0.8316 | 0.7727 | baseline logistic regression |
| `gender_only.csv` | 0.7870 | 0.7655 | pure gender baseline |
| `rbf_svm_16.csv` | 0.8296 | 0.7655 | RBF SVM (C=100, gamma=0.1), 35 test flips |
| `logreg_12a_segment_sex.csv` | 0.8384 | 0.7608 | segmented-by-sex logistic regression |
| `xgb_thresh_55.csv` | n/a | 0.7608 | thresholded XGBoost |
| `random_forest.csv` | 0.8316 | 0.7584 | random forest |
| `xgboost_tuned.csv` | 0.8552 | 0.7584 | tuned XGBoost |
| `ensemble.csv` | 0.8563 | 0.7560 | hard-vote ensemble |

### Confirmed technical findings

- `v1` feature pipeline used a transductive `TicketGroupSize` computed on combined train + test. This was initially called "leakage" but is more precisely a transductive feature — it uses test-set metadata (ticket strings) without touching survival labels. It is allowed in competition workflows but should be used intentionally.
- `v2` removed the transductive feature and simplified to an EDA-aligned set.
- `v3` added systematic interactions and grouped deck features. Locally it was coherent, but it was not separately submitted to Kaggle.
- `12a` segmented the logistic model by sex. This improved local CV but overcorrected on the leaderboard.
- `12b` added `FareRankInClass` and did not improve over `12a`.
- `12c` threshold tuning did not improve over `12a`.
- `15` tested leakage-safe ticket group features (3 candidates). All flat or negative — the signal exists in training data but is not transferable to test because 64% of test passengers have tickets unseen in train.
- `16` tested RBF SVM as a nonlinear model class probe. CV tied logistic regression but flipped 35 passengers (18 female Pclass=3 survived→died). Kaggle score: 0.7655, same as gender-only baseline. Confirms the ceiling is in the features, not the model class.
- `17` tested full-manifest TicketGroupSize + broader C sweep. C=0.01 confirmed genuinely optimal across all feature sets (not noisy selection). Full-manifest groups still didn't help as features.
- `18` tested ticket-mate survival propagation — using training survival labels from ticket-mates as signal. Adding TicketSurvRate as a feature hurt CV badly (-3.2%). But **hybrid rules (override logistic when 2+ ticket-mates have unanimous outcomes) scored 0.7751 on Kaggle** — new best.
- `19` tested surname-based survival propagation — extending the hybrid rule pattern to surname+Pclass family groups. Three approaches: 19a (surname rules only, CV 0.8247), 19b (ticket priority + surname fallback, CV 0.8227), 19c (connected components via union-find, CV 0.8223). 19b submitted to Kaggle: **0.7751 — tied with 18b**. The 8 new surname flips netted to zero (some correct, some false positives from common names like Smith, White). Connected components confirmed false-merge risk (Johnson component merged 8 unrelated people via common surname + shared LINE ticket).

### Stable model behavior pattern

- Plain logistic models are conservative and leaderboard-safe.
- More flexible models change more passengers and consistently do worse on Kaggle.
- The leaderboard punishes large subgroup-specific corrections — but **rewards small, domain-motivated group-based corrections**.
- Group survival propagation (ticket-mate rules) is the first approach to beat the v2 baseline on the actual leaderboard.
- Surname-based group extension adds coverage (24 new test passengers) but the signal is noisy — common names create false matches that cancel out genuine family signal.

## Best Current Submission

Use `18_ticket_survival_propagation.py` and `submissions/logreg_18_18b_hybrid_2mate.csv` as the current best.

Architecture:

- v2 logistic regression (C=0.01, 15 features) as the base model
- Rule override: when a test passenger has 2+ ticket-mates in training with unanimous survival outcome, override the logistic prediction with the group outcome
- 48 test passengers receive a rule override; 13 of those change from the v2 baseline prediction

Relevant files:

- `scripts/modeling/18_ticket_survival_propagation.py`
- `submissions/logreg_18_18b_hybrid_2mate.csv`
- `scripts/modeling/10_logreg_v2.py` (base model)
- `submissions/logreg_v2.csv` (previous best, used for comparison)

## What Failed And Why

### Tree / ensemble family

Files:

- `02_random_forest.py`
- `03_xgboost.py`
- `04_lightgbm.py`
- `06_xgboost_tuned.py`
- `07_random_forest_tuned.py`
- `09_ensemble.py`

Why they are not useful as the main path:

- highest CV in repo
- consistently worse Kaggle scores than plain logistic regression
- likely overfit small-sample subgroup patterns

Recommendation: archive as completed dead-end experiments, not active candidates.

### Aggressive segmented logistic

Files:

- `12a_segment_by_sex.py`
- `12b_segment_fare_rank.py`
- `12c_segment_threshold.py`

What happened:

- `12a` improved CV from `0.8305` to `0.8384`
- but leaderboard dropped from `0.7727` to `0.7608`
- the model made a large, concentrated set of flips, mostly turning `female, Pclass 3` passengers from survive to die

Interpretation:

- segmentation found a real training-set pattern
- the correction was too aggressive for the hidden test set
- this path is informative, but not a production baseline

Recommendation: keep for reference, but archive away from the active path.

### Surname features (as model input)

Files:

- `11c_logreg_v3_surname.py`
- `12b_segment_fare_rank.py` is unrelated; keep surname conclusions tied to 11c only

What happened:

- leakage-safe surname experiments did not help
- `SurnameSurvHint` was noisy and unstable
- shuffle ablation did not support shipping surname features

Recommendation: treat as explored and rejected for now.

### Surname-based survival propagation (step 19, 2026-03-14) — NEUTRAL

Files:

- `scripts/modeling/19_surname_survival_propagation.py`
- `results/models/19_surname_propagation.txt`
- `submissions/logreg_19_19a_surname_pc.csv`
- `submissions/logreg_19_19b_ticket_surname.csv`
- `submissions/logreg_19_19c_connected.csv`

What happened:

- extended the 18b hybrid rule pattern to identify family groups via surname+Pclass
- 19a (surname rules only): CV 0.8247, best of the three — surname groups may be a slightly better grouping for some families
- 19b (ticket priority + surname fallback): CV 0.8227, 8 new flips vs 18b on test. **Kaggle 0.7751 — tied with 18b**
- 19c (connected components via union-find): CV 0.8223, worst — false merges from common names (Johnson component: 8 unrelated people merged via shared surname + LINE ticket)
- surname rules add 24 NEW test passengers not covered by ticket rules (of 418 total)
- of the 8 new surname flips in 19b: 7 survived→died, 1 died→survived. Net impact: zero.

Why it didn't improve:

- common surnames (Smith, White, Hansen) create false-positive family matches
- surname+Pclass constraint helps but doesn't eliminate the problem
- the genuine family signal (Ford, Cacic, Oreskovic) is canceled by the false matches
- 17 total surname overrides on test exceeded the 15-risk threshold

Recommendation: surname propagation as implemented is explored and neutral. A stricter variant (3+ mates, or common-name exclusion) might extract the genuine signal, but diminishing returns are likely.

### Ticket group features (step 15, 2026-03-13)

Files:

- `scripts/modeling/15_ticket_group_experiment.py`

What happened:

- computed TicketGroupSize from training data only (no leakage)
- tested 3 candidates: raw count, EDA-aligned buckets, non-family-companion indicator
- the signal is real in training (solo 30%, small group 57-63%, large 0-20%)
- but 64% of test passengers have tickets not seen in train → feature has no signal for most test passengers
- all 3 candidates flat or negative vs baseline (best delta: +0.0007 with 0 test flips)

Why it failed:

- train-only ticket counting is too lossy: groups spanning train/test get undercounted
- strong regularization (C=0.01) suppresses the weak remaining signal
- the non-family-companion subset (75 train, 15 test passengers) is too sparse

Recommendation: explored and rejected. Ticket group features require combined train+test computation to be useful, which is the leakage that v2 already removed.

### RBF SVM nonlinear probe (step 16, 2026-03-13)

Files:

- `scripts/modeling/16_rbf_svm_probe.py`
- `results/models/16_rbf_svm_probe.txt`
- `submissions/rbf_svm_16.csv`

What happened:

- grid search selected C=100, gamma=0.1 (best inner CV: 0.8406)
- repeated CV through harness: 0.8296 ± 0.0214 (vs baseline 0.8301 ± 0.0239)
- paired delta: -0.0005, W/L/T: 21/22/7 — dead even
- 35 test flips: 18 female_Pclass3 survived→died, 11 male_Pclass1 mixed
- instability 6x worse than logistic (0.054 vs 0.009)
- Kaggle score: **0.7655** (same as gender-only baseline)

Why it failed:

- the decision boundary on v2 features is approximately linear
- the SVM's "corrections" in female_Pclass3 are wrong on the hidden test set (same pattern as segmented 12a)
- higher model flexibility = more aggressive subgroup corrections = worse leaderboard

Key conclusion: the ceiling is in the features, not the model class. Nonlinear models on the same features don't help.

### Full-manifest groups + relaxed regularization (step 17, 2026-03-14)

Files:

- `scripts/modeling/17_full_manifest_groups.py`

What happened:

- tested TicketGroupSize computed on full train+test manifest (not just train)
- tested broader C sweep: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
- C=0.01 is genuinely optimal across ALL feature sets — not noisy selection
- full-manifest ticket groups still don't help as features (delta +0.0001 at best)
- confirmed that higher C = worse CV consistently (more noise fitting)

Key conclusion: the v2 regularization choice was correct. The features are noisy enough that strong regularization genuinely helps.

### Ticket survival propagation (step 18, 2026-03-14) — BREAKTHROUGH

Files:

- `scripts/modeling/18_ticket_survival_propagation.py`
- `submissions/logreg_18_18b_hybrid_2mate.csv`

What happened:

- tested using training survival labels from ticket-mates as signal
- 18a: TicketSurvRate as a feature — CV dropped catastrophically (-3.2%) because OOF encoding makes the feature noisy during CV
- 18b: Hybrid rules — override logistic prediction when 2+ ticket-mates have unanimous outcomes
  - CV 0.8228 (slightly below baseline 0.8301 due to group splitting across folds)
  - 13 test flips: 7 died→survived (male Pc1/Pc2 whose families survived), 6 survived→died (Pc3 whose families died)
  - **Kaggle score: 0.7751 — NEW BEST** (up from 0.7727)
- 18b-loose: 1+ mates — too aggressive (128 overrides, CV -3.7%)

Why it worked:

- rules bypass regularization — C=0.01 can't suppress a hard override
- unanimous ticket-mate outcomes are a strong, domain-legitimate signal
- families and travel parties survived or died together (historical fact)
- the 13 flips are spread across subgroups and domain-motivated, unlike previous concentrated female_Pclass3 corrections

Why the CV was pessimistic:

- random CV splits break ticket groups across folds
- a group of 3 gets split ~60/40, so often <2 mates available in the training fold during CV
- at test time, ALL 891 training passengers are available, so the rule fires more often and more accurately
- **this means standard CV underestimates group-based signals**

### Evaluation infrastructure (step 20, 2026-03-14)

Added two new validators to `shared/evaluate.py`:

1. **Repeated grouped CV** (`repeated_grouped_cv`, `grouped_cv_splits`): keeps entire ticket groups in the same fold. Useful for testing whether the base logistic model leaks information through group structure. Result: v2 logistic scores 0.8295 under grouped CV vs 0.8301 under stratified CV — no leakage detected.

2. **Leave-one-out grouped CV** (`leave_one_out_grouped_cv`): holds out one member per multi-member group while keeping mates in training. Designed to simulate the deployment scenario for propagation rules. **However, testing revealed a structural limitation:** for unanimous-outcome rules like 18b, LOO produces only trivially-correct fires (from truly unanimous groups) and structurally-wrong fires (from dissenter removal in mixed groups). Neither informs the real question: "will a test passenger from a training-unanimous group follow the group pattern?"

Key finding: **propagation rules on small grouped datasets cannot be meaningfully validated locally.** The Kaggle leaderboard remains the only real evidence for the rule layer. Local evaluation is useful for comparing rule variants (fire rate, coverage, structural diagnostics) but not for absolute accuracy estimates.

Group composition in training: 134 eligible ticket groups (2+ members), of which 85 are truly unanimous and 49 are mixed.

## What Still Looks Plausible

The group survival propagation direction has been validated on Kaggle but local evaluation has inherent limits. Remaining directions:

1. **Refine surname filtering** — exclude common names, require 3+ mates, or use name-rarity weighting to keep genuine families and drop false matches
2. **Relaxed unanimity threshold** — test majority rules (e.g., 3 out of 4 mates survived → predict survived). Higher noise risk.
3. **Cabin-based groups** — passengers sharing a cabin number are likely family/travel party. Coverage may be low (many missing cabin values).
4. **Use local evaluation for comparative diagnostics** — fire rate, coverage, structural quality of groups — not for absolute accuracy. Kaggle is the final arbiter for rule-based changes.

## What Not To Chase Next

- more tree tuning or ensembling
- large segmented-model swings
- broad feature proliferation that improves CV but changes many passengers
- nonlinear model classes on the current feature set (explored step 16, confirmed linear ceiling)
- TicketSurvRate as a continuous feature (explored step 18a, catastrophic CV drop)
- any model that aggressively reclassifies female_Pclass3 as died (confirmed harmful 3 times)
- relaxed regularization (C=0.01 confirmed optimal in step 17)
- hybrid rules with 1+ mates (too aggressive, step 18b-loose)
- naive surname+Pclass propagation (step 19: common-name false positives cancel genuine signal, net zero on Kaggle)
- connected components merging ticket + surname groups (step 19c: false merges from common names make it worse than ticket-only)

## Recommended Active Workflow

Keep the active path small:

1. Preprocess with `build_features.py` (outputs v3 artifacts to `data/train_processed.csv`, `data/test_processed.csv`)
2. The v2 baseline is frozen via `13_v2_rebaseline.py` + `reconstruct_v2_features()` in `shared/evaluate.py:432`. This reconstructs the 15-feature v2 matrix from v3 processed data and is verified against the saved `logreg_v2.csv`. Experiment scripts (16-19) import `reconstruct_v2_features` directly — do NOT re-run `10_logreg_v2.py` to reproduce the baseline.
3. The current best submission is `18_ticket_survival_propagation.py` (hybrid rules + v2 logistic). Use this as the practical baseline for new experiments.
4. Any new experiment must answer:
   - how many test passengers changed?
   - which subgroup changed?
   - did the change stay small and interpretable?

Note: `10_logreg_v2.py` is the script that originally produced `logreg_v2.csv`, but it reads whatever is in `train_processed.csv` directly. Since `build_features.py` now outputs v3 features, re-running step 10 would silently train on v3 features. The baseline is maintained through saved artifacts and `reconstruct_v2_features()`, not by re-running step 10.

If a new model changes 20+ passengers, especially concentrated in one subgroup, treat it as high-risk even if CV improves.

## Suggested Cleanup / Refactor Plan

Do not delete anything yet. First reorganize the repo so active work and historical work are separated.

### 1. Keep these as active

- `scripts/preprocessing/build_features.py`
- `scripts/modeling/10_logreg_v2.py`
- `results/models/10_logreg_v2.txt`
- `results/models/10_logreg_v2_coefficients.csv`
- `submissions/logreg_v2.csv`
- `NOTES.md`
- this file

### 2. Move these into `scripts/modeling/archive/`

Archive completed experiments that are no longer on the active path:

- `01_logistic_regression.py`
- `02_random_forest.py`
- `03_xgboost.py`
- `04_lightgbm.py`
- `05_neural_network.py`
- `06_xgboost_tuned.py`
- `07_random_forest_tuned.py`
- `08_logistic_regression_tuned.py`
- `09_ensemble.py`
- `11_logreg_v3.py`
- `11a_logreg_v3_interactions.py`
- `11b_logreg_v3_fare_deck.py`
- `11c_logreg_v3_surname.py`
- `12a_segment_by_sex.py`
- `12b_segment_fare_rank.py`
- `12c_segment_threshold.py`

Leave `10_logreg_v2.py`, `13_v2_rebaseline.py`, `14_male_pclass1_interactions.py`, `15_ticket_group_experiment.py`, and `16_rbf_svm_probe.py` in place as active reference.

### 3. Move stale results into `results/models/archive/`

Archive anything not tied to the active baseline or the most recent decision-making.

Keep outside archive:

- `10_logreg_v2.txt`
- `10_logreg_v2_coefficients.csv`
- optionally `12a_segment_by_sex.txt` as the main cautionary example

Everything else can move into `archive/`.

### 4. Move stale submissions into `submissions/archive/`

Keep at top level only:

- `logreg_v2.csv`
- optionally `logreg_12a_segment_sex.csv` as the important failed challenger
- optionally `gender_only.csv` as the trivial baseline

Archive the rest.

### 5. Add a naming convention

For future experiments:

- `NN_description.py` for scripts
- same `NN_` prefix for result files
- submission names should match the script name closely

Avoid names like `FIXED`, `selected`, or multiple similarly named variants without a matching result file.

### 6. Reduce duplication in modeling scripts

If more experiments are added, create a small shared helper module under `src/` or `shared/` for:

- loading processed train/test data
- Tee logging
- CV reporting
- subgroup reporting
- submission diff reporting

Right now each modeling script reimplements the same reporting logic.

## Short Practical Summary

- Best known submission: `logreg_18_18b_hybrid_2mate.csv` (Kaggle **0.7751**)
- The breakthrough came from group survival propagation — not better features or fancier models
- Hybrid rules (ticket-mate unanimous outcomes) override the logistic model for ~48 test passengers
- Individual-level features and model classes are at their ceiling — confirmed across steps 14-17
- Standard CV underestimates group-based signals (splits groups across folds)
- The active direction is refining and extending group survival rules
- Use step 18's hybrid as the new baseline for future work
