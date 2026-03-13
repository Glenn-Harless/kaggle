# Titanic State Of The World

Last updated: 2026-03-12

## Current Read

This project has explored a wide range of feature engineering and model families for the Kaggle Titanic competition. The clearest result so far is that conservative logistic regression models generalize better than more expressive models on the hidden Kaggle test set.

The best confirmed Kaggle score is `0.7727`. That score has been achieved by two plain logistic-regression submissions:

- `submissions/logreg_v2.csv`
- `submissions/logistic_regression.csv`

Higher local CV has **not** translated into better leaderboard performance:

- tuned XGBoost and ensemble models posted the best CV scores and worse Kaggle scores
- segmented logistic regression improved CV meaningfully and still scored worse on Kaggle

The repo therefore has a strong CV-to-leaderboard divergence problem. Local CV is useful for rejecting obviously bad ideas, but it is not reliably ranking the final submissions once models become more expressive.

## What We Know

### Confirmed leaderboard ordering

Sorted by Kaggle score:

| Submission | Local CV | Kaggle | Notes |
| --- | ---: | ---: | --- |
| `logreg_v2.csv` | 0.8305 | 0.7727 | v2 logistic regression, 15 features |
| `logistic_regression.csv` | 0.8316 | 0.7727 | baseline logistic regression |
| `gender_only.csv` | 0.7870 | 0.7655 | pure gender baseline |
| `logreg_12a_segment_sex.csv` | 0.8384 | 0.7608 | segmented-by-sex logistic regression |
| `xgb_thresh_55.csv` | n/a | 0.7608 | thresholded XGBoost |
| `random_forest.csv` | 0.8316 | 0.7584 | random forest |
| `xgboost_tuned.csv` | 0.8552 | 0.7584 | tuned XGBoost |
| `ensemble.csv` | 0.8563 | 0.7560 | hard-vote ensemble |

### Confirmed technical findings

- `v1` feature pipeline had leakage through `TicketGroupSize` computed on combined train + test.
- `v2` removed the leakage and simplified features to an EDA-aligned set.
- `v3` added systematic interactions and grouped deck features. Locally it was coherent, but it was not separately submitted to Kaggle.
- `12a` segmented the logistic model by sex. This improved local CV but overcorrected on the leaderboard.
- `12b` added `FareRankInClass` and did not improve over `12a`.
- `12c` threshold tuning did not improve over `12a`.

### Stable model behavior pattern

- Plain logistic models are conservative and leaderboard-safe.
- More flexible models change more passengers and consistently do worse on Kaggle.
- The leaderboard appears to punish large subgroup-specific corrections.

## Best Current Baseline

Use `10_logreg_v2.py` and `submissions/logreg_v2.csv` as the operational baseline.

Why:

- best confirmed Kaggle score
- no known leakage
- simple enough to reason about
- changes less aggressively than later experiments

Relevant files:

- `scripts/modeling/10_logreg_v2.py`
- `results/models/10_logreg_v2.txt`
- `results/models/10_logreg_v2_coefficients.csv`
- `submissions/logreg_v2.csv`

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

### Surname features

Files:

- `11c_logreg_v3_surname.py`
- `12b_segment_fare_rank.py` is unrelated; keep surname conclusions tied to 11c only

What happened:

- leakage-safe surname experiments did not help
- `SurnameSurvHint` was noisy and unstable
- shuffle ablation did not support shipping surname features

Recommendation: treat as explored and rejected for now.

## What Still Looks Plausible

These are the only directions that still look worth testing:

1. Small, conservative logistic-regression changes off the `v2` baseline
2. Tiny interaction additions that change very few passengers
3. Better local validation focused on stability of changed passengers, not just mean CV
4. Targeted `female Pclass 3` features only if they do not produce large submission swings

## What Not To Chase Next

- more tree tuning
- more ensembling
- large segmented-model swings
- surname survival hints
- broad feature proliferation that improves CV but changes many passengers

## Recommended Active Workflow

Keep the active path small:

1. Baseline preprocess with `build_features.py`
2. Baseline model with `10_logreg_v2.py`
3. Any new experiment must answer:
   - how many test passengers changed?
   - which subgroup changed?
   - did the change stay small and interpretable?

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

Leave `10_logreg_v2.py` in place as the current baseline.

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

- Best known submission: `logreg_v2.csv`
- Best local CV does not mean best Kaggle score
- Conservative logistic regression is the current winning approach
- Archive the tree, ensemble, v3, and segmented branches
- Use `v2` as the baseline for any future work
