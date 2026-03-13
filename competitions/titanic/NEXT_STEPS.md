# Titanic Handoff And Next Steps

Last updated: 2026-03-12

## Where We Are

This project has reached a clear intermediate conclusion:

- the best confirmed Kaggle score is still `0.7727`
- the best confirmed submissions are conservative logistic-regression models
- local CV improvements have repeatedly failed to translate to the leaderboard once models became more flexible

At this point, the problem is not "we need to try more random models." The problem is that the local validation loop and the hidden-test behavior diverge, especially when a model makes large subgroup-specific corrections.

## Best Known Submission

Current best confirmed Kaggle submission:

- `submissions/logreg_v2.csv`

Supporting files:

- `scripts/modeling/10_logreg_v2.py`
- `results/models/10_logreg_v2.txt`
- `results/models/10_logreg_v2_coefficients.csv`

This remains the operational baseline.

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

### 1. Leakage-safe family / ticket-group experiment

This is the highest-upside remaining direction.

Why:

- Titanic often benefits from group context
- individual-level linear features appear close to tapped out
- this is still one of the most plausible routes to `0.80+`

Constraints:

- strict out-of-fold feature construction
- compare against `logreg_v2.csv`
- reject immediately if the feature changes too many passengers or leaks target information

Candidate ideas:

- family / surname group size
- ticket-party size, but only if computed leakage-safe
- family-level survival hints, only out-of-fold

### 2. One bounded nonlinear single-model probe

Try exactly one nonlinear model, with the harness.

Best candidate:

- RBF SVM

Why:

- it is a genuine change in model class
- it can capture interactions without explicitly exploding features

Constraints:

- do not branch into multiple nonlinear families at once
- use repeated CV, flip audits, and subgroup change analysis
- reject if it changes many passengers without a very strong case

### 3. One tiny hybrid rules-plus-model baseline

Only if done conservatively.

Why:

- Titanic is one of the few datasets where a small amount of rules-based structure can be legitimate

Constraints:

- keep rules limited to near-deterministic cases
- use the logistic baseline for everything else
- evaluate through the same harness

### 4. Keep the harness as the default gate

Any future experiment should report:

- repeated CV mean and std
- paired delta vs baseline
- total test flips
- flips by subgroup
- flips outside intended target subgroup
- changed PassengerIds

No future candidate should go to Kaggle without passing through that process.

## Recommended Immediate Order

If work resumes later, do this in order:

1. design a leakage-safe family / ticket-group experiment
2. run one bounded nonlinear single-model probe
3. consider a tiny hybrid rules-plus-model baseline only after seeing those results

## Practical Baseline To Resume From

Start from:

- `scripts/modeling/10_logreg_v2.py`
- `submissions/logreg_v2.csv`
- `shared/evaluate.py`
- `scripts/modeling/13_v2_rebaseline.py`

Treat these as the reference stack.

## Bottom Line

We are leaving off in a good place analytically:

- the repo now has a stronger understanding of why local CV and Kaggle diverge
- the highest-risk branches have been tested and rejected
- the active baseline is clear
- the next useful work should focus on qualitatively different signal sources, not more of the same feature tinkering
