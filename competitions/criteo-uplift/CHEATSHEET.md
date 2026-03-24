# Uplift Modeling & Causal Inference Cheatsheet

A quick-reference guide from the Criteo Uplift project.

---

## Core Concepts

### Prediction vs Causation

| | Predictive Modeling | Causal / Uplift Modeling |
|---|---|---|
| Question | "Will this user convert?" | "Will showing this ad CAUSE them to convert?" |
| Output | P(Y=1) | τ(x) = P(Y=1\|ad) - P(Y=1\|no ad) |
| Targets | High-propensity users (Sure Things) | Users most influenced by the ad (Persuadables) |
| Ground truth | Observable — did they convert? | Never observable for an individual |

### The Fundamental Problem of Causal Inference

For any person, you can only observe ONE outcome — with the ad or without. Never both. The individual treatment effect is logically impossible to observe. Randomization (RCTs) solves this at the group level by making treatment/control groups statistically identical.

### Key Terms

| Term | Definition | Example |
|---|---|---|
| **ATE** | Average Treatment Effect. Mean outcome difference between treatment and control groups. One number for the whole population. | "Ads increased visits by 1.03% on average" |
| **CATE** | Conditional Average Treatment Effect. Treatment effect for a subgroup defined by features. | "Users with low f8 have 3.6% lift vs 0.1% for high f8" |
| **Counterfactual** | The outcome that would have happened under the alternative scenario. Never observed. | "What would Dallas sales have been without the campaign?" |
| **Treatment** | The intervention being tested. | Showing an ad |
| **Control** | The group that doesn't receive the intervention. | Not showing the ad |
| **RCT** | Randomized Controlled Trial. Random assignment guarantees unbiased ATE. | A/B test, geo experiment |
| **Propensity score** | P(treatment=1\|features). Probability of receiving treatment. | In Criteo: ~0.85 for everyone |
| **MDE** | Minimum Detectable Effect. Smallest effect an experiment can reliably detect (design stage). | "We can detect 18% lift at 60% power" |
| **AUUC** | Area Under Uplift Curve. Measures how well a model ranks users by treatment effect. Higher = better. | X-learner: 0.0294 |

### The Four Quadrants

```
                          Would convert WITHOUT ad?
                          No                Yes
                    +------------------+------------------+
  Converts WITH ad  | PERSUADABLES     | SURE THINGS      |
  Yes               | tau > 0          | tau ~ 0          |
                    | TARGET THESE     | waste of budget   |
                    +------------------+------------------+
  No                | LOST CAUSES      | SLEEPING DOGS    |
                    | tau ~ 0          | tau < 0          |
                    | don't bother     | ads HURT, avoid  |
                    +------------------+------------------+
```

---

## The Models: When to Use What

### Decision Flowchart

```
Do you have RCT data (randomized treatment/control)?
  No  --> Can't do uplift modeling. Run an experiment first.
  Yes --> Continue.

Is your treatment/control split balanced (roughly 50/50)?
  Yes --> S-learner or T-learner both work well
  No  --> X-learner (designed for imbalanced splits)

Is the treatment effect likely small relative to the outcome?
  Yes --> Avoid S-learner (regularization bias). Use T-learner or X-learner.
  No  --> S-learner is fine and simplest.
```

### Model Summary

| Model | How It Works | Strengths | Weaknesses | When to Use |
|---|---|---|---|---|
| **Standard Classifier** | Train P(Y) ignoring treatment. Rank by propensity. | Simple. Sometimes accidentally works. | No causal reasoning. Targets Sure Things in most datasets. | Don't use for uplift. Use as a baseline only. |
| **S-Learner** | One model: P(Y) = f(features, treatment). Uplift = f(x,T=1) - f(x,T=0). | Data-efficient (one model, all data). Simple to implement. Best calibration. | Regularization bias — can ignore weak treatment effects. Treatment ranked 10/13 in our experiment. | Balanced treatment groups. Strong treatment effects. Quick first attempt. |
| **T-Learner** | Two models: Model_T on treatment data, Model_C on control. Uplift = Model_T(x) - Model_C(x). | Treatment effect can't be regularized away. | Noisy when groups are imbalanced (control model has less data). Overpredicts. | Balanced treatment groups. When S-learner's treatment importance is low. |
| **X-Learner** | Three stages: (1) T-learner base models, (2) cross-impute individual effects using real outcomes, (3) model the imputed effects, combine with propensity weights. | Handles imbalanced groups. Uses real outcomes (less noise). Best AUUC in our experiment. | More complex (4 models). Can still overpredict at extremes. | Imbalanced treatment groups (like 85/15). Default choice when unsure. |

### How Each Model Predicts Uplift

```
S-Learner:
  1. Train f(features, treatment) on ALL data
  2. Predict f(x, T=1) - f(x, T=0)

T-Learner:
  1. Train Model_T on treatment data
  2. Train Model_C on control data
  3. Predict Model_T(x) - Model_C(x)

X-Learner:
  1. Train Model_T and Model_C (same as T-learner)
  2. D_treated = actual_outcome - Model_C(features)   [for treated users]
     D_control = Model_T(features) - actual_outcome    [for control users]
  3. Train tau_1 to predict D_treated from features
     Train tau_0 to predict D_control from features
  4. tau_final = g * tau_0(x) + (1-g) * tau_1(x)
     where g = P(treatment)
```

---

## Evaluation

### How to Read an Uplift Curve

- X-axis: % of users targeted (sorted by model score, highest first)
- Y-axis: cumulative incremental outcome (extra visits/conversions caused by ads)
- A curve above random = model is front-loading persuadables
- All curves converge at 100% (total ATE)
- Steeper early climb = better model

### How to Read a Decile Table

Split test users into 10 equal buckets by predicted score. For each bucket:

| Column | Meaning |
|---|---|
| Decile | Range of predicted scores in this bucket |
| N | Number of users |
| Visit(T) | Actual visit rate among treated users in this bucket |
| Visit(C) | Actual visit rate among control users in this bucket |
| ATE | Visit(T) - Visit(C) = actual treatment effect in this bucket |
| Pred | Average predicted uplift score in this bucket |

**What to look for:**
- ATE should increase from bottom decile to top decile (model ranking is correct)
- Pred should be close to ATE (model is well-calibrated)
- Wide spread in ATE = model successfully separates persuadables from non-responders

### Uplift@K%

"If I target the top K% of users (ranked by model score), what's the actual ATE in that group?"

- Computed entirely from test set actuals — model only determines the ranking
- Higher is better at each K%
- All models converge to overall ATE at @100%

### Key Metrics

| Metric | What It Measures | How to Use |
|---|---|---|
| **AUUC** | Overall ranking quality (area under uplift curve) | Primary comparison metric. Higher = better. |
| **Qini** | Similar to AUUC, accounts for group sizes | Secondary metric. Confirms AUUC findings. |
| **Uplift@K%** | Actual ATE when targeting top K% | Practical — aligns with budget constraints. |
| **Decile ATE spread** | Difference in actual ATE between top and bottom decile | Measures discrimination — can the model tell users apart? |
| **Calibration** | How close predicted uplift is to actual ATE per decile | Matters if using scores for thresholds/budget decisions. |

---

## Practical Application

### How to Deploy an Uplift Model

```
1. Score all users: compute uplift = f(features, T=1) - f(features, T=0)
2. Rank users by uplift score (highest first)
3. Apply targeting rule:
   - Budget-based: "target top N users our budget allows"
   - Threshold-based: "target users with uplift > break-even"
   - Exclusion: "never target users with uplift < 0 (sleeping dogs)"
```

### Break-Even Uplift

```
break_even = ad_cost_per_user / value_per_conversion
```

Any user with predicted uplift above this is profitable to target.

### When Uplift Modeling Matters Most

- **High ad costs** — expensive impressions mean waste is costly
- **Imbalanced treatment effects** — some users respond, many don't
- **Budget constraints** — can't target everyone, need to prioritize
- **Sleeping dogs exist** — ads can hurt some users

### When It Matters Less

- **Very cheap ads** (email, push) — waste is cheap, target everyone
- **Homogeneous effects** — everyone responds similarly
- **No RCT data available** — can't train the model

---

## Connection to Geo Experiments

```
Geo Experiments                    Uplift Modeling
----------------------------       ----------------------------
DMAs as units                      Users as units
10 treatment + 10 control          85% treatment + 15% control
Diff-in-diff --> one ATE           Meta-learners --> per-user CATE
"Did the campaign work?"           "WHO did it work for?"
MDE = planning tool                AUUC = evaluation tool
$70K for one answer                Reusable model for targeting
```

**The virtuous cycle:**
1. Run one good experiment (collect treatment/control data)
2. Build uplift model (identify who responds)
3. Target smarter in future campaigns
4. Validate with a follow-up experiment
5. Refine the model → repeat

---

## File Reference

```
scripts/preprocessing/01_eda.py           EDA with causal lens
scripts/modeling/02_naive_baselines.py    LogReg baselines + uplift curves
scripts/modeling/03_s_learner.py          S-learner (LightGBM)
scripts/modeling/04_t_learner.py          T-learner (LightGBM)
scripts/modeling/05_x_learner.py          X-learner (LightGBM)
scripts/modeling/06_model_comparison.py   Head-to-head + Uplift@K%
scripts/modeling/08_targeting_analysis.py Cost-benefit + business analysis

results/models/*.txt                     All printed output (logged via Tee)
results/analysis/*.png                   All plots
```
