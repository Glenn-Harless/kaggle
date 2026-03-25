# Marketing Mix Modeling & Bayesian Inference Cheatsheet

A quick-reference guide from the DT Mart MMM project.

---

## What Is MMM?

**The business question:** "We spent $X on advertising across multiple channels. How much revenue did each channel drive? Where should we spend next?"

**The method:** Time-series regression with marketing-specific transformations (adstock + saturation), fitted with Bayesian inference for honest uncertainty.

| | Predictive ML | Marketing Mix Modeling |
|---|---|---|
| Question | "What will sales be next week?" | "How much did each channel contribute to sales?" |
| Goal | Prediction accuracy | Causal decomposition |
| Output | One number (predicted sales) | Waterfall: baseline + channel 1 + channel 2 + ... |
| Validation | Test set accuracy, CV | Posterior predictive checks, parameter recovery, prior sensitivity |
| Key metric | RMSE, R² | Channel contribution shares with credible intervals |
| Data size | Thousands+ rows | ~50-150 weekly observations |

---

## The MMM Equation

```
sales(t) = intercept
         + β₁ × saturation(adstock(spend_channel1(t)))
         + β₂ × saturation(adstock(spend_channel2(t)))
         + γ₁ × control₁(t)
         + seasonality(t)
         + noise(t)
```

Each channel's spend goes through a two-stage pipeline before affecting sales:

```
Raw spend → Adstock (carry-over) → Saturation (diminishing returns) → β × effect
```

---

## Adstock (Carry-Over)

**What:** Past advertising still influences today's sales. A TV ad on Monday → purchase on Thursday.

**Formula:** `adstock[t] = spend[t] + α × adstock[t-1]`

**Parameter:** α (alpha), range 0-1
- α = 0 → no carry-over (only this week's spend matters)
- α = 0.5 → last week retains 50% of its effect
- α = 0.9 → last week retains 90% — very slow decay

**Channel intuition:**

| Channel | α Range | Why |
|---|---|---|
| Google Search | 0.05-0.2 | Intent-driven, click→buy fast |
| TikTok | 0.1-0.3 | Short-form, quick attention span |
| Facebook | 0.2-0.4 | Discovery, some recall |
| TV / CTV | 0.4-0.7 | Premium content, memorable, jingle sticks |
| Brand campaigns | 0.7-0.9 | Slow brand building over months |

**l_max:** Maximum number of lag weeks to consider. Rule of thumb: set until α^l_max < 0.01.

| α | l_max needed |
|---|---|
| 0.3 | 4 |
| 0.5 | 7 |
| 0.7 | 13 |
| 0.9 | 44 |

**Key insight:** Adstock is NOT autocorrelation. Autocorrelation is a *problem* in residuals. Adstock is a deliberate *feature transformation* that, when modeled correctly, actually *reduces* residual autocorrelation.

---

## Saturation (Diminishing Returns)

**What:** Each additional dollar of advertising is less effective than the last. The 1st ad impression matters more than the 50th.

**Formula:** `saturation(x) = 1 - exp(-λx)` (logistic)

**Parameter:** λ (lambda), range >0
- λ small → slow saturation (big budget before hitting ceiling)
- λ large → fast saturation (small budget already near ceiling)

**Channel intuition:**

| Channel | λ Range | Why |
|---|---|---|
| National TV | Low (1-3) | Massive audience, takes a lot to saturate |
| CTV/OTT | Low-Med (1-3) | Broad reach, slow saturation |
| Facebook | Medium (2-4) | Large audience, moderate saturation |
| TikTok | Med-High (3-6) | Younger demographic, faster saturation |
| Google Search | High (4-8) | Limited keyword inventory saturates fast |
| Retargeting | Very High (6+) | Small audience, exhaustion happens quickly |

**Business implication:** If Channel A has λ=2 (room to grow) and Channel B has λ=8 (saturated), shift budget A→B. This is what budget optimization does.

**How to read a response curve:**
- X-axis: spend level
- Y-axis: effect on sales
- Steep part (left): every dollar counts
- Flat part (right): saturated — more spend doesn't help
- Red dot: where your current spend sits

---

## Bayesian Inference — The Core Idea

```
1. State prior beliefs          "TV ROI is probably 1-5x"
2. Observe data                 52 weeks of sales + spend
3. Bayes' theorem combines      prior + data → posterior
4. MCMC discovers the posterior  8,000 samples mapping the landscape
5. Use samples for everything    means, intervals, probabilities
```

### Prior → Posterior

```
P(parameters | data) = P(data | parameters) × P(parameters) / P(data)
   ↑ POSTERIOR            ↑ LIKELIHOOD         ↑ PRIOR        ↑ (ignore)
```

- **Prior:** Your starting belief before seeing data. Encoded as a probability distribution.
- **Posterior:** Your updated belief after seeing data. Also a distribution.
- More data → posterior dominated by data, prior doesn't matter much
- Less data → posterior influenced by prior (our DT Mart situation)

### Beta Distribution Priors (for α)

α and β parameters = "fake observations"
- Beta(1,1): 2 fake obs → uninformative, data dominates
- Beta(5,5): 10 fake obs → mild regularization
- Beta(50,50): 100 fake obs → strong prior, fights the data

Mean = α / (α + β). Strength = α + β.

### MCMC (Markov Chain Monte Carlo)

**Not optimization.** MCMC explores the posterior landscape; gradient descent finds one point.

```
Gradient descent:  Uses gradients to CONVERGE to one point
MCMC (NUTS):       Uses gradients to EXPLORE the entire distribution
```

- **Chains:** 4 independent MCMC runs (4 hikers exploring independently)
- **Warmup/tune:** Initial samples discarded (hiker finding the region)
- **Draws:** Kept samples that map the posterior (typically 1000-2000 per chain)
- The posterior already exists (defined by prior + data). MCMC discovers it.

---

## Convergence Diagnostics

Your validation checklist (replaces train/test accuracy):

| Check | What It Means | Good | Bad |
|---|---|---|---|
| **R-hat** | Do all chains agree? | < 1.01 | > 1.01 — chains found different answers |
| **ESS** | Effective independent samples | > 400 | < 100 — estimates too noisy |
| **Divergences** | Sampler hit pathological geometry | 0 | 50+ — model needs fixing |
| **Trace plots** | Visual check of chain mixing | Fuzzy caterpillars | Trending, stuck, or chains at different levels |

**If any fail, do not interpret results. Fix the model first.**

Fixes: increase `target_accept` (0.90→0.95), increase `tune`, reduce model complexity, reparameterize.

---

## The Full Workflow

```
1. PRIOR PREDICTIVE CHECK     "Do my priors produce plausible sales?"
   → Generate predictions from priors alone (no fitting)
   → If predictions are wildly implausible, fix priors

2. FIT (MCMC SAMPLING)        "Run 4 chains × 2000 draws"
   → pm.sample() with target_accept=0.95

3. CONVERGENCE DIAGNOSTICS    "Did the sampler work?"
   → R-hat < 1.01, ESS > 400, divergences ≈ 0, fuzzy caterpillars

4. POSTERIOR PREDICTIVE CHECK  "Does the model reproduce the data?"
   → Generate fake data from posterior, compare to real data

5. CHANNEL CONTRIBUTIONS      "How much did each channel drive?"
   → Decomposition: baseline + ch1 + ch2 + ... = total sales
   → HDI intervals: "Channel 1 contributed 52-77% of media sales"
   → Pairwise probabilities: "P(ch1 > ch2) = 98%"

6. PRIOR SENSITIVITY           "Do results change with different priors?"
   → Re-run with different priors, compare channel shares
   → < 5% change = robust (data-driven)
   → > 15% change = prior-dependent (be transparent)

7. BUDGET OPTIMIZATION         "Where should we shift spend?"
   → Response curves show where each channel sits on saturation
   → ROAS comparison across channels
   → Scenario analysis: "what if we cut X by 50%?"
```

---

## Reading MMM Output

### Contribution Shares with HDI

```
Channel 1:  65% [52% — 77%]  (95% HDI)
Channel 2:  35% [23% — 48%]
```

- 65% = mean (average across all posterior samples)
- [52% — 77%] = 95% HDI (we're 95% sure the true share is in this range)
- Ranges can overlap between channels — doesn't mean they're equal

### Pairwise Probabilities

```
P(ch1 > ch2) = 98%
```

Computed by counting: out of 8,000 posterior samples, how many had ch1 > ch2? This is the direct answer to "which channel drives more?" — more actionable than overlapping HDIs.

### Waterfall Chart

Read left to right: Baseline + Ch1 + Ch2 + ... + Controls = Total Sales.
Baseline = sales that would happen without any advertising.
Channel blocks = sales attributed to each channel's advertising.

---

## Why Bayesian > OLS for MMM

| Problem | OLS Failure | Bayesian Fix |
|---|---|---|
| **Negative coefficients** | "Spending on Digital *decreases* sales" (obviously wrong) | HalfNormal priors force coefficients positive |
| **Multicollinearity** | Correlated channels get arbitrary coefficient splits | Priors regularize and stabilize estimates |
| **Overfitting** | R²=0.51 on 48 observations with 9 features | Priors act as structured regularization |
| **No uncertainty** | One number per channel, no confidence measure | Posterior distributions with HDI intervals |
| **No domain knowledge** | Can't encode adstock, saturation, positivity | Model structure encodes marketing physics |

VIF (Variance Inflation Factor) quantifies multicollinearity:
- VIF=1: no problem
- VIF=10+: severe — coefficient estimates unreliable
- VIF=50+: extreme — estimates are meaningless

---

## Data Requirements

| Factor | Minimum | Ideal | DT Mart | Your Work |
|---|---|---|---|---|
| Weekly observations | 52 | 104+ (2 years) | 47 | 80-100 |
| Spend granularity | Weekly | Daily | Monthly (!) | Weekly |
| Channels | 3-4 groups | 3-6 | 4 (grouped from 9) | 3-4 |
| Obs/parameter ratio | 3 | 5-10 | 2.8 | 5-7 |
| Channel correlation | < 0.7 | < 0.5 | 0.86 (Brand/SEM) | TBD |

**Each channel costs 3 parameters** (α, λ, β). Adding a 5th channel with tiny spend wastes 3 degrees of freedom. Drop or group small channels.

**Monthly spend distributed to weekly** means within-month variation = 0. The model can only learn from between-month variation. Get weekly spend from ad platforms whenever possible.

---

## Connection to Other Work

```
Geo experiments:       MMM:                        Uplift Modeling (Criteo):
─────────────────────────────         ────────────────────────    ────────────────────────
"Did the campaign work?"              "How much did each          "WHO should we target?"
                                       channel contribute?"
One channel at a time                 All channels simultaneously All users simultaneously
High internal validity                Broad but less certain      Individual-level targeting
Ground truth for ONE channel          Estimates for ALL channels  Ranks users by persuadability
$70K for one answer                   Ongoing model, updated      Reusable scoring model
                                       quarterly
```

**How they work together:**
1. Run geo experiment → get ground truth lift for one channel
2. Feed lift result into MMM as informative prior (`mmm.add_lift_test_measurements()`)
3. MMM estimates all channels, calibrated by the experiment
4. Uplift model targets the right users within each channel

**Your CTV plan:** Build MMM on Google/Facebook/TikTok (80-100 weeks). Use MMM predictions as counterfactual baseline during CTV flight. Compare actual vs predicted = CTV-attributable estimate. Geo experiment provides causal confirmation. Three methods triangulate.

---

## Libraries

```
pymc              Bayesian modeling framework (defines models, runs MCMC/NUTS)
pymc-marketing    Marketing-specific models (MMM class, adstock, saturation)
arviz             Bayesian visualization/diagnostics (trace plots, HDI, R-hat)
statsmodels       OLS baseline, VIF calculation
```

**PyMC-Marketing API (key classes):**
```python
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation

mmm = MMM(
    date_column="date",
    channel_columns=["google", "facebook", "tiktok"],
    control_columns=["is_holiday", "trend"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)
mmm.sample_prior_predictive(X, y)     # before fitting
mmm.fit(X, y, target_accept=0.95)     # MCMC
mmm.sample_posterior_predictive(X)     # after fitting
```

---

## File Reference

```
scripts/preprocessing/01_eda.py              EDA with marketing measurement lens
scripts/modeling/02_frequentist_baseline.py   OLS failure demonstration
scripts/modeling/03_bayesian_primer.py        Bayesian mechanics on toy problems
scripts/modeling/04_first_mmm.py              First MMM on synthetic data (parameter recovery)
scripts/modeling/05_adstock_saturation.py      Adstock & saturation deep dive
scripts/modeling/06_diagnostics.py            Diagnostics & contribution decomposition
scripts/preprocessing/07_data_prep.py         DT Mart data engineering (9→4 channels)
scripts/modeling/08_real_mmm.py               Full MMM on DT Mart + prior sensitivity
scripts/modeling/09_budget_optimization.py     ROAS, response curves, reallocation

results/models/*.txt                         All printed output (logged via Tee)
results/analysis/*.png                       All plots
data/mmm_weekly.csv                          MMM-ready weekly dataset
```
