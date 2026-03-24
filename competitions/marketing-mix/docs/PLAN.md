# Marketing Mix Modeling: Plan of Attack

**Datasets:** PyMC-Marketing Synthetic Data (learning) + [DT Mart Market Mix Modeling](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling) (real-world practice)
**Goal:** Learning-focused — understand Bayesian inference and marketing mix modeling from scratch
**Approach:** Progression from frequentist failure to full Bayesian MMM, building intuition at every step
**Date:** 2026-03-23

---

## Why Marketing Mix Modeling?

Every company that spends money on advertising asks the same question: **"How much did each channel contribute to sales?"** MMM answers this by decomposing total sales into contributions from each marketing channel (TV, digital, search, etc.), accounting for two real-world effects:

1. **Adstock (carry-over):** A TV ad you see Monday still influences you on Thursday. Effects linger and decay over time.
2. **Saturation (diminishing returns):** The first $10K on Facebook drives lots of conversions. The next $10K drives fewer. The 10th $10K drives almost none.

MMM is the **hottest skill in marketing data science** right now. Apple's iOS 14.5 privacy changes killed cookie-based attribution, so the entire industry is returning to MMM for marketing measurement. Google open-sourced Meridian, Meta built Robyn, and PyMC Labs built PyMC-Marketing — all in the last 2-3 years.

### Why Bayesian?

This is where it gets interesting. Traditional (frequentist) regression **breaks** on MMM data:

- You have ~100-150 weekly observations but 20+ parameters → severe overfitting
- Marketing channels are correlated (you increase TV and Facebook together in Q4) → multicollinearity produces negative ROI estimates for channels that clearly work
- You have domain knowledge ("TV ads have carry-over effects") but no way to encode it in OLS

Bayesian inference solves all three problems by letting you:
- **Encode domain knowledge as priors** ("TV effects probably decay with a half-life of 2-4 weeks")
- **Get distributions, not point estimates** ("TV contributed 28-34% of sales" instead of just "31%")
- **Regularize with structure** (priors prevent overfitting far better than generic L1/L2)

---

## The Datasets

We use **two datasets** strategically — synthetic for learning, real for practice:

### Dataset 1: PyMC-Marketing Synthetic Data (Steps 4-6)

| Property | Value |
|---|---|
| Observations | 179 weeks (Apr 2018 - Sep 2021) |
| Channels | 2 (`x1`, `x2`) |
| Controls | 2 events + trend |
| Target | `y` (synthetic sales) |
| Key feature | **Known ground truth** — adstock α=0.4/0.2, saturation λ=4.0/3.0 |

Purpose: Learn the full Bayesian MMM workflow on clean data where you can **verify the model recovers the true parameters**. This is like having the answer key while you learn the method.

### Dataset 2: DT Mart (Steps 7-9)

| Property | Value |
|---|---|
| Transaction data | ~1.5M rows, daily, Jul 2015 - Jun 2016 |
| Aggregated to weekly | ~52 observations (we build this ourselves) |
| Channels | 9 (TV, Digital, Sponsorship, Content Marketing, Online Marketing, Affiliates, SEM, Radio, Other) |
| Product categories | 5 (Camera, CameraAccessory, EntertainmentSmall, GameCDDVD, GamingHardware) |
| Controls | Special sales/promotions (Diwali, Eid, Independence Day, etc.) |
| Additional | Monthly NPS scores, product MRP and discount data |

Purpose: Grapple with real-world messiness — limited data, many channels, multicollinearity, and the honest realization that **52 weeks with 9 channels isn't enough data for a reliable MMM.** Learning when your data can't support your model is a critical practitioner skill.

### Key Columns (DT Mart)

| Source File | Key Columns | Description |
|---|---|---|
| `firstfile.csv` | Date, gmv_new, units, product_category, discount, Sales_name | Transaction-level sales data |
| `MediaInvestment.csv` | TV, Digital, Sponsorship, Content.Marketing, Online.marketing, Affiliates, SEM, Radio, Other | Monthly media spend by channel |
| `SpecialSale.csv` | Sale dates and names | Indian market promotions (Diwali, Eid, Republic Day, etc.) |
| `MonthlyNPSscore.csv` | NPS | Net Promoter Score by month |

---

## Key Differences from Previous Projects

| | Your Previous Projects | Marketing Mix Modeling |
|---|---|---|
| Paradigm | **Frequentist** — one best answer | **Bayesian** — distribution of plausible answers |
| Parameters | Point estimates (coefficient = 2.3) | Posterior distributions (coefficient ≈ Normal(2.3, 0.4)) |
| Regularization | Generic (L1/L2 shrink toward zero) | Structured (priors encode domain knowledge) |
| Output | Predictions | **Decomposition** — how much did each channel contribute? |
| Data size | Thousands to millions of rows | ~50-150 weekly aggregated observations |
| Validation | Train/test split, CV | Parameter recovery, posterior predictive checks, lift test calibration |
| Key tool | scikit-learn, LightGBM | PyMC, PyMC-Marketing, ArviZ |

---

## The Big Idea: Why Bayesian Thinking Is Different

In every project you've done so far, the workflow has been:

```
Define model → Fit (minimize loss) → Get ONE answer → Evaluate on test set
```

Bayesian inference flips this:

```
Define model + PRIOR BELIEFS → Update beliefs with data → Get a DISTRIBUTION of answers → Every downstream decision carries uncertainty
```

### Bayes' Theorem — The Only Formula You Need

```
P(parameters | data) = P(data | parameters) × P(parameters) / P(data)
   ↑ POSTERIOR            ↑ LIKELIHOOD         ↑ PRIOR        ↑ EVIDENCE
   "what I believe        "how likely is        "what I        (normalizing
    after seeing data"     this data given       believed       constant —
                           these parameters?"    before"        ignore this)
```

**Concrete MMM example:**

- **Prior:** "I believe TV's adstock decay rate is probably between 0.3 and 0.7" → `Beta(3, 3)`
- **Likelihood:** "Given decay=0.5, how well does the model explain observed sales?"
- **Posterior:** "After seeing 2 years of sales data, TV's decay rate is probably 0.45-0.65"

The prior is your **starting belief**. The data **updates** that belief. The posterior is your **informed belief**. With more data, the posterior is dominated by the data. With less data, the prior matters more. This is exactly the right behavior for MMM — we have limited data, so domain knowledge (priors) should matter.

### Why Can't You Just Compute This?

In theory, Bayes' theorem gives you the exact posterior. In practice, the integral in the denominator (P(data)) is intractable for models with more than a few parameters. You'd need to integrate over a 20-dimensional space. So instead, we **sample** from the posterior using MCMC.

**MCMC (Markov Chain Monte Carlo) in plain English:**

Imagine you're a hiker exploring a mountain range in dense fog. You can't see the entire landscape, but at each step you can check the elevation where you stand. MCMC works like this:

1. Start somewhere random in parameter space
2. Propose a random step
3. Check: is the posterior probability higher here? If yes, move. If no, maybe move (with probability proportional to how much worse it is)
4. Record where you are
5. Repeat 4,000 times
6. The collection of recorded positions IS your posterior distribution

The places you visit most frequently are the high-probability parameter values. After enough steps, your "hiking trail" traces out the posterior distribution.

**PyMC uses NUTS (No-U-Turn Sampler)** — a smarter version that uses gradient information to make intelligent proposals instead of random walks. Think of it as gradient descent, except instead of finding ONE minimum, it explores the ENTIRE high-probability region.

### Key Vocabulary

| Term | What It Means | Analogy |
|---|---|---|
| **Prior** | Your belief before seeing data | Your hypothesis before running an experiment |
| **Posterior** | Your updated belief after data | Your conclusion after seeing results |
| **Credible interval** | "95% probability the parameter is in this range" | What people THINK confidence intervals mean (but they don't) |
| **Chains** | Independent MCMC runs (usually 4) | 4 hikers exploring independently |
| **Warmup/burn-in** | Initial samples discarded | Hikers finding the interesting region |
| **Divergence** | Sampler hit a pathological region | Hiker fell off a cliff — model might need fixing |
| **R-hat** | Convergence check (should be < 1.01) | Did all 4 hikers find the same peaks? |
| **Prior predictive check** | Generate data from priors alone | "Do my assumptions produce sensible predictions?" |
| **Posterior predictive check** | Generate data from fitted model | "Does my model reproduce the patterns in real data?" |

---

## Connection to Your Work

```
Geo experiments (your day job):          MMM (what you'll learn):
─────────────────────────────            ────────────────────────────
Treatment: Dallas sees ads               Input: weekly spend by channel
Control: Houston doesn't                 Control: seasonal effects, promos
Measure: aggregate lift                  Measure: channel contribution %
Question: "Did the campaign work?"       Question: "How much did each channel contribute?"
Output: one number (total lift)          Output: decomposition + uncertainty
Limitation: one experiment at a time     Strength: estimates all channels simultaneously
```

MMM and geo experiments are **complementary**:
- Geo experiments give you **ground truth** for one channel at a time (high internal validity)
- MMM gives you **estimates for all channels** simultaneously (broad but less certain)
- In practice, you use geo experiments to **calibrate MMM priors** — this is called lift test calibration, and PyMC-Marketing has built-in support for it

The Criteo uplift project asks "WHO should we target?" MMM asks "WHERE should we spend the budget?" Together, they cover the two biggest questions in marketing measurement.

---

## Steps

### Phase 1: Foundations (Steps 1-3)

#### Step 1: EDA — Understanding Marketing Time-Series Data

Explore the DT Mart dataset with a marketing measurement lens:

- **Data engineering:** Aggregate transaction-level data (`firstfile.csv`, ~1.5M rows) to weekly granularity
- **Join media spend:** Map monthly `MediaInvestment.csv` to weekly observations (discuss: why monthly spend data for weekly models is a limitation)
- **Channel exploration:** Spend distribution across 9 channels — which channels vary enough to be identifiable?
- **Correlation analysis:** How correlated are the channels? (this foreshadows the multicollinearity problem)
- **Seasonality:** Identify weekly/monthly patterns, special sale effects (Diwali, etc.)
- **Target distribution:** Total GMV (gross merchandise value) over time — trends, spikes, patterns
- **Sample size reality check:** Count weekly observations, count parameters we'd need → is this enough? (Spoiler: barely)

This EDA is different from predictive EDA — we're assessing whether the data can support causal decomposition, not looking for predictive features.

- Script: `scripts/preprocessing/01_eda.py`

**What you learn:** How marketing time-series data differs from cross-sectional data, the multicollinearity problem, why aggregation granularity matters, and why data quantity is the #1 limiting factor in MMM.

#### Step 2: The Frequentist Baseline — Why OLS Breaks

Build a standard OLS regression to intentionally show why MMM needs Bayesian methods:

1. **Simple OLS:** `sales ~ TV + Digital + SEM + ...` — just raw spend as features
2. **Inspect coefficients:** At least one channel will likely have a negative coefficient (negative ROI — this is wrong, spending money shouldn't decrease sales)
3. **Check VIF (Variance Inflation Factor):** Quantify multicollinearity between channels
4. **Overfit check:** With ~52 observations and 9+ features, R² will be deceptively high
5. **No uncertainty:** OLS gives point estimates — you can't say "how confident am I that TV contributed more than Digital?"
6. **No domain knowledge:** OLS doesn't know that adstock exists, that effects decay, or that channels saturate

The point of this step is to **viscerally experience the failure mode** that motivated Bayesian MMM. Keep this script short — it's a demonstration, not a model we'll use.

- Script: `scripts/modeling/02_frequentist_baseline.py`

**What you learn:** Why standard regression fails for marketing attribution. Multicollinearity, overfitting on small time-series, and the inability to encode domain knowledge are not minor annoyances — they fundamentally produce wrong answers.

#### Step 3: Bayesian Primer — Learning to Think in Distributions

Before touching MMM, build Bayesian intuition on simple problems:

1. **Coin flip:** Start with a Beta prior, observe flips, watch the posterior update. Visualize how the posterior narrows with more data. Change the prior strength — see how informative vs uninformative priors affect results.

2. **Bayesian linear regression:** Fit a simple `y = mx + b` with PyMC. Compare posterior distributions for slope and intercept vs. OLS point estimates. Introduce: trace plots, posterior summary, credible intervals.

3. **Prior predictive checks:** Before fitting, sample from priors and generate synthetic predictions. Ask: "Do my priors produce plausible data?" If priors on the slope allow slopes of 1000, the prior is bad.

4. **Posterior predictive checks:** After fitting, generate synthetic data from the fitted model. Compare to real data. Does the model capture the right patterns?

5. **MCMC mechanics:** Run the coin flip example with different numbers of samples (100, 1000, 10000). Watch convergence. Introduce chains, R-hat, effective sample size.

This step is ONLY about building Bayesian intuition. No marketing, no MMM — just the statistical framework. Every concept introduced here will be used in Steps 4-9.

- Script: `scripts/modeling/03_bayesian_primer.py`

**Primer — Key Libraries:**

```
PyMC          — The Bayesian modeling framework (defines models, runs MCMC)
ArviZ         — Visualization and diagnostics for Bayesian models (trace plots, forest plots, R-hat)
PyMC-Marketing — Marketing-specific models built on PyMC (MMM, CLV)
```

PyMC is to Bayesian modeling what scikit-learn is to ML. ArviZ is like matplotlib but for posteriors. PyMC-Marketing is like a scikit-learn estimator that wraps the complexity into a clean API.

**What you learn:** Priors, posteriors, MCMC sampling, convergence diagnostics, prior/posterior predictive checks. These are the building blocks for everything that follows.

### Phase 2: Bayesian MMM on Synthetic Data (Steps 4-6)

#### Step 4: First Bayesian MMM — The Full Workflow

Use PyMC-Marketing's synthetic dataset (179 weeks, 2 channels, known ground truth) to learn the complete MMM workflow:

1. **Load and explore:** The synthetic data has known parameters (adstock α=0.4/0.2, saturation λ=4.0/3.0) — our goal is to recover these
2. **Build the model:** Instantiate `MMM` with `GeometricAdstock` and `LogisticSaturation`
3. **Prior predictive check:** Does the model generate plausible sales before seeing data?
4. **Fit the model:** Run MCMC (4 chains, 1000 draws, 1500 warmup)
5. **Check convergence:** R-hat < 1.01, no divergences, reasonable ESS
6. **Posterior analysis:** Did we recover the true parameters? Compare posterior distributions to known ground truth
7. **Channel contributions:** Decompose total sales into channel 1 + channel 2 + baseline contributions

This is the **parameter recovery exercise** — the Bayesian equivalent of a unit test. If the model can't recover known parameters on clean data, something is wrong with the setup.

- Script: `scripts/modeling/04_first_mmm.py`

**Primer — The MMM Equation:**

```
sales(t) = intercept
         + β₁ × saturation(adstock(spend_channel1(t)))
         + β₂ × saturation(adstock(spend_channel2(t)))
         + γ₁ × control₁(t)
         + seasonality(t)
         + ε(t)
```

Each channel's spend goes through two transformations:
1. **Adstock** — carries forward past spend: `adstock[t] = spend[t] + α × adstock[t-1]`
   - α = 0 → no carry-over (search ads: click today, forget tomorrow)
   - α = 0.8 → strong carry-over (TV: see the ad Monday, remember it Friday)

2. **Saturation** — diminishing returns: `saturation(x) = 1 - exp(-λx)`
   - Low spend → near-linear returns (every dollar counts)
   - High spend → flat returns (you've saturated the audience)

The β coefficients capture the overall effectiveness of each channel. The prior on β is positive-only (`HalfNormal`) because spending money shouldn't decrease sales.

**What you learn:** The complete Bayesian MMM workflow end-to-end. Parameter recovery builds trust that the method works before applying it to messy real data.

#### Step 5: Adstock & Saturation Deep Dive

Isolate and deeply understand the two transformations that make MMM work:

1. **Adstock exploration:**
   - Visualize geometric decay with different α values (0.1, 0.3, 0.5, 0.7, 0.9)
   - Show how a single week's spend "smears" across future weeks
   - Compare `GeometricAdstock` (1 parameter, constant decay) vs `WeibullAdstock` (2 parameters, flexible decay shape)
   - Connect to marketing intuition: TV has long adstock, paid search has short adstock
   - Show `l_max` (maximum lag) choice matters — too short truncates the effect, too long adds noise

2. **Saturation exploration:**
   - Visualize logistic saturation with different λ values
   - Show the response curve: spend on x-axis, incremental effect on y-axis
   - Compare `LogisticSaturation` (concave only) vs `HillSaturation` (can be S-shaped — initially accelerating then saturating)
   - Connect to marketing intuition: the first ad you see is more impactful than the 50th

3. **The interaction:** Show how adstock + saturation together create the full response curve. Adstock determines WHEN the effect happens. Saturation determines HOW MUCH effect each dollar produces. Together, they model the reality of how advertising works.

4. **Prior sensitivity:** Change the priors on adstock and saturation parameters. How much do the posteriors change? When data is limited (like DT Mart), priors dominate — this is important to understand honestly.

- Script: `scripts/modeling/05_adstock_saturation.py`

**What you learn:** The marketing-specific domain knowledge encoded in the MMM model structure. Adstock and saturation aren't arbitrary math — they directly model how advertising works in the real world.

#### Step 6: Model Diagnostics & Posterior Interpretation

Learn to critically evaluate a Bayesian model (this replaces "test set accuracy" from supervised ML):

1. **Trace plots:** Visualize MCMC chains. Good: "fuzzy caterpillars" (well-mixed). Bad: trending, stuck, or divergent chains.
2. **R-hat and ESS:** Compute for all parameters. R-hat > 1.01 means chains haven't converged — don't trust the results.
3. **Divergences:** What they mean, why they happen, how to fix them (reparameterization, increase target_accept).
4. **Posterior predictive checks:** Generate synthetic sales from the posterior. Overlay on real data. Does the model capture the seasonal patterns? The variance? The spikes?
5. **Channel contribution decomposition:** Break down total sales into: baseline + channel 1 + channel 2 + seasonality + controls. Visualize as stacked area chart.
6. **Credible intervals on contributions:** "Channel 1 contributed 22-31% of sales (95% HDI)." This uncertainty is a FEATURE, not a bug — it tells you how confident to be.
7. **Compare to ground truth:** Since we're on synthetic data, directly verify the contribution estimates against the known truth.

- Script: `scripts/modeling/06_diagnostics.py`

**Primer — Credible Intervals vs Confidence Intervals:**

```
Frequentist 95% CI:  "If we repeated this experiment 100 times,
                      ~95 of the computed intervals would contain the true value."
                      Says NOTHING about THIS particular interval.

Bayesian 95% HDI:    "Given our data, there is a 95% probability
                      the parameter lies in this range."
                      This is what everyone THINKS confidence intervals mean.
```

For marketing decisions, the Bayesian interpretation is what you actually want. "There's a 95% probability TV ROAS is between 2.1x and 3.8x" is directly actionable. A frequentist CI doesn't technically let you say this.

**What you learn:** How to evaluate Bayesian models (not accuracy — convergence, predictive checks, uncertainty quantification), how to interpret and present channel contributions with honest uncertainty.

### Phase 3: Real-World MMM (Steps 7-9)

#### Step 7: Data Preparation — Building the MMM-Ready Dataset

Transform DT Mart's raw transaction data into a weekly time series suitable for MMM:

1. **Aggregate transactions to weekly:** Sum GMV and units per week from `firstfile.csv`
2. **Join media spend:** Map monthly `MediaInvestment.csv` to weekly observations
   - Discuss the limitation: monthly spend distributed evenly across weeks is a strong assumption
   - In practice, you'd get weekly or daily spend reports from ad platforms
3. **Create control variables:**
   - Special sale binary indicators from `SpecialSale.csv` (Diwali, Eid, Republic Day, etc.)
   - Monthly NPS as a control (customer satisfaction proxy)
   - Trend variable
4. **Channel selection:** With only ~52 observations, we can't fit 9 channels reliably
   - Combine correlated/small channels (e.g., Affiliates + Online Marketing → "Performance")
   - Aim for 3-4 channel groups maximum
   - This is a real-world decision every MMM practitioner faces
5. **Final dataset:** ~52 weekly rows with: date, total_gmv, channel_1_spend, ..., channel_4_spend, is_diwali, is_sale, trend

- Script: `scripts/preprocessing/07_data_prep.py`

**What you learn:** The data engineering side of MMM — aggregation, joining disparate sources, channel grouping decisions. Also the hard truth: **most real-world MMM data isn't ready-made.** The practitioner assembles it from transactions, ad platform exports, and marketing calendars.

#### Step 8: Full MMM on DT Mart — Priors, Fitting, Honest Assessment

Apply the full Bayesian MMM workflow to real data:

1. **Set priors with domain reasoning:**
   - TV adstock: `Beta(3, 3)` — moderate carry-over, "TV effects linger"
   - Digital adstock: `Beta(1, 3)` — fast decay, "digital is more immediate"
   - Channel coefficients: `HalfNormal` — positive only (spending shouldn't hurt sales)
   - Discuss: with only 52 observations, priors will HEAVILY influence the posterior. This is honest and important.

2. **Prior predictive check:** Generate fake sales from priors alone. Do they look plausible?

3. **Fit the model:** MCMC sampling with careful convergence monitoring

4. **Diagnose HONESTLY:**
   - Wide posteriors? That means the data can't distinguish channels well (expected with 52 obs)
   - Overlapping contribution intervals? We can't confidently rank channels (this is CORRECT — the data doesn't support strong claims)
   - Good posterior predictive fit? The model at least captures the overall pattern

5. **Compare to frequentist baseline:** Show the Bayesian model's honest uncertainty vs OLS's false precision

6. **Sensitivity analysis:** How much do results change with different priors? If a lot → results are prior-driven, not data-driven. Be transparent about this.

- Script: `scripts/modeling/08_real_mmm.py`

**Primer — The Prior Dominance Problem:**

With 52 observations and 15+ parameters, the posterior will be strongly influenced by the priors. This isn't a failure — it's the model correctly telling you "I don't have enough data to be confident." In production MMM:
- You'd have 2-3+ years of weekly data (104-156+ observations)
- You'd calibrate priors with geo experiment results ("we ran a lift test and TV ROAS was ~3x")
- PyMC-Marketing has `mmm.add_lift_test_measurements()` for this

The DT Mart exercise teaches you what "not enough data" looks like and why longer time series and lift test calibration matter.

**What you learn:** How to apply Bayesian MMM to real data with honest assessment of limitations. Prior sensitivity analysis. The critical importance of data quality and quantity in causal estimation.

#### Step 9: Channel Attribution & Budget Optimization

Extract business value from the MMM — this is the payoff:

1. **Channel contribution waterfall:** Decompose total sales into: baseline + TV + Digital + SEM + ... + seasonality. Visualize as waterfall chart.
2. **ROAS by channel:** For each channel, compute return on ad spend with credible intervals: "TV ROAS = 2.1-3.8x (95% HDI)"
3. **Response curves:** Plot the saturation curve per channel — incremental sales per dollar at current spend level vs. double the spend
4. **Budget optimization:** Given a fixed budget, what's the optimal allocation across channels?
   - Use PyMC-Marketing's `BudgetOptimizer`
   - Compare current allocation vs. optimal allocation
   - Show the expected lift from reallocation with uncertainty bands
5. **Scenario analysis:** "What happens if we cut TV by 50%?" "What if we double SEM?" Generate counterfactual predictions with credible intervals.
6. **Honest caveats:** Document what we're confident about vs. what's driven by priors. In a real engagement, what additional data would we need?

- Script: `scripts/modeling/09_budget_optimization.py`

**What you learn:** How to translate Bayesian MMM outputs into marketing decisions. Budget optimization under uncertainty. How to present results to stakeholders with appropriate confidence levels.

### Phase 4: Wrap-up (Step 10)

#### Step 10: Retrospective

Reflect on the full journey from frequentist failure to Bayesian MMM:

- How did your understanding of Bayesian inference evolve?
- What surprised you about the Bayesian workflow vs. ML?
- What were the biggest limitations of the DT Mart data, and how would you fix them in practice?
- How does MMM complement your Criteo uplift project? Your geo experiment work?
- If you had to explain MMM to a marketing VP in 5 minutes, what would you say?
- What would you do differently next time?
- How would you apply this at work — what data would you need, what would the first model look like?

- File: `RETROSPECTIVE.md`

---

## Validation Protocol

MMM validation is fundamentally different from supervised ML. There is no test set for causal decomposition — you can't observe the true contribution of each channel. Instead:

- **Parameter recovery (synthetic data):** Does the model recover known ground truth parameters?
- **Posterior predictive checks:** Does the model generate realistic-looking sales data?
- **Convergence diagnostics:** R-hat < 1.01, no divergences, sufficient ESS for all parameters
- **Prior sensitivity analysis:** How much do results change when priors change? (stability check)
- **Out-of-sample fit:** Hold out the last 8-12 weeks, fit on the rest, check predictive accuracy
- **Cross-validation:** Time-series cross-validation (expanding window) — but primarily for predictive fit, NOT for causal attribution quality
- **Lift test calibration (production):** Compare model estimates to known experimental results

## Libraries

- `pymc` — Bayesian modeling framework (defines models, runs MCMC/NUTS)
- `pymc-marketing` — Marketing-specific models (MMM class, adstock, saturation, budget optimizer)
- `arviz` — Bayesian visualization and diagnostics (trace plots, forest plots, R-hat, posterior predictive)
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `statsmodels` — OLS baseline, VIF calculation
- `scikit-learn` — Preprocessing (MaxAbsScaler, used internally by PyMC-Marketing)

## Principles (carried forward)

1. **Every change gets validated** — posterior predictive checks and convergence diagnostics, not just accuracy
2. **Understand before optimizing** — the point is learning Bayesian inference, not getting the best MAPE
3. **Explain deeply** — every step includes the *why*, not just the *what*
4. **Honest about limitations** — if the data can't support a conclusion, say so. Wide posteriors are informative.
5. **Connect to real work** — relate every concept back to geo experiments and marketing decisions
6. **Clean code, clear output** — numbered scripts, Tee logging

## File Structure

```
competitions/marketing-mix/
├── data/                         # Raw data (gitignored)
├── docs/
│   └── PLAN.md                   # This file
├── scripts/
│   ├── preprocessing/
│   │   ├── 01_eda.py
│   │   └── 07_data_prep.py
│   └── modeling/
│       ├── 02_frequentist_baseline.py
│       ├── 03_bayesian_primer.py
│       ├── 04_first_mmm.py
│       ├── 05_adstock_saturation.py
│       ├── 06_diagnostics.py
│       ├── 08_real_mmm.py
│       └── 09_budget_optimization.py
├── results/
│   ├── models/                   # Saved model traces, convergence logs
│   └── analysis/                 # Contribution plots, response curves, optimization results
└── RETROSPECTIVE.md              # Final reflection
```

## Key References

### Textbooks (in order of accessibility)
- [Think Bayes](https://greenteapress.com/wp/think-bayes/) (Allen Downey) — Python-first, uses code instead of math, ideal for programmers
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) (McElreath) — Gold standard for building Bayesian intuition, video lectures on YouTube
- [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/) (Gelman et al.) — Comprehensive reference, not a starting point

### Blog Posts & Tutorials
- [Jake VanderPlas: Frequentism and Bayesianism](http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/) — Best intro for ML practitioners
- [Thomas Wiecki: MCMC Sampling for Dummies](https://twiecki.io/blog/2015/11/10/mcmc-sampling/) — Best intuitive MCMC explanation
- [Juan Orduz: Media Effect Estimation with PyMC](https://juanitorduz.github.io/pymc_mmm/) — Best hands-on MMM tutorial
- [Recast: Intro to Bayesian Methods for MMM](https://getrecast.com/bayesian-methods-for-mmm/) — Excellent conceptual overview
- [PyMC Labs: MMM Complete Guide](https://www.pymc-labs.com/blog-posts/marketing-mix-modeling-a-complete-guide) — Full mathematical framework + case studies

### Documentation
- [PyMC-Marketing docs](https://www.pymc-marketing.io/) — Official library documentation
- [PyMC-Marketing MMM example notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html) — Full workflow tutorial
- [ArviZ docs](https://python.arviz.org/) — Bayesian visualization reference
- [PyMC docs](https://www.pymc.io/) — Core Bayesian framework

### Papers
- [Jin et al. (2017): Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects](https://research.google/pubs/bayesian-methods-for-media-mix-modeling-with-carryover-and-shape-effects/) — The foundational paper PyMC-Marketing implements
- [Kunzel et al. (2017): Meta-learners for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/1706.03461) — Connection to your Criteo uplift work

### Kaggle Dataset
- [DT Mart Market Mix Modeling](https://www.kaggle.com/datasets/datatattle/dt-mart-market-mix-modeling) — Real-world transaction + media spend data
