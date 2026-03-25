# Marketing Mix Modeling — Retrospective

**Date:** 2026-03-24
**Duration:** Single session, Steps 1-9
**Dataset:** DT Mart Market Mix Modeling (Kaggle) + PyMC-Marketing synthetic data

---

## The Journey

```
Phase 1: Foundations
  Step 1 — EDA: Discovered 47 usable weeks, monthly spend granularity,
                Aug 2015 data hole, massive multicollinearity (r=0.99)
  Step 2 — OLS failure: 2/7 negative coefficients, 7/7 VIF>10,
                0/7 statistically significant. The controlled demolition.
  Step 3 — Bayesian primer: Coin flips, regression, priors/posteriors,
                MCMC mechanics, predictive checks

Phase 2: Synthetic MMM (known ground truth)
  Step 4 — First MMM: Recovered adstock α values (0.40→0.48, 0.20→0.15)
                Channel shares within 2% of truth. 0 divergences.
  Step 5 — Adstock & saturation: Visualized the full pipeline.
                Built channel-specific intuition for work.
  Step 6 — Diagnostics: Contribution decomposition, waterfall chart,
                channel share uncertainty plot. The business deliverable.

Phase 3: Real-world MMM (DT Mart)
  Step 7 — Data prep: 1.58M transactions → 47 weekly rows,
                9 channels → 4 groups, 2.8 obs/param (tight)
  Step 8 — Real MMM: Performance dominates (62%), 27 divergences,
                wide posteriors, prior sensitivity PASSED (<1.5% change)
  Step 9 — Budget optimization: ROAS, response curves, reallocation.
                Saturation artifact from monthly spend data.
```

---

## What I Learned

### Bayesian inference
- Not optimization — MCMC explores a landscape rather than finding a minimum
- The posterior is a fixed mathematical object (defined by prior + data). MCMC discovers it.
- Priors are "fake observations" — Beta(5,5) = 10 fake coin flips
- With limited data, priors matter. This is a feature, not a bug.
- Prior predictive checks catch bad assumptions BEFORE they corrupt results

### MMM-specific
- Adstock and saturation are feature transformations, not magic — they encode real marketing physics
- Each channel costs 3 parameters (α, λ, β). Budget your degrees of freedom.
- Monthly spend distributed to weekly kills within-month variation — get weekly data
- Multicollinearity means the model can't split credit between correlated channels
- The model LEARNS adstock decay from data — you provide priors, not fixed values
- Channel contribution shares are more trustworthy than raw parameter values (scaling)

### Practical skills
- The full workflow: prior predictive → fit → convergence check → posterior predictive → contributions
- How to read trace plots (fuzzy caterpillars = good)
- How to interpret HDI vs pairwise probabilities (HDI = per-channel range, pairwise = comparison)
- Prior sensitivity analysis: run twice with different priors, check if conclusions change
- When to trust results (converged, low divergences, robust to priors) vs when to caveat

---

## What Surprised Me

1. **OLS was even worse than expected.** Not just imprecise — actively wrong. Negative coefficients on channels with clear positive spend. 7/7 non-significant.

2. **Prior sensitivity was good on DT Mart.** Expected priors to dominate with only 47 weeks. Channel shares changed <1.5% between Beta(1,3) and Beta(3,3) priors. The data was actually informative about relative contributions even if not about absolute values.

3. **Saturation curves were useless with monthly spend.** All channels showed 100% saturated — not because they actually are, but because the model only sees one spend level per month. This was the clearest demonstration of why weekly spend data matters.

4. **The baseline absorbed too much.** On synthetic data, baseline was ~66% (correct). On DT Mart, baseline was ~5% (the model attributed 95% to media — unrealistically high). This is a known issue with limited data.

---

## DT Mart Limitations — What Would Fix Them

| Limitation | Impact | Fix |
|---|---|---|
| 47 weeks (need 100+) | Wide posteriors, prior-dependent | Get 2+ years of data |
| Monthly spend | No within-month variation, flat saturation curves | Weekly spend from ad platforms |
| 9 correlated channels | Had to group to 4, still r=0.86 | More diverse channel mix, or run sequentially |
| No experimental data | Can't calibrate priors with ground truth | Geo experiments → `add_lift_test_measurements()` |
| Aug 2015 data hole | Lost 5 weeks (~10% of data) | Better data collection QA |

---

## How This Connects to My Work

**MMM + Geo Experiments + Uplift Modeling = The Full Marketing Measurement Stack**

```
Question                         Method              Project
─────────────────────────────────────────────────────────────
"Where should we spend?"         MMM                 This project
"Did the campaign work?"         Geo experiment      Day job
"Who should we target?"          Uplift modeling      Criteo project
```

**My CTV plan (evolved):**
1. Build MMM on Google/Facebook/TikTok (80-100 weeks, weekly spend)
2. Use MMM as counterfactual baseline during CTV flight
3. Residual (actual - predicted) = CTV-attributable estimate
4. Geo experiment provides causal confirmation
5. Once CTV has more flight data, add to MMM directly with geo-calibrated priors

**Advantages over DT Mart:**
- 80-100 weeks vs 47 (2x more data)
- Weekly spend from ad platforms vs monthly distribution
- 3 channels vs 4 groups (fewer parameters)
- Planned geo experiment for CTV prior calibration

---

## If I Had to Explain MMM to a Marketing VP in 5 Minutes

"We built a model that looks at your weekly sales alongside your weekly advertising spend across all channels. It accounts for two things regular analysis misses: (1) ads have a delayed effect — a TV spot today might drive a purchase next week, and (2) there are diminishing returns — the 50th ad impression is less valuable than the 1st.

The model tells us how much of your sales each channel is responsible for, with honest uncertainty ranges. For example, 'Performance marketing drives roughly 50-80% of your media-attributable sales, and we're 94% confident it outperforms SEM.'

The key insight is that you're spending 47% of your budget on Sponsorship, but it's only driving about 12% of the effect. Performance marketing gets 33% of the budget but drives 62% of the effect. Shifting budget from Sponsorship to Performance would likely improve total ROI.

Caveat: these are directional recommendations. Before making major budget changes, we'd want to validate with a controlled geo experiment."

---

## What I'd Do Differently Next Time

1. **Start with synthetic data** — this was the right call. Building on known ground truth first built confidence in the method before confronting messy real data.
2. **Demand weekly spend data** — the single biggest limitation was monthly granularity. At work, insist on weekly or daily from ad platforms.
3. **Fewer channels, longer time series** — 3 channels with 100 weeks beats 7 channels with 47 weeks. Simplify the model to match the data.
4. **Geo experiment first, then MMM** — the ideal workflow is: experiment → ground truth → calibrated MMM priors → more reliable decomposition.
