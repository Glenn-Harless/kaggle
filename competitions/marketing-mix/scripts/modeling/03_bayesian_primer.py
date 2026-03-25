"""
Marketing Mix Modeling: Step 3 — Bayesian Primer

Learn to think in distributions using simple toy problems.
NO marketing content, NO MMM — pure Bayesian mechanics.

  1. Coin flip — Beta prior, observe data, posterior update
  2. Bayesian linear regression — PyMC model, trace plots, credible intervals
  3. Prior predictive checks — "do my assumptions make sense?"
  4. Posterior predictive checks — "does my model reproduce the data?"
  5. MCMC mechanics — chains, R-hat, ESS, convergence

Every concept here will be used in Steps 4-9.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import pymc as pm
import arviz as az

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/03_bayesian_primer.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 3 — Bayesian Primer")
print("=" * 60)
print("\nGoal: Build Bayesian intuition on simple problems before")
print("applying it to marketing data.")


# ============================================================
# 1. COIN FLIP — YOUR FIRST BAYESIAN MODEL
# ============================================================
# The coin flip is to Bayesian statistics what "Hello World" is
# to programming. It's simple enough to verify by hand, but
# demonstrates every core concept:
#
# - Prior: what do you believe before seeing any flips?
# - Likelihood: how probable is the observed data given a bias?
# - Posterior: what do you believe after seeing the flips?
#
# The math works out analytically here (Beta-Binomial conjugacy),
# so we can verify that MCMC gives the right answer.
#
# ANALOGY TO MMM:
#   Coin bias (θ)    →  Channel effectiveness (β)
#   Prior on θ       →  Prior on β ("TV probably has positive ROI")
#   Coin flip data   →  Weekly sales data
#   Posterior on θ   →  Posterior on β ("TV ROI is probably 2-4x")

print("\n\n" + "=" * 60)
print("1. COIN FLIP — BAYESIAN UPDATING")
print("=" * 60)

# Simulate a slightly biased coin (true probability = 0.65)
np.random.seed(42)
TRUE_BIAS = 0.65
flips = np.random.binomial(1, TRUE_BIAS, size=100)
n_heads = flips.sum()
n_tails = len(flips) - n_heads

print(f"\n  True coin bias: {TRUE_BIAS}")
print(f"  Flipped {len(flips)} times: {n_heads} heads, {n_tails} tails")
print(f"  Observed frequency: {n_heads/len(flips):.2f}")

# --- 1a. The analytical (exact) solution ---
# With a Beta(a,b) prior and observing h heads and t tails:
#   Posterior = Beta(a + h, b + t)
#
# Beta(1,1) = uniform prior ("I have no idea, any bias is equally likely")
# Beta(5,5) = weak prior ("I think it's probably fair, but I'm not sure")
# Beta(50,50) = strong prior ("I'm very confident it's fair")

priors = {
    "Uniform Beta(1,1)": (1, 1),
    "Weak Beta(5,5)": (5, 5),
    "Strong Beta(50,50)": (50, 50),
}

print(f"\n  Analytical posteriors (exact — no MCMC needed):")
print(f"  {'Prior':25s}  {'Prior Mean':>12s}  {'Posterior Mean':>15s}  {'95% HDI':>20s}")
print(f"  {'-'*75}")
for name, (a, b) in priors.items():
    post_a = a + n_heads
    post_b = b + n_tails
    post_mean = post_a / (post_a + post_b)
    # HDI (Highest Density Interval) for Beta distribution
    hdi_low, hdi_high = stats.beta.ppf([0.025, 0.975], post_a, post_b)
    print(f"  {name:25s}  {a/(a+b):>12.2f}  {post_mean:>15.3f}  [{hdi_low:.3f}, {hdi_high:.3f}]")

print(f"\n  KEY OBSERVATIONS:")
print(f"  - Uniform prior: posterior ≈ observed frequency (data dominates)")
print(f"  - Weak prior: posterior pulled slightly toward 0.5 (prior has mild influence)")
print(f"  - Strong prior: posterior pulled strongly toward 0.5 (prior fights the data)")
print(f"  - With 100 observations, even the strong prior yields to the data somewhat")
print(f"  - With only 10 observations, the strong prior would dominate (try it!)")

# --- 1b. Verify with MCMC ---
# Now let's do the same thing with PyMC to verify MCMC recovers
# the analytical answer. This builds confidence that MCMC works.
print(f"\n  Now verifying with MCMC (PyMC)...")

with pm.Model() as coin_model:
    # Prior: Beta(1,1) = uniform
    theta = pm.Beta("theta", alpha=1, beta=1)
    # Likelihood: Binomial
    obs = pm.Binomial("obs", n=len(flips), p=theta, observed=n_heads)
    # Sample from posterior using NUTS
    coin_trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                           progressbar=False)

coin_summary = az.summary(coin_trace, var_names=["theta"],
                          hdi_prob=0.95, round_to=4)
print(f"\n  MCMC result (should match Uniform prior analytical solution):")
print(f"  {coin_summary.to_string()}")
print(f"\n  Analytical: mean={n_heads/(n_heads+n_tails+2):.4f}")
print(f"  MCMC:       mean={coin_trace.posterior['theta'].values.mean():.4f}")
print(f"  Match? {'Yes' if abs(coin_trace.posterior['theta'].values.mean() - (n_heads+1)/(len(flips)+2)) < 0.01 else 'No'}")

# --- 1c. Visualize prior → posterior updating ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.linspace(0, 1, 200)

for ax, (name, (a, b)) in zip(axes, priors.items()):
    # Prior
    prior_pdf = stats.beta.pdf(x, a, b)
    ax.plot(x, prior_pdf, "b--", linewidth=2, label=f"Prior: {name}")

    # Posterior
    post_a = a + n_heads
    post_b = b + n_tails
    post_pdf = stats.beta.pdf(x, post_a, post_b)
    ax.plot(x, post_pdf, "r-", linewidth=2,
            label=f"Posterior: Beta({post_a},{post_b})")

    # True value
    ax.axvline(TRUE_BIAS, color="green", linewidth=2, linestyle=":",
               label=f"True bias = {TRUE_BIAS}")

    ax.set_xlabel("Coin bias (θ)")
    ax.set_ylabel("Density")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Bayesian Updating: Prior → Posterior\n(100 flips of a coin with true bias = 0.65)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_coin_flip_updating.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 03_coin_flip_updating.png")


# --- 1d. Effect of sample size on posterior ---
# With more data, the posterior narrows and the prior matters less.
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
sample_sizes = [5, 20, 50, 100]

for ax, n in zip(axes, sample_sizes):
    subset = flips[:n]
    h = subset.sum()
    t = n - h

    for name, (a, b) in priors.items():
        post_a = a + h
        post_b = b + t
        post_pdf = stats.beta.pdf(x, post_a, post_b)
        ax.plot(x, post_pdf, linewidth=2, label=name)

    ax.axvline(TRUE_BIAS, color="black", linewidth=2, linestyle=":")
    ax.set_title(f"n = {n} flips", fontsize=12, fontweight="bold")
    ax.set_xlabel("θ")
    if n == 5:
        ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("How Sample Size Overcomes Prior Influence\n"
             "(all three priors converge as n grows)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_sample_size_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 03_sample_size_effect.png")

print(f"\n  THIS IS THE KEY LESSON FOR MMM:")
print(f"  With n=5, your prior heavily influences the result → like our DT Mart (52 weeks)")
print(f"  With n=100, data dominates and the prior barely matters → like 3+ years of data")
print(f"  Our DT Mart data is on the LEFT side of this spectrum. Priors will matter a lot.")


# ============================================================
# 2. BAYESIAN LINEAR REGRESSION — FROM OLS TO POSTERIORS
# ============================================================
# Now let's do something you already understand (linear regression)
# but the Bayesian way. Instead of ONE best-fit line, we get a
# DISTRIBUTION of plausible lines.
#
# OLS answer:      slope = 2.3, intercept = 5.1
# Bayesian answer: slope ~ Normal(2.3, 0.4), intercept ~ Normal(5.1, 1.2)
#
# The Bayesian answer says "the slope is probably around 2.3, but
# it could reasonably be anywhere from 1.5 to 3.1."

print("\n\n" + "=" * 60)
print("2. BAYESIAN LINEAR REGRESSION")
print("=" * 60)

# Generate synthetic data: y = 3x + 10 + noise
np.random.seed(42)
N = 50
x_data = np.random.uniform(0, 10, N)
TRUE_SLOPE = 3.0
TRUE_INTERCEPT = 10.0
TRUE_SIGMA = 5.0
y_data = TRUE_SLOPE * x_data + TRUE_INTERCEPT + np.random.normal(0, TRUE_SIGMA, N)

print(f"\n  Synthetic data: y = {TRUE_SLOPE}x + {TRUE_INTERCEPT} + N(0, {TRUE_SIGMA})")
print(f"  {N} data points")

# --- 2a. OLS baseline ---
from numpy.polynomial.polynomial import polyfit
ols_intercept, ols_slope = polyfit(x_data, y_data, 1)
print(f"\n  OLS result (point estimates):")
print(f"    Slope:     {ols_slope:.3f}  (true: {TRUE_SLOPE})")
print(f"    Intercept: {ols_intercept:.3f}  (true: {TRUE_INTERCEPT})")

# --- 2b. Bayesian regression with PyMC ---
print(f"\n  Fitting Bayesian regression with PyMC...")

with pm.Model() as linear_model:
    # Priors — weakly informative
    # Normal(0, 10) means "the slope is probably between -20 and +20"
    # This is deliberately vague — we're saying "I don't know much"
    slope = pm.Normal("slope", mu=0, sigma=10)
    intercept = pm.Normal("intercept", mu=0, sigma=20)
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Expected value (the regression line)
    mu = slope * x_data + intercept

    # Likelihood — how probable is the observed y given the line?
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

    # Sample from the posterior
    linear_trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                             progressbar=False)

# --- 2c. Results ---
linear_summary = az.summary(linear_trace, var_names=["slope", "intercept", "sigma"],
                            hdi_prob=0.95, round_to=3)
print(f"\n  Bayesian result (posterior distributions):")
print(f"  {linear_summary.to_string()}")

post_slope = linear_trace.posterior["slope"].values.flatten()
post_intercept = linear_trace.posterior["intercept"].values.flatten()

print(f"\n  Comparison:")
print(f"  {'':15s}  {'True':>8s}  {'OLS':>8s}  {'Bayesian Mean':>15s}  {'Bayesian 95% HDI':>20s}")
print(f"  {'-'*70}")
print(f"  {'Slope':15s}  {TRUE_SLOPE:>8.2f}  {ols_slope:>8.2f}  {post_slope.mean():>15.3f}  "
      f"[{np.percentile(post_slope, 2.5):.3f}, {np.percentile(post_slope, 97.5):.3f}]")
print(f"  {'Intercept':15s}  {TRUE_INTERCEPT:>8.2f}  {ols_intercept:>8.2f}  {post_intercept.mean():>15.3f}  "
      f"[{np.percentile(post_intercept, 2.5):.3f}, {np.percentile(post_intercept, 97.5):.3f}]")

print(f"\n  The Bayesian mean ≈ OLS estimate (with weak priors and enough data).")
print(f"  But Bayesian also gives you the SPREAD — how uncertain are we?")
print(f"  OLS just says 'slope = {ols_slope:.2f}'. Bayesian says 'slope is")
print(f"  probably between {np.percentile(post_slope, 2.5):.1f} and {np.percentile(post_slope, 97.5):.1f}'.")

# --- 2d. Visualize: distribution of plausible lines ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Data + OLS line vs Bayesian lines
ax = axes[0]
ax.scatter(x_data, y_data, alpha=0.5, color="gray", edgecolors="black", linewidth=0.5)
x_plot = np.linspace(0, 10, 100)

# Draw 100 posterior lines (showing uncertainty as a bundle of lines)
for i in range(100):
    idx = np.random.randint(len(post_slope))
    ax.plot(x_plot, post_slope[idx] * x_plot + post_intercept[idx],
            color="steelblue", alpha=0.05, linewidth=1)

# OLS line
ax.plot(x_plot, ols_slope * x_plot + ols_intercept,
        "r-", linewidth=2, label="OLS (single line)")
# True line
ax.plot(x_plot, TRUE_SLOPE * x_plot + TRUE_INTERCEPT,
        "g--", linewidth=2, label="True relationship")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("OLS (red) vs Bayesian (blue bundle)\nvs True (green dashed)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Posterior distribution of slope
ax = axes[1]
ax.hist(post_slope, bins=50, density=True, alpha=0.7, color="steelblue",
        edgecolor="white", linewidth=0.5)
ax.axvline(TRUE_SLOPE, color="green", linewidth=2, linestyle="--",
           label=f"True = {TRUE_SLOPE}")
ax.axvline(ols_slope, color="red", linewidth=2, linestyle="-",
           label=f"OLS = {ols_slope:.2f}")
ax.axvline(post_slope.mean(), color="steelblue", linewidth=2, linestyle="-",
           label=f"Bayes mean = {post_slope.mean():.2f}")
ax.set_xlabel("Slope value")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Slope",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Posterior distribution of intercept
ax = axes[2]
ax.hist(post_intercept, bins=50, density=True, alpha=0.7, color="coral",
        edgecolor="white", linewidth=0.5)
ax.axvline(TRUE_INTERCEPT, color="green", linewidth=2, linestyle="--",
           label=f"True = {TRUE_INTERCEPT}")
ax.axvline(ols_intercept, color="red", linewidth=2, linestyle="-",
           label=f"OLS = {ols_intercept:.2f}")
ax.set_xlabel("Intercept value")
ax.set_ylabel("Density")
ax.set_title("Posterior Distribution of Intercept",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("Bayesian Linear Regression: Distributions, Not Point Estimates",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_bayesian_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 03_bayesian_regression.png")


# ============================================================
# 3. PRIOR PREDICTIVE CHECKS — "DO MY PRIORS MAKE SENSE?"
# ============================================================
# Before fitting the model to data, ask: "If my priors are all
# I know, what kind of data would the model predict?"
#
# This catches bad priors BEFORE they corrupt your posterior.
#
# Example of a bad prior: slope ~ Normal(0, 1000)
#   This says "the slope could be 3000" — which would predict
#   y values of 30,000+ for x=10. If your data is in the range
#   0-50, those priors are clearly wrong.
#
# Example of a good prior: slope ~ Normal(0, 10)
#   This says "the slope is probably between -20 and +20" — which
#   produces y values in a plausible range.
#
# ANALOGY TO MMM:
#   Bad prior:  "TV's adstock decay could be 100" (nonsensical)
#   Good prior: "TV's adstock decay is probably 0.2-0.8" (plausible)

print("\n\n" + "=" * 60)
print("3. PRIOR PREDICTIVE CHECKS")
print("=" * 60)

print(f"\n  Generating predictions from priors alone (no data used)...")

# Compare good vs bad priors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (prior_name, slope_sigma) in zip(axes, [("Good: Normal(0,10)", 10),
                                                  ("Bad: Normal(0,1000)", 1000)]):
    with pm.Model() as prior_check_model:
        slope_pc = pm.Normal("slope", mu=0, sigma=slope_sigma)
        intercept_pc = pm.Normal("intercept", mu=0, sigma=20)
        sigma_pc = pm.HalfNormal("sigma", sigma=10)
        mu_pc = slope_pc * x_data + intercept_pc
        y_pc = pm.Normal("y_pred", mu=mu_pc, sigma=sigma_pc)

        prior_samples = pm.sample_prior_predictive(samples=200, random_seed=42)

    # Plot prior predictive samples
    y_prior = prior_samples.prior["y_pred"].values[0]  # (samples, N)
    for i in range(50):
        ax.plot(x_data, y_prior[i], alpha=0.05, color="steelblue")
    ax.scatter(x_data, y_data, color="red", s=20, zorder=5, label="Actual data")
    ax.set_xlabel("x")
    ax.set_ylabel("y (prior predictive)")
    ax.set_title(f"Prior: slope ~ {prior_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    y_range = np.abs(y_prior).max()
    print(f"\n  {prior_name}:")
    print(f"    Prior predictive y range: [{y_prior.min():.0f}, {y_prior.max():.0f}]")
    print(f"    Actual data y range:      [{y_data.min():.0f}, {y_data.max():.0f}]")
    if y_range > 10000:
        print(f"    PROBLEM: Prior produces values up to ±{y_range:.0f}")
        print(f"    The model thinks y could be in the thousands — clearly wrong.")
    else:
        print(f"    Good: Prior predictive range is in the same ballpark as data.")

plt.suptitle("Prior Predictive Check: Do My Priors Produce Plausible Predictions?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_prior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 03_prior_predictive.png")

print(f"\n  TAKEAWAY: Always check your priors BEFORE fitting.")
print(f"  In MMM, this means: 'Do my priors on adstock decay and channel")
print(f"  effectiveness produce weekly sales in a plausible range?'")
print(f"  If the prior predictive shows sales of $10 billion/week, fix your priors.")


# ============================================================
# 4. POSTERIOR PREDICTIVE CHECKS — "DOES MY MODEL WORK?"
# ============================================================
# After fitting, ask: "If I generate new data from my fitted model,
# does it look like the real data?"
#
# This is the Bayesian equivalent of checking residuals or test
# set performance. If the model captures the right patterns (mean,
# variance, shape), it's a reasonable description of the data.
#
# In MMM: "Does the fitted model reproduce the seasonal pattern
# in weekly sales? The variance? The spikes?"

print("\n\n" + "=" * 60)
print("4. POSTERIOR PREDICTIVE CHECKS")
print("=" * 60)

print(f"\n  Generating predictions from the fitted model...")

with linear_model:
    ppc = pm.sample_posterior_predictive(linear_trace, random_seed=42,
                                         progressbar=False)

y_ppc = ppc.posterior_predictive["y_obs"].values  # (chains, draws, N)
# Flatten chains and draws
y_ppc_flat = y_ppc.reshape(-1, N)  # (chains*draws, N)

print(f"\n  Posterior predictive shape: {y_ppc_flat.shape}")
print(f"  (each row is one 'simulated dataset' from the fitted model)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Overlay simulated datasets on real data
ax = axes[0]
for i in range(100):
    sorted_idx = np.argsort(x_data)
    ax.plot(x_data[sorted_idx], y_ppc_flat[i][sorted_idx],
            alpha=0.03, color="steelblue")
ax.scatter(x_data, y_data, color="red", s=20, zorder=5,
           label="Actual data", edgecolors="black", linewidth=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Posterior Predictive: Simulated Data vs Actual",
             fontsize=11, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Distribution of y — simulated vs actual
ax = axes[1]
# Plot histogram of some simulated datasets
for i in range(200):
    ax.hist(y_ppc_flat[i], bins=20, alpha=0.01, color="steelblue", density=True)
ax.hist(y_data, bins=20, alpha=0.7, color="red", density=True,
        edgecolor="black", linewidth=0.5, label="Actual data")
ax.set_xlabel("y values")
ax.set_ylabel("Density")
ax.set_title("Does Simulated Data Match Real Data Distribution?",
             fontsize=11, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Posterior Predictive Check: Model Reproduces the Data Pattern",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 03_posterior_predictive.png")

# Quantitative check
y_pred_mean = y_ppc_flat.mean(axis=0)
residuals = y_data - y_pred_mean
print(f"\n  Quantitative posterior predictive check:")
print(f"    Mean of actual data:    {y_data.mean():.2f}")
print(f"    Mean of simulated data: {y_ppc_flat.mean():.2f}")
print(f"    Std of actual data:     {y_data.std():.2f}")
print(f"    Std of simulated data:  {y_ppc_flat.std(axis=1).mean():.2f}")
print(f"    Residual MAE:           {np.abs(residuals).mean():.2f}")
print(f"\n  If the simulated data has similar mean, std, and shape as the")
print(f"  actual data, the model is a reasonable description of reality.")


# ============================================================
# 5. MCMC MECHANICS — UNDERSTANDING THE SAMPLER
# ============================================================
# So far we've used pm.sample() as a black box. Let's open it up
# and understand what the sampler is actually doing.
#
# Key concepts:
#   CHAINS: Independent MCMC runs (default 4). Like 4 hikers
#           exploring the same mountain range independently.
#   WARMUP: Initial samples discarded. The hiker is still finding
#           the interesting region — early steps are unreliable.
#   R-HAT:  Did all chains find the same answer? R-hat < 1.01 = good.
#           If chains disagree, something is wrong.
#   ESS:    Effective Sample Size. MCMC samples are autocorrelated
#           (each step depends on the previous), so 2000 draws might
#           only give you ~1000 "effective" independent samples.

print("\n\n" + "=" * 60)
print("5. MCMC MECHANICS — CHAINS, R-HAT, CONVERGENCE")
print("=" * 60)

# --- 5a. Trace plots ---
# A trace plot shows the sampler's path through parameter space.
# GOOD: looks like "fuzzy caterpillars" — random, stationary, well-mixed
# BAD: trending, stuck in one region, or chains that disagree

print(f"\n  Generating trace plots for the linear regression model...")

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
params = ["slope", "intercept", "sigma"]
true_vals = [TRUE_SLOPE, TRUE_INTERCEPT, TRUE_SIGMA]

for row, (param, true_val) in enumerate(zip(params, true_vals)):
    samples = linear_trace.posterior[param].values  # (chains, draws)

    # Left: trace plot (sampler path)
    ax = axes[row, 0]
    for chain in range(samples.shape[0]):
        ax.plot(samples[chain], alpha=0.5, linewidth=0.3)
    ax.axhline(true_val, color="red", linewidth=2, linestyle="--",
               label=f"True = {true_val}")
    ax.set_ylabel(param)
    ax.set_xlabel("Sample index")
    ax.set_title(f"Trace: {param}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: posterior density
    ax = axes[row, 1]
    for chain in range(samples.shape[0]):
        ax.hist(samples[chain], bins=40, alpha=0.3, density=True,
                label=f"Chain {chain+1}")
    ax.axvline(true_val, color="red", linewidth=2, linestyle="--",
               label=f"True = {true_val}")
    ax.set_xlabel(param)
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior: {param}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Trace Plots & Posterior Densities\n"
             "(4 chains should overlap — 'fuzzy caterpillars')",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_trace_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 03_trace_plots.png")

# --- 5b. Convergence diagnostics ---
print(f"\n  Convergence diagnostics:")
full_summary = az.summary(linear_trace, var_names=params, hdi_prob=0.95)
print(f"\n{full_summary.to_string()}")

print(f"\n  Reading the diagnostics:")
for param in params:
    rhat = full_summary.loc[param, "r_hat"]
    ess_bulk = full_summary.loc[param, "ess_bulk"]
    ess_tail = full_summary.loc[param, "ess_tail"]
    rhat_ok = "GOOD" if rhat < 1.01 else "BAD — chains disagree!"
    ess_ok = "GOOD" if ess_bulk > 400 else "LOW — need more samples"
    print(f"    {param:12s}  R-hat={rhat:.3f} ({rhat_ok})  "
          f"ESS_bulk={ess_bulk:.0f} ({ess_ok})")

print(f"""
  WHAT THESE MEAN:
  ─────────────────────────────────────────────────────────
  R-hat < 1.01    → All 4 chains found the same answer (converged)
  R-hat > 1.01    → Chains disagree — don't trust the results!
                    Fix: run longer, reparameterize, or simplify model

  ESS > 400       → Enough effective independent samples for reliable estimates
  ESS < 100       → Estimates are noisy — run more samples or fix the model

  No divergences  → The sampler navigated the posterior smoothly
  Divergences > 0 → The sampler hit pathological geometry — model may need fixing
                    Fix: increase target_accept (e.g., 0.95), reparameterize
""")


# --- 5c. Effect of number of samples ---
print(f"  Effect of sample count on posterior precision:")
print(f"  (Using the coin flip model with fewer samples)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sample_counts = [100, 500, 2000]

for ax, n_samples in zip(axes, sample_counts):
    with pm.Model():
        theta_s = pm.Beta("theta", alpha=1, beta=1)
        obs_s = pm.Binomial("obs", n=len(flips), p=theta_s, observed=n_heads)
        trace_s = pm.sample(n_samples, tune=500, chains=4, random_seed=42,
                            progressbar=False)

    theta_samples = trace_s.posterior["theta"].values.flatten()
    ax.hist(theta_samples, bins=40, density=True, alpha=0.7,
            color="steelblue", edgecolor="white")
    ax.axvline(TRUE_BIAS, color="green", linewidth=2, linestyle="--")
    ax.set_title(f"{n_samples} samples/chain\n"
                 f"ESS={az.ess(trace_s, var_names=['theta'])['theta'].values:.0f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("θ")
    ax.grid(True, alpha=0.3)

plt.suptitle("More Samples → Smoother Posterior Estimate",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/03_sample_count_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 03_sample_count_effect.png")


# ============================================================
# 6. SUMMARY — WHAT YOU NOW KNOW
# ============================================================

print("\n\n" + "=" * 60)
print("6. SUMMARY — BAYESIAN BUILDING BLOCKS")
print("=" * 60)

print(f"""
  CONCEPT                  WHAT IT IS                           MMM ANALOGY
  ─────────────────────────────────────────────────────────────────────────────
  Prior                    Your belief before data              "TV ROI is probably 1-5x"
  Posterior                Your updated belief after data       "TV ROI is 2.1-3.8x (95%)"
  Prior predictive check   Do priors produce plausible data?    "Do my priors generate realistic weekly sales?"
  Posterior predictive     Does fitted model reproduce data?    "Does the model capture seasonal patterns?"
  Trace plot               Sampler's path through param space   Check for "fuzzy caterpillars"
  R-hat                    Did chains converge? (< 1.01)        All 4 independent runs agree
  ESS                      Effective independent samples        Enough to trust the estimates
  Divergences              Sampler hit pathological geometry    Model may need fixing

  KEY TAKEAWAYS:
  1. Bayesian = distributions, not point estimates
  2. Priors matter more with less data (our MMM situation!)
  3. Always check priors BEFORE fitting (prior predictive)
  4. Always check the fit AFTER fitting (posterior predictive)
  5. MCMC convergence diagnostics replace train/test accuracy
  6. With weak priors and enough data, Bayesian ≈ OLS + uncertainty

  You now have the vocabulary and intuition for Steps 4-9.
  Next: apply these concepts to a real MMM on synthetic data.
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
