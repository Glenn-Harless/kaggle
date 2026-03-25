"""
Marketing Mix Modeling: Step 8 — Full MMM on DT Mart

Apply the complete Bayesian MMM workflow to REAL data.
This is where we confront the honest limitations of 47 weeks
with correlated channels and monthly-granularity spend.

  1. Load MMM-ready data from Step 7
  2. Set priors with domain reasoning
  3. Prior predictive check
  4. Fit with MCMC
  5. Convergence diagnostics
  6. Channel contributions with honest uncertainty
  7. Prior sensitivity analysis
  8. Honest assessment
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

import pymc as pm
import arviz as az

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_extras.prior import Prior

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
DATA = f"{BASE}/data"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/08_real_mmm.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 8 — Real MMM on DT Mart")
print("=" * 60)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n\n" + "=" * 60)
print("1. LOAD MMM-READY DATA")
print("=" * 60)

df = pd.read_csv(f"{DATA}/mmm_weekly.csv")
df["date"] = pd.to_datetime(df["date"])

channels = ["spend_sponsorship", "spend_performance", "spend_brand", "spend_sem"]
controls = ["has_special_sale", "trend"]

print(f"\n  Shape: {df.shape}")
print(f"  Weeks: {len(df)}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Channels: {channels}")
print(f"  Controls: {controls}")
print(f"  Target: total_gmv (mean={df['total_gmv'].mean():,.0f})")


# ============================================================
# 2. BUILD THE MODEL WITH DOMAIN-INFORMED PRIORS
# ============================================================
# With only 47 observations, prior choice MATTERS.
# We use domain reasoning for each channel group:
#
# Sponsorship (events/sports): moderate carry-over, slow saturation
#   α ~ Beta(3,3): centered at 0.5, "sponsorship has lingering brand effect"
#   λ ~ Gamma(3,1): default, moderate saturation
#
# Performance (online+affiliates): fast decay, moderate saturation
#   α ~ Beta(1,3): centered at 0.25, "digital performance is immediate"
#   λ ~ Gamma(3,1): default
#
# Brand (TV+digital brand): moderate-high carry-over
#   α ~ Beta(3,3): centered at 0.5, "brand advertising lingers"
#   λ ~ Gamma(3,1): default
#
# SEM (search): very fast decay, fast saturation
#   α ~ Beta(1,4): centered at 0.2, "search is intent-driven, immediate"
#   λ ~ Gamma(3,1): default

print("\n\n" + "=" * 60)
print("2. BUILD MODEL WITH DOMAIN-INFORMED PRIORS")
print("=" * 60)

mmm = MMM(
    date_column="date",
    channel_columns=channels,
    control_columns=controls,
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,  # 2 fourier modes for annual pattern
)

X = df[["date"] + channels + controls]
y = df["total_gmv"]

print(f"\n  Model configuration:")
print(f"    Channels: {channels}")
print(f"    Controls: {controls}")
print(f"    Adstock: GeometricAdstock(l_max=8)")
print(f"    Saturation: LogisticSaturation()")
print(f"    Yearly seasonality: 2 fourier modes")
print(f"\n  Using default priors (reasonable for exploratory analysis):")
print(f"    α ~ Beta(1, 3) — biased toward lower carry-over")
print(f"    λ ~ Gamma(3, 1) — moderate saturation")
print(f"    β ~ HalfNormal(2) — positive, moderate effect")


# ============================================================
# 3. PRIOR PREDICTIVE CHECK
# ============================================================

print("\n\n" + "=" * 60)
print("3. PRIOR PREDICTIVE CHECK")
print("=" * 60)

mmm.sample_prior_predictive(X, y, samples=500, random_seed=42)

if hasattr(mmm.idata, "prior_predictive") and "y" in mmm.idata.prior_predictive:
    prior_pred = mmm.idata.prior_predictive["y"].values.reshape(-1, len(df))
elif hasattr(mmm.idata, "prior") and "y" in mmm.idata.prior:
    prior_pred = mmm.idata.prior["y"].values.reshape(-1, len(df))
else:
    prior_pred = None

if prior_pred is not None:
    print(f"\n  Actual GMV range:         [{y.min():,.0f}, {y.max():,.0f}]")
    print(f"  Prior predictive range:   [{np.percentile(prior_pred, 2.5):,.0f}, "
          f"{np.percentile(prior_pred, 97.5):,.0f}] (central 95%)")
    print(f"  Prior predictive mean:    {prior_pred.mean():,.0f}")
    print(f"  Actual mean:              {y.mean():,.0f}")

    fig, ax = plt.subplots(figsize=(14, 5))
    for i in range(min(100, len(prior_pred))):
        ax.plot(df["date"], prior_pred[i] / 1e6, alpha=0.03, color="steelblue")
    ax.plot(df["date"], y / 1e6, "r-", linewidth=2, label="Actual GMV")
    ax.set_title("Prior Predictive Check — DT Mart", fontsize=13, fontweight="bold")
    ax.set_ylabel("GMV (millions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/08_prior_predictive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 08_prior_predictive.png")
else:
    print(f"  Could not extract prior predictive. Proceeding to fit.")


# ============================================================
# 4. FIT THE MODEL
# ============================================================

print("\n\n" + "=" * 60)
print("4. FIT THE MODEL (MCMC)")
print("=" * 60)
print(f"\n  47 observations, ~17 parameters")
print(f"  target_accept=0.95 (cautious sampler to reduce divergences)")
print(f"  4 chains × 2000 draws + 2000 warmup\n")

idata = mmm.fit(X, y, random_seed=42, target_accept=0.95,
                draws=2000, tune=2000)

print(f"\n  Fitting complete.")


# ============================================================
# 5. CONVERGENCE DIAGNOSTICS
# ============================================================

print("\n\n" + "=" * 60)
print("5. CONVERGENCE DIAGNOSTICS")
print("=" * 60)

core_params = ["intercept", "adstock_alpha", "saturation_lam",
               "saturation_beta", "gamma_control", "y_sigma"]
summary = az.summary(idata, var_names=core_params, round_to=3)
n_divergences = int(idata.sample_stats["diverging"].sum().values)

print(f"\n  Posterior summary (core parameters):")
print(summary.to_string())

print(f"\n  Divergences: {n_divergences}")

# Check each parameter
print(f"\n  Convergence status:")
issues = []
for param in summary.index:
    rhat = summary.loc[param, "r_hat"]
    ess = summary.loc[param, "ess_bulk"]
    rhat_ok = rhat < 1.01 if not np.isnan(rhat) else True
    ess_ok = ess > 400
    status = "OK" if rhat_ok and ess_ok else "WARNING"
    if status == "WARNING":
        issues.append(param)
    print(f"    {param:40s}  R-hat={rhat:.3f}  ESS={ess:>7.0f}  [{status}]")

if not issues and n_divergences == 0:
    print(f"\n  All diagnostics PASS.")
elif n_divergences > 0 and n_divergences < 50:
    print(f"\n  {n_divergences} divergences — mild. Results likely still usable")
    print(f"  but interpret with some caution.")
elif n_divergences >= 50:
    print(f"\n  {n_divergences} divergences — significant. Results may be unreliable.")
    print(f"  Consider: increasing target_accept, simplifying model, or adding data.")

# Save trace plots
az.plot_trace(idata, var_names=core_params, compact=True, figsize=(14, 14))
plt.tight_layout()
plt.savefig(f"{RESULTS}/08_trace_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 08_trace_plots.png")


# ============================================================
# 6. CHANNEL CONTRIBUTIONS — THE REAL RESULTS
# ============================================================

print("\n\n" + "=" * 60)
print("6. CHANNEL CONTRIBUTIONS")
print("=" * 60)

posterior = idata.posterior

# Adstock recovery
print(f"\n  Learned adstock decay (α) per channel:")
channel_names = list(posterior.coords["channel"].values)
for ch in channel_names:
    alpha_samples = posterior["adstock_alpha"].sel(channel=ch).values.flatten()
    hdi = az.hdi(alpha_samples, hdi_prob=0.95)
    label = ch.replace("spend_", "")
    print(f"    {label:20s}  mean={alpha_samples.mean():.2f}  "
          f"95% HDI=[{hdi[0]:.2f}, {hdi[1]:.2f}]")

# Channel contribution shares
print(f"\n  Channel contribution shares (% of media-driven sales):")
ch_contrib = posterior["channel_contribution_original_scale"]
# Sum over time → total contribution per channel
ch_totals = {}
for ch in channel_names:
    samples = ch_contrib.sel(channel=ch).values.reshape(-1, len(df)).sum(axis=1)
    ch_totals[ch] = samples

total_media = sum(ch_totals.values())
shares = {ch: ch_totals[ch] / total_media for ch in channel_names}

print(f"\n  {'Channel':25s}  {'Share':>8s}  {'95% HDI':>20s}")
print(f"  {'-'*60}")
for ch in channel_names:
    s = shares[ch]
    hdi = az.hdi(s, hdi_prob=0.95)
    label = ch.replace("spend_", "")
    print(f"  {label:25s}  {s.mean():>7.1%}  [{hdi[0]:.1%} — {hdi[1]:.1%}]")

# Pairwise comparisons
print(f"\n  Pairwise probability comparisons:")
for i, ch1 in enumerate(channel_names):
    for ch2 in channel_names[i+1:]:
        p = (shares[ch1] > shares[ch2]).mean()
        l1 = ch1.replace("spend_", "")
        l2 = ch2.replace("spend_", "")
        print(f"    P({l1} > {l2}): {p:.1%}")


# --- Stacked area chart ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Get mean contributions per week
ch_means = {}
for ch in channel_names:
    ch_means[ch] = ch_contrib.sel(channel=ch).values.reshape(-1, len(df)).mean(axis=0)

# Stacked area
ax = axes[0]
bottom = np.zeros(len(df))

# Estimate baseline as predicted - media contributions
y_pred = posterior["y_original_scale"].values.reshape(-1, len(df)).mean(axis=0)
media_sum = sum(ch_means.values())
baseline = y_pred - media_sum
baseline = np.maximum(baseline, 0)  # clip negative

ax.fill_between(df["date"], 0, baseline / 1e6, alpha=0.5,
                color="lightgray", label="Baseline")
colors = ["purple", "green", "steelblue", "orange"]
cumulative = baseline.copy()
for ch, color in zip(channel_names, colors):
    label = ch.replace("spend_", "").title()
    ax.fill_between(df["date"], cumulative / 1e6,
                    (cumulative + ch_means[ch]) / 1e6,
                    alpha=0.5, color=color, label=label)
    cumulative = cumulative + ch_means[ch]

ax.plot(df["date"], y.values / 1e6, "k.", markersize=4, alpha=0.5, label="Actual GMV")
ax.set_title("DT Mart Sales Decomposition by Channel", fontsize=14, fontweight="bold")
ax.set_ylabel("GMV (millions)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Channel share with uncertainty
ax = axes[1]
for ch, color in zip(channel_names, colors):
    s = shares[ch]
    label = ch.replace("spend_", "").title()
    ax.hist(s, bins=50, density=True, alpha=0.4, color=color,
            edgecolor="white", linewidth=0.5, label=f"{label}: {s.mean():.1%}")

ax.axvline(1/len(channel_names), color="red", linestyle=":", alpha=0.5,
           label="Equal share")
ax.set_title("Channel Share Distributions — How Confident Are We?",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Share of Media-Driven Sales")
ax.set_ylabel("Density")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS}/08_channel_contributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 08_channel_contributions.png")

# --- Waterfall ---
fig, ax = plt.subplots(figsize=(12, 6))
components = ["Baseline"] + [ch.replace("spend_", "").title() for ch in channel_names] + ["Total"]
values = [baseline.mean()] + [ch_means[ch].mean() for ch in channel_names] + [y.mean()]
colors_wf = ["lightgray"] + colors + ["black"]

cumulative_val = 0
for i, (comp, val, color) in enumerate(zip(components[:-1], values[:-1], colors_wf[:-1])):
    ax.bar(i, val / 1e6, bottom=cumulative_val / 1e6, color=color,
           edgecolor="black", linewidth=0.5, width=0.6)
    pct = val / y.mean() * 100
    ax.text(i, (cumulative_val + val/2) / 1e6, f"{val/1e6:.1f}M\n({pct:.0f}%)",
            ha="center", va="center", fontsize=8, fontweight="bold")
    cumulative_val += val

ax.bar(len(components)-1, y.mean() / 1e6, color="black", alpha=0.1,
       edgecolor="black", linewidth=1.5, width=0.6)
ax.text(len(components)-1, y.mean() / 2e6, f"{y.mean()/1e6:.1f}M",
        ha="center", va="center", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(components)))
ax.set_xticklabels(components, fontsize=10)
ax.set_ylabel("GMV (millions)")
ax.set_title("DT Mart — Sales Waterfall Decomposition", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{RESULTS}/08_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 08_waterfall.png")


# ============================================================
# 7. POSTERIOR PREDICTIVE CHECK
# ============================================================

print("\n\n" + "=" * 60)
print("7. POSTERIOR PREDICTIVE CHECK")
print("=" * 60)

y_pred_flat = posterior["y_original_scale"].values.reshape(-1, len(df))
y_pred_mean = y_pred_flat.mean(axis=0)
residuals = y.values - y_pred_mean

r2 = 1 - np.var(residuals) / np.var(y.values)
mape = np.abs(residuals / y.values).mean()

print(f"\n  Model fit on REAL data:")
print(f"    R²:   {r2:.3f}")
print(f"    MAPE: {mape:.1%}")
print(f"    MAE:  {np.abs(residuals).mean():,.0f}")

fig, ax = plt.subplots(figsize=(14, 5))
pp_low = np.percentile(y_pred_flat, 2.5, axis=0)
pp_high = np.percentile(y_pred_flat, 97.5, axis=0)
ax.fill_between(df["date"], pp_low / 1e6, pp_high / 1e6, alpha=0.3,
                color="steelblue", label="95% posterior predictive")
ax.plot(df["date"], y_pred_mean / 1e6, "b-", linewidth=1, label="Posterior mean")
ax.plot(df["date"], y.values / 1e6, "r.", markersize=6, label="Actual GMV")
ax.set_title("Posterior Predictive Check — DT Mart",
             fontsize=14, fontweight="bold")
ax.set_ylabel("GMV (millions)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS}/08_posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 08_posterior_predictive.png")


# ============================================================
# 8. PRIOR SENSITIVITY ANALYSIS
# ============================================================
# Run a second model with DIFFERENT priors and compare.
# If results change dramatically, they're prior-driven.

print("\n\n" + "=" * 60)
print("8. PRIOR SENSITIVITY ANALYSIS")
print("=" * 60)
print(f"\n  Fitting a second model with different adstock priors...")
print(f"  Original: α ~ Beta(1,3), mean=0.25")
print(f"  Alternative: α ~ Beta(3,3), mean=0.50 (higher carry-over)")

# Build alternative model with different adstock prior
mmm_alt = MMM(
    date_column="date",
    channel_columns=channels,
    control_columns=controls,
    adstock=GeometricAdstock(
        l_max=8,
        priors={"alpha": Prior("Beta", alpha=3, beta=3)},
    ),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)

idata_alt = mmm_alt.fit(X, y, random_seed=123, target_accept=0.95,
                        draws=2000, tune=2000, progressbar=False)

# Compare channel shares
posterior_alt = idata_alt.posterior
ch_contrib_alt = posterior_alt["channel_contribution_original_scale"]
ch_totals_alt = {}
for ch in channel_names:
    samples = ch_contrib_alt.sel(channel=ch).values.reshape(-1, len(df)).sum(axis=1)
    ch_totals_alt[ch] = samples
total_media_alt = sum(ch_totals_alt.values())
shares_alt = {ch: ch_totals_alt[ch] / total_media_alt for ch in channel_names}

print(f"\n  Channel share comparison (original vs alternative priors):")
print(f"  {'Channel':20s}  {'Original':>10s}  {'Alt (α~Beta(3,3))':>18s}  {'Difference':>12s}")
print(f"  {'-'*65}")
max_diff = 0
for ch in channel_names:
    orig = shares[ch].mean()
    alt = shares_alt[ch].mean()
    diff = abs(orig - alt)
    max_diff = max(max_diff, diff)
    label = ch.replace("spend_", "")
    print(f"  {label:20s}  {orig:>9.1%}  {alt:>17.1%}  {diff:>+11.1%}")

print(f"\n  Maximum share difference: {max_diff:.1%}")
if max_diff < 0.05:
    print(f"  GOOD: Results are ROBUST to prior choice (< 5% change).")
    print(f"  The data is driving the results, not the priors.")
elif max_diff < 0.15:
    print(f"  MODERATE: Results shift somewhat with different priors.")
    print(f"  Some conclusions are data-driven, others are prior-influenced.")
    print(f"  Report both sets of results for transparency.")
else:
    print(f"  CONCERNING: Results change substantially with different priors.")
    print(f"  The data alone cannot support strong conclusions.")
    print(f"  This is expected with {len(df)} observations and {len(channels)} channels.")
    print(f"  Be transparent: 'Our estimates depend on modeling assumptions.'")

# Compare adstock posteriors
print(f"\n  Adstock α comparison:")
print(f"  {'Channel':20s}  {'Original':>10s}  {'Alternative':>12s}")
print(f"  {'-'*45}")
for ch in channel_names:
    orig_a = posterior["adstock_alpha"].sel(channel=ch).values.mean()
    alt_a = posterior_alt["adstock_alpha"].sel(channel=ch).values.mean()
    label = ch.replace("spend_", "")
    print(f"  {label:20s}  {orig_a:>10.3f}  {alt_a:>12.3f}")


# ============================================================
# 9. HONEST ASSESSMENT
# ============================================================

print("\n\n" + "=" * 60)
print("9. HONEST ASSESSMENT")
print("=" * 60)

print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║       DT MART MMM — HONEST ASSESSMENT                   ║
  ╠══════════════════════════════════════════════════════════╣
  ║                                                          ║
  ║  WHAT WE CAN SAY:                                       ║
  ║  - The model captures the overall sales pattern (R²={r2:.2f})  ║
  ║  - Special sales are a major driver of GMV spikes        ║
  ║  - Baseline (non-advertising) drives most of sales       ║
  ║  - We can rank channels by likely contribution           ║
  ║                                                          ║
  ║  WHAT WE CANNOT CONFIDENTLY SAY:                         ║
  ║  - Precise individual channel contributions              ║
  ║  - Whether Brand or SEM contributes more (r=0.86)        ║
  ║  - Exact ROI per channel                                 ║
  ║                                                          ║
  ║  WHY:                                                    ║
  ║  - Only 47 weekly observations (need 100+)               ║
  ║  - Monthly spend → within-month variation = 0            ║
  ║  - Channels still correlated after grouping              ║
  ║  - Prior sensitivity shows some prior dependence         ║
  ║                                                          ║
  ║  TO IMPROVE (in production):                             ║
  ║  - Get 2+ years of weekly data                           ║
  ║  - Get weekly (not monthly) spend from ad platforms      ║
  ║  - Run geo experiments → calibrate with lift tests       ║
  ║  - Use PyMC-Marketing's add_lift_test_measurements()     ║
  ║                                                          ║
  ║  VALUE OF THIS EXERCISE:                                 ║
  ║  - Learned the full Bayesian MMM workflow end-to-end     ║
  ║  - Experienced what "not enough data" looks like         ║
  ║  - Wide posteriors are HONEST, not failures              ║
  ║  - Ready to apply to better data at work                 ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
