"""
Marketing Mix Modeling: Step 5 — Adstock & Saturation Deep Dive

Isolate and deeply understand the two transformations that make MMM
different from regular regression. No model fitting here — just
building intuition about what these transformations do and why
they exist.

  1. Adstock — how advertising effects carry over time
  2. Saturation — how advertising hits diminishing returns
  3. The interaction — adstock + saturation together
  4. Prior sensitivity — how much do priors influence results?
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/marketing-mix"
RESULTS = f"{BASE}/results/analysis"

tee = Tee(f"{BASE}/results/models/05_adstock_saturation.txt")
sys.stdout = tee


print("Marketing Mix Modeling: Step 5 — Adstock & Saturation Deep Dive")
print("=" * 60)


# ============================================================
# 1. ADSTOCK — HOW ADVERTISING CARRIES OVER TIME
# ============================================================
# When you see a TV ad on Monday, you don't immediately buy.
# Maybe you think about it for a few days. Maybe you mention it
# to a friend. The ad's influence DECAYS over time but doesn't
# disappear instantly.
#
# Geometric adstock models this with one parameter: α (alpha)
#   adstock[t] = spend[t] + α × adstock[t-1]
#
# Unrolling the recursion:
#   adstock[t] = spend[t] + α×spend[t-1] + α²×spend[t-2] + ...
#
# α controls the decay rate:
#   α = 0.0  →  no memory (only today's spend matters)
#   α = 0.5  →  last week retains 50% of its effect
#   α = 0.9  →  last week retains 90% — very slow decay
#
# MARKETING INTUITION:
#   Google Search ads:  α ≈ 0.1  (click today, buy today)
#   Facebook/TikTok:    α ≈ 0.3  (scroll past, maybe remember tomorrow)
#   TV/CTV:             α ≈ 0.5-0.7  (jingle sticks in your head for weeks)
#   Brand campaigns:    α ≈ 0.7-0.9  (slow brand building over months)

print("\n\n" + "=" * 60)
print("1. ADSTOCK — CARRY-OVER EFFECTS")
print("=" * 60)


def geometric_adstock(x, alpha, l_max=12):
    """Apply geometric adstock: each week carries forward α of the previous."""
    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(t + 1, l_max)):
            result[t] += x[t - lag] * (alpha ** lag)
    return result


# --- 1a. Single impulse response ---
# What happens when you spend $100 in ONE week and nothing else?
# The adstock transformation "smears" that $100 across future weeks.

print("\n  1a. Single impulse — how a one-time spend smears over time")

impulse = np.zeros(20)
impulse[2] = 100  # $100 spent in week 3

alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, alpha in zip(axes.flat, alphas):
    adstocked = geometric_adstock(impulse, alpha)
    ax.bar(range(len(impulse)), impulse, alpha=0.3, color="gray",
           label="Raw spend", width=0.4)
    ax.bar(np.arange(len(impulse)) + 0.4, adstocked, alpha=0.7,
           color="steelblue", label="After adstock", width=0.4)
    ax.set_title(f"α = {alpha}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Effect")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(110, adstocked.max() * 1.1))
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate: how many weeks until 90% of effect has occurred?
    if alpha > 0:
        cumulative = np.cumsum([alpha**i for i in range(50)])
        total = sum(alpha**i for i in range(200))
        matches = [i for i, c in enumerate(cumulative) if c >= 0.9 * total]
        weeks_90 = (matches[0] + 1) if matches else 50
        ax.annotate(f"90% of effect\nwithin {weeks_90} weeks",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

plt.suptitle("Adstock: How a Single $100 Spend Smears Over Time\n"
             "Higher α = longer carry-over",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_adstock_impulse.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_adstock_impulse.png")

# Print the decay weights
print(f"\n  Decay weights for each α (week 0 = spend week, week 1 = next week, ...):")
print(f"  {'Week':>6s}", end="")
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print(f"  {'α='+str(alpha):>8s}", end="")
print()
for week in range(8):
    print(f"  {week:>6d}", end="")
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        weight = alpha ** week
        print(f"  {weight:>8.3f}", end="")
    print()

print(f"\n  At α=0.9, week 7 still retains {0.9**7:.0%} of the original effect!")
print(f"  At α=0.1, week 7 retains only {0.1**7:.6%} — effectively zero.")


# --- 1b. Realistic spend pattern ---
# What does adstock look like on actual varying spend?
print(f"\n  1b. Realistic spend pattern with different α values")

np.random.seed(42)
weeks = 52
spend = np.random.exponential(scale=50, size=weeks)
# Add some campaigns (spend spikes)
spend[10:13] += 200  # Q4 campaign
spend[35:38] += 150  # Summer campaign

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, alpha in zip(axes.flat, [0.1, 0.3, 0.5, 0.7]):
    adstocked = geometric_adstock(spend, alpha)
    ax.fill_between(range(weeks), spend, alpha=0.2, color="gray", label="Raw spend")
    ax.plot(range(weeks), spend, "gray", alpha=0.5, linewidth=1)
    ax.plot(range(weeks), adstocked, "steelblue", linewidth=2, label="Adstocked")
    ax.set_title(f"α = {alpha} — {'Search ads' if alpha==0.1 else 'Social media' if alpha==0.3 else 'TV/CTV' if alpha==0.5 else 'Brand'}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Spend / Adstocked")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Adstock on Realistic Spend — Higher α Smooths and Delays the Signal",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_adstock_realistic.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_adstock_realistic.png")

print(f"""
  KEY OBSERVATIONS:
  - α=0.1 (search): adstocked curve closely tracks raw spend. Little smoothing.
    The effect is almost entirely in the week you spend.
  - α=0.3 (social): mild smoothing. Campaign spikes linger for 2-3 weeks.
  - α=0.5 (TV/CTV): significant smoothing. Campaigns create broad humps.
    This is why TV campaigns are planned in "flights" — the effect builds.
  - α=0.7 (brand): very smooth. Individual weeks are hard to distinguish.
    The adstocked signal is dominated by accumulated past spend.
""")


# --- 1c. l_max — maximum lag ---
print(f"  1c. l_max — how many weeks of history to consider")

fig, ax = plt.subplots(figsize=(12, 5))
impulse_long = np.zeros(30)
impulse_long[2] = 100
alpha = 0.5
for l_max in [4, 8, 12, 20]:
    adstocked = geometric_adstock(impulse_long, alpha, l_max=l_max)
    ax.plot(adstocked, label=f"l_max={l_max}", linewidth=2, alpha=0.7)
ax.bar(range(len(impulse_long)), impulse_long, alpha=0.15, color="gray", width=0.5)
ax.set_title(f"Effect of l_max on Adstock (α={alpha})", fontsize=13, fontweight="bold")
ax.set_xlabel("Week")
ax.set_ylabel("Adstocked value")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_lmax_effect.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_lmax_effect.png")

print(f"\n  l_max=4: truncates the tail — underestimates carry-over")
print(f"  l_max=8: captures most of the effect for α≤0.5")
print(f"  l_max=12-20: needed for high α (brand campaigns)")
print(f"  Rule of thumb: l_max should cover until α^l_max < 0.01")
for alpha in [0.3, 0.5, 0.7, 0.9]:
    import math
    l_needed = math.ceil(math.log(0.01) / math.log(alpha))
    print(f"    α={alpha}: need l_max ≥ {l_needed}")


# ============================================================
# 2. SATURATION — DIMINISHING RETURNS
# ============================================================
# The first $10K you spend on Facebook reaches new people.
# The 10th $10K reaches the same people for the 10th time.
# Each additional dollar is less effective.
#
# Logistic saturation: f(x) = 1 - exp(-λx)
#   λ controls HOW FAST saturation kicks in:
#   λ = 0.5  →  slow saturation (big budget before hitting ceiling)
#   λ = 2.0  →  moderate saturation
#   λ = 10.0 →  fast saturation (small budget already near ceiling)
#
# MARKETING INTUITION:
#   Large market (national TV):  λ small (takes a LOT of spend to saturate)
#   Small market (local search): λ large (small market saturates quickly)
#   Niche audience (retargeting): λ very large (audience exhaustion)

print("\n\n" + "=" * 60)
print("2. SATURATION — DIMINISHING RETURNS")
print("=" * 60)


def logistic_saturation(x, lam):
    """1 - exp(-λx): concave, starts linear, flattens toward 1."""
    return 1 - np.exp(-lam * x)


# --- 2a. Saturation curves at different λ ---
print(f"\n  2a. How λ controls the shape of diminishing returns")

x_spend = np.linspace(0, 3, 200)
lambdas = [0.5, 1.0, 2.0, 4.0, 8.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: saturation curves (total effect)
ax = axes[0]
for lam in lambdas:
    ax.plot(x_spend, logistic_saturation(x_spend, lam),
            linewidth=2, label=f"λ={lam}")
ax.set_title("Saturation Curves: Total Effect vs Spend",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Spend (normalized)")
ax.set_ylabel("Effect (0 to 1)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color="red", linestyle="--", alpha=0.3, label="Ceiling")

# Right: marginal effect (derivative = additional effect per additional dollar)
ax = axes[1]
dx = x_spend[1] - x_spend[0]
for lam in lambdas:
    sat = logistic_saturation(x_spend, lam)
    marginal = np.gradient(sat, dx)
    ax.plot(x_spend, marginal, linewidth=2, label=f"λ={lam}")
ax.set_title("Marginal Effect: Each Additional Dollar's Impact",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Spend (normalized)")
ax.set_ylabel("Marginal effect (slope)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Saturation: Higher λ = Faster Diminishing Returns",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_saturation_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_saturation_curves.png")

print(f"\n  Reading the LEFT panel (total effect):")
print(f"    All curves start at 0 and approach 1 (the ceiling).")
print(f"    λ=0.5: slowly climbs — you can spend a LOT before hitting the ceiling.")
print(f"    λ=8.0: rockets up immediately — tiny spend already near ceiling.")
print(f"\n  Reading the RIGHT panel (marginal effect — the ROI at each spend level):")
print(f"    All curves START at their λ value (the initial ROI).")
print(f"    All curves DECLINE toward 0 (every additional dollar does less).")
print(f"    λ=8.0 starts high but crashes fast — first dollar is amazing, 10th is worthless.")
print(f"    λ=0.5 starts low but declines slowly — consistent moderate returns.")

# --- 2b. Business interpretation ---
print(f"""
  BUSINESS TRANSLATION:
  ─────────────────────────────────────────────────────
  λ=0.5 (slow saturation):
    "We can keep increasing Google Ads spend significantly
     before hitting diminishing returns. Large addressable market."

  λ=8.0 (fast saturation):
    "Our retargeting audience is small. Even modest spend
     reaches everyone. Increasing budget won't help much."

  THIS IS WHY SATURATION MATTERS FOR BUDGET DECISIONS:
    If Channel A has λ=2 and Channel B has λ=8, and both
    are at the same spend level, shifting budget FROM B TO A
    will increase total sales (B is saturated, A isn't).
    This is exactly what budget optimization does in Step 9.
""")


# ============================================================
# 3. THE INTERACTION — ADSTOCK + SATURATION TOGETHER
# ============================================================
# In the full MMM, spend goes through BOTH transformations:
#   effect = β × saturation(adstock(spend))
#
# Adstock answers: WHEN does the effect happen?
# Saturation answers: HOW MUCH effect per dollar?
# Together they model the complete advertising response.

print("=" * 60)
print("3. ADSTOCK + SATURATION — THE FULL RESPONSE")
print("=" * 60)

# Show the pipeline on realistic spend
np.random.seed(42)
spend_real = np.random.exponential(50, 52)
spend_real[10:13] += 200  # Campaign spike

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Stage 1: Raw spend
axes[0].fill_between(range(52), spend_real, alpha=0.5, color="gray")
axes[0].plot(spend_real, "gray", linewidth=1)
axes[0].set_title("Stage 1: Raw Weekly Spend", fontsize=12, fontweight="bold")
axes[0].set_ylabel("$")
axes[0].grid(True, alpha=0.3)
axes[0].annotate("Campaign spike", xy=(11, spend_real[11]), fontsize=9,
                 arrowprops=dict(arrowstyle="->"), xytext=(20, 250))

# Stage 2: After adstock (α=0.5)
alpha = 0.5
adstocked = geometric_adstock(spend_real, alpha)
axes[1].fill_between(range(52), adstocked, alpha=0.5, color="steelblue")
axes[1].plot(adstocked, "steelblue", linewidth=1)
axes[1].set_title(f"Stage 2: After Adstock (α={alpha}) — Effect Smeared Over Time",
                  fontsize=12, fontweight="bold")
axes[1].set_ylabel("Adstocked spend")
axes[1].grid(True, alpha=0.3)
axes[1].annotate("Campaign effect lingers\nfor weeks after spike",
                 xy=(16, adstocked[16]), fontsize=9,
                 arrowprops=dict(arrowstyle="->"), xytext=(25, adstocked.max()*0.8))

# Stage 3: After saturation (λ=3.0) — normalize first
adstock_norm = adstocked / adstocked.max()
saturated = logistic_saturation(adstock_norm, 3.0)
axes[2].fill_between(range(52), saturated, alpha=0.5, color="coral")
axes[2].plot(saturated, "coral", linewidth=1)
axes[2].set_title(f"Stage 3: After Saturation (λ=3.0) — Diminishing Returns Applied",
                  fontsize=12, fontweight="bold")
axes[2].set_ylabel("Saturated effect")
axes[2].grid(True, alpha=0.3)
axes[2].annotate("Campaign spike is\ncapped by saturation",
                 xy=(11, saturated[11]), fontsize=9,
                 arrowprops=dict(arrowstyle="->"), xytext=(20, 0.9))

# Stage 4: Final effect = β × saturated
beta = 2.0
final_effect = beta * saturated
axes[3].fill_between(range(52), final_effect, alpha=0.5, color="green")
axes[3].plot(final_effect, "green", linewidth=1)
axes[3].set_title(f"Stage 4: Final Channel Effect = β × saturated (β={beta})",
                  fontsize=12, fontweight="bold")
axes[3].set_ylabel("Sales contribution")
axes[3].set_xlabel("Week")
axes[3].grid(True, alpha=0.3)

plt.suptitle("The Full MMM Pipeline: spend → adstock → saturation → effect",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_full_pipeline.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 05_full_pipeline.png")

print(f"""
  THE PIPELINE IN WORDS:
  ─────────────────────────────────────────────────────
  Stage 1 (Raw spend):     Sharp spike during campaign week 10-12.
  Stage 2 (Adstock):       Spike smears forward — effect lingers for weeks.
                           The carry-over captures delayed conversions.
  Stage 3 (Saturation):    High-spend weeks get capped — diminishing returns.
                           Low-spend weeks are nearly linear (full value).
  Stage 4 (β × effect):    Scale to actual sales contribution.

  Without adstock:  model only credits sales in the week you spend.
  Without saturation:  model assumes doubling spend doubles sales.
  Both are wrong. The pipeline corrects both simultaneously.
""")


# ============================================================
# 4. CHANNEL COMPARISON — YOUR WORK CHANNELS
# ============================================================
# Let's apply these concepts to your actual work channels to
# build intuition for when you build your own MMM.

print("=" * 60)
print("4. CHANNEL COMPARISON — APPLYING TO REAL CHANNELS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

channels = {
    "Google Search Ads": {"alpha": 0.1, "lam": 6.0, "color": "#4285F4",
                          "note": "Click today, convert today.\nSmall audience saturates fast."},
    "Facebook Ads": {"alpha": 0.3, "lam": 3.0, "color": "#1877F2",
                     "note": "Scroll past, maybe remember.\nLarge audience, moderate saturation."},
    "TikTok Ads": {"alpha": 0.25, "lam": 4.0, "color": "#000000",
                   "note": "Short attention, quick decay.\nYounger demographic, fast saturation."},
    "CTV/OTT Ads": {"alpha": 0.6, "lam": 1.5, "color": "#E50914",
                    "note": "Premium content = memorable.\nBroad reach, slow saturation."},
}

np.random.seed(42)
spend_example = np.random.exponential(30, 30)

for ax, (name, params) in zip(axes.flat, channels.items()):
    adstocked = geometric_adstock(spend_example, params["alpha"])
    adstock_norm = adstocked / adstocked.max()
    saturated = logistic_saturation(adstock_norm, params["lam"])

    ax.fill_between(range(30), spend_example / spend_example.max(),
                    alpha=0.15, color="gray", label="Raw spend")
    ax.plot(adstock_norm, "--", color=params["color"], alpha=0.5,
            linewidth=1, label=f"Adstocked (α={params['alpha']})")
    ax.plot(saturated, color=params["color"], linewidth=2,
            label=f"Final effect (λ={params['lam']})")
    ax.set_title(name, fontsize=12, fontweight="bold", color=params["color"])
    ax.set_xlabel("Week")
    ax.set_ylabel("Normalized effect")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.annotate(params["note"], xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, fontstyle="italic", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Channel Comparison: Different Channels Have Different Adstock & Saturation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_channel_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 05_channel_comparison.png")

print(f"""
  CHANNEL PRIORS FOR YOUR WORK MMM:
  ─────────────────────────────────────────────────────
  Channel          α (carry-over)     λ (saturation)     Why
  ─────────────────────────────────────────────────────
  Google Search    Low (0.05-0.2)     High (4-8)         Intent-driven, click→buy fast
  Facebook         Medium (0.2-0.4)   Medium (2-4)       Discovery, some recall
  TikTok           Low-Med (0.1-0.3)  Medium-High (3-6)  Short-form, quick attention
  CTV/OTT          High (0.4-0.7)     Low (1-3)          Premium, memorable, broad reach

  These aren't exact — they're PRIORS. The model will update them
  with your data. But starting with reasonable domain knowledge
  helps the model converge faster and produces better results
  when data is limited (your 80-100 weeks).
""")


# ============================================================
# 5. PRIOR SENSITIVITY — HOW MUCH DO PRIORS MATTER?
# ============================================================
# With limited data, priors heavily influence the posterior.
# This section shows HOW MUCH.

print("=" * 60)
print("5. PRIOR SENSITIVITY — WHEN PRIORS DOMINATE")
print("=" * 60)

# Simulate: same data, different priors → different posteriors
# Using the coin flip analogy but framed as adstock
from scipy import stats

# Fake scenario: 52 weeks of data suggest α ≈ 0.4
# But what if we use different priors?
true_alpha = 0.4
n_obs = 52  # like DT Mart

# Simulate "observations" as if we had a noisy estimate
np.random.seed(42)
obs_alphas = np.clip(np.random.normal(true_alpha, 0.1, n_obs), 0.01, 0.99)
obs_mean = obs_alphas.mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.linspace(0, 1, 200)

prior_configs = [
    ("Uninformative\nBeta(1,1)", 1, 1),
    ("Reasonable\nBeta(3,3)", 3, 3),
    ("Strong wrong\nBeta(1,8)", 1, 8),  # peaked at ~0.11, away from truth
]

for ax, (name, a, b) in zip(axes, prior_configs):
    # Prior
    prior = stats.beta.pdf(x, a, b)
    ax.plot(x, prior, "b--", linewidth=2, label=f"Prior: Beta({a},{b})")

    # Approximate posterior (using Beta-Binomial intuition)
    # Treat obs as successes out of total
    pseudo_successes = int(obs_mean * n_obs)
    pseudo_failures = n_obs - pseudo_successes
    post_a = a + pseudo_successes
    post_b = b + pseudo_failures
    posterior = stats.beta.pdf(x, post_a, post_b)
    ax.plot(x, posterior, "r-", linewidth=2, label=f"Posterior")

    ax.axvline(true_alpha, color="green", linewidth=2, linestyle=":",
               label=f"True α = {true_alpha}")
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("α (adstock decay)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Compute posterior mean
    post_mean = post_a / (post_a + post_b)
    ax.annotate(f"Post. mean = {post_mean:.2f}", xy=(0.95, 0.95),
                xycoords="axes fraction", fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))

plt.suptitle(f"Prior Sensitivity with n={n_obs} Observations (like DT Mart)\n"
             f"True α = {true_alpha}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/05_prior_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved: 05_prior_sensitivity.png")

print(f"""
  RESULTS:
  ─────────────────────────────────────────────────────
  Uninformative Beta(1,1):  posterior ≈ data (mean ≈ {obs_mean:.2f})
    → Prior stays out of the way. Good if you truly don't know.

  Reasonable Beta(3,3):     posterior ≈ data, slightly pulled to 0.5
    → Mild regularization. Good default for most channels.

  Strong wrong Beta(1,8):   posterior pulled AWAY from truth toward 0.1
    → Bad prior actively misleads! With only {n_obs} observations,
      wrong priors cause real damage.

  LESSON FOR YOUR MMM:
  With 52-100 weeks, your priors meaningfully affect results.
  This is OK if your priors are reasonable (based on domain knowledge).
  This is DANGEROUS if your priors are wrong.
  Always do prior sensitivity analysis (Step 8): run the model with
  different priors and check if the conclusions change.
  If they do, be honest about it — "our results depend on assumptions."
""")


# ============================================================
# 6. SUMMARY
# ============================================================

print("=" * 60)
print("6. SUMMARY")
print("=" * 60)

print(f"""
  ADSTOCK (carry-over):
    What:   Past spend still influences today's sales
    Param:  α (0-1) — higher = longer memory
    Why:    TV ad on Monday → purchase on Thursday
    Impact: Without it, model underestimates long-running channels

  SATURATION (diminishing returns):
    What:   Each additional dollar is less effective
    Param:  λ (>0) — higher = faster saturation
    Why:    50th ad impression is less impactful than the 1st
    Impact: Without it, model says "double spend = double sales" (wrong)

  TOGETHER:
    spend → adstock(α) → saturation(λ) → β × effect → contribution
    Adstock:    WHEN does the effect happen?
    Saturation: HOW MUCH effect per dollar?
    β:          OVERALL scale of this channel's contribution

  FOR YOUR WORK:
    Google Search: low α, high λ (immediate, small market)
    Facebook:      medium α, medium λ (discovery, broad)
    TikTok:        low-med α, med-high λ (short attention)
    CTV:           high α, low λ (memorable, broad reach)

  PRIOR SENSITIVITY:
    With limited data, priors matter. Use domain knowledge.
    Always test: "do my conclusions change with different priors?"
""")


# ============================================================
# CLEANUP
# ============================================================
tee.close()
print("Done. Log saved.")
