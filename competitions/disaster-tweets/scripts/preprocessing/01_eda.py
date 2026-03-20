"""
Disaster Tweets: Step 1 — EDA
Explore the dataset with an NLP lens: text patterns, keywords, and structure.
"""

import sys
sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")

import re
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.evaluate import Tee

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/disaster-tweets"

tee = Tee(f"{BASE}/results/models/01_eda.txt")
sys.stdout = tee


# ============================================================
# DATA LOADING
# ============================================================

train = pd.read_csv(f"{BASE}/data/train.csv")
test = pd.read_csv(f"{BASE}/data/test.csv")

print("Disaster Tweets: Step 1 — EDA")
print("=" * 60)
print(f"\n  Train: {train.shape[0]} rows, {train.shape[1]} columns")
print(f"  Test:  {test.shape[0]} rows, {test.shape[1]} columns")
print(f"  Columns: {list(train.columns)}")
print(f"\n  First 3 rows:")
print(train.head(3).to_string(index=False))


# ============================================================
# 1. TARGET DISTRIBUTION
# ============================================================

print("\n\n" + "=" * 60)
print("1. TARGET DISTRIBUTION")
print("=" * 60)

counts = train["target"].value_counts()
print(f"\n  Not disaster (0): {counts[0]:,} ({counts[0]/len(train):.1%})")
print(f"  Disaster (1):     {counts[1]:,} ({counts[1]/len(train):.1%})")
print(f"  Ratio:            {counts[0]/counts[1]:.2f}:1")
print(f"\n  Class balance is moderate — not heavily imbalanced.")
print(f"  F1 metric already accounts for this by balancing precision and recall.")


# ============================================================
# 2. MISSING VALUES
# ============================================================

print("\n\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

for col in train.columns:
    n_miss = train[col].isna().sum()
    if n_miss > 0:
        print(f"  {col:12s}: {n_miss:,} missing ({n_miss/len(train):.1%})")

print(f"\n  text:     0 missing — always present")
print(f"  keyword:  {train['keyword'].isna().sum()} missing — mostly present, usable feature")
print(f"  location: {train['location'].isna().sum()} missing — 33%, very messy (user-entered)")


# ============================================================
# 3. KEYWORD ANALYSIS
# ============================================================

print("\n\n" + "=" * 60)
print("3. KEYWORD ANALYSIS")
print("=" * 60)

# Decode URL-encoded keywords (e.g., "body%20bags" -> "body bags")
train["keyword_clean"] = train["keyword"].str.replace("%20", " ", regex=False)

kw_stats = (
    train.groupby("keyword_clean")["target"]
    .agg(["count", "mean"])
    .rename(columns={"count": "n", "mean": "disaster_rate"})
    .sort_values("disaster_rate", ascending=False)
)

print(f"\n  Unique keywords: {train['keyword_clean'].nunique()}")
print(f"  Missing keyword: {train['keyword'].isna().sum()} tweets")

# Most disaster-predictive keywords
print(f"\n  Top 15 keywords (highest disaster rate, n >= 10):")
top_disaster = kw_stats[kw_stats["n"] >= 10].head(15)
print(f"  {'Keyword':30s} {'Count':>6s} {'Disaster%':>10s}")
print(f"  {'-' * 50}")
for kw, row in top_disaster.iterrows():
    print(f"  {kw:30s} {int(row['n']):>6d} {row['disaster_rate']:>10.1%}")

# Least disaster-predictive keywords
print(f"\n  Bottom 15 keywords (lowest disaster rate, n >= 10):")
bottom_disaster = kw_stats[kw_stats["n"] >= 10].tail(15)
print(f"  {'Keyword':30s} {'Count':>6s} {'Disaster%':>10s}")
print(f"  {'-' * 50}")
for kw, row in bottom_disaster.iterrows():
    print(f"  {kw:30s} {int(row['n']):>6d} {row['disaster_rate']:>10.1%}")

# How good is keyword alone?
kw_only = train.dropna(subset=["keyword_clean"]).copy()
kw_median_rate = kw_stats["disaster_rate"].median()
print(f"\n  Median keyword disaster rate: {kw_median_rate:.1%}")
print(f"  If we predict disaster when keyword rate > 50%:")
kw_pred_map = (kw_stats["disaster_rate"] > 0.5).to_dict()
kw_only["kw_pred"] = kw_only["keyword_clean"].map(kw_pred_map).astype(int)
kw_acc = (kw_only["kw_pred"] == kw_only["target"]).mean()
from sklearn.metrics import f1_score
kw_f1 = f1_score(kw_only["target"], kw_only["kw_pred"])
print(f"  Keyword-only accuracy: {kw_acc:.3f}")
print(f"  Keyword-only F1:       {kw_f1:.3f}")
print(f"  (Only on tweets that have a keyword — {len(kw_only)}/{len(train)})")


# ============================================================
# 4. TEXT LENGTH PATTERNS
# ============================================================

print("\n\n" + "=" * 60)
print("4. TEXT LENGTH PATTERNS")
print("=" * 60)

train["char_len"] = train["text"].str.len()
train["word_count"] = train["text"].str.split().str.len()

for label, group in train.groupby("target"):
    name = "Disaster" if label == 1 else "Not disaster"
    print(f"\n  {name} tweets:")
    print(f"    Char length — mean: {group['char_len'].mean():.1f}, "
          f"median: {group['char_len'].median():.0f}, "
          f"std: {group['char_len'].std():.1f}")
    print(f"    Word count  — mean: {group['word_count'].mean():.1f}, "
          f"median: {group['word_count'].median():.0f}, "
          f"std: {group['word_count'].std():.1f}")

# Statistical test
from scipy import stats
d_chars = train[train["target"] == 1]["char_len"]
nd_chars = train[train["target"] == 0]["char_len"]
t_stat, p_val = stats.ttest_ind(d_chars, nd_chars)
print(f"\n  T-test (char length): t={t_stat:.2f}, p={p_val:.4f}")
if p_val < 0.01:
    print(f"  Significant difference — disaster tweets tend to be "
          f"{'longer' if d_chars.mean() > nd_chars.mean() else 'shorter'}.")


# ============================================================
# 5. URL, HASHTAG, AND @MENTION PATTERNS
# ============================================================

print("\n\n" + "=" * 60)
print("5. URL, HASHTAG, AND @MENTION PATTERNS")
print("=" * 60)

train["n_urls"] = train["text"].str.count(r"https?://\S+")
train["n_hashtags"] = train["text"].str.count(r"#\w+")
train["n_mentions"] = train["text"].str.count(r"@\w+")
train["has_url"] = (train["n_urls"] > 0).astype(int)
train["has_hashtag"] = (train["n_hashtags"] > 0).astype(int)
train["has_mention"] = (train["n_mentions"] > 0).astype(int)

for feature in ["has_url", "has_hashtag", "has_mention"]:
    print(f"\n  {feature}:")
    ct = pd.crosstab(train[feature], train["target"], normalize="index")
    for val in [0, 1]:
        if val in ct.index:
            pct_disaster = ct.loc[val, 1] if 1 in ct.columns else 0
            n = (train[feature] == val).sum()
            print(f"    {feature}={val}: {n:>5d} tweets, {pct_disaster:.1%} disaster")

# Summarize
url_d_rate = train[train["has_url"] == 1]["target"].mean()
url_nd_rate = train[train["has_url"] == 0]["target"].mean()
print(f"\n  Summary:")
print(f"    Tweets with URLs:     {url_d_rate:.1%} are disasters (vs {url_nd_rate:.1%} without)")
ht_d_rate = train[train["has_hashtag"] == 1]["target"].mean()
ht_nd_rate = train[train["has_hashtag"] == 0]["target"].mean()
print(f"    Tweets with hashtags: {ht_d_rate:.1%} are disasters (vs {ht_nd_rate:.1%} without)")
mn_d_rate = train[train["has_mention"] == 1]["target"].mean()
mn_nd_rate = train[train["has_mention"] == 0]["target"].mean()
print(f"    Tweets with @mentions:{mn_d_rate:.1%} are disasters (vs {mn_nd_rate:.1%} without)")


# ============================================================
# 6. COMMON WORDS — DISASTER VS NOT
# ============================================================

print("\n\n" + "=" * 60)
print("6. COMMON WORDS — DISASTER VS NOT")
print("=" * 60)


def get_top_words(texts, n=25):
    """Extract top-n words from a series of texts (lowercased, alpha only)."""
    words = []
    for text in texts:
        tokens = re.findall(r"[a-z]+", text.lower())
        words.extend(tokens)
    return Counter(words).most_common(n)


disaster_texts = train[train["target"] == 1]["text"]
non_disaster_texts = train[train["target"] == 0]["text"]

top_disaster = get_top_words(disaster_texts)
top_non_disaster = get_top_words(non_disaster_texts)

print(f"\n  Top 25 words in DISASTER tweets:")
print(f"  {'Word':20s} {'Count':>8s}")
print(f"  {'-' * 30}")
for word, count in top_disaster:
    print(f"  {word:20s} {count:>8d}")

print(f"\n  Top 25 words in NON-DISASTER tweets:")
print(f"  {'Word':20s} {'Count':>8s}")
print(f"  {'-' * 30}")
for word, count in top_non_disaster:
    print(f"  {word:20s} {count:>8d}")

# Words most distinctive to each class (ratio-based)
print(f"\n  Most distinctive words (disaster vs non-disaster):")
all_d_words = Counter()
all_nd_words = Counter()
for text in disaster_texts:
    all_d_words.update(re.findall(r"[a-z]+", text.lower()))
for text in non_disaster_texts:
    all_nd_words.update(re.findall(r"[a-z]+", text.lower()))

# Compute ratio: P(word | disaster) / P(word | non-disaster)
vocab = set(all_d_words.keys()) | set(all_nd_words.keys())
n_d_total = sum(all_d_words.values())
n_nd_total = sum(all_nd_words.values())

ratios = {}
for w in vocab:
    d_freq = (all_d_words[w] + 1) / (n_d_total + len(vocab))   # Laplace smoothing
    nd_freq = (all_nd_words[w] + 1) / (n_nd_total + len(vocab))
    if all_d_words[w] + all_nd_words[w] >= 10:  # minimum total frequency
        ratios[w] = d_freq / nd_freq

sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 20 words that signal DISASTER (highest P(w|disaster)/P(w|not)):")
print(f"  {'Word':20s} {'Ratio':>8s} {'Disaster#':>10s} {'Non-dis#':>10s}")
print(f"  {'-' * 52}")
for word, ratio in sorted_ratios[:20]:
    print(f"  {word:20s} {ratio:>8.2f} {all_d_words[word]:>10d} {all_nd_words[word]:>10d}")

print(f"\n  Top 20 words that signal NOT-DISASTER:")
print(f"  {'Word':20s} {'Ratio':>8s} {'Disaster#':>10s} {'Non-dis#':>10s}")
print(f"  {'-' * 52}")
for word, ratio in sorted_ratios[-20:]:
    print(f"  {word:20s} {ratio:>8.2f} {all_d_words[word]:>10d} {all_nd_words[word]:>10d}")


# ============================================================
# 7. LOCATION FIELD ANALYSIS
# ============================================================

print("\n\n" + "=" * 60)
print("7. LOCATION FIELD")
print("=" * 60)

n_loc = train["location"].notna().sum()
n_unique = train["location"].nunique()
print(f"\n  Present: {n_loc}/{len(train)} ({n_loc/len(train):.1%})")
print(f"  Unique values: {n_unique}")
print(f"\n  Top 15 locations:")
loc_counts = train["location"].value_counts().head(15)
for loc, count in loc_counts.items():
    rate = train[train["location"] == loc]["target"].mean()
    print(f"    {loc:35s} n={count:>4d}  disaster_rate={rate:.1%}")

print(f"\n  Verdict: {n_unique} unique values is very messy (user-entered free text).")
print(f"  Too noisy to use directly. Could extract country/state with geocoding")
print(f"  but likely diminishing returns for this project.")


# ============================================================
# 8. EXAMPLE TWEETS
# ============================================================

print("\n\n" + "=" * 60)
print("8. EXAMPLE TWEETS")
print("=" * 60)

print(f"\n  5 random DISASTER tweets:")
for _, row in train[train["target"] == 1].sample(5, random_state=42).iterrows():
    kw = row["keyword_clean"] if pd.notna(row["keyword_clean"]) else "—"
    print(f"    [{kw:20s}] {row['text'][:100]}")

print(f"\n  5 random NON-DISASTER tweets:")
for _, row in train[train["target"] == 0].sample(5, random_state=42).iterrows():
    kw = row["keyword_clean"] if pd.notna(row["keyword_clean"]) else "—"
    print(f"    [{kw:20s}] {row['text'][:100]}")


# ============================================================
# 9. PLOTS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Disaster Tweets: Step 1 — EDA", fontsize=14, fontweight="bold")

# 9a: Target distribution
ax = axes[0, 0]
colors = ["#4ECDC4", "#FF6B6B"]
train["target"].value_counts().sort_index().plot.bar(ax=ax, color=colors)
ax.set_xlabel("Target")
ax.set_ylabel("Count")
ax.set_title("Target Distribution")
ax.set_xticklabels(["Not Disaster (0)", "Disaster (1)"], rotation=0)

# 9b: Character length by class
ax = axes[0, 1]
for label, color in [(0, "#4ECDC4"), (1, "#FF6B6B")]:
    name = "Disaster" if label == 1 else "Not Disaster"
    train[train["target"] == label]["char_len"].hist(
        ax=ax, bins=40, alpha=0.6, color=color, label=name
    )
ax.set_xlabel("Character Length")
ax.set_ylabel("Count")
ax.set_title("Text Length Distribution")
ax.legend()

# 9c: Word count by class
ax = axes[0, 2]
for label, color in [(0, "#4ECDC4"), (1, "#FF6B6B")]:
    name = "Disaster" if label == 1 else "Not Disaster"
    train[train["target"] == label]["word_count"].hist(
        ax=ax, bins=30, alpha=0.6, color=color, label=name
    )
ax.set_xlabel("Word Count")
ax.set_ylabel("Count")
ax.set_title("Word Count Distribution")
ax.legend()

# 9d: URL/Hashtag/Mention rates by class
ax = axes[1, 0]
features = ["has_url", "has_hashtag", "has_mention"]
labels_short = ["URL", "Hashtag", "@mention"]
x = np.arange(len(features))
width = 0.35
rates_0 = [train[train["target"] == 0][f].mean() for f in features]
rates_1 = [train[train["target"] == 1][f].mean() for f in features]
ax.bar(x - width / 2, rates_0, width, label="Not Disaster", color="#4ECDC4")
ax.bar(x + width / 2, rates_1, width, label="Disaster", color="#FF6B6B")
ax.set_xticks(x)
ax.set_xticklabels(labels_short)
ax.set_ylabel("Proportion with Feature")
ax.set_title("Structural Features by Class")
ax.legend()

# 9e: Top 15 keywords by disaster rate
ax = axes[1, 1]
top_kw = kw_stats[kw_stats["n"] >= 20].head(15)
y_pos = np.arange(len(top_kw))
ax.barh(y_pos, top_kw["disaster_rate"], color="#FF6B6B", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_kw.index, fontsize=8)
ax.set_xlabel("Disaster Rate")
ax.set_title("Top Keywords (Disaster Rate, n>=20)")
ax.invert_yaxis()

# 9f: Bottom 15 keywords by disaster rate
ax = axes[1, 2]
bottom_kw = kw_stats[kw_stats["n"] >= 20].tail(15)
y_pos = np.arange(len(bottom_kw))
ax.barh(y_pos, bottom_kw["disaster_rate"], color="#4ECDC4", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(bottom_kw.index, fontsize=8)
ax.set_xlabel("Disaster Rate")
ax.set_title("Bottom Keywords (Disaster Rate, n>=20)")
ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{BASE}/results/analysis/01_eda.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Saved: {plot_path}")


# ============================================================
# 10. KEY TAKEAWAYS
# ============================================================

print("\n\n" + "=" * 60)
print("10. KEY TAKEAWAYS FOR MODELING")
print("=" * 60)
print("""
  1. CLASS BALANCE: 57/43 split — moderate imbalance. F1 metric handles this
     well. No need for oversampling.

  2. KEYWORD is a strong signal: keyword-only baseline already gets decent F1.
     Some keywords (earthquake, wildfire) are nearly 100% disaster.
     But keyword alone can't solve it — many keywords are ambiguous.

  3. TEXT LENGTH: Disaster tweets are slightly longer on average. Weak signal
     but could help as a feature alongside text content.

  4. URLs: Disaster tweets more often contain URLs (linking to news sources).
     @mentions: Non-disaster tweets use more @mentions (social interaction).
     Both are structural features that complement word content.

  5. WORD CONTENT: Clear signal — disaster tweets use "fire," "killed,"
     "storm," "earthquake"; non-disaster uses "love," "like," "new," "body"
     (metaphorical usage). This is exactly what bag-of-words will capture.

  6. LOCATION: Too messy to use (3000+ unique user-entered strings). Skip it.

  7. THE CORE CHALLENGE: Words like "fire," "crash," "body" have both
     literal (disaster) and metaphorical (non-disaster) meanings.
     Bag-of-words can't distinguish these — that's what embeddings and
     transformers will fix in later phases.
""")

tee.close()
print("Done. Results saved to results/models/01_eda.txt")
