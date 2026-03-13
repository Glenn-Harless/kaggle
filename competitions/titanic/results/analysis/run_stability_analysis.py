"""
Prediction Stability Analysis for Titanic Test Set
Analyzes which test passengers have unstable/borderline predictions.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"
DATA_DIR = f"{BASE}/data"
SUBMISSIONS_DIR = f"{BASE}/submissions"
OUT_DIR = f"{BASE}/results/analysis"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. REPRODUCE THE v2 MODEL PIPELINE
# ============================================================

# Load processed data
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")

test_ids = test["PassengerId"]
X_test = test.drop(columns=["PassengerId"])

X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]

# Train the v2 model (C=0.01)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
])
pipe.fit(X_train, y_train)

# Get predicted probabilities for test set
test_proba = pipe.predict_proba(X_test)[:, 1]
test_pred = pipe.predict(X_test)

# Verify against existing submission
v2_sub = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_v2.csv")
match = (test_pred == v2_sub["Survived"].values).all()
print(f"Predictions match logreg_v2.csv: {match}")
if not match:
    n_diff = (test_pred != v2_sub["Survived"].values).sum()
    print(f"  Differences: {n_diff}")

# ============================================================
# 2. IDENTIFY BUBBLE PASSENGERS (0.3 < p < 0.7)
# ============================================================

import re
def extract_title(name):
    m = re.search(r', (\w+)\.', name)
    if m:
        title = m.group(1)
        title_map = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
            "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
            "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare",
            "Mme": "Mrs", "Capt": "Rare", "Sir": "Rare",
        }
        return title_map.get(title, "Rare")
    return "Rare"

bubble_df = pd.DataFrame({
    "PassengerId": test_ids.values,
    "Probability": test_proba,
    "Predicted": test_pred,
    "Sex": raw_test["Sex"].values,
    "Pclass": raw_test["Pclass"].values,
    "Age": raw_test["Age"].values,
    "Fare": raw_test["Fare"].values,
    "Name": raw_test["Name"].values,
    "SibSp": raw_test["SibSp"].values,
    "Parch": raw_test["Parch"].values,
    "Cabin": raw_test["Cabin"].values,
    "Embarked": raw_test["Embarked"].values,
})
bubble_df["Title"] = raw_test["Name"].apply(extract_title)
bubble_df["FamilySize"] = raw_test["SibSp"] + raw_test["Parch"]

bubble_mask = (test_proba > 0.3) & (test_proba < 0.7)
tight_bubble_mask = (test_proba > 0.4) & (test_proba < 0.6)

# bubble_passengers will be created after 12a columns are added

print(f"\nBubble passengers (0.3 < p < 0.7): {bubble_mask.sum()} / {len(test_proba)}")
print(f"Tight bubble (0.4 < p < 0.6): {tight_bubble_mask.sum()} / {len(test_proba)}")

# ============================================================
# 3. BOOTSTRAP STABILITY ANALYSIS (100 resamples)
# ============================================================

N_BOOTSTRAP = 100
np.random.seed(42)

bootstrap_preds = np.zeros((N_BOOTSTRAP, len(X_test)), dtype=int)
bootstrap_proba = np.zeros((N_BOOTSTRAP, len(X_test)))

print(f"\nRunning {N_BOOTSTRAP} bootstrap resamples...")
for i in range(N_BOOTSTRAP):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[idx]
    y_boot = y_train.iloc[idx]

    pipe_boot = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42))
    ])
    pipe_boot.fit(X_boot, y_boot)
    bootstrap_preds[i] = pipe_boot.predict(X_test)
    bootstrap_proba[i] = pipe_boot.predict_proba(X_test)[:, 1]
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N_BOOTSTRAP} done")

base_pred = test_pred
flip_count = np.sum(bootstrap_preds != base_pred[np.newaxis, :], axis=0)
flip_rate = flip_count / N_BOOTSTRAP

boot_mean_proba = bootstrap_proba.mean(axis=0)
boot_std_proba = bootstrap_proba.std(axis=0)

bubble_df["FlipRate"] = flip_rate
bubble_df["BootMeanProba"] = boot_mean_proba
bubble_df["BootStdProba"] = boot_std_proba

print(f"\nPassengers that flip in any bootstrap: {(flip_rate > 0).sum()}")
print(f"Passengers with flip rate > 10%: {(flip_rate > 0.1).sum()}")
print(f"Passengers with flip rate > 25%: {(flip_rate > 0.25).sum()}")

# ============================================================
# 4. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

thresholds = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]
threshold_preds = {}
for t in thresholds:
    threshold_preds[t] = (test_proba >= t).astype(int)

threshold_changes = {}
for t in thresholds:
    diff = (threshold_preds[t] != threshold_preds[0.50]).sum()
    threshold_changes[t] = diff

flip_at_45 = (threshold_preds[0.45] != threshold_preds[0.50])
flip_at_55 = (threshold_preds[0.55] != threshold_preds[0.50])

bubble_df["PredAt0.45"] = threshold_preds[0.45]
bubble_df["PredAt0.55"] = threshold_preds[0.55]

# ============================================================
# 5. SUBGROUP ANALYSIS
# ============================================================

subgroup_stats = []
for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (bubble_df["Sex"] == sex) & (bubble_df["Pclass"] == pclass)
        sub = bubble_df[mask]
        n = len(sub)
        n_bubble = ((sub["Probability"] > 0.3) & (sub["Probability"] < 0.7)).sum()
        n_flippers = (sub["FlipRate"] > 0).sum()
        n_high_flip = (sub["FlipRate"] > 0.1).sum()
        mean_prob = sub["Probability"].mean()
        mean_flip = sub["FlipRate"].mean()
        pred_surv_rate = sub["Predicted"].mean()
        subgroup_stats.append({
            "Sex": sex, "Pclass": pclass, "N": n,
            "N_Bubble": n_bubble, "N_AnyFlip": n_flippers,
            "N_HighFlip": n_high_flip,
            "MeanProb": mean_prob, "MeanFlipRate": mean_flip,
            "PredSurvRate": pred_surv_rate
        })

subgroup_df = pd.DataFrame(subgroup_stats)

# ============================================================
# 6. CROSS-REFERENCE WITH 12a
# ============================================================

seg_sub = pd.read_csv(f"{SUBMISSIONS_DIR}/logreg_12a_segment_sex.csv")
bubble_df["Pred_v2"] = v2_sub["Survived"].values
bubble_df["Pred_12a"] = seg_sub["Survived"].values
bubble_df["Flipped_by_12a"] = (bubble_df["Pred_v2"] != bubble_df["Pred_12a"])

flipped_by_12a = bubble_df[bubble_df["Flipped_by_12a"]].copy()
n_12a_flips = len(flipped_by_12a)

flipped_in_bubble = flipped_by_12a[(flipped_by_12a["Probability"] > 0.3) & (flipped_by_12a["Probability"] < 0.7)]
flipped_outside_bubble = flipped_by_12a[~((flipped_by_12a["Probability"] > 0.3) & (flipped_by_12a["Probability"] < 0.7))]

# Now create bubble_passengers (after all columns are added to bubble_df)
bubble_passengers = bubble_df[bubble_mask].sort_values("Probability")

# ============================================================
# 7. GENERATE THE REPORT
# ============================================================

report = []
report.append("# Prediction Stability Analysis Report")
report.append("")
report.append("## Executive Summary")
report.append("")
report.append(f"- **Model**: Logistic Regression (C=0.01) on v2 features")
report.append(f"- **Test set size**: {len(test_proba)} passengers")
report.append(f"- **Bubble passengers** (P between 0.3-0.7): **{bubble_mask.sum()}** ({100*bubble_mask.sum()/len(test_proba):.1f}%)")
report.append(f"- **Tight bubble** (P between 0.4-0.6): **{tight_bubble_mask.sum()}** ({100*tight_bubble_mask.sum()/len(test_proba):.1f}%)")
report.append(f"- **Bootstrap unstable** (flip in >0% of {N_BOOTSTRAP} resamples): **{(flip_rate > 0).sum()}** passengers")
report.append(f"- **Highly unstable** (flip in >10% of resamples): **{(flip_rate > 0.1).sum()}** passengers")
report.append(f"- **Predictions match logreg_v2.csv**: {match}")
report.append("")

# Probability distribution
report.append("## 1. Probability Distribution")
report.append("")
report.append("| Probability Range | Count | % of Test Set |")
report.append("|:---|---:|---:|")
ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
          (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
for lo, hi in ranges:
    n = ((test_proba >= lo) & (test_proba < hi)).sum()
    pct = 100 * n / len(test_proba)
    label = f"{lo:.1f} - {hi:.1f}" if hi <= 1.0 else f"{lo:.1f} - 1.0"
    report.append(f"| {label} | {n} | {pct:.1f}% |")
report.append("")

# Bubble passengers subgroup
report.append("## 2. Bubble Passengers by Subgroup")
report.append("")
report.append("| Sex | Pclass | Total N | N in Bubble | % Bubble | Mean P(Surv) | Pred Surv Rate |")
report.append("|:---|---:|---:|---:|---:|---:|---:|")
for _, row in subgroup_df.iterrows():
    bubble_pct = 100*row['N_Bubble']/row['N'] if row['N'] > 0 else 0
    report.append(f"| {row['Sex']} | {int(row['Pclass'])} | {int(row['N'])} | {int(row['N_Bubble'])} | {bubble_pct:.0f}% | {row['MeanProb']:.3f} | {row['PredSurvRate']:.3f} |")
report.append("")

# Key finding
fem3_mask = (bubble_df["Sex"] == "female") & (bubble_df["Pclass"] == 3)
fem3 = bubble_df[fem3_mask]
fem3_bubble = fem3[(fem3["Probability"] > 0.3) & (fem3["Probability"] < 0.7)]
report.append(f"**Key finding**: Of the {bubble_mask.sum()} bubble passengers, **{len(fem3_bubble)}** ({100*len(fem3_bubble)/max(bubble_mask.sum(),1):.0f}%) are 3rd-class females.")
report.append("")

# Complete bubble passenger list
report.append("## 3. Complete Bubble Passenger List")
report.append("")
report.append("| PID | P(Surv) | Pred | Sex | Pcl | Age | Fare | Title | FamSz | FlipRate | Flipped by 12a |")
report.append("|---:|---:|---:|:---|---:|---:|---:|:---|---:|---:|:---|")
for _, row in bubble_passengers.iterrows():
    age_str = f"{row['Age']:.0f}" if pd.notna(row['Age']) else "?"
    flipped_12a = "YES" if row['Flipped_by_12a'] else ""
    report.append(f"| {int(row['PassengerId'])} | {row['Probability']:.3f} | {int(row['Predicted'])} | {row['Sex']} | {int(row['Pclass'])} | {age_str} | {row['Fare']:.1f} | {row['Title']} | {int(row['FamilySize'])} | {row['FlipRate']:.0%} | {flipped_12a} |")
report.append("")

# Bootstrap stability
report.append("## 4. Bootstrap Stability Analysis")
report.append("")
report.append(f"Retrained on {N_BOOTSTRAP} bootstrap resamples of the training data.")
report.append("")
report.append("| Flip Rate Threshold | Passengers | % of Test |")
report.append("|:---|---:|---:|")
for threshold_val in [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
    n = (flip_rate > threshold_val).sum()
    report.append(f"| > {threshold_val:.0%} | {n} | {100*n/len(test_proba):.1f}% |")
report.append("")

# Top unstable
report.append("### Most Unstable Passengers (by bootstrap flip rate)")
report.append("")
top_unstable = bubble_df.nlargest(25, "FlipRate")
top_unstable = top_unstable[top_unstable["FlipRate"] > 0]
if len(top_unstable) > 0:
    report.append("| PID | P(Surv) | Pred | FlipRate | BootStd | Sex | Pcl | Age | Title | Flipped 12a |")
    report.append("|---:|---:|---:|---:|---:|:---|---:|---:|:---|:---|")
    for _, row in top_unstable.iterrows():
        age_str = f"{row['Age']:.0f}" if pd.notna(row['Age']) else "?"
        fl12a = "YES" if row['Flipped_by_12a'] else ""
        report.append(f"| {int(row['PassengerId'])} | {row['Probability']:.3f} | {int(row['Predicted'])} | {row['FlipRate']:.0%} | {row['BootStdProba']:.3f} | {row['Sex']} | {int(row['Pclass'])} | {age_str} | {row['Title']} | {fl12a} |")
report.append("")

# Subgroup instability
report.append("### Instability by Subgroup")
report.append("")
report.append("| Sex | Pclass | N | Any Flip | Flip >10% | Mean Flip Rate |")
report.append("|:---|---:|---:|---:|---:|---:|")
for _, row in subgroup_df.iterrows():
    report.append(f"| {row['Sex']} | {int(row['Pclass'])} | {int(row['N'])} | {int(row['N_AnyFlip'])} | {int(row['N_HighFlip'])} | {row['MeanFlipRate']:.1%} |")
report.append("")

# Threshold sensitivity
report.append("## 5. Threshold Sensitivity Analysis")
report.append("")
report.append("| Threshold | Predicted Survived | Change from 0.50 |")
report.append("|---:|---:|---:|")
for t in thresholds:
    n_surv = threshold_preds[t].sum()
    delta = threshold_changes[t]
    marker = " (baseline)" if t == 0.50 else ""
    report.append(f"| {t:.2f} | {n_surv} | {delta:+d}{marker} |")
report.append("")

# 12a cross-reference
report.append("## 6. Cross-Reference with 12a Segmented Model")
report.append("")
report.append(f"The 12a segmented model changed **{n_12a_flips}** predictions vs v2.")
report.append(f"It scored 0.7608 on Kaggle vs v2's 0.7727 (worse by ~5 correct predictions).")
report.append("")
report.append(f"- Flips in bubble zone (P 0.3-0.7): **{len(flipped_in_bubble)}**")
report.append(f"- Flips OUTSIDE bubble zone: **{len(flipped_outside_bubble)}**")
report.append("")

flipped_to_died = flipped_by_12a[flipped_by_12a["Pred_12a"] == 0]
flipped_to_surv = flipped_by_12a[flipped_by_12a["Pred_12a"] == 1]
report.append(f"- Flipped survived -> died: **{len(flipped_to_died)}**")
report.append(f"- Flipped died -> survived: **{len(flipped_to_surv)}**")
report.append("")

report.append("### 12a Flipped Passengers")
report.append("")
report.append("| PID | P(Surv) | v2 Pred | 12a Pred | Sex | Pcl | Title | FlipRate | In Bubble? |")
report.append("|---:|---:|---:|---:|:---|---:|:---|---:|:---|")
for _, row in flipped_by_12a.sort_values("Probability").iterrows():
    in_bubble = "YES" if 0.3 < row["Probability"] < 0.7 else "NO"
    report.append(f"| {int(row['PassengerId'])} | {row['Probability']:.3f} | {int(row['Pred_v2'])} | {int(row['Pred_12a'])} | {row['Sex']} | {int(row['Pclass'])} | {row['Title']} | {row['FlipRate']:.0%} | {in_bubble} |")
report.append("")

# Key insights
report.append("## 7. Key Insights")
report.append("")
report.append("### Why 12a Scored Worse")
report.append("")
report.append("The 12a model aggressively flipped 3rd-class females from survived to died.")
report.append("Since 12a scored worse on Kaggle, the v2 model's more optimistic predictions")
report.append("for 3rd-class females appear closer to the test set truth.")
report.append("")
report.append("### Where Improvement Points Can Come From")
report.append("")
very_close = ((test_proba >= 0.48) & (test_proba <= 0.52)).sum()
close = ((test_proba >= 0.45) & (test_proba <= 0.55)).sum()
report.append(f"- **{very_close}** passengers have P within 2% of decision boundary (0.48-0.52)")
report.append(f"- **{close}** passengers have P within 5% of decision boundary (0.45-0.55)")
report.append(f"- **{bubble_mask.sum()}** passengers in broad bubble zone (0.3-0.7)")
report.append("")
report.append("### Actionable Takeaway")
report.append("")
report.append("1. **Male 1st class borderline cases**: Safer targets for improvement than female 3rd class")
report.append("2. **Don't flip 3rd-class females to died**: The 12a experiment proved this hurts")
report.append("3. **Threshold tuning**: Small threshold changes could flip a few borderline cases")
report.append("4. **Any improvement must change <15 test predictions** to be trustworthy")

# Write report
report_text = "\n".join(report)
report_path = f"{OUT_DIR}/stability_analysis_report.md"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\nReport written to: {report_path}")
