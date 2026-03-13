"""
Titanic Error Analysis: Out-of-Fold Misclassification Patterns
Reproduces the v2 logistic regression pipeline and analyzes errors.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/data"
OUT_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/results/analysis"

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load data ----
train_processed = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
train_raw = pd.read_csv(f"{DATA_DIR}/train.csv")

X = train_processed.drop(columns=["Survived"])
y = train_processed["Survived"]

feature_cols = list(X.columns)
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Training samples: {len(X)}")

# ---- Out-of-fold predictions ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_C = 0.01  # as selected by the v2 model

oof_preds = np.zeros(len(X))
oof_probs = np.zeros(len(X))
fold_assignments = np.zeros(len(X), dtype=int)

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=best_C, max_iter=2000, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    oof_preds[val_idx] = pipe.predict(X_val)
    oof_probs[val_idx] = pipe.predict_proba(X_val)[:, 1]
    fold_assignments[val_idx] = fold_idx + 1

    fold_acc = accuracy_score(y.iloc[val_idx], oof_preds[val_idx])
    print(f"  Fold {fold_idx+1}: accuracy={fold_acc:.4f}")

overall_acc = accuracy_score(y, oof_preds)
print(f"\nOverall OOF accuracy: {overall_acc:.4f}")

# ---- Build analysis dataframe ----
# Merge raw passenger info with predictions
analysis = train_raw[["PassengerId", "Survived", "Sex", "Pclass", "Age", "Fare", "Name", "SibSp", "Parch", "Cabin", "Embarked"]].copy()
analysis["Title"] = train_raw["Name"].str.extract(r", (\w+)\.", expand=False)
title_map = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Countess": "Rare", "Ms": "Miss", "Lady": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare",
    "Mme": "Mrs", "Capt": "Rare", "Sir": "Rare",
}
analysis["Title"] = analysis["Title"].map(title_map).fillna("Rare")
analysis["FamilySize"] = analysis["SibSp"] + analysis["Parch"]
analysis["Predicted"] = oof_preds.astype(int)
analysis["PredProb"] = oof_probs
analysis["Correct"] = (analysis["Survived"] == analysis["Predicted"]).astype(int)
analysis["Fold"] = fold_assignments
analysis["ErrorType"] = "Correct"
analysis.loc[(analysis["Survived"] == 0) & (analysis["Predicted"] == 1), "ErrorType"] = "FP"
analysis.loc[(analysis["Survived"] == 1) & (analysis["Predicted"] == 0), "ErrorType"] = "FN"

# ---- Summary stats ----
misclassified = analysis[analysis["Correct"] == 0]
correct = analysis[analysis["Correct"] == 1]
n_errors = len(misclassified)
n_total = len(analysis)

# ---- Build report ----
report = []
report.append("# Error Analysis Report: v2 Logistic Regression")
report.append("")
report.append(f"**Model**: LogisticRegression(C=0.01), 5-fold StratifiedKFold (random_state=42)")
report.append(f"**Overall OOF accuracy**: {overall_acc:.4f} ({n_errors} errors / {n_total} passengers)")
report.append("")

# 1. FP vs FN breakdown
fp = misclassified[misclassified["ErrorType"] == "FP"]
fn = misclassified[misclassified["ErrorType"] == "FN"]
report.append("## 1. Error Type Breakdown")
report.append("")
report.append(f"| Type | Count | % of Errors | Description |")
report.append(f"|:---|---:|---:|:---|")
report.append(f"| False Positive | {len(fp)} | {100*len(fp)/n_errors:.1f}% | Predicted survived, actually died |")
report.append(f"| False Negative | {len(fn)} | {100*len(fn)/n_errors:.1f}% | Predicted died, actually survived |")
report.append("")

# 2. Error rates by Sex x Pclass
report.append("## 2. Error Rates by Sex x Pclass")
report.append("")
report.append("| Group | Total | Errors | Error Rate | FP | FN | Survival Rate |")
report.append("|:---|---:|---:|---:|---:|---:|---:|")
for sex in ["male", "female"]:
    for pclass in [1, 2, 3]:
        mask = (analysis["Sex"] == sex) & (analysis["Pclass"] == pclass)
        group = analysis[mask]
        group_errors = group[group["Correct"] == 0]
        group_fp = group[group["ErrorType"] == "FP"]
        group_fn = group[group["ErrorType"] == "FN"]
        survival_rate = group["Survived"].mean()
        if len(group) > 0:
            err_rate = len(group_errors) / len(group)
            report.append(f"| {sex} Pclass={pclass} | {len(group)} | {len(group_errors)} | {err_rate:.1%} | {len(group_fp)} | {len(group_fn)} | {survival_rate:.1%} |")
report.append("")

# 3. Error rates by Title
report.append("## 3. Error Rates by Title")
report.append("")
report.append("| Title | Total | Errors | Error Rate | FP | FN |")
report.append("|:---|---:|---:|---:|---:|---:|")
for title in ["Mr", "Mrs", "Miss", "Master", "Rare"]:
    mask = analysis["Title"] == title
    group = analysis[mask]
    group_errors = group[group["Correct"] == 0]
    group_fp = group[group["ErrorType"] == "FP"]
    group_fn = group[group["ErrorType"] == "FN"]
    if len(group) > 0:
        err_rate = len(group_errors) / len(group)
        report.append(f"| {title} | {len(group)} | {len(group_errors)} | {err_rate:.1%} | {len(group_fp)} | {len(group_fn)} |")
report.append("")

# 4. Probability distribution of errors
report.append("## 4. Probability Distribution of Misclassified Passengers")
report.append("")
report.append("| Probability Range | Errors | % of Errors |")
report.append("|:---|---:|---:|")
prob_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in prob_bins:
    if hi == 1.0:
        mask = (misclassified["PredProb"] >= lo) & (misclassified["PredProb"] <= hi)
    else:
        mask = (misclassified["PredProb"] >= lo) & (misclassified["PredProb"] < hi)
    count = mask.sum()
    if count > 0:
        report.append(f"| [{lo:.1f}-{hi:.1f}) | {count} | {100*count/n_errors:.1f}% |")
report.append("")

near_boundary = misclassified[(misclassified["PredProb"] >= 0.3) & (misclassified["PredProb"] <= 0.7)]
confident_errors = misclassified[(misclassified["PredProb"] < 0.3) | (misclassified["PredProb"] > 0.7)]
report.append(f"- **Near boundary (0.3-0.7)**: {len(near_boundary)} ({100*len(near_boundary)/n_errors:.1f}%)")
report.append(f"- **Confident errors (<0.3 or >0.7)**: {len(confident_errors)} ({100*len(confident_errors)/n_errors:.1f}%)")
report.append("")

# 5. Most confident errors
report.append("## 5. Most Confident Errors (Top 20)")
report.append("")
report.append("| PID | True | Pred | Prob | Sex | Pcl | Age | Fare | Title | FamSz | Cabin | Emb |")
report.append("|---:|---:|---:|---:|:---|---:|---:|---:|:---|---:|:---|:---|")
confident_sorted = misclassified.copy()
confident_sorted["ErrorConfidence"] = np.where(
    confident_sorted["ErrorType"] == "FN",
    1 - confident_sorted["PredProb"],
    confident_sorted["PredProb"]
)
confident_sorted = confident_sorted.sort_values("ErrorConfidence", ascending=False)
for _, row in confident_sorted.head(20).iterrows():
    cabin = str(row["Cabin"])[:6] if pd.notna(row["Cabin"]) else "-"
    age = f"{row['Age']:.0f}" if pd.notna(row["Age"]) else "?"
    report.append(f"| {int(row['PassengerId'])} | {int(row['Survived'])} | {int(row['Predicted'])} | {row['PredProb']:.3f} | {row['Sex']} | {int(row['Pclass'])} | {age} | {row['Fare']:.2f} | {row['Title']} | {int(row['FamilySize'])} | {cabin} | {row['Embarked']} |")
report.append("")

# 6. Passengers barely correct (prob near 0.5)
report.append("## 6. Passengers Barely Correct (prob 0.35-0.65, correct)")
report.append("")
barely_correct = correct[(correct["PredProb"] >= 0.35) & (correct["PredProb"] <= 0.65)]
report.append(f"**Count**: {len(barely_correct)}")
report.append("")
report.append("| Sex x Pclass | Count |")
report.append("|:---|---:|")
for sex in ["male", "female"]:
    for pclass in [1, 2, 3]:
        mask = (barely_correct["Sex"] == sex) & (barely_correct["Pclass"] == pclass)
        count = mask.sum()
        if count > 0:
            report.append(f"| {sex} Pclass={pclass} | {count} |")
report.append("")

# 7. Error rate by fold
report.append("## 7. Error Rate by Fold")
report.append("")
report.append("| Fold | Errors | Total | Error Rate |")
report.append("|---:|---:|---:|---:|")
for fold in range(1, 6):
    fold_mask = analysis["Fold"] == fold
    fold_data = analysis[fold_mask]
    fold_errors = fold_data[fold_data["Correct"] == 0]
    report.append(f"| {fold} | {len(fold_errors)} | {len(fold_data)} | {100*len(fold_errors)/len(fold_data):.1f}% |")
report.append("")

# 8. FP analysis (predicted survived but died)
report.append("## 8. False Positive Analysis (Predicted Survived, Actually Died)")
report.append("")
report.append(f"**Total FPs**: {len(fp)}")
report.append("")
if len(fp) > 0:
    report.append(f"- By Sex: {fp['Sex'].value_counts().to_dict()}")
    report.append(f"- By Pclass: {fp['Pclass'].value_counts().to_dict()}")
    report.append(f"- By Title: {fp['Title'].value_counts().to_dict()}")
    report.append(f"- Median age: {fp['Age'].median():.1f}")
    report.append(f"- Median fare: {fp['Fare'].median():.2f}")
    report.append(f"- HasCabin rate: {fp['Cabin'].notna().mean():.2f}")
    report.append("")

# 9. FN analysis (predicted died but survived)
report.append("## 9. False Negative Analysis (Predicted Died, Actually Survived)")
report.append("")
report.append(f"**Total FNs**: {len(fn)}")
report.append("")
if len(fn) > 0:
    report.append(f"- By Sex: {fn['Sex'].value_counts().to_dict()}")
    report.append(f"- By Pclass: {fn['Pclass'].value_counts().to_dict()}")
    report.append(f"- By Title: {fn['Title'].value_counts().to_dict()}")
    report.append(f"- Median age: {fn['Age'].median():.1f}")
    report.append(f"- Median fare: {fn['Fare'].median():.2f}")
    report.append(f"- HasCabin rate: {fn['Cabin'].notna().mean():.2f}")
    report.append("")

# 10. Special subgroups
report.append("## 10. Special Subgroup Analysis")
report.append("")

# Women who died
women_died = analysis[(analysis["Sex"] == "female") & (analysis["Survived"] == 0)]
women_died_fp = women_died[women_died["Predicted"] == 1]
report.append(f"### Women who died: {len(women_died)}")
report.append(f"- Model predicted survived (FP): {len(women_died_fp)} ({100*len(women_died_fp)/len(women_died):.1f}%)")
report.append(f"- By Pclass: {women_died['Pclass'].value_counts().to_dict()}")
report.append("")

# Men who survived
men_survived = analysis[(analysis["Sex"] == "male") & (analysis["Survived"] == 1)]
men_survived_fn = men_survived[men_survived["Predicted"] == 0]
report.append(f"### Men who survived: {len(men_survived)}")
report.append(f"- Model predicted died (FN): {len(men_survived_fn)} ({100*len(men_survived_fn)/len(men_survived):.1f}%)")
report.append(f"- By Pclass: {men_survived['Pclass'].value_counts().to_dict()}")
report.append("")

# 3rd class women
women_3rd = analysis[(analysis["Sex"] == "female") & (analysis["Pclass"] == 3)]
women_3rd_errors = women_3rd[women_3rd["Correct"] == 0]
report.append(f"### 3rd Class Women: {len(women_3rd)}")
report.append(f"- Survival rate: {women_3rd['Survived'].mean():.1%}")
report.append(f"- Error rate: {len(women_3rd_errors)/len(women_3rd):.1%}")
report.append(f"- FP: {len(women_3rd[women_3rd['ErrorType']=='FP'])}, FN: {len(women_3rd[women_3rd['ErrorType']=='FN'])}")
report.append("")

# 1st/2nd class men
men_upper = analysis[(analysis["Sex"] == "male") & (analysis["Pclass"].isin([1, 2]))]
men_upper_errors = men_upper[men_upper["Correct"] == 0]
report.append(f"### 1st/2nd Class Men: {len(men_upper)}")
report.append(f"- Survival rate: {men_upper['Survived'].mean():.1%}")
report.append(f"- Error rate: {len(men_upper_errors)/len(men_upper):.1%}")
report.append(f"- FP: {len(men_upper[men_upper['ErrorType']=='FP'])}, FN: {len(men_upper[men_upper['ErrorType']=='FN'])}")
report.append("")

# Children
children = analysis[analysis["Age"] <= 12]
children_errors = children[children["Correct"] == 0]
report.append(f"### Children (age <= 12): {len(children)}")
report.append(f"- Survival rate: {children['Survived'].mean():.1%}")
report.append(f"- Error rate: {len(children_errors)/len(children):.1%}" if len(children) > 0 else "- N/A")
report.append(f"- FP: {len(children[children['ErrorType']=='FP'])}, FN: {len(children[children['ErrorType']=='FN'])}")
report.append("")

# 11. Family size
report.append("## 11. Error Rate by Family Size")
report.append("")
report.append("| FamSize | Total | Errors | Error Rate | Survival Rate |")
report.append("|---:|---:|---:|---:|---:|")
for fam_size in sorted(analysis["FamilySize"].unique()):
    mask = analysis["FamilySize"] == fam_size
    group = analysis[mask]
    group_errors = group[group["Correct"] == 0]
    if len(group) >= 5:
        report.append(f"| {fam_size} | {len(group)} | {len(group_errors)} | {100*len(group_errors)/len(group):.1f}% | {group['Survived'].mean():.1%} |")
report.append("")

# 12. Fare quartile
report.append("## 12. Error Rate by Fare Quartile")
report.append("")
analysis["FareQ"] = pd.qcut(analysis["Fare"], 4, labels=["Q1(low)", "Q2", "Q3", "Q4(high)"])
report.append("| Quartile | Fare Range | Errors | Total | Error Rate |")
report.append("|:---|:---|---:|---:|---:|")
for q in ["Q1(low)", "Q2", "Q3", "Q4(high)"]:
    mask = analysis["FareQ"] == q
    group = analysis[mask]
    group_errors = group[group["Correct"] == 0]
    if len(group) > 0:
        fare_range = f"[{group['Fare'].min():.1f}-{group['Fare'].max():.1f}]"
        report.append(f"| {q} | {fare_range} | {len(group_errors)} | {len(group)} | {100*len(group_errors)/len(group):.1f}% |")
report.append("")

# 13. All FP women
report.append("## 13. All False Positive Women (Predicted Survived, Actually Died)")
report.append("")
fp_women = fp[fp["Sex"] == "female"].sort_values("PredProb", ascending=False)
if len(fp_women) > 0:
    report.append("| PID | Prob | Pcl | Age | Fare | Title | FamSz | Cabin | Emb | Name |")
    report.append("|---:|---:|---:|---:|---:|:---|---:|:---|:---|:---|")
    for _, row in fp_women.iterrows():
        cabin = str(row["Cabin"])[:6] if pd.notna(row["Cabin"]) else "-"
        age = f"{row['Age']:.0f}" if pd.notna(row["Age"]) else "?"
        name = str(row["Name"])[:40]
        report.append(f"| {int(row['PassengerId'])} | {row['PredProb']:.3f} | {int(row['Pclass'])} | {age} | {row['Fare']:.2f} | {row['Title']} | {int(row['FamilySize'])} | {cabin} | {row['Embarked']} | {name} |")
report.append("")

# 14. All FN men
report.append("## 14. All False Negative Men (Predicted Died, Actually Survived)")
report.append("")
fn_men = fn[fn["Sex"] == "male"].sort_values("PredProb", ascending=True)
if len(fn_men) > 0:
    report.append("| PID | Prob | Pcl | Age | Fare | Title | FamSz | Cabin | Emb | Name |")
    report.append("|---:|---:|---:|---:|---:|:---|---:|:---|:---|:---|")
    for _, row in fn_men.iterrows():
        cabin = str(row["Cabin"])[:6] if pd.notna(row["Cabin"]) else "-"
        age = f"{row['Age']:.0f}" if pd.notna(row["Age"]) else "?"
        name = str(row["Name"])[:40]
        report.append(f"| {int(row['PassengerId'])} | {row['PredProb']:.3f} | {int(row['Pclass'])} | {age} | {row['Fare']:.2f} | {row['Title']} | {int(row['FamilySize'])} | {cabin} | {row['Embarked']} | {name} |")
report.append("")

# 15. Actionable summary
report.append("## 15. Actionable Findings Summary")
report.append("")
report.append(f"**Total errors**: {n_errors}/{n_total} (FP: {len(fp)}, FN: {len(fn)})")
report.append("")

subgroup_errors = []
for sex in ["male", "female"]:
    for pclass in [1, 2, 3]:
        mask = (analysis["Sex"] == sex) & (analysis["Pclass"] == pclass)
        group = analysis[mask]
        group_errors = group[group["Correct"] == 0]
        if len(group) > 0:
            subgroup_errors.append({
                "group": f"{sex}_Pclass{pclass}",
                "n_errors": len(group_errors),
                "n_total": len(group),
                "error_rate": len(group_errors)/len(group),
                "pct_of_all_errors": len(group_errors)/n_errors
            })

subgroup_errors.sort(key=lambda x: x["n_errors"], reverse=True)
report.append("### Error Concentration by Sex x Pclass")
report.append("")
report.append("| Group | Errors | Error Rate | % of All Errors |")
report.append("|:---|---:|---:|---:|")
for sg in subgroup_errors:
    report.append(f"| {sg['group']} | {sg['n_errors']} | {sg['error_rate']:.1%} | {sg['pct_of_all_errors']:.1%} |")
report.append("")

report.append(f"**Largest error source**: {subgroup_errors[0]['group']} with {subgroup_errors[0]['n_errors']} errors")
report.append(f"If we could fix all errors in this group: +{100*subgroup_errors[0]['n_errors']/n_total:.1f}pp accuracy")
report.append("")

# Swing passengers
swing = analysis[(analysis["PredProb"] >= 0.4) & (analysis["PredProb"] <= 0.6)]
swing_correct = swing[swing["Correct"] == 1]
swing_wrong = swing[swing["Correct"] == 0]
report.append("### Swing Passengers (prob 0.4-0.6)")
report.append("")
report.append(f"- Total: {len(swing)}")
report.append(f"- Correct: {len(swing_correct)}")
report.append(f"- Wrong: {len(swing_wrong)}")
if len(swing) > 0:
    report.append(f"- Accuracy in swing zone: {len(swing_correct)/len(swing):.1%}")
report.append("")
report.append("| Sex x Pclass | Count | Accuracy |")
report.append("|:---|---:|---:|")
for sex in ["male", "female"]:
    for pclass in [1, 2, 3]:
        mask = (swing["Sex"] == sex) & (swing["Pclass"] == pclass)
        count = mask.sum()
        if count > 0:
            swing_sub = swing[mask]
            report.append(f"| {sex} Pclass={pclass} | {count} | {swing_sub['Correct'].mean():.1%} |")
report.append("")

# Write report
report_text = "\n".join(report)
report_path = f"{OUT_DIR}/error_analysis_report.md"
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\nReport written to: {report_path}")
print(f"Report length: {len(report_text)} characters")
