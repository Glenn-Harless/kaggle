"""
Titanic Model 21: v3 Base + Ticket Rules

Tests whether the v3 feature set (21 features: one-hot Pclass, Sex×Pclass
interactions, grouped deck) improves over v2 when combined with ticket-mate
survival rules (the 18b hybrid approach).

Hypothesis: v3 interaction features capture the non-linear class-gender
pattern better than v2's ordinal Pclass. v3 + ticket rules may outperform
v2 + ticket rules (current best: 0.7751 on Kaggle).

Prior evidence: Step 11 found v3 logistic matched v2 CV (0.8305) and
changed 4 test passengers. Expected test difference vs 18b: 0-4 passengers.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import (
    Tee, repeated_cv, report_cv, paired_comparison, report_paired,
    flip_analysis, report_flips, reconstruct_v2_features,
)

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# Tee output to log file
tee = Tee(f"{BASE}/results/models/21_v3_ticket_rules.txt")
sys.stdout = tee

# ---- Load data ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_train = pd.read_csv(f"{BASE}/data/train.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

# v3 features (primary model)
y = train_v3["Survived"]
X_v3 = train_v3.drop(columns=["Survived"])
test_ids = test_v3["PassengerId"]
X_test_v3 = test_v3.drop(columns=["PassengerId"])

# v2 features (for inline comparison only)
train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)
X_v2 = train_v2.drop(columns=["Survived"])
X_test_v2 = test_v2.drop(columns=["PassengerId"])

# Baseline CV scores
baseline_cv = np.load(f"{BASE}/results/models/13_v2_baseline_cv_scores.npy")

# Ticket data
train_tickets = raw_train["Ticket"].values
test_tickets = raw_test["Ticket"].values
train_surv_by_ticket = raw_train.groupby("Ticket")["Survived"].agg(["mean", "count"])

print("=" * 60)
print("STEP 21: v3 BASE + TICKET RULES")
print("=" * 60)
print(f"v3 features ({X_v3.shape[1]}): {list(X_v3.columns)}")
print(f"v2 features ({X_v2.shape[1]}): {list(X_v2.columns)}")
print(f"v2 baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print()


# ================================================================
# PHASE 1: v3 BASELINE REPEATED CV
# ================================================================
print("=" * 60)
print("PHASE 1: v3 BASELINE REPEATED CV (C=0.01)")
print("=" * 60)
print()

pipe_v3 = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42)),
])

v3_baseline_cv = repeated_cv(pipe_v3, X_v3, y)
print("--- v3 Baseline (C=0.01) ---")
report_cv(v3_baseline_cv, "v3_base")
print()

# Paired comparison vs v2 baseline
print("--- Paired: v3 baseline vs v2 baseline ---")
comp_v3_v2 = paired_comparison(baseline_cv, v3_baseline_cv)
report_paired(comp_v3_v2, "v2_base", "v3_base")
print()

# Save v3 baseline CV scores
np.save(f"{BASE}/results/models/21_v3_baseline_cv_scores.npy", v3_baseline_cv)
print(f"Saved: results/models/21_v3_baseline_cv_scores.npy")
print()


# ================================================================
# PHASE 2: C SWEEP ON v3
# ================================================================
print("=" * 60)
print("PHASE 2: C SWEEP ON v3")
print("=" * 60)
print()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_splits = list(cv.split(X_v3, y))

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs v2 BL':>9}")
print(f"  {'-' * 38}")

best_C = {"C": 0.01, "cv_mean": 0, "cv_scores": None}
for C in C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v3.iloc[train_idx]
        X_vl = X_v3.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        fold_scores.append(accuracy_score(y_vl, pipe.predict(X_vl)))

    scores = np.array(fold_scores)
    delta = scores.mean() - baseline_cv.mean()
    marker = " <-- best" if scores.mean() > best_C["cv_mean"] else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+9.4f}{marker}")
    if scores.mean() > best_C["cv_mean"]:
        best_C = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print()
print(f"  Best C for v3: {best_C['C']}, CV={best_C['cv_mean']:.4f}")
print()


# ================================================================
# PHASE 3: v3 + TICKET RULES (HYBRID CV)
# ================================================================
print("=" * 60)
print("PHASE 3: v3 + TICKET RULES (HYBRID CV)")
print("=" * 60)
print()

# Determine which C values to test for hybrid
hybrid_C_values = [0.01]
if best_C["C"] != 0.01:
    hybrid_C_values.append(best_C["C"])

# ---- v3 + ticket rules ----
print("--- v3 + ticket rules (2+ mates) ---")
print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs v2 BL':>9}")
print(f"  {'-' * 38}")

best_v3_hybrid = {"C": 0.01, "cv_mean": 0, "cv_scores": None}
for C in hybrid_C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v3.iloc[train_idx]
        X_vl = X_v3.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_vl)

        # Ticket rule overrides (exact 18b logic)
        tr_tickets = train_tickets[train_idx]
        tr_df = pd.DataFrame({"Ticket": tr_tickets, "Survived": y_tr.values})
        ticket_stats = tr_df.groupby("Ticket")["Survived"].agg(["mean", "count"])

        for i, vi in enumerate(val_idx):
            ticket = train_tickets[vi]
            if ticket in ticket_stats.index:
                rate = ticket_stats.loc[ticket, "mean"]
                count = ticket_stats.loc[ticket, "count"]
                if count >= 2 and rate == 1.0:
                    preds[i] = 1
                elif count >= 2 and rate == 0.0:
                    preds[i] = 0

        fold_scores.append(accuracy_score(y_vl, preds))

    scores = np.array(fold_scores)
    delta = scores.mean() - baseline_cv.mean()
    marker = " <-- best" if scores.mean() > best_v3_hybrid["cv_mean"] else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+9.4f}{marker}")
    if scores.mean() > best_v3_hybrid["cv_mean"]:
        best_v3_hybrid = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print()
print(f"  Best v3+rules: C={best_v3_hybrid['C']}, CV={best_v3_hybrid['cv_mean']:.4f}")
print()

# ---- v2 + ticket rules (18b inline, same splits for valid paired comparison) ----
print("--- v2 + ticket rules (18b inline, same splits) ---")
scores_v2_rules = []
for train_idx, val_idx in cv_splits:
    X_tr = X_v2.iloc[train_idx]
    X_vl = X_v2.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_vl)

    # Ticket rule overrides
    tr_tickets = train_tickets[train_idx]
    tr_df = pd.DataFrame({"Ticket": tr_tickets, "Survived": y_tr.values})
    ticket_stats = tr_df.groupby("Ticket")["Survived"].agg(["mean", "count"])

    for i, vi in enumerate(val_idx):
        ticket = train_tickets[vi]
        if ticket in ticket_stats.index:
            rate = ticket_stats.loc[ticket, "mean"]
            count = ticket_stats.loc[ticket, "count"]
            if count >= 2 and rate == 1.0:
                preds[i] = 1
            elif count >= 2 and rate == 0.0:
                preds[i] = 0

    scores_v2_rules.append(accuracy_score(y_vl, preds))

scores_v2_rules = np.array(scores_v2_rules)
delta_v2_rules = scores_v2_rules.mean() - baseline_cv.mean()
print(f"  CV Mean: {scores_v2_rules.mean():.4f} ± {scores_v2_rules.std():.4f}  (delta vs BL: {delta_v2_rules:+.4f})")
print()

# ---- KEY COMPARISON: v3+rules vs v2+rules ----
print("--- KEY COMPARISON: v3+rules vs v2+rules ---")
comp_hybrid = paired_comparison(scores_v2_rules, best_v3_hybrid["cv_scores"])
report_paired(comp_hybrid, "v2+rules", "v3+rules")
print()


# ================================================================
# PHASE 4: TEST PREDICTIONS + FLIP ANALYSIS
# ================================================================
print("=" * 60)
print("PHASE 4: TEST PREDICTIONS AND FLIP ANALYSIS")
print("=" * 60)
print()

# Fit final v3 logistic on full training set with best hybrid C
best_hybrid_C = best_v3_hybrid["C"]
print(f"Final model: v3 LogReg C={best_hybrid_C} + ticket rules (2+ mates)")
print()

pipe_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=best_hybrid_C, max_iter=2000, random_state=42)),
])
pipe_final.fit(X_v3, y)
base_test_preds = pipe_final.predict(X_test_v3)
base_test_proba = pipe_final.predict_proba(X_test_v3)[:, 1]

# Apply ticket rule overrides (exact 18b logic)
test_preds = base_test_preds.copy()
test_proba = base_test_proba.copy()
overrides = []

for i in range(len(test_preds)):
    ticket = test_tickets[i]
    if ticket in train_surv_by_ticket.index:
        rate = train_surv_by_ticket.loc[ticket, "mean"]
        count = train_surv_by_ticket.loc[ticket, "count"]
        if count >= 2 and rate == 1.0:
            old = test_preds[i]
            test_preds[i] = 1
            test_proba[i] = 0.95
            overrides.append({
                "PID": int(test_ids.iloc[i]),
                "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Ticket": ticket,
                "Direction": f"{old}->1" if old != 1 else "no_change",
            })
        elif count >= 2 and rate == 0.0:
            old = test_preds[i]
            test_preds[i] = 0
            test_proba[i] = 0.05
            overrides.append({
                "PID": int(test_ids.iloc[i]),
                "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Ticket": ticket,
                "Direction": f"{old}->0" if old != 0 else "no_change",
            })

print(f"Ticket rule overrides on test: {len(overrides)} passengers")
for o in overrides:
    print(f"  PID {o['PID']}: {o['Sex']:>6s} Pc{o['Pclass']} ticket={o['Ticket']} {o['Direction']}")
print()

# Flip analysis vs plain v2 baseline
print("--- Flip analysis vs logreg_v2.csv (plain v2 baseline) ---")
flips_vs_v2 = flip_analysis(test_preds, test_proba, f"{BASE}/submissions/logreg_v2.csv",
                             test_ids, raw_test)
report_flips(flips_vs_v2)
print()

# Flip analysis vs current best (18b)
print("--- Flip analysis vs logreg_18_18b_hybrid_2mate.csv (current best) ---")
flips_vs_18b = flip_analysis(test_preds, test_proba,
                              f"{BASE}/submissions/logreg_18_18b_hybrid_2mate.csv",
                              test_ids, raw_test)
report_flips(flips_vs_18b)
print()


# ================================================================
# PHASE 5: SAVE SUBMISSION + SUMMARY
# ================================================================
print("=" * 60)
print("PHASE 5: SAVE SUBMISSION + SUMMARY")
print("=" * 60)
print()

sub_path = f"{BASE}/submissions/logreg_21_v3_hybrid_2mate.csv"
sub = pd.DataFrame({"PassengerId": test_ids, "Survived": test_preds})
sub.to_csv(sub_path, index=False)
print(f"Saved: {sub_path}")
print(f"Survival rate: {test_preds.mean():.3f} ({int(test_preds.sum())}/{len(test_preds)})")
print()

# Summary table
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print()
print(f"{'Model':<35} {'CV Mean':>8} {'CV Std':>7} {'Δ v2 BL':>8} {'Δ v2+rules':>11} {'W/L/T':>13}")
print("-" * 85)
print(f"{'v2 baseline (C=0.01)':<35} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>8} {'--':>11} {'--':>13}")

summary_rows = [
    ("v3 baseline (C=0.01)", v3_baseline_cv, None),
    ("v2 + rules (18b inline)", scores_v2_rules, None),
    (f"v3 + rules (C={best_v3_hybrid['C']})", best_v3_hybrid["cv_scores"], scores_v2_rules),
]

for label, scores, compare_to in summary_rows:
    d_bl = scores.mean() - baseline_cv.mean()
    if compare_to is not None:
        d_rules = scores.mean() - compare_to.mean()
        comp = paired_comparison(compare_to, scores)
        w, l, t = comp["n_candidate_wins"], comp["n_baseline_wins"], comp["n_ties"]
        wlt = f"{w}/{l}/{t}"
    else:
        d_rules = scores.mean() - scores_v2_rules.mean() if scores_v2_rules is not None else 0
        wlt = "--"
    print(f"{label:<35} {scores.mean():>8.4f} {scores.std():>7.4f} {d_bl:>+8.4f} {d_rules:>+11.4f} {wlt:>13}")

print()

# Flip count vs 18b
n_diff_18b = flips_vs_18b["n_flips"]
print(f"Test differences vs 18b: {n_diff_18b} passengers (expected 0-4)")
if n_diff_18b > 10:
    print("*** WARNING: More than 10 test differences vs 18b — unexpected ***")
print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")

tee.close()
