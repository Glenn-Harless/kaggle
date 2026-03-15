"""
Titanic Model 18: Ticket Group Survival Propagation

Uses training survival labels from ticket-mates as features/rules.
This is standard supervised learning — not target leakage.

The key insight: families and travel parties survived or died together.
If 3 out of 4 members of a ticket group survived (in training), the 4th
(in test) likely survived too.

Two approaches tested:
  18a: Add TicketSurvRate as a feature in the logistic model (OOF encoded)
  18b: Hybrid — override the logistic prediction with a rule when
       ticket-mates have unanimous outcomes (all survived or all died)

Out-of-fold encoding during CV prevents score inflation.
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
    reconstruct_v2_features, flip_analysis, report_flips,
    paired_comparison, report_paired,
)

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# ---- Load data ----
train_v3 = pd.read_csv(f"{BASE}/data/train_processed.csv")
test_v3 = pd.read_csv(f"{BASE}/data/test_processed.csv")
raw_train = pd.read_csv(f"{BASE}/data/train.csv")
raw_test = pd.read_csv(f"{BASE}/data/test.csv")

train_v2 = reconstruct_v2_features(train_v3)
test_v2 = reconstruct_v2_features(test_v3)

y = train_v2["Survived"]
X_v2 = train_v2.drop(columns=["Survived"])
test_ids = test_v2["PassengerId"]
X_test_v2 = test_v2.drop(columns=["PassengerId"])

baseline_cv = np.load(f"{BASE}/results/models/13_v2_baseline_cv_scores.npy")

train_tickets = raw_train["Ticket"].values
test_tickets = raw_test["Ticket"].values

# Full manifest ticket group sizes
all_tickets = pd.concat([raw_train["Ticket"], raw_test["Ticket"]], ignore_index=True)
ticket_counts = all_tickets.value_counts()
train_tgs = raw_train["Ticket"].map(ticket_counts).values
test_tgs = raw_test["Ticket"].map(ticket_counts).values

# Pre-compute ticket survival stats from full training set (for test predictions)
train_surv_by_ticket = raw_train.groupby("Ticket")["Survived"].agg(["mean", "count"])

print("=" * 60)
print("STEP 18: TICKET GROUP SURVIVAL PROPAGATION")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print()


# ================================================================
# PHASE 1: DIAGNOSTICS
# ================================================================
print("--- Diagnostics: Test passengers with ticket-mates in training ---")

train_ticket_set = set(raw_train["Ticket"])
test_has_mate = raw_test["Ticket"].isin(train_ticket_set)
n_hint = test_has_mate.sum()
print(f"  Can receive survival hint: {n_hint} / {len(raw_test)}")
print(f"  No ticket-mates in train:  {(~test_has_mate).sum()} / {len(raw_test)}")
print()

# Unanimous cases among test passengers with mates
test_mates = raw_test[test_has_mate].copy()
test_mates["mate_rate"] = test_mates["Ticket"].map(train_surv_by_ticket["mean"])
test_mates["mate_count"] = test_mates["Ticket"].map(train_surv_by_ticket["count"])

# Require 2+ mates for a meaningful signal
strong = test_mates[test_mates["mate_count"] >= 2]
all_surv = strong[strong["mate_rate"] == 1.0]
all_died = strong[strong["mate_rate"] == 0.0]
mixed = strong[(strong["mate_rate"] > 0) & (strong["mate_rate"] < 1)]

print(f"--- Unanimous cases (2+ ticket-mates in training) ---")
print(f"  All mates survived (rate=1.0): {len(all_surv)} test passengers")
print(f"  All mates died (rate=0.0):     {len(all_died)} test passengers")
print(f"  Mixed outcomes:                {len(mixed)} test passengers")
print(f"  Only 1 mate (weak signal):     {(test_mates['mate_count'] == 1).sum()} test passengers")
print()

if len(all_surv) > 0:
    print("  Unanimous SURVIVED:")
    for _, row in all_surv.iterrows():
        print(f"    PID {int(row['PassengerId'])}: {row['Sex']:>6s} Pc{int(row['Pclass'])} "
              f"ticket={row['Ticket']} ({int(row['mate_count'])} mates)")
    print()

if len(all_died) > 0:
    print(f"  Unanimous DIED (first 15):")
    for _, row in all_died.head(15).iterrows():
        print(f"    PID {int(row['PassengerId'])}: {row['Sex']:>6s} Pc{int(row['Pclass'])} "
              f"ticket={row['Ticket']} ({int(row['mate_count'])} mates)")
    print()

# What does the baseline predict for these passengers?
baseline_sub = pd.read_csv(f"{BASE}/submissions/logreg_v2.csv")
baseline_preds = baseline_sub.set_index("PassengerId")["Survived"]

if len(all_surv) > 0:
    surv_ids = all_surv["PassengerId"].values
    bl_says_died = sum(baseline_preds.loc[pid] == 0 for pid in surv_ids)
    print(f"  Of {len(all_surv)} 'all mates survived': baseline predicts {bl_says_died} as DIED")
    print(f"  → These are the passengers a rule could FLIP to survived")

if len(all_died) > 0:
    died_ids = all_died["PassengerId"].values
    bl_says_surv = sum(baseline_preds.loc[pid] == 1 for pid in died_ids)
    print(f"  Of {len(all_died)} 'all mates died': baseline predicts {bl_says_surv} as SURVIVED")
    print(f"  → These are the passengers a rule could FLIP to died")
print()


# ================================================================
# PHASE 2: CUSTOM REPEATED CV
# ================================================================
print("=" * 60)
print("PHASE 2: CUSTOM REPEATED CV (10x5-fold)")
print("=" * 60)
print()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_splits = list(cv.split(X_v2, y))

C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def compute_ticket_surv_hint(source_tickets, source_y, target_tickets):
    """Compute mean survival of ticket-mates from source data."""
    stats = pd.DataFrame({"Ticket": source_tickets, "Survived": source_y})
    ticket_means = stats.groupby("Ticket")["Survived"].mean()
    global_mean = source_y.mean()
    return pd.Series(target_tickets).map(ticket_means).fillna(global_mean).values


# ---- 18a: TicketSurvRate as feature ----
print("--- 18a: v2 + TicketSurvRate (OOF) ---")
print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs BL':>7}")
print(f"  {'-' * 35}")

best_a = {"cv_mean": 0}
for C in C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v2.iloc[train_idx].copy()
        X_vl = X_v2.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        # OOF: compute hint from training fold only
        X_tr["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets[train_idx], y_tr.values, train_tickets[train_idx])
        X_vl["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets[train_idx], y_tr.values, train_tickets[val_idx])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        fold_scores.append(accuracy_score(y_vl, pipe.predict(X_vl)))

    scores = np.array(fold_scores)
    delta = scores.mean() - baseline_cv.mean()
    marker = " ***" if delta > 0.002 else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+7.4f}{marker}")
    if scores.mean() > best_a["cv_mean"]:
        best_a = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print(f"  Best: C={best_a['C']}, CV={best_a['cv_mean']:.4f}")
print()


# ---- 18a+TGS: TicketSurvRate + TicketGroupSize ----
print("--- 18a+TGS: v2 + TicketSurvRate + TicketGroupSize ---")
print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs BL':>7}")
print(f"  {'-' * 35}")

best_a_tgs = {"cv_mean": 0}
for C in C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v2.iloc[train_idx].copy()
        X_vl = X_v2.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        X_tr["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets[train_idx], y_tr.values, train_tickets[train_idx])
        X_vl["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets[train_idx], y_tr.values, train_tickets[val_idx])
        X_tr["TicketGroupSize"] = train_tgs[train_idx]
        X_vl["TicketGroupSize"] = train_tgs[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        fold_scores.append(accuracy_score(y_vl, pipe.predict(X_vl)))

    scores = np.array(fold_scores)
    delta = scores.mean() - baseline_cv.mean()
    marker = " ***" if delta > 0.002 else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+7.4f}{marker}")
    if scores.mean() > best_a_tgs["cv_mean"]:
        best_a_tgs = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print(f"  Best: C={best_a_tgs['C']}, CV={best_a_tgs['cv_mean']:.4f}")
print()


# ---- 18b: Hybrid rules + logistic ----
print("--- 18b: Hybrid Rules + Logistic ---")
print("  Rule: override when 2+ ticket-mates have unanimous outcome")
print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs BL':>7}")
print(f"  {'-' * 35}")

best_b = {"cv_mean": 0}
for C in C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v2.iloc[train_idx].copy()
        X_vl = X_v2.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_vl)

        # Override with ticket-mate rules
        tr_tickets = train_tickets[train_idx]
        vl_tickets = train_tickets[val_idx]
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
    marker = " ***" if delta > 0.002 else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+7.4f}{marker}")
    if scores.mean() > best_b["cv_mean"]:
        best_b = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print(f"  Best: C={best_b['C']}, CV={best_b['cv_mean']:.4f}")
print()


# ---- 18b-loose: Hybrid with 1+ mates (looser threshold) ----
print("--- 18b-loose: Hybrid Rules (1+ mates) + Logistic ---")
print("  Rule: override when 1+ ticket-mates have unanimous outcome")
print(f"  {'C':>6}  {'CV Mean':>8}  {'CV Std':>7}  {'vs BL':>7}")
print(f"  {'-' * 35}")

best_b_loose = {"cv_mean": 0}
for C in C_values:
    fold_scores = []
    for train_idx, val_idx in cv_splits:
        X_tr = X_v2.iloc[train_idx].copy()
        X_vl = X_v2.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_vl)

        tr_tickets = train_tickets[train_idx]
        tr_df = pd.DataFrame({"Ticket": tr_tickets, "Survived": y_tr.values})
        ticket_stats = tr_df.groupby("Ticket")["Survived"].agg(["mean", "count"])

        for i, vi in enumerate(val_idx):
            ticket = train_tickets[vi]
            if ticket in ticket_stats.index:
                rate = ticket_stats.loc[ticket, "mean"]
                count = ticket_stats.loc[ticket, "count"]
                if count >= 1 and rate == 1.0:
                    preds[i] = 1
                elif count >= 1 and rate == 0.0:
                    preds[i] = 0

        fold_scores.append(accuracy_score(y_vl, preds))

    scores = np.array(fold_scores)
    delta = scores.mean() - baseline_cv.mean()
    marker = " ***" if delta > 0.002 else ""
    print(f"  {C:>6.2f}  {scores.mean():>8.4f}  {scores.std():>7.4f}  {delta:>+7.4f}{marker}")
    if scores.mean() > best_b_loose["cv_mean"]:
        best_b_loose = {"C": C, "cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}

print(f"  Best: C={best_b_loose['C']}, CV={best_b_loose['cv_mean']:.4f}")
print()


# ================================================================
# PHASE 3: TEST PREDICTIONS + FLIP ANALYSIS FOR TOP CANDIDATES
# ================================================================
print("=" * 60)
print("PHASE 3: TEST PREDICTIONS AND FLIP ANALYSIS")
print("=" * 60)

candidates = [
    ("18a_SurvRate", best_a, "feature"),
    ("18a_SurvRate_TGS", best_a_tgs, "feature_tgs"),
    ("18b_Hybrid_2mate", best_b, "hybrid_2"),
    ("18b_Hybrid_1mate", best_b_loose, "hybrid_1"),
]

for label, cfg, mode in candidates:
    print(f"\n--- {label} (C={cfg['C']}, CV={cfg['cv_mean']:.4f}) ---")

    # Paired comparison
    comp = paired_comparison(baseline_cv, cfg["cv_scores"])
    report_paired(comp)
    print()

    # Generate test predictions
    X_tr_final = X_v2.copy()
    X_te_final = X_test_v2.copy()

    if mode.startswith("feature"):
        # Add survival hint as feature
        X_tr_final["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets, y.values, train_tickets)
        X_te_final["TicketSurvRate"] = compute_ticket_surv_hint(
            train_tickets, y.values, test_tickets)
        if mode == "feature_tgs":
            X_tr_final["TicketGroupSize"] = train_tgs
            X_te_final["TicketGroupSize"] = test_tgs

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=cfg["C"], max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr_final, y)
        test_preds = pipe.predict(X_te_final)
        test_proba = pipe.predict_proba(X_te_final)[:, 1]

    else:
        # Hybrid: logistic + rule override
        min_mates = 2 if mode == "hybrid_2" else 1
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=cfg["C"], max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr_final, y)
        test_preds = pipe.predict(X_te_final)
        test_proba = pipe.predict_proba(X_te_final)[:, 1]

        n_overrides = 0
        for i in range(len(test_preds)):
            ticket = test_tickets[i]
            if ticket in train_surv_by_ticket.index:
                rate = train_surv_by_ticket.loc[ticket, "mean"]
                count = train_surv_by_ticket.loc[ticket, "count"]
                if count >= min_mates and rate == 1.0:
                    test_preds[i] = 1
                    test_proba[i] = 0.95
                    n_overrides += 1
                elif count >= min_mates and rate == 0.0:
                    test_preds[i] = 0
                    test_proba[i] = 0.05
                    n_overrides += 1
        print(f"  Rule overrides on test: {n_overrides} passengers")

    # Flip analysis
    flips = flip_analysis(
        test_preds, test_proba,
        f"{BASE}/submissions/logreg_v2.csv",
        test_ids, raw_test, target_subgroup=None,
    )
    report_flips(flips)

    # Save submission
    sub_path = f"{BASE}/submissions/logreg_18_{label.lower()}.csv"
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": test_preds})
    sub.to_csv(sub_path, index=False)
    print(f"\n  Saved: {sub_path}")
    print(f"  Survival rate: {test_preds.mean():.3f} ({int(test_preds.sum())}/{len(test_preds)})")


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print()
print(f"{'Model':<30} {'CV Mean':>8} {'CV Std':>7} {'Delta':>7} {'W/L/T':>10}")
print("-" * 65)
print(f"{'v2 baseline (C=0.01)':<30} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'--':>10}")

for label, cfg, _ in candidates:
    comp = paired_comparison(baseline_cv, cfg["cv_scores"])
    d = comp["mean_delta"]
    w, l, t = comp["n_candidate_wins"], comp["n_baseline_wins"], comp["n_ties"]
    print(f"{label + ' C=' + str(cfg['C']):<30} {cfg['cv_mean']:>8.4f} {cfg['cv_std']:>7.4f} {d:>+7.4f} {f'{w}/{l}/{t}':>10}")

print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")
