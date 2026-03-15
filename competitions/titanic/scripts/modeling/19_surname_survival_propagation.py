"""
Titanic Model 19: Surname-Based Survival Propagation

Extends the hybrid-rule pattern from 18b by identifying family groups via
surname+Pclass in addition to ticket number. This catches families who
bought separate tickets but share a surname and passenger class.

Three approaches tested:
  19a: Surname+Pclass rules only (no ticket rules)
  19b: Ticket rules + surname rules (ticket priority) — primary candidate
  19c: Connected components (ticket ∪ surname+Pclass via union-find)

All approaches use the same v2 LogReg base (C=0.01) with hard rule overrides.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from collections import defaultdict

sys.path.insert(0, "/Users/glennharless/dev-brain/kaggle")
from shared.evaluate import (
    Tee, reconstruct_v2_features, flip_analysis, report_flips,
    paired_comparison, report_paired,
)

BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"

# Tee output to log file
tee = Tee(f"{BASE}/results/models/19_surname_propagation.txt")
sys.stdout = tee

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

# Pre-compute ticket survival stats from full training set (for test predictions)
train_surv_by_ticket = raw_train.groupby("Ticket")["Survived"].agg(["mean", "count"])

# ================================================================
# PHASE 0: SURNAME EXTRACTION + DIAGNOSTICS
# ================================================================
print("=" * 60)
print("STEP 19: SURNAME-BASED SURVIVAL PROPAGATION")
print("=" * 60)
print(f"Baseline: v2 LogReg (15 features, C=0.01)")
print(f"Baseline repeated CV: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")
print()

# Extract surnames
raw_train["Surname"] = raw_train["Name"].str.split(",").str[0].str.strip()
raw_test["Surname"] = raw_test["Name"].str.split(",").str[0].str.strip()

# Build SurnamePclass keys
raw_train["SurnamePclass"] = raw_train["Surname"] + "_Pc" + raw_train["Pclass"].astype(str)
raw_test["SurnamePclass"] = raw_test["Surname"] + "_Pc" + raw_test["Pclass"].astype(str)

train_sp_keys = raw_train["SurnamePclass"].values
test_sp_keys = raw_test["SurnamePclass"].values

# Pre-compute surname+Pclass survival stats from full training set
train_surv_by_sp = raw_train.groupby("SurnamePclass")["Survived"].agg(["mean", "count"])

print("--- Diagnostics: Surname+Pclass groups ---")
sp_counts = raw_train["SurnamePclass"].value_counts()
print(f"  Unique Surname+Pclass keys in training: {len(sp_counts)}")
print(f"  Keys with 2+ members: {(sp_counts >= 2).sum()}")
print(f"  Keys with 3+ members: {(sp_counts >= 3).sum()}")
print(f"  Keys with 5+ members: {(sp_counts >= 5).sum()}")
print(f"  Largest group: {sp_counts.iloc[0]} ({sp_counts.index[0]})")
print()

# Coverage: how many test passengers get surname-mates vs ticket-mates
train_sp_set = set(raw_train["SurnamePclass"])
train_ticket_set = set(raw_train["Ticket"])

test_has_ticket_mate = raw_test["Ticket"].isin(train_ticket_set)
test_has_surname_mate = raw_test["SurnamePclass"].isin(train_sp_set)
# Require 2+ mates for meaningful signal
test_has_strong_ticket = pd.Series(False, index=raw_test.index)
test_has_strong_surname = pd.Series(False, index=raw_test.index)

for idx, row in raw_test.iterrows():
    ticket = row["Ticket"]
    if ticket in train_surv_by_ticket.index:
        if train_surv_by_ticket.loc[ticket, "count"] >= 2:
            test_has_strong_ticket.loc[idx] = True
    sp_key = row["SurnamePclass"]
    if sp_key in train_surv_by_sp.index:
        if train_surv_by_sp.loc[sp_key, "count"] >= 2:
            test_has_strong_surname.loc[idx] = True

both = test_has_strong_ticket & test_has_strong_surname
surname_only = test_has_strong_surname & ~test_has_strong_ticket
ticket_only = test_has_strong_ticket & ~test_has_strong_surname

print("--- Coverage: Test passengers with 2+ mates in training ---")
print(f"  Ticket-mates (2+):       {test_has_strong_ticket.sum()} / {len(raw_test)}")
print(f"  Surname+Pclass mates (2+): {test_has_strong_surname.sum()} / {len(raw_test)}")
print(f"  Both ticket AND surname:   {both.sum()}")
print(f"  Surname-ONLY (new adds):   {surname_only.sum()}")
print(f"  Ticket-ONLY:               {ticket_only.sum()}")
print()

# Show the NEW surname-only passengers (these are the ones 19b could help with)
if surname_only.sum() > 0:
    print("--- NEW: Test passengers with surname mates but NO ticket mates ---")
    new_passengers = raw_test[surname_only].copy()
    new_passengers["mate_rate"] = new_passengers["SurnamePclass"].map(train_surv_by_sp["mean"])
    new_passengers["mate_count"] = new_passengers["SurnamePclass"].map(train_surv_by_sp["count"])
    unanimous = new_passengers[(new_passengers["mate_rate"].isin([0.0, 1.0])) & (new_passengers["mate_count"] >= 2)]
    print(f"  Total new passengers:    {len(new_passengers)}")
    print(f"  With unanimous mates:    {len(unanimous)}")
    if len(unanimous) > 0:
        for _, row in unanimous.iterrows():
            direction = "SURVIVED" if row["mate_rate"] == 1.0 else "DIED"
            print(f"    PID {int(row['PassengerId'])}: {row['Sex']:>6s} Pc{int(row['Pclass'])} "
                  f"surname={row['Surname']} ({int(row['mate_count'])} mates → {direction})")
    print()

# Baseline predictions for comparison
baseline_sub = pd.read_csv(f"{BASE}/submissions/logreg_v2.csv")
baseline_preds_map = baseline_sub.set_index("PassengerId")["Survived"]


# ================================================================
# PHASE 1: CUSTOM REPEATED CV (10x5-fold, C=0.01 only)
# ================================================================
print("=" * 60)
print("PHASE 1: CUSTOM REPEATED CV (10x5-fold, C=0.01)")
print("=" * 60)
print()

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_splits = list(cv.split(X_v2, y))
C = 0.01


# ---- 18b inline: Ticket rules only (for comparison) ----
print("--- 18b (inline): Ticket rules only (2+ mates) ---")
scores_18b = []
for train_idx, val_idx in cv_splits:
    X_tr = X_v2.iloc[train_idx]
    X_vl = X_v2.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
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

    scores_18b.append(accuracy_score(y_vl, preds))

scores_18b = np.array(scores_18b)
delta_18b = scores_18b.mean() - baseline_cv.mean()
print(f"  CV Mean: {scores_18b.mean():.4f} ± {scores_18b.std():.4f}  (delta vs BL: {delta_18b:+.4f})")
print()


# ---- 19a: Surname+Pclass rules only ----
print("--- 19a: Surname+Pclass rules only (2+ mates) ---")
scores_19a = []
for train_idx, val_idx in cv_splits:
    X_tr = X_v2.iloc[train_idx]
    X_vl = X_v2.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_vl)

    # Surname+Pclass rule overrides (leakage-safe: training fold only)
    tr_sp = train_sp_keys[train_idx]
    tr_surv = y_tr.values
    tr_df = pd.DataFrame({"SP": tr_sp, "Survived": tr_surv})
    sp_stats = tr_df.groupby("SP")["Survived"].agg(["mean", "count"])

    for i, vi in enumerate(val_idx):
        sp_key = train_sp_keys[vi]
        if sp_key in sp_stats.index:
            rate = sp_stats.loc[sp_key, "mean"]
            count = sp_stats.loc[sp_key, "count"]
            if count >= 2 and rate == 1.0:
                preds[i] = 1
            elif count >= 2 and rate == 0.0:
                preds[i] = 0

    scores_19a.append(accuracy_score(y_vl, preds))

scores_19a = np.array(scores_19a)
delta_19a = scores_19a.mean() - baseline_cv.mean()
print(f"  CV Mean: {scores_19a.mean():.4f} ± {scores_19a.std():.4f}  (delta vs BL: {delta_19a:+.4f})")
comp_19a = paired_comparison(scores_18b, scores_19a)
print(f"  vs 18b: {comp_19a['mean_delta']:+.4f}  W/L/T: {comp_19a['n_candidate_wins']}/{comp_19a['n_baseline_wins']}/{comp_19a['n_ties']}")
print()


# ---- 19b: Ticket + Surname (ticket priority) ----
print("--- 19b: Ticket + Surname rules (ticket priority, 2+ mates) ---")
scores_19b = []
override_counts_19b = {"ticket": 0, "surname": 0, "total_val": 0}
for train_idx, val_idx in cv_splits:
    X_tr = X_v2.iloc[train_idx]
    X_vl = X_v2.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_vl)

    # Ticket stats (training fold only)
    tr_tickets_fold = train_tickets[train_idx]
    tr_df_ticket = pd.DataFrame({"Ticket": tr_tickets_fold, "Survived": y_tr.values})
    ticket_stats = tr_df_ticket.groupby("Ticket")["Survived"].agg(["mean", "count"])

    # Surname stats (training fold only)
    tr_sp_fold = train_sp_keys[train_idx]
    tr_df_sp = pd.DataFrame({"SP": tr_sp_fold, "Survived": y_tr.values})
    sp_stats = tr_df_sp.groupby("SP")["Survived"].agg(["mean", "count"])

    for i, vi in enumerate(val_idx):
        overridden = False

        # Ticket rule first
        ticket = train_tickets[vi]
        if ticket in ticket_stats.index:
            rate = ticket_stats.loc[ticket, "mean"]
            count = ticket_stats.loc[ticket, "count"]
            if count >= 2 and rate in (0.0, 1.0):
                preds[i] = int(rate)
                overridden = True
                override_counts_19b["ticket"] += 1

        # Surname rule second (only if ticket didn't fire)
        if not overridden:
            sp_key = train_sp_keys[vi]
            if sp_key in sp_stats.index:
                rate = sp_stats.loc[sp_key, "mean"]
                count = sp_stats.loc[sp_key, "count"]
                if count >= 2 and rate in (0.0, 1.0):
                    preds[i] = int(rate)
                    override_counts_19b["surname"] += 1

    override_counts_19b["total_val"] += len(val_idx)
    scores_19b.append(accuracy_score(y_vl, preds))

scores_19b = np.array(scores_19b)
delta_19b = scores_19b.mean() - baseline_cv.mean()
print(f"  CV Mean: {scores_19b.mean():.4f} ± {scores_19b.std():.4f}  (delta vs BL: {delta_19b:+.4f})")
comp_19b_vs_18b = paired_comparison(scores_18b, scores_19b)
print(f"  vs 18b: {comp_19b_vs_18b['mean_delta']:+.4f}  W/L/T: {comp_19b_vs_18b['n_candidate_wins']}/{comp_19b_vs_18b['n_baseline_wins']}/{comp_19b_vs_18b['n_ties']}")
print(f"  Overrides across all folds: ticket={override_counts_19b['ticket']}, "
      f"surname={override_counts_19b['surname']}, "
      f"total_val={override_counts_19b['total_val']}")
print()


# ---- 19c: Connected components (union-find) ----
print("--- 19c: Connected components (ticket ∪ surname+Pclass) ---")
scores_19c = []
component_size_log = []

for train_idx, val_idx in cv_splits:
    X_tr = X_v2.iloc[train_idx]
    X_vl = X_v2.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_vl)

    # Build union-find from training fold only
    n_tr = len(train_idx)
    parent = list(range(n_tr))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Index passengers by ticket and surname+Pclass
    ticket_to_idxs = defaultdict(list)
    sp_to_idxs = defaultdict(list)
    for local_i, global_i in enumerate(train_idx):
        ticket_to_idxs[train_tickets[global_i]].append(local_i)
        sp_to_idxs[train_sp_keys[global_i]].append(local_i)

    # Union passengers sharing a ticket
    for idxs in ticket_to_idxs.values():
        for j in range(1, len(idxs)):
            union(idxs[0], idxs[j])

    # Union passengers sharing surname+Pclass
    for idxs in sp_to_idxs.values():
        for j in range(1, len(idxs)):
            union(idxs[0], idxs[j])

    # Compute component survival stats
    comp_members = defaultdict(list)
    comp_survival = defaultdict(list)
    for local_i, global_i in enumerate(train_idx):
        root = find(local_i)
        comp_members[root].append(global_i)
        comp_survival[root].append(y.iloc[global_i])

    comp_stats = {}  # root -> (mean, count)
    for root, survivals in comp_survival.items():
        arr = np.array(survivals)
        comp_stats[root] = (arr.mean(), len(arr))

    # Track component sizes for diagnostics (first fold only)
    if len(component_size_log) == 0:
        sizes = [len(v) for v in comp_members.values() if len(v) >= 2]
        component_size_log = sorted(sizes, reverse=True)

    # Build reverse lookup: ticket -> root, sp_key -> root
    ticket_to_root = {}
    for local_i, global_i in enumerate(train_idx):
        root = find(local_i)
        ticket_to_root[train_tickets[global_i]] = root
        # sp_key -> root (same root since they're unioned)

    sp_to_root = {}
    for local_i, global_i in enumerate(train_idx):
        root = find(local_i)
        sp_to_root[train_sp_keys[global_i]] = root

    # Override validation passengers
    for i, vi in enumerate(val_idx):
        # Find which component this val passenger would join
        ticket = train_tickets[vi]
        sp_key = train_sp_keys[vi]

        # Try ticket first, then surname
        root = ticket_to_root.get(ticket)
        if root is None:
            root = sp_to_root.get(sp_key)

        if root is not None:
            rate, count = comp_stats[root]
            if count >= 2 and rate in (0.0, 1.0):
                preds[i] = int(rate)

    scores_19c.append(accuracy_score(y_vl, preds))

scores_19c = np.array(scores_19c)
delta_19c = scores_19c.mean() - baseline_cv.mean()
print(f"  CV Mean: {scores_19c.mean():.4f} ± {scores_19c.std():.4f}  (delta vs BL: {delta_19c:+.4f})")
comp_19c_vs_18b = paired_comparison(scores_18b, scores_19c)
print(f"  vs 18b: {comp_19c_vs_18b['mean_delta']:+.4f}  W/L/T: {comp_19c_vs_18b['n_candidate_wins']}/{comp_19c_vs_18b['n_baseline_wins']}/{comp_19c_vs_18b['n_ties']}")

# Component size diagnostics
if component_size_log:
    print(f"  Component sizes (first fold, 2+ members): {len(component_size_log)} components")
    print(f"  Largest: {component_size_log[0]}  Top-5: {component_size_log[:5]}")
    if component_size_log[0] >= 5:
        print(f"  *** WARNING: Component with {component_size_log[0]} members — possible false merges ***")
print()


# ================================================================
# PHASE 2: TEST PREDICTIONS + FLIP ANALYSIS
# ================================================================
print("=" * 60)
print("PHASE 2: TEST PREDICTIONS AND FLIP ANALYSIS")
print("=" * 60)
print()

# Fit final model on full training set
pipe_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=C, max_iter=2000, random_state=42)),
])
pipe_final.fit(X_v2, y)
base_test_preds = pipe_final.predict(X_test_v2)
base_test_proba = pipe_final.predict_proba(X_test_v2)[:, 1]


# ---- 19a: Surname+Pclass rules only ----
print("--- 19a: Surname+Pclass rules only ---")
preds_19a = base_test_preds.copy()
proba_19a = base_test_proba.copy()
overrides_19a = []

for i in range(len(preds_19a)):
    sp_key = test_sp_keys[i]
    if sp_key in train_surv_by_sp.index:
        rate = train_surv_by_sp.loc[sp_key, "mean"]
        count = train_surv_by_sp.loc[sp_key, "count"]
        if count >= 2 and rate == 1.0:
            old = preds_19a[i]
            preds_19a[i] = 1
            proba_19a[i] = 0.95
            overrides_19a.append({
                "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                "Ticket": test_tickets[i], "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Direction": f"{old}->1" if old != 1 else "no_change",
            })
        elif count >= 2 and rate == 0.0:
            old = preds_19a[i]
            preds_19a[i] = 0
            proba_19a[i] = 0.05
            overrides_19a.append({
                "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                "Ticket": test_tickets[i], "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Direction": f"{old}->0" if old != 0 else "no_change",
            })

print(f"  Surname rule overrides on test: {len(overrides_19a)} passengers")
for o in overrides_19a:
    print(f"    PID {o['PID']}: {o['Sex']:>6s} Pc{o['Pclass']} surname={o['Surname']} "
          f"ticket={o['Ticket']} {o['Direction']}")
print()

print("  Flip analysis vs logreg_v2.csv (plain baseline):")
flips_19a = flip_analysis(preds_19a, proba_19a, f"{BASE}/submissions/logreg_v2.csv",
                          test_ids, raw_test)
report_flips(flips_19a)
print()

print("  Flip analysis vs logreg_18_18b_hybrid_2mate.csv (current best):")
flips_19a_vs_18b = flip_analysis(preds_19a, proba_19a,
                                 f"{BASE}/submissions/logreg_18_18b_hybrid_2mate.csv",
                                 test_ids, raw_test)
report_flips(flips_19a_vs_18b)
print()


# ---- 19b: Ticket + Surname (ticket priority) ----
print("--- 19b: Ticket + Surname rules (ticket priority) ---")
preds_19b = base_test_preds.copy()
proba_19b = base_test_proba.copy()
overrides_19b_ticket = []
overrides_19b_surname = []

for i in range(len(preds_19b)):
    overridden = False

    # Ticket rule first (exact 18b logic)
    ticket = test_tickets[i]
    if ticket in train_surv_by_ticket.index:
        rate = train_surv_by_ticket.loc[ticket, "mean"]
        count = train_surv_by_ticket.loc[ticket, "count"]
        if count >= 2 and rate == 1.0:
            old = preds_19b[i]
            preds_19b[i] = 1
            proba_19b[i] = 0.95
            overridden = True
            overrides_19b_ticket.append({
                "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                "Ticket": ticket, "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Direction": f"{old}->1" if old != 1 else "no_change",
                "Source": "ticket",
            })
        elif count >= 2 and rate == 0.0:
            old = preds_19b[i]
            preds_19b[i] = 0
            proba_19b[i] = 0.05
            overridden = True
            overrides_19b_ticket.append({
                "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                "Ticket": ticket, "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Direction": f"{old}->0" if old != 0 else "no_change",
                "Source": "ticket",
            })

    # Surname rule second (only if ticket didn't fire)
    if not overridden:
        sp_key = test_sp_keys[i]
        if sp_key in train_surv_by_sp.index:
            rate = train_surv_by_sp.loc[sp_key, "mean"]
            count = train_surv_by_sp.loc[sp_key, "count"]
            if count >= 2 and rate == 1.0:
                old = preds_19b[i]
                preds_19b[i] = 1
                proba_19b[i] = 0.95
                overrides_19b_surname.append({
                    "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                    "Ticket": ticket, "Sex": raw_test["Sex"].iloc[i],
                    "Pclass": int(raw_test["Pclass"].iloc[i]),
                    "Direction": f"{old}->1" if old != 1 else "no_change",
                    "Source": "surname",
                })
            elif count >= 2 and rate == 0.0:
                old = preds_19b[i]
                preds_19b[i] = 0
                proba_19b[i] = 0.05
                overrides_19b_surname.append({
                    "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                    "Ticket": ticket, "Sex": raw_test["Sex"].iloc[i],
                    "Pclass": int(raw_test["Pclass"].iloc[i]),
                    "Direction": f"{old}->0" if old != 0 else "no_change",
                    "Source": "surname",
                })

print(f"  Ticket overrides: {len(overrides_19b_ticket)} passengers")
for o in overrides_19b_ticket:
    print(f"    PID {o['PID']}: {o['Sex']:>6s} Pc{o['Pclass']} surname={o['Surname']} "
          f"ticket={o['Ticket']} {o['Direction']} [{o['Source']}]")
print(f"  Surname overrides (NEW): {len(overrides_19b_surname)} passengers")
for o in overrides_19b_surname:
    print(f"    PID {o['PID']}: {o['Sex']:>6s} Pc{o['Pclass']} surname={o['Surname']} "
          f"ticket={o['Ticket']} {o['Direction']} [{o['Source']}]")
print()

# Verify: ticket overrides should match 18b exactly
best_18b_sub = pd.read_csv(f"{BASE}/submissions/logreg_18_18b_hybrid_2mate.csv")
best_18b_preds = best_18b_sub.set_index("PassengerId")["Survived"]

# Rebuild 18b overrides for comparison
preds_18b_check = base_test_preds.copy()
for i in range(len(preds_18b_check)):
    ticket = test_tickets[i]
    if ticket in train_surv_by_ticket.index:
        rate = train_surv_by_ticket.loc[ticket, "mean"]
        count = train_surv_by_ticket.loc[ticket, "count"]
        if count >= 2 and rate == 1.0:
            preds_18b_check[i] = 1
        elif count >= 2 and rate == 0.0:
            preds_18b_check[i] = 0

# Check ticket layer matches
ticket_match = (preds_18b_check == best_18b_preds.values).all()
print(f"  Ticket layer matches 18b submission: {ticket_match}")
if not ticket_match:
    diffs = np.where(preds_18b_check != best_18b_preds.values)[0]
    print(f"  *** WARNING: {len(diffs)} mismatches in ticket layer vs 18b submission ***")
print()

print("  Flip analysis vs logreg_v2.csv (plain baseline):")
flips_19b = flip_analysis(preds_19b, proba_19b, f"{BASE}/submissions/logreg_v2.csv",
                          test_ids, raw_test)
report_flips(flips_19b)
print()

print("  Flip analysis vs logreg_18_18b_hybrid_2mate.csv (current best):")
flips_19b_vs_18b = flip_analysis(preds_19b, proba_19b,
                                 f"{BASE}/submissions/logreg_18_18b_hybrid_2mate.csv",
                                 test_ids, raw_test)
report_flips(flips_19b_vs_18b)
print()


# ---- 19c: Connected components ----
print("--- 19c: Connected components (ticket ∪ surname+Pclass) ---")
preds_19c = base_test_preds.copy()
proba_19c = base_test_proba.copy()

# Build union-find from full training set
n_train = len(raw_train)
parent_full = list(range(n_train))

def find_full(x):
    while parent_full[x] != x:
        parent_full[x] = parent_full[parent_full[x]]
        x = parent_full[x]
    return x

def union_full(a, b):
    ra, rb = find_full(a), find_full(b)
    if ra != rb:
        parent_full[ra] = rb

# Index by ticket and surname+Pclass
ticket_to_train_idxs = defaultdict(list)
sp_to_train_idxs = defaultdict(list)
for i in range(n_train):
    ticket_to_train_idxs[train_tickets[i]].append(i)
    sp_to_train_idxs[train_sp_keys[i]].append(i)

for idxs in ticket_to_train_idxs.values():
    for j in range(1, len(idxs)):
        union_full(idxs[0], idxs[j])

for idxs in sp_to_train_idxs.values():
    for j in range(1, len(idxs)):
        union_full(idxs[0], idxs[j])

# Compute component stats
comp_members_full = defaultdict(list)
for i in range(n_train):
    root = find_full(i)
    comp_members_full[root].append(i)

comp_stats_full = {}
for root, members in comp_members_full.items():
    survivals = y.iloc[members].values
    comp_stats_full[root] = (survivals.mean(), len(survivals))

# Component size distribution
comp_sizes = sorted([len(v) for v in comp_members_full.values() if len(v) >= 2], reverse=True)
print(f"  Components with 2+ members: {len(comp_sizes)}")
if comp_sizes:
    print(f"  Largest: {comp_sizes[0]}  Top-5: {comp_sizes[:5]}")
    print(f"  Size distribution: ", end="")
    for threshold in [2, 3, 4, 5, 6, 7, 8, 10]:
        n_at = sum(1 for s in comp_sizes if s >= threshold)
        if n_at > 0:
            print(f"{threshold}+:{n_at}  ", end="")
    print()
    if comp_sizes[0] >= 5:
        # Show the mega-component members
        for root, members in comp_members_full.items():
            if len(members) >= 5:
                names = raw_train.iloc[members]["Name"].values
                surnames = raw_train.iloc[members]["Surname"].values
                tickets = raw_train.iloc[members]["Ticket"].values
                print(f"    Component (root={root}, size={len(members)}):")
                for name, surname, ticket in zip(names, surnames, tickets):
                    print(f"      {surname} | ticket={ticket} | {name}")
print()

# Build reverse lookup for test passengers
ticket_to_root_full = {}
for i in range(n_train):
    root = find_full(i)
    ticket_to_root_full[train_tickets[i]] = root

sp_to_root_full = {}
for i in range(n_train):
    root = find_full(i)
    sp_to_root_full[train_sp_keys[i]] = root

overrides_19c = []
for i in range(len(preds_19c)):
    ticket = test_tickets[i]
    sp_key = test_sp_keys[i]

    # Find component via ticket first, then surname
    root = ticket_to_root_full.get(ticket)
    if root is None:
        root = sp_to_root_full.get(sp_key)

    if root is not None:
        rate, count = comp_stats_full[root]
        if count >= 2 and rate in (0.0, 1.0):
            old = preds_19c[i]
            preds_19c[i] = int(rate)
            proba_19c[i] = 0.95 if rate == 1.0 else 0.05
            overrides_19c.append({
                "PID": int(test_ids.iloc[i]), "Surname": raw_test["Surname"].iloc[i],
                "Ticket": ticket, "Sex": raw_test["Sex"].iloc[i],
                "Pclass": int(raw_test["Pclass"].iloc[i]),
                "Direction": f"{old}->{int(rate)}" if old != int(rate) else "no_change",
                "CompSize": count,
            })

print(f"  Rule overrides on test: {len(overrides_19c)} passengers")
for o in overrides_19c:
    print(f"    PID {o['PID']}: {o['Sex']:>6s} Pc{o['Pclass']} surname={o['Surname']} "
          f"ticket={o['Ticket']} {o['Direction']} (comp_size={o['CompSize']})")
print()

print("  Flip analysis vs logreg_v2.csv (plain baseline):")
flips_19c = flip_analysis(preds_19c, proba_19c, f"{BASE}/submissions/logreg_v2.csv",
                          test_ids, raw_test)
report_flips(flips_19c)
print()

print("  Flip analysis vs logreg_18_18b_hybrid_2mate.csv (current best):")
flips_19c_vs_18b = flip_analysis(preds_19c, proba_19c,
                                 f"{BASE}/submissions/logreg_18_18b_hybrid_2mate.csv",
                                 test_ids, raw_test)
report_flips(flips_19c_vs_18b)
print()


# ================================================================
# PHASE 3: SAVE SUBMISSIONS + SUMMARY
# ================================================================
print("=" * 60)
print("PHASE 3: SAVE SUBMISSIONS + SUMMARY")
print("=" * 60)
print()

submissions = [
    ("logreg_19_19a_surname_pc.csv", preds_19a, scores_19a, "19a_Surname_Pc"),
    ("logreg_19_19b_ticket_surname.csv", preds_19b, scores_19b, "19b_Ticket+Surname"),
    ("logreg_19_19c_connected.csv", preds_19c, scores_19c, "19c_Connected"),
]

for filename, preds, _, label in submissions:
    sub_path = f"{BASE}/submissions/{filename}"
    sub = pd.DataFrame({"PassengerId": test_ids, "Survived": preds})
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    print(f"    Survival rate: {preds.mean():.3f} ({int(preds.sum())}/{len(preds)})")

print()

# Summary table
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print()
print(f"{'Model':<35} {'CV Mean':>8} {'CV Std':>7} {'Δ BL':>7} {'Δ 18b':>7} {'W/L/T vs 18b':>13}")
print("-" * 80)
print(f"{'v2 baseline (C=0.01)':<35} {baseline_cv.mean():>8.4f} {baseline_cv.std():>7.4f} {'--':>7} {'--':>7} {'--':>13}")

summary_rows = [
    ("18b ticket (inline)", scores_18b),
    ("19a surname+Pc only", scores_19a),
    ("19b ticket+surname", scores_19b),
    ("19c connected comp", scores_19c),
]

for label, scores in summary_rows:
    d_bl = scores.mean() - baseline_cv.mean()
    d_18b = scores.mean() - scores_18b.mean()
    comp = paired_comparison(scores_18b, scores)
    w, l, t = comp["n_candidate_wins"], comp["n_baseline_wins"], comp["n_ties"]
    print(f"{label:<35} {scores.mean():>8.4f} {scores.std():>7.4f} {d_bl:>+7.4f} {d_18b:>+7.4f} {f'{w}/{l}/{t}':>13}")

print()

# Risk flags
n_surname_flips = len(overrides_19b_surname)
if n_surname_flips > 15:
    print(f"*** RISK: {n_surname_flips} surname overrides on test (>15 threshold) ***")
print()
print("Results presented for human review.")
print("Do NOT auto-accept or auto-reject based on automated metrics alone.")

tee.close()
