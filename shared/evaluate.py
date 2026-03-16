"""
Titanic Evaluation Harness

Provides multi-metric model evaluation for small-dataset model selection.
Primary signals: repeated CV, paired comparison, flip analysis.
Supporting context: prediction instability, bootstrap .632+.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")


BASE = "/Users/glennharless/dev-brain/kaggle/competitions/titanic"


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout


# ============================================================
# PRIMARY: Repeated Stratified CV
# ============================================================

def repeated_cv(pipeline, X, y, n_splits=5, n_repeats=10, random_state=42):
    """Run repeated stratified K-fold CV. Returns array of fold scores."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return scores


def report_cv(scores, label=""):
    """Print CV summary."""
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Mean:   {scores.mean():.4f}")
    print(f"  {prefix}Std:    {scores.std():.4f}")
    print(f"  {prefix}Min:    {scores.min():.4f}")
    print(f"  {prefix}Max:    {scores.max():.4f}")
    print(f"  {prefix}Range:  {scores.max() - scores.min():.4f}")


# ============================================================
# PRIMARY: Repeated Grouped CV
# ============================================================
#
# Scope: Use grouped CV to evaluate BASE MODELS and FEATURE VARIANTS
# for group-structure leakage. It answers: "does the model benefit from
# ticket-mates leaking across folds?"
#
# NOT suitable for evaluating propagation rules (ticket-mate survival
# overrides), because grouping by ticket prevents the rule from ever
# firing. For rule evaluation, use leave_one_out_grouped_cv() below.
# ============================================================

def grouped_cv_splits(groups, n_splits=5, n_repeats=10, random_state=42):
    """
    Generate (train_idx, val_idx) splits for repeated grouped K-fold.

    Keeps all members of a group in the same fold. Groups are shuffled
    between repeats to produce different fold assignments.

    Use this directly when experiment scripts need custom per-fold logic
    (e.g., OOF encoding). NOT suitable for propagation rules — use
    leave_one_out_grouped_cv() instead.

    Args:
        groups: array-like of group labels (e.g., ticket strings), one per sample.
        n_splits: number of folds per repeat.
        n_repeats: number of times to repeat with different group shuffles.
        random_state: base random seed (incremented per repeat).

    Yields:
        (train_idx, val_idx) index arrays, n_splits * n_repeats total.
    """
    unique_groups = np.unique(groups)

    for repeat in range(n_repeats):
        rng = np.random.RandomState(random_state + repeat)
        shuffled = rng.permutation(unique_groups)
        group_folds = np.array_split(shuffled, n_splits)

        for fold_i in range(n_splits):
            val_groups = set(group_folds[fold_i])
            val_mask = np.array([g in val_groups for g in groups])
            train_idx = np.where(~val_mask)[0]
            val_idx = np.where(val_mask)[0]
            yield train_idx, val_idx


def repeated_grouped_cv(pipeline, X, y, groups, n_splits=5, n_repeats=10,
                        random_state=42):
    """
    Repeated grouped K-fold CV. Keeps all members of a group in the same fold.

    Unlike RepeatedStratifiedKFold, this does NOT stratify by class — fold
    class balance depends on random group assignment. For 891 samples with
    ~680 unique ticket groups, fold sizes and class balance are adequate.

    Returns array of fold scores (n_splits * n_repeats total), compatible
    with paired_comparison() against baseline_cv scores of the same length.
    """
    from sklearn.metrics import accuracy_score

    scores = []
    for train_idx, val_idx in grouped_cv_splits(groups, n_splits, n_repeats,
                                                random_state):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        score = accuracy_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
        scores.append(score)

    return np.array(scores)


# ============================================================
# PRIMARY: Leave-One-Out Grouped CV (for propagation rules)
# ============================================================
#
# Holds out one member per multi-member group while keeping remaining
# members in training. Structurally closer to deployment than GroupKFold.
#
# KNOWN LIMITATION (2026-03-14): For unanimous-outcome rules like 18b,
# LOO has a structural bias. Truly unanimous groups → rule fires and
# is trivially correct (100%). Mixed groups → holding out the dissenter
# creates spurious unanimity, rule fires and is always wrong. Neither
# case informs the real question: "will a test passenger from a
# training-unanimous group follow the group pattern?" Only the Kaggle
# leaderboard answers that. This validator is still useful for:
# - measuring fire rate and coverage
# - comparing rule variants (e.g., 2-mate vs 3-mate thresholds)
# - detecting structural problems (low coverage, high false-fire rate)
#
# NOT suitable for base model evaluation (use repeated_cv or
# repeated_grouped_cv for that).
# ============================================================

def leave_one_out_grouped_cv(y, groups, rule_fn, n_repeats=10, random_state=42):
    """
    Evaluate a propagation rule by holding out one member per eligible group.

    For each repeat: shuffle eligible groups, hold out one random member
    from each multi-member group, keep remaining members in training.
    The rule_fn is called to produce predictions for held-out passengers.
    Solo passengers (group size 1) are excluded — the rule can never
    apply to them.

    Args:
        y: Series of survival labels (indexed 0..N-1).
        groups: array of group labels (e.g., ticket strings), one per sample.
        rule_fn: callable(train_idx, val_idx, groups) -> predictions array.
            Must return an array of 0/1 predictions for val_idx passengers,
            or np.nan for passengers where the rule did not fire.
        n_repeats: number of random holdout rounds.
        random_state: base seed.

    Returns:
        dict with accuracy, coverage, and per-repeat diagnostics.
    """
    from collections import defaultdict

    # Identify multi-member groups
    group_to_idxs = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_idxs[g].append(i)

    eligible_groups = {g: idxs for g, idxs in group_to_idxs.items()
                       if len(idxs) >= 2}

    all_idx = np.arange(len(y))
    solo_idx = [i for i, g in enumerate(groups) if len(group_to_idxs[g]) == 1]

    repeat_results = []

    for repeat in range(n_repeats):
        rng = np.random.RandomState(random_state + repeat)

        val_idx = []
        train_idx_from_groups = []

        for g, idxs in eligible_groups.items():
            # Hold out one random member
            holdout_pos = rng.randint(len(idxs))
            for j, idx in enumerate(idxs):
                if j == holdout_pos:
                    val_idx.append(idx)
                else:
                    train_idx_from_groups.append(idx)

        # Training set = solo passengers + non-held-out group members
        train_idx = np.array(sorted(solo_idx + train_idx_from_groups))
        val_idx = np.array(sorted(val_idx))

        # Get rule predictions
        preds = rule_fn(train_idx, val_idx, groups)

        # Separate fired vs not-fired
        y_val = y.iloc[val_idx].values
        fired_mask = ~np.isnan(preds)
        n_fired = fired_mask.sum()
        n_val = len(val_idx)

        if n_fired > 0:
            fired_correct = (preds[fired_mask] == y_val[fired_mask]).sum()
            fired_acc = fired_correct / n_fired
        else:
            fired_correct = 0
            fired_acc = np.nan

        repeat_results.append({
            "n_val": n_val,
            "n_fired": n_fired,
            "n_correct": fired_correct,
            "fired_acc": fired_acc,
        })

    # Aggregate
    total_fired = sum(r["n_fired"] for r in repeat_results)
    total_correct = sum(r["n_correct"] for r in repeat_results)
    total_val = sum(r["n_val"] for r in repeat_results)

    result = {
        "n_eligible_groups": len(eligible_groups),
        "n_eligible_passengers": sum(len(v) for v in eligible_groups.values()),
        "n_repeats": n_repeats,
        "total_val": total_val,
        "total_fired": total_fired,
        "total_correct": total_correct,
        "fire_rate": total_fired / total_val if total_val > 0 else 0,
        "fired_accuracy": total_correct / total_fired if total_fired > 0 else np.nan,
        "per_repeat": repeat_results,
    }
    return result


def report_loo_grouped(result, label=""):
    """Print leave-one-out grouped CV summary."""
    prefix = f"[{label}] " if label else ""
    r = result
    print(f"  {prefix}Eligible groups: {r['n_eligible_groups']} "
          f"({r['n_eligible_passengers']} passengers)")
    print(f"  {prefix}Repeats: {r['n_repeats']}")
    print(f"  {prefix}Total held-out: {r['total_val']} "
          f"(~{r['total_val'] // r['n_repeats']} per repeat)")
    print(f"  {prefix}Rule fired: {r['total_fired']} / {r['total_val']} "
          f"({r['fire_rate']:.1%})")
    if r['total_fired'] > 0:
        print(f"  {prefix}Accuracy on fired: {r['fired_accuracy']:.4f} "
              f"({r['total_correct']}/{r['total_fired']})")
    else:
        print(f"  {prefix}Rule never fired")

    # Per-repeat breakdown
    per = r["per_repeat"]
    fired_accs = [p["fired_acc"] for p in per if not np.isnan(p["fired_acc"])]
    if fired_accs:
        print(f"  {prefix}Fired acc per repeat: "
              f"mean={np.mean(fired_accs):.4f}, "
              f"min={np.min(fired_accs):.4f}, "
              f"max={np.max(fired_accs):.4f}")


# ============================================================
# PRIMARY: Paired Comparison
# ============================================================

def paired_comparison(baseline_scores, candidate_scores):
    """
    Paired comparison of two models' repeated CV scores.
    Both must have the same length (same CV splits).
    Returns dict with delta stats.
    """
    assert len(baseline_scores) == len(candidate_scores), \
        f"Score arrays must match: {len(baseline_scores)} vs {len(candidate_scores)}"

    deltas = candidate_scores - baseline_scores
    result = {
        "mean_delta": deltas.mean(),
        "std_delta": deltas.std(),
        "min_delta": deltas.min(),
        "max_delta": deltas.max(),
        "n_candidate_wins": (deltas > 0).sum(),
        "n_baseline_wins": (deltas < 0).sum(),
        "n_ties": (deltas == 0).sum(),
    }
    return result


def report_paired(comparison, baseline_label="baseline", candidate_label="candidate"):
    """Print paired comparison summary."""
    c = comparison
    print(f"  Delta ({candidate_label} - {baseline_label}):")
    print(f"    Mean:  {c['mean_delta']:+.4f}")
    print(f"    Std:   {c['std_delta']:.4f}")
    print(f"    Range: [{c['min_delta']:+.4f}, {c['max_delta']:+.4f}]")
    print(f"    Wins:  candidate {c['n_candidate_wins']}, baseline {c['n_baseline_wins']}, ties {c['n_ties']}")


# ============================================================
# PRIMARY: Test Prediction Flip Analysis
# ============================================================

def flip_analysis(candidate_preds, candidate_proba, baseline_csv, test_ids,
                  raw_test_df, target_subgroup=None):
    """
    Compare candidate test predictions against baseline submission.

    Args:
        candidate_preds: array of 0/1 predictions
        candidate_proba: array of P(survived) probabilities
        baseline_csv: path to baseline submission CSV
        test_ids: PassengerId array
        raw_test_df: raw test DataFrame with Sex, Pclass columns
        target_subgroup: dict like {"Sex": "male", "Pclass": 1} — warn on changes outside

    Returns dict with flip details.
    """
    baseline = pd.read_csv(baseline_csv)
    baseline_preds = baseline["Survived"].values

    diff_mask = candidate_preds != baseline_preds
    n_flips = diff_mask.sum()

    # Build flip detail dataframe
    flip_df = pd.DataFrame({
        "PassengerId": test_ids,
        "Sex": raw_test_df["Sex"].values,
        "Pclass": raw_test_df["Pclass"].values,
        "Baseline": baseline_preds,
        "Candidate": candidate_preds,
        "Probability": candidate_proba,
    })
    flip_df["Flipped"] = diff_mask
    flip_df["InBubble"] = (candidate_proba > 0.3) & (candidate_proba < 0.7)

    # Subgroup breakdown of flips
    flipped = flip_df[flip_df["Flipped"]].copy()
    flipped["Direction"] = np.where(
        flipped["Candidate"] > flipped["Baseline"],
        "died->survived", "survived->died"
    )

    subgroup_flips = {}
    for sex in ["male", "female"]:
        for pclass in [1, 2, 3]:
            key = f"{sex}_Pclass{pclass}"
            mask = (flipped["Sex"] == sex) & (flipped["Pclass"] == pclass)
            subgroup_flips[key] = mask.sum()

    # Target subgroup analysis
    outside_target = 0
    outside_target_ids = []
    if target_subgroup is not None and n_flips > 0:
        target_mask = pd.Series(True, index=flipped.index)
        for col, val in target_subgroup.items():
            target_mask &= (flipped[col] == val)
        outside = flipped[~target_mask]
        outside_target = len(outside)
        outside_target_ids = outside["PassengerId"].tolist()

    result = {
        "n_flips": n_flips,
        "n_test": len(test_ids),
        "flip_rate": n_flips / len(test_ids) if len(test_ids) > 0 else 0,
        "flipped_ids": flipped["PassengerId"].tolist(),
        "subgroup_flips": subgroup_flips,
        "n_flips_in_bubble": flipped["InBubble"].sum() if n_flips > 0 else 0,
        "n_flips_outside_bubble": (~flipped["InBubble"]).sum() if n_flips > 0 else 0,
        "outside_target": outside_target,
        "outside_target_ids": outside_target_ids,
        "flip_details": flipped,
    }
    return result


def report_flips(flip_result, target_subgroup=None):
    """Print flip analysis summary."""
    f = flip_result
    print(f"  Total flips: {f['n_flips']} / {f['n_test']} ({f['flip_rate']:.1%})")

    if f["n_flips"] > 0:
        print(f"  In bubble (P 0.3-0.7): {f['n_flips_in_bubble']}")
        print(f"  Outside bubble: {f['n_flips_outside_bubble']}")

        print(f"\n  Flips by subgroup:")
        for key, count in sorted(f["subgroup_flips"].items()):
            if count > 0:
                print(f"    {key}: {count}")

        if target_subgroup is not None:
            target_str = ", ".join(f"{k}={v}" for k, v in target_subgroup.items())
            if f["outside_target"] > 0:
                print(f"\n  *** WARNING: {f['outside_target']} flip(s) OUTSIDE target subgroup ({target_str}) ***")
                print(f"  *** Outside IDs: {f['outside_target_ids']} ***")
            else:
                print(f"\n  All flips within target subgroup ({target_str}): OK")

        print(f"\n  Changed PassengerIds: {f['flipped_ids']}")

        # Show flip details
        details = f["flip_details"]
        print(f"\n  {'PID':>5} {'Baseline':>8} {'Candidate':>9} {'Prob':>6} {'Sex':>6} {'Pcl':>4} {'Direction'}")
        print(f"  {'-'*55}")
        for _, row in details.iterrows():
            print(f"  {int(row['PassengerId']):>5} {int(row['Baseline']):>8} "
                  f"{int(row['Candidate']):>9} {row['Probability']:>6.3f} "
                  f"{row['Sex']:>6} {int(row['Pclass']):>4} {row['Direction']}")


# ============================================================
# SUPPORTING: Prediction Instability
# ============================================================

def prediction_instability(pipeline, X, y, n_splits=5, n_repeats=10, random_state=42):
    """
    Measure per-sample prediction instability across repeated CV folds.
    Returns dict with instability scores and summary stats.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    n_samples = len(y)
    pred_counts = np.zeros(n_samples, dtype=int)
    pred_sum = np.zeros(n_samples, dtype=float)

    for train_idx, val_idx in cv.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])

        for i, idx in enumerate(val_idx):
            pred_sum[idx] += preds[i]
            pred_counts[idx] += 1

    pred_rate = pred_sum / pred_counts
    instability = 2 * np.minimum(pred_rate, 1 - pred_rate)

    return {
        "instability_scores": instability,
        "mean_instability": instability.mean(),
        "n_unstable": (instability > 0).sum(),
        "n_highly_unstable": (instability > 0.4).sum(),
    }


def report_instability(inst_result):
    """Print instability summary."""
    i = inst_result
    print(f"  Mean instability:   {i['mean_instability']:.4f}")
    print(f"  Unstable samples:   {i['n_unstable']} (any disagreement across folds)")
    print(f"  Highly unstable:    {i['n_highly_unstable']} (instability > 0.4)")


# ============================================================
# SUPPORTING: Bootstrap .632+
# ============================================================

def bootstrap_632plus(pipeline, X, y, n_bootstraps=200, random_state=42):
    """Compute bootstrap .632+ accuracy estimate."""
    rng = np.random.RandomState(random_state)
    n = len(y)

    train_errors = []
    oob_errors = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) == 0:
            continue

        model = clone(pipeline)
        model.fit(X.iloc[boot_idx], y.iloc[boot_idx])

        train_errors.append(1 - model.score(X.iloc[boot_idx], y.iloc[boot_idx]))
        oob_errors.append(1 - model.score(X.iloc[oob_idx], y.iloc[oob_idx]))

    mean_train_err = np.mean(train_errors)
    mean_oob_err = np.mean(oob_errors)

    # .632 estimate
    err_632 = 0.368 * mean_train_err + 0.632 * mean_oob_err

    # No-information rate
    model = clone(pipeline)
    model.fit(X, y)
    p = y.mean()
    p_hat = model.predict(X).mean()
    gamma = p * (1 - p_hat) + (1 - p) * p_hat

    # Relative overfitting rate
    R = (mean_oob_err - mean_train_err) / (gamma - mean_train_err + 1e-10)
    R = np.clip(R, 0, 1)

    w = 0.632 / (1 - 0.368 * R)
    err_632plus = (1 - w) * mean_train_err + w * mean_oob_err

    return {
        "train_acc": 1 - mean_train_err,
        "oob_acc": 1 - mean_oob_err,
        "acc_632": 1 - err_632,
        "acc_632plus": 1 - err_632plus,
        "overfitting_rate_R": R,
    }


def report_632plus(boot_result):
    """Print .632+ summary."""
    b = boot_result
    print(f"  Train accuracy:     {b['train_acc']:.4f}")
    print(f"  OOB accuracy:       {b['oob_acc']:.4f}")
    print(f"  .632 accuracy:      {b['acc_632']:.4f}")
    print(f"  .632+ accuracy:     {b['acc_632plus']:.4f}")
    print(f"  Overfitting rate R: {b['overfitting_rate_R']:.4f}")


# ============================================================
# MAIN: evaluate_model
# ============================================================

def evaluate_model(pipeline, X_train, y_train, X_test, test_ids, raw_test_df,
                   baseline_csv=f"{BASE}/submissions/logreg_v2.csv",
                   baseline_cv_scores=None,
                   target_subgroup=None,
                   report_path=None,
                   label="candidate"):
    """
    Full evaluation of a model against the v2 baseline.

    Args:
        pipeline: sklearn Pipeline (unfitted)
        X_train, y_train: training data
        X_test: test features (no PassengerId)
        test_ids: PassengerId array for test set
        raw_test_df: raw test DataFrame with Sex, Pclass columns
        baseline_csv: path to baseline submission
        baseline_cv_scores: array of baseline repeated CV scores (for paired comparison)
        target_subgroup: dict for outside-target warnings
        report_path: if set, tee output to this file
        label: name for this model in output

    Returns dict with all metrics.
    """
    tee = None
    if report_path:
        tee = Tee(report_path)
        sys.stdout = tee

    print("=" * 60)
    print(f"EVALUATION: {label}")
    print("=" * 60)
    print(f"Features: {X_train.shape[1]}")
    print(f"Feature list: {list(X_train.columns)}")
    print()

    results = {"label": label}

    # ---- PRIMARY: Repeated CV ----
    print("--- PRIMARY: Repeated Stratified CV (10x5-fold) ---")
    cv_scores = repeated_cv(pipeline, X_train, y_train)
    report_cv(cv_scores, label)
    results["cv_scores"] = cv_scores
    results["cv_mean"] = cv_scores.mean()
    results["cv_std"] = cv_scores.std()
    print()

    # ---- PRIMARY: Paired Comparison ----
    if baseline_cv_scores is not None:
        print("--- PRIMARY: Paired Comparison vs Baseline ---")
        comparison = paired_comparison(baseline_cv_scores, cv_scores)
        report_paired(comparison)
        results["paired"] = comparison
        print()

    # ---- PRIMARY: Test Prediction Flips ----
    print("--- PRIMARY: Test Prediction Flip Analysis ---")
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)
    test_preds = fitted.predict(X_test)
    test_proba = fitted.predict_proba(X_test)[:, 1]

    flips = flip_analysis(
        test_preds, test_proba, baseline_csv, test_ids,
        raw_test_df, target_subgroup
    )
    report_flips(flips, target_subgroup)
    results["flips"] = flips
    results["test_preds"] = test_preds
    results["test_proba"] = test_proba
    print()

    # ---- SUPPORTING: Prediction Instability ----
    print("--- SUPPORTING: Prediction Instability ---")
    instability = prediction_instability(pipeline, X_train, y_train)
    report_instability(instability)
    results["instability"] = instability
    print()

    # ---- SUPPORTING: Bootstrap .632+ ----
    print("--- SUPPORTING: Bootstrap .632+ (200 resamples) ---")
    boot = bootstrap_632plus(pipeline, X_train, y_train)
    report_632plus(boot)
    results["bootstrap"] = boot
    print()

    # ---- Summary ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Repeated CV:  {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"  .632+ acc:    {boot['acc_632plus']:.4f}")
    print(f"  Test flips:   {flips['n_flips']}")
    if baseline_cv_scores is not None:
        print(f"  Delta vs baseline: {results['paired']['mean_delta']:+.4f}")
    print(f"  Instability:  {instability['mean_instability']:.4f}")
    print()

    if tee:
        tee.close()

    return results


# ============================================================
# HELPERS: v2 feature reconstruction
# ============================================================

def reconstruct_v2_features(df):
    """
    Reconstruct the real v2 feature matrix from v3 processed data.
    Drops v3-only columns, reconstructs ordinal Pclass.

    Works for both train (has Survived) and test (has PassengerId).
    Returns a copy with exactly the v2 feature columns.
    """
    df = df.copy()

    # Reconstruct ordinal Pclass from one-hot
    df["Pclass"] = 1 + df["Pclass_2"].astype(int) + 2 * df["Pclass_3"].astype(int)

    # Select exactly the v2 feature columns
    v2_features = [
        "Pclass", "Sex", "Fare", "IsAlone", "IsLargeFamily", "HasCabin",
        "IsChild", "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs",
        "Title_Rare", "Emb_C", "Emb_Q", "Emb_S",
    ]

    # For train, keep Survived; for test, keep PassengerId
    extra_cols = []
    if "Survived" in df.columns:
        extra_cols.append("Survived")
    if "PassengerId" in df.columns:
        extra_cols.append("PassengerId")

    return df[extra_cols + v2_features]


# ============================================================
# REGRESSION: RMSLE and Repeated KFold CV
# ============================================================

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error.

    RMSLE = sqrt(mean((log(pred+1) - log(actual+1))^2))
    Negative predictions are clipped to 0 before log.
    """
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, None)
    y_true = np.array(y_true, dtype=float)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def regression_cv(model, X, y, n_splits=5, n_repeats=5, random_state=42,
                  scoring_fn=None):
    """
    Repeated KFold CV for regression. Returns array of per-fold scores.

    scoring_fn: callable(y_true, y_pred) -> float. Default: RMSLE.
    Lower is better.
    """
    from sklearn.model_selection import KFold

    if scoring_fn is None:
        scoring_fn = rmsle

    scores = []
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True,
                    random_state=random_state + repeat)
        for train_idx, val_idx in kf.split(X):
            m = clone(model)
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = m.predict(X.iloc[val_idx])
            score = scoring_fn(y.iloc[val_idx], preds)
            scores.append(score)

    return np.array(scores)


def report_regression_cv(scores, label=""):
    """Print regression CV summary (lower is better for RMSLE)."""
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Mean RMSLE: {scores.mean():.5f}")
    print(f"  {prefix}Std:        {scores.std():.5f}")
    print(f"  {prefix}Min:        {scores.min():.5f}")
    print(f"  {prefix}Max:        {scores.max():.5f}")
    print(f"  {prefix}Range:      {scores.max() - scores.min():.5f}")
