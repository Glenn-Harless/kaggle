"""
Titanic Model 11c: Logistic Regression v3 — Stage C (Surname Features)

Tests surname-based features with strict out-of-fold encoding.
Uses a CUSTOM CV LOOP (not cross_val_score) to prevent target leakage.

Features tested (in order):
  1. SurnameGroupSize — count of passengers sharing a surname (safe, no target info)
  2. SurnameSurvHint — mean survival of surname group (requires OOF encoding)

Safety checks:
  - Shuffle ablation: randomize surname labels and confirm lift disappears
  - SurnameGroupSize and SurnameSurvHint tested separately before combining
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/data"
RESULTS_DIR = "/Users/glennharless/dev-brain/kaggle/competitions/titanic/results/models"


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


tee = Tee(f"{RESULTS_DIR}/11c_logreg_v3_surname.txt")
sys.stdout = tee

# ---- Load data ----
train_proc = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test_proc = pd.read_csv(f"{DATA_DIR}/test_processed.csv")
raw_train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")

test_ids = test_proc["PassengerId"]
test_proc = test_proc.drop(columns=["PassengerId"])

X_base = train_proc.drop(columns=["Survived"])
y = train_proc["Survived"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---- Feature engineering helpers ----

def add_interaction_features(df):
    """Stage A: Pclass one-hot + Sex x Pclass interactions."""
    df = df.copy()
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
    df["Sex_x_Pclass_2"] = df["Sex"] * df["Pclass_2"]
    df["Sex_x_Pclass_3"] = df["Sex"] * df["Pclass_3"]
    df = df.drop(columns=["Pclass"])
    return df


def add_deck_features(df, cabin_series):
    """Stage B: grouped deck dummies from cabin letter."""
    df = df.copy()
    deck = cabin_series.fillna("U").str[0].replace("T", "U")
    df["Deck_ABC"] = deck.isin(["A", "B", "C"]).astype(int)
    df["Deck_DE"] = deck.isin(["D", "E"]).astype(int)
    df["Deck_FG"] = deck.isin(["F", "G"]).astype(int)
    return df


def extract_surname(name_series):
    """Extract surname from Name column (text before the comma)."""
    return name_series.str.split(",").str[0].str.strip()


def compute_surname_group_size(surnames):
    """Count passengers per surname. No target info — always safe."""
    counts = surnames.value_counts()
    return surnames.map(counts).values


def compute_surname_surv_hint_oof(X_df, y_series, surnames, cv_splits):
    """Compute out-of-fold surname survival hints.

    For each fold:
      - On the train split: compute mean survival per surname
      - On the val split: map surname -> train survival mean
      - Unseen surnames in val get the global train mean as fallback

    Returns array of OOF-encoded survival hints.
    """
    oof_hints = np.full(len(X_df), np.nan)

    for train_idx, val_idx in cv_splits:
        train_surnames = surnames.iloc[train_idx]
        train_y = y_series.iloc[train_idx]

        # Compute surname survival stats on train fold only
        surname_stats = pd.DataFrame({
            "surname": train_surnames,
            "survived": train_y,
        }).groupby("surname")["survived"].mean()

        # Map to validation fold
        val_surnames = surnames.iloc[val_idx]
        global_mean = train_y.mean()  # fallback for unseen surnames
        oof_hints[val_idx] = val_surnames.map(surname_stats).fillna(global_mean).values

    return oof_hints


def compute_surname_surv_hint_full(y_series, train_surnames, target_surnames):
    """Compute surname survival hint using full training set.

    For final test predictions: compute stats on all training data,
    map to target (test) set. Unseen surnames get global mean.
    """
    surname_stats = pd.DataFrame({
        "surname": train_surnames,
        "survived": y_series,
    }).groupby("surname")["survived"].mean()

    global_mean = y_series.mean()
    return target_surnames.map(surname_stats).fillna(global_mean).values


# ---- Build base features (Stage A+B) ----
X = add_interaction_features(X_base)
X = add_deck_features(X, raw_train["Cabin"])
test_X = add_interaction_features(test_proc)
test_X = add_deck_features(test_X, raw_test["Cabin"])

# Extract surnames
train_surnames = extract_surname(raw_train["Name"])
test_surnames = extract_surname(raw_test["Name"])

print("=" * 60)
print("STAGE C: SURNAME FEATURES")
print("=" * 60)
print(f"Base features (A+B): {X.shape[1]}")
print(f"Feature list: {list(X.columns)}")
print()

# ---- Surname statistics ----
train_group_sizes = compute_surname_group_size(train_surnames)
print("--- Surname Group Size Distribution (train) ---")
for sz in sorted(pd.Series(train_group_sizes).unique()):
    n_passengers = (train_group_sizes == sz).sum()
    n_groups = n_passengers // sz if sz > 0 else 0
    surv_rate = y[train_group_sizes == sz].mean()
    print(f"  Size {sz}: {n_passengers:3d} passengers ({n_groups:3d} groups)  surv_rate={surv_rate:.3f}")
print()


# ---- Custom CV evaluation function ----

def custom_cv_evaluate(X_df, y_series, C, extra_train_fn=None, extra_test_fn=None,
                       extra_cols_train=None, extra_cols_test=None):
    """Run custom CV loop with optional per-fold feature computation.

    extra_train_fn: function(train_idx, val_idx) -> (train_extra, val_extra) arrays
    extra_cols_train/test: pre-computed columns to add (no per-fold computation needed)
    """
    fold_scores = []
    splits = list(cv.split(X_df, y_series))

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train = X_df.iloc[train_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        y_train = y_series.iloc[train_idx]
        y_val = y_series.iloc[val_idx]

        # Add pre-computed extra columns
        if extra_cols_train is not None:
            for col_name, col_values in extra_cols_train.items():
                X_train[col_name] = col_values[train_idx]
                X_val[col_name] = col_values[val_idx]

        # Add per-fold computed features
        if extra_train_fn is not None:
            train_extra, val_extra = extra_train_fn(train_idx, val_idx)
            for col_name in train_extra:
                X_train[col_name] = train_extra[col_name]
                X_val[col_name] = val_extra[col_name]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        fold_scores.append(accuracy_score(y_val, y_pred))

    return np.array(fold_scores)


def sweep_and_report(label, X_df, y_series, extra_train_fn=None,
                     extra_cols_train=None, C_values=None):
    """Run C sweep with custom CV and report results."""
    if C_values is None:
        C_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"\n--- {label} ---")
    n_feat = X_df.shape[1]
    if extra_cols_train:
        n_feat += len(extra_cols_train)
    print(f"Features: {n_feat}")
    print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}")
    print("-" * 30)

    best_C = None
    best_score = 0
    best_scores = None
    for C in C_values:
        scores = custom_cv_evaluate(
            X_df, y_series, C,
            extra_train_fn=extra_train_fn,
            extra_cols_train=extra_cols_train,
        )
        marker = ""
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
            best_scores = scores
            marker = " <-- best"
        print(f"{C:>8.2f}  {scores.mean():>8.4f}  {scores.std():>8.4f}{marker}")

    print(f"  Best: C={best_C}, CV={best_score:.4f} ± {best_scores.std():.4f}")
    return best_score, best_C, best_scores


# ================================================================
# C0: Baseline (Stage A+B, no surname features) via custom CV
# ================================================================
c0_score, c0_C, c0_scores = sweep_and_report("C0: Baseline (A+B, no surname)", X, y)

# ================================================================
# C1: SurnameGroupSize only (safe — no target info)
# ================================================================
extra_c1 = {"SurnameGroupSize": train_group_sizes.astype(float)}
c1_score, c1_C, c1_scores = sweep_and_report(
    "C1: + SurnameGroupSize", X, y, extra_cols_train=extra_c1
)

# ================================================================
# C2: SurnameSurvHint only (OOF encoded — requires per-fold computation)
# ================================================================
cv_splits = list(cv.split(X, y))


def surv_hint_oof_fn(train_idx, val_idx):
    """Compute SurnameSurvHint per fold."""
    train_sn = train_surnames.iloc[train_idx]
    train_y = y.iloc[train_idx]

    surname_stats = pd.DataFrame({
        "surname": train_sn,
        "survived": train_y,
    }).groupby("surname")["survived"].mean()

    global_mean = train_y.mean()

    val_sn = train_surnames.iloc[val_idx]
    train_hints = train_sn.map(surname_stats).fillna(global_mean).values
    val_hints = val_sn.map(surname_stats).fillna(global_mean).values

    return {"SurnameSurvHint": train_hints}, {"SurnameSurvHint": val_hints}


c2_score, c2_C, c2_scores = sweep_and_report(
    "C2: + SurnameSurvHint (OOF)", X, y, extra_train_fn=surv_hint_oof_fn
)

# ================================================================
# C3: Both SurnameGroupSize + SurnameSurvHint
# ================================================================

def both_surname_oof_fn(train_idx, val_idx):
    """Compute SurnameSurvHint per fold (GroupSize is pre-computed)."""
    train_sn = train_surnames.iloc[train_idx]
    train_y = y.iloc[train_idx]

    surname_stats = pd.DataFrame({
        "surname": train_sn,
        "survived": train_y,
    }).groupby("surname")["survived"].mean()

    global_mean = train_y.mean()

    val_sn = train_surnames.iloc[val_idx]
    train_hints = train_sn.map(surname_stats).fillna(global_mean).values
    val_hints = val_sn.map(surname_stats).fillna(global_mean).values

    return {"SurnameSurvHint": train_hints}, {"SurnameSurvHint": val_hints}


c3_score, c3_C, c3_scores = sweep_and_report(
    "C3: + SurnameGroupSize + SurnameSurvHint (OOF)", X, y,
    extra_train_fn=both_surname_oof_fn,
    extra_cols_train=extra_c1,
)

# ================================================================
# SHUFFLE ABLATION: Confirm SurnameSurvHint lift isn't leakage
# ================================================================
print("\n" + "=" * 60)
print("SHUFFLE ABLATION")
print("=" * 60)
print("Randomize surname labels — if lift persists, it's leakage.\n")

np.random.seed(42)
shuffled_surnames = train_surnames.sample(frac=1, random_state=42).reset_index(drop=True)


def shuffled_surv_hint_fn(train_idx, val_idx):
    """Same as surv_hint_oof_fn but with shuffled surnames."""
    train_sn = shuffled_surnames.iloc[train_idx]
    train_y = y.iloc[train_idx]

    surname_stats = pd.DataFrame({
        "surname": train_sn,
        "survived": train_y,
    }).groupby("surname")["survived"].mean()

    global_mean = train_y.mean()

    val_sn = shuffled_surnames.iloc[val_idx]
    train_hints = train_sn.map(surname_stats).fillna(global_mean).values
    val_hints = val_sn.map(surname_stats).fillna(global_mean).values

    return {"SurnameSurvHint": train_hints}, {"SurnameSurvHint": val_hints}


shuffled_group_sizes = compute_surname_group_size(shuffled_surnames).astype(float)
extra_shuffled = {"SurnameGroupSize": shuffled_group_sizes}

shuf_score, shuf_C, shuf_scores = sweep_and_report(
    "Shuffled: GroupSize + SurvHint (should match baseline)",
    X, y,
    extra_train_fn=shuffled_surv_hint_fn,
    extra_cols_train=extra_shuffled,
)

print(f"\nShuffle ablation result:")
print(f"  Baseline (no surname):  {c0_score:.4f}")
print(f"  Real surnames (C3):     {c3_score:.4f}  delta={c3_score - c0_score:+.4f}")
print(f"  Shuffled surnames:      {shuf_score:.4f}  delta={shuf_score - c0_score:+.4f}")
if abs(shuf_score - c0_score) < 0.005:
    print("  PASS: Shuffled lift is negligible — feature is safe")
else:
    print("  WARNING: Shuffled surnames still show lift — potential leakage!")
print()

# ================================================================
# STAGE C SUMMARY
# ================================================================
print("=" * 60)
print("STAGE C SUMMARY")
print("=" * 60)

all_configs = [
    ("C0: Baseline (A+B)", c0_score, c0_scores.std()),
    ("C1: + GroupSize", c1_score, c1_scores.std()),
    ("C2: + SurvHint (OOF)", c2_score, c2_scores.std()),
    ("C3: + Both", c3_score, c3_scores.std()),
    ("Shuffled: Both", shuf_score, shuf_scores.std()),
]

for name, score, std in sorted(all_configs, key=lambda x: x[1], reverse=True):
    print(f"  {name:35s}  {score:.4f} ± {std:.4f}")

# ---- Decide what to keep ----
# Keep the best config that improves over baseline
best_surname_config = None
best_surname_score = c0_score
features_to_add = []

if c3_score > c0_score + 0.002:
    best_surname_config = "C3"
    best_surname_score = c3_score
    features_to_add = ["SurnameGroupSize", "SurnameSurvHint"]
elif c2_score > c0_score + 0.002:
    best_surname_config = "C2"
    best_surname_score = c2_score
    features_to_add = ["SurnameSurvHint"]
elif c1_score > c0_score + 0.002:
    best_surname_config = "C1"
    best_surname_score = c1_score
    features_to_add = ["SurnameGroupSize"]
else:
    best_surname_config = "None"
    features_to_add = []

print(f"\nRecommendation: {best_surname_config}")
print(f"Features to add: {features_to_add}")
print()

# ---- Generate submission from best config ----
# Train on full training set, predict test
print("--- Generating Submission ---")

# Use best C from the winning config
if best_surname_config == "C3":
    final_C = c3_C
elif best_surname_config == "C2":
    final_C = c2_C
elif best_surname_config == "C1":
    final_C = c1_C
else:
    final_C = c0_C

X_final = X.copy()
test_X_final = test_X.copy()

# Add surname features computed on full training set
if "SurnameGroupSize" in features_to_add:
    X_final["SurnameGroupSize"] = train_group_sizes.astype(float)
    test_group_sizes = compute_surname_group_size(
        pd.concat([train_surnames, test_surnames])
    )[len(train_surnames):]  # Only test portion
    # Actually, for test: count occurrences of each surname in TRAIN+TEST combined?
    # No — SurnameGroupSize should be computed on train+test combined since it's
    # just a count (no target info). But to be safe, compute on train only and
    # map to test.
    train_counts = train_surnames.value_counts()
    test_group_sizes_safe = test_surnames.map(train_counts).fillna(0).values + \
        test_surnames.map(test_surnames.value_counts()).fillna(0).values
    # Actually simplest: count in train+test combined (no target leakage for counts)
    all_surnames = pd.concat([train_surnames, test_surnames], ignore_index=True)
    all_counts = all_surnames.value_counts()
    test_X_final["SurnameGroupSize"] = test_surnames.map(all_counts).values.astype(float)
    # Re-compute train with combined counts too for consistency
    X_final["SurnameGroupSize"] = train_surnames.map(all_counts).values.astype(float)

if "SurnameSurvHint" in features_to_add:
    # For final model: compute on full training set, map to test
    X_final["SurnameSurvHint"] = compute_surname_surv_hint_full(
        y, train_surnames, train_surnames
    )
    test_X_final["SurnameSurvHint"] = compute_surname_surv_hint_full(
        y, train_surnames, test_surnames
    )

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=final_C, max_iter=2000, random_state=42))
])
pipe.fit(X_final, y)
train_acc = pipe.score(X_final, y)
test_pred = pipe.predict(test_X_final)

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logreg_v3c_surname.csv", index=False)
print(f"Submission saved: submissions/logreg_v3c_surname.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Best CV: {best_surname_score:.4f}")
print(f"Gap: {train_acc - best_surname_score:.4f}")
print()

# ---- Test prediction profile ----
print("--- Test Prediction Profile ---")
test_profile = pd.DataFrame({
    "Sex": raw_test["Sex"],
    "Pclass": raw_test["Pclass"],
    "Predicted": test_pred,
})
for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (test_profile["Sex"] == sex) & (test_profile["Pclass"] == pclass)
        sub = test_profile[mask]
        n = len(sub)
        pred_surv = sub["Predicted"].sum()
        pred_rate = sub["Predicted"].mean()
        print(f"  {sex:6s} Pclass {pclass}: n={n:3d}  predicted_survived={pred_surv:3.0f}  rate={pred_rate:.3f}")

total_female_dead = ((test_profile["Sex"] == "female") & (test_profile["Predicted"] == 0)).sum()
print(f"\n  Total female predicted dead: {total_female_dead}")

# ---- Diff vs v2 ----
v2_sub = pd.read_csv(f"{DATA_DIR}/../submissions/logreg_v2.csv")
diff = (submission["Survived"] != v2_sub["Survived"])
n_diff = diff.sum()
print(f"\n--- Diff vs v2 ---")
print(f"  Passengers changed: {n_diff} / {len(submission)}")
if n_diff > 0:
    changed_ids = submission.loc[diff, "PassengerId"].values
    changed_v2 = v2_sub.loc[diff, "Survived"].values
    changed_v3 = submission.loc[diff, "Survived"].values
    for pid, old, new in zip(changed_ids, changed_v2, changed_v3):
        sex = raw_test.loc[raw_test["PassengerId"] == pid, "Sex"].values[0]
        pclass = raw_test.loc[raw_test["PassengerId"] == pid, "Pclass"].values[0]
        name = raw_test.loc[raw_test["PassengerId"] == pid, "Name"].values[0]
        print(f"    PassengerId {pid}: {old} -> {new}  ({sex}, Pclass {pclass}, {name})")
print()

# ---- Coefficients ----
model = pipe.named_steps["model"]
coef_df = pd.DataFrame({
    "feature": X_final.columns,
    "coefficient": model.coef_[0],
    "abs_coefficient": np.abs(model.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

print("--- Coefficients ---")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {direction} {row['feature']:>20s}  {row['coefficient']:+.4f}  {bar}")

coef_df.to_csv(f"{RESULTS_DIR}/11c_logreg_v3_surname_coefficients.csv", index=False)
print(f"\nResults saved to: results/models/11c_logreg_v3_surname.txt")

tee.close()
