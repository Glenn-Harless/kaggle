"""
Titanic Model 11b: Logistic Regression v3 — Stage B (Fare & Deck)

Starts from Stage A's best feature set (interactions).
Tests fare refinement and deck encoding separately, in order:
  B1: log1p(Fare) vs raw Fare
  B2: Grouped deck features (only after B1 is resolved)

Deck groups:
  Deck_ABC (high decks, 1st class)
  Deck_DE  (mid decks)
  Deck_FG  (low decks)
  Deck_U   (unknown/missing — reference category, dropped)
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
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


tee = Tee(f"{RESULTS_DIR}/11b_logreg_v3_fare_deck.txt")
sys.stdout = tee

# ---- Load data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")
raw_train = pd.read_csv(f"{DATA_DIR}/train.csv")
raw_test = pd.read_csv(f"{DATA_DIR}/test.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def add_interaction_features(df):
    """Stage A interaction features."""
    df = df.copy()
    df["Pclass_2"] = (df["Pclass"] == 2).astype(int)
    df["Pclass_3"] = (df["Pclass"] == 3).astype(int)
    df["Sex_x_Pclass_2"] = df["Sex"] * df["Pclass_2"]
    df["Sex_x_Pclass_3"] = df["Sex"] * df["Pclass_3"]
    df = df.drop(columns=["Pclass"])
    return df


def evaluate(X_eval, y_eval, label, C_values=None):
    """Run regularization sweep and return best score, best C, and scores."""
    if C_values is None:
        C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    print(f"\n--- {label} ---")
    print(f"Features ({X_eval.shape[1]}): {list(X_eval.columns)}")
    print(f"{'C':>8}  {'CV Mean':>8}  {'CV Std':>8}  {'Gap':>8}")
    print("-" * 40)

    best_C = None
    best_score = 0
    best_scores = None
    for C in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000, random_state=42))
        ])
        scores = cross_val_score(pipe, X_eval, y_eval, cv=cv, scoring="accuracy")
        pipe.fit(X_eval, y_eval)
        train_acc = pipe.score(X_eval, y_eval)
        gap = train_acc - scores.mean()
        marker = ""
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C = C
            best_scores = scores
            marker = " <-- best"
        print(f"{C:>8.2f}  {scores.mean():>8.4f}  {scores.std():>8.4f}  {gap:>+8.4f}{marker}")

    print(f"  Best: C={best_C}, CV={best_score:.4f} ± {best_scores.std():.4f}")
    return best_score, best_C, best_scores


# ================================================================
# STAGE A BASELINE (for comparison)
# ================================================================
print("=" * 60)
print("STAGE B: FARE & DECK REFINEMENT")
print("=" * 60)

X_base = add_interaction_features(X)
test_base = add_interaction_features(test)

base_score, base_C, base_scores = evaluate(X_base, y, "B0: Stage A baseline (raw Fare)")

# ================================================================
# B1: log1p(Fare) INSTEAD OF raw Fare
# ================================================================
X_b1 = X_base.copy()
X_b1["Fare"] = np.log1p(X_b1["Fare"])
test_b1 = test_base.copy()
test_b1["Fare"] = np.log1p(test_b1["Fare"])

b1_score, b1_C, b1_scores = evaluate(X_b1, y, "B1: log1p(Fare) replacing raw Fare")

# ================================================================
# B1-alt: log1p(Fare) ALONGSIDE raw Fare
# ================================================================
X_b1alt = X_base.copy()
X_b1alt["LogFare"] = np.log1p(X_b1alt["Fare"])
test_b1alt = test_base.copy()
test_b1alt["LogFare"] = np.log1p(test_b1alt["Fare"])

b1alt_score, b1alt_C, b1alt_scores = evaluate(X_b1alt, y, "B1-alt: log1p(Fare) + raw Fare (both)")

# ---- B1 Summary ----
print("\n" + "=" * 60)
print("B1 FARE SUMMARY")
print("=" * 60)
fare_results = [
    ("Raw Fare (baseline)", base_score, base_scores.std()),
    ("log1p(Fare) only", b1_score, b1_scores.std()),
    ("Both raw + log1p", b1alt_score, b1alt_scores.std()),
]
for name, score, std in sorted(fare_results, key=lambda x: x[1], reverse=True):
    print(f"  {name:25s}  {score:.4f} ± {std:.4f}")

# Pick the best fare variant for B2
if b1_score >= base_score and b1_score >= b1alt_score:
    fare_winner = "log1p"
    X_fare_best = X_b1
    test_fare_best = test_b1
    fare_best_C = b1_C
    fare_best_score = b1_score
elif b1alt_score > base_score and b1alt_score > b1_score:
    fare_winner = "both"
    X_fare_best = X_b1alt
    test_fare_best = test_b1alt
    fare_best_C = b1alt_C
    fare_best_score = b1alt_score
else:
    fare_winner = "raw"
    X_fare_best = X_base
    test_fare_best = test_base
    fare_best_C = base_C
    fare_best_score = base_score

print(f"\nFare winner: {fare_winner} (CV={fare_best_score:.4f})")
print()

# ================================================================
# B2: Grouped Deck features (added to best fare variant)
# ================================================================
print("=" * 60)
print("B2: DECK FEATURES")
print("=" * 60)


def extract_deck(cabin_series):
    """Extract deck letter from Cabin, group into ABC/DE/FG/U."""
    deck = cabin_series.fillna("U").str[0]
    # Map T -> U (only 1 occurrence, treat as unknown)
    deck = deck.replace("T", "U")
    return deck


def add_deck_features(df, deck_series):
    """Add grouped deck dummies to feature dataframe."""
    df = df.copy()
    df["Deck_ABC"] = deck_series.isin(["A", "B", "C"]).astype(int)
    df["Deck_DE"] = deck_series.isin(["D", "E"]).astype(int)
    df["Deck_FG"] = deck_series.isin(["F", "G"]).astype(int)
    # Deck_U is reference (dropped)
    return df


train_deck = extract_deck(raw_train["Cabin"])
test_deck = extract_deck(raw_test["Cabin"])

# Show deck distribution
print("\nDeck distribution (train):")
for d in ["A", "B", "C", "D", "E", "F", "G", "U"]:
    n = (train_deck == d).sum()
    surv = y[train_deck == d].mean() if n > 0 else 0
    print(f"  {d}: n={n:3d}  surv_rate={surv:.3f}")

print("\nGrouped deck distribution (train):")
for group, letters in [("ABC", ["A","B","C"]), ("DE", ["D","E"]), ("FG", ["F","G"]), ("U", ["U"])]:
    mask = train_deck.isin(letters)
    n = mask.sum()
    surv = y[mask].mean() if n > 0 else 0
    print(f"  {group}: n={n:3d}  surv_rate={surv:.3f}")

# B2a: Deck only (no fare change from winner)
X_b2 = add_deck_features(X_fare_best, train_deck)
test_b2 = add_deck_features(test_fare_best, test_deck)

b2_score, b2_C, b2_scores = evaluate(X_b2, y, f"B2: Best fare ({fare_winner}) + Deck groups")

# But also test deck with the base (raw fare + interactions) as a sanity check
X_b2_raw = add_deck_features(X_base, train_deck)
test_b2_raw = add_deck_features(test_base, test_deck)

b2_raw_score, b2_raw_C, b2_raw_scores = evaluate(X_b2_raw, y, "B2-alt: Raw Fare + Deck groups (no fare change)")

# ---- B2 Summary ----
print("\n" + "=" * 60)
print("B2 DECK SUMMARY")
print("=" * 60)
deck_results = [
    (f"No deck ({fare_winner} fare)", fare_best_score),
    (f"+ Deck groups ({fare_winner} fare)", b2_score),
    ("+ Deck groups (raw fare)", b2_raw_score),
]
for name, score in sorted(deck_results, key=lambda x: x[1], reverse=True):
    print(f"  {name:35s}  {score:.4f}")

# ================================================================
# FINAL STAGE B SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("STAGE B FINAL SUMMARY")
print("=" * 60)

all_configs = [
    ("A: Interactions only (raw Fare)", base_score, base_scores.std(), X_base, test_base, base_C),
    ("B1: log1p(Fare)", b1_score, b1_scores.std(), X_b1, test_b1, b1_C),
    ("B1-alt: Both fares", b1alt_score, b1alt_scores.std(), X_b1alt, test_b1alt, b1alt_C),
    (f"B2: {fare_winner} fare + Deck", b2_score, b2_scores.std(), X_b2, test_b2, b2_C),
    ("B2-alt: raw fare + Deck", b2_raw_score, b2_raw_scores.std(), X_b2_raw, test_b2_raw, b2_raw_C),
]

for name, score, std, _, _, _ in sorted(all_configs, key=lambda x: x[1], reverse=True):
    marker = ""
    print(f"  {name:38s}  {score:.4f} ± {std:.4f}{marker}")

# Pick the overall best
best_config = max(all_configs, key=lambda x: x[1])
print(f"\nBest config: {best_config[0]} (CV={best_config[1]:.4f})")

# ---- Generate submission from best config ----
best_name, _, _, X_best, test_best, best_C_final = best_config
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=best_C_final, max_iter=2000, random_state=42))
])
pipe.fit(X_best, y)
test_pred = pipe.predict(test_best)

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/logreg_v3b_fare_deck.csv", index=False)
print(f"\nSubmission saved: submissions/logreg_v3b_fare_deck.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

# ---- Subgroup analysis on test ----
print("\n--- Test Prediction Profile (best config) ---")
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
        print(f"    PassengerId {pid}: {old} -> {new}  ({sex}, Pclass {pclass})")
print()

# ---- Coefficients for best config ----
model = pipe.named_steps["model"]
coef_df = pd.DataFrame({
    "feature": X_best.columns,
    "coefficient": model.coef_[0],
    "abs_coefficient": np.abs(model.coef_[0])
}).sort_values("abs_coefficient", ascending=False)

print("--- Coefficients (best config) ---")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    bar = "#" * int(row["abs_coefficient"] * 10)
    print(f"  {direction} {row['feature']:>18s}  {row['coefficient']:+.4f}  {bar}")

coef_df.to_csv(f"{RESULTS_DIR}/11b_logreg_v3_fare_deck_coefficients.csv", index=False)
print(f"\nResults saved to: results/models/11b_logreg_v3_fare_deck.txt")

tee.close()
