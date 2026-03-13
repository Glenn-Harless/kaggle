"""
Titanic Model 5: Neural Network

A simple feedforward neural network using scikit-learn's MLPClassifier.
Requires scaling — neural networks are sensitive to feature magnitude.
On a dataset this small, this is more of a learning exercise than a
competitive approach.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

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


tee = Tee(f"{RESULTS_DIR}/05_neural_network.txt")
sys.stdout = tee

# ---- Load processed data ----
train = pd.read_csv(f"{DATA_DIR}/train_processed.csv")
test = pd.read_csv(f"{DATA_DIR}/test_processed.csv")

test_ids = test["PassengerId"]
test = test.drop(columns=["PassengerId"])

X = train.drop(columns=["Survived"])
y = train["Survived"]

print("=" * 60)
print("MODEL 5: NEURAL NETWORK (MLPClassifier)")
print("=" * 60)
print(f"Features: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print()

# ---- Architecture ----
# Two hidden layers: 64 neurons -> 32 neurons
# ReLU activation, Adam optimizer
# Early stopping to prevent overfitting
hidden_layers = (64, 32)
print(f"Architecture: {X.shape[1]} -> {hidden_layers[0]} -> {hidden_layers[1]} -> 1")
print(f"Activation: ReLU")
print(f"Optimizer: Adam")
print(f"Early stopping: enabled (patience=20)")
print()

# ---- Cross-validation ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,        # hold out 10% of training for validation
        validation_fraction=0.1,
        n_iter_no_change=20,         # stop if no improvement for 20 epochs
        alpha=0.001,                 # L2 regularization
        batch_size=32,
        random_state=42,
    ))
])

scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("--- 5-Fold Cross-Validation ---")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.4f}")
print(f"\n  Mean:  {scores.mean():.4f}")
print(f"  Std:   {scores.std():.4f}")
print()

# ---- Compare to all models ----
print("--- Model Comparison ---")
results = {
    "Gender-only":          0.7870,
    "Logistic Regression":  0.8316,
    "Random Forest":        0.8316,
    "LightGBM":             0.8384,
    "XGBoost":              0.8451,
    "Neural Network":       scores.mean(),
}
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    marker = " <-- best" if score == max(results.values()) else ""
    print(f"  {name:25s}  {score:.4f}{marker}")
print()

# ---- Train on full data ----
pipeline.fit(X, y)
model = pipeline.named_steps["model"]

print(f"--- Training Details ---")
print(f"  Epochs run: {model.n_iter_}")
print(f"  Final loss: {model.loss_:.4f}")
print()

# ---- Confusion matrix on full training set ----
y_pred = pipeline.predict(X)
cm = confusion_matrix(y, y_pred)
print("--- Confusion Matrix (on training data) ---")
print(f"  Predicted:     Died  Survived")
print(f"  Actual Died:   {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"  Actual Surv:   {cm[1][0]:4d}    {cm[1][1]:4d}")
print()

print("--- Classification Report (on training data) ---")
print(classification_report(y, y_pred, target_names=["Died", "Survived"]))

# ---- Overfitting check ----
train_accuracy = pipeline.score(X, y)
print(f"--- Overfitting Check ---")
print(f"  Training accuracy:  {train_accuracy:.4f}")
print(f"  CV accuracy:        {scores.mean():.4f}")
print(f"  Gap:                {train_accuracy - scores.mean():.4f}")
if train_accuracy - scores.mean() > 0.05:
    print(f"  Warning: Gap > 5% — potential overfitting")
else:
    print(f"  Gap < 5% — looks healthy")
print()

# ---- Prediction confidence distribution ----
y_proba = pipeline.predict_proba(X)[:, 1]
print("--- Prediction Confidence ---")
print(f"  Predictions > 0.9 (very confident survive): {(y_proba > 0.9).sum()}")
print(f"  Predictions < 0.1 (very confident die):     {(y_proba < 0.1).sum()}")
print(f"  Predictions 0.4-0.6 (uncertain):            {((y_proba >= 0.4) & (y_proba <= 0.6)).sum()}")
print()

# ---- Neural network specific: layer weights analysis ----
print("--- Layer Weights Summary ---")
for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"  Layer {i+1}: {weights.shape[0]} -> {weights.shape[1]}  "
          f"(weight range: {weights.min():.3f} to {weights.max():.3f}, "
          f"mean abs: {np.abs(weights).mean():.3f})")
print()

# ---- Generate submission ----
test_pred = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_pred})
submission.to_csv(f"{DATA_DIR}/../submissions/neural_network.csv", index=False)
print(f"Submission saved: submissions/neural_network.csv")
print(f"Predicted survival rate in test: {test_pred.mean():.3f}")

print(f"\nResults saved to: results/models/05_neural_network.txt")

tee.close()
