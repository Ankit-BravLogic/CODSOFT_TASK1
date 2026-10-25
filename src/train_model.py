
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the labeled training dataset
df = pd.read_csv(
    DATA_DIR / "train_data.txt",
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"],
    on_bad_lines="error",
)

# Combine title and plot description
df["text"] = (
    df["title"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
)

X = df["text"]
y = df["genre"].str.strip()

# Reserve 20% for validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Number of genres:", y.nunique())

# TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=100_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    ),
])

print("\nTraining model...")
model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
print(f"\nValidation accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_val, predictions, zero_division=0))

# Save the complete fitted pipeline
joblib.dump(model, MODEL_DIR / "genre_classifier.pkl")

# Save validation metrics
report = classification_report(
    y_val, predictions, output_dict=True, zero_division=0
)
with open(REPORT_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Plot genre distribution
plt.figure(figsize=(12, 8))
df["genre"].value_counts().sort_values().plot(kind="bar")
plt.title("Movie Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig(REPORT_DIR / "genre_distribution.png", dpi=150)
plt.close()

# Plot confusion matrix for the most frequent genres
top_genres = y.value_counts().head(10).index.tolist()
mask = y_val.isin(top_genres)

cm = confusion_matrix(
    y_val[mask],
    pd.Series(predictions, index=y_val.index)[mask],
    labels=top_genres,
)

plt.figure(figsize=(12, 9))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=top_genres,
    yticklabels=top_genres,
    cmap="Blues",
)
plt.title("Validation Confusion Matrix — Top 10 Genres")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("\nSaved model:", MODEL_DIR / "genre_classifier.pkl")
print("Saved reports in:", REPORT_DIR)
