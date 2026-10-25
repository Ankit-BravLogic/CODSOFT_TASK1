
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(
    DATA_DIR / "train_data.txt",
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"],
    on_bad_lines="error",
)

X = (
    df["title"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
)
y = df["genre"].str.strip()

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Compare two classifiers with identical TF-IDF settings.
for name, classifier in [
    (
        "Logistic Regression",
        LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight="balanced",
            random_state=42,
        ),
    ),
    (
        "Linear SVM",
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ),
    ),
]:
    print(f"\nTraining {name}...", flush=True)

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
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted", zero_division=0
    )

    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")
    print(f"Weighted F1:{weighted_f1:.4f}")

    if name == "Linear SVM":
        joblib.dump(model, MODEL_DIR / "genre_classifier_svm.pkl")
        print("Saved models/genre_classifier_svm.pkl")
