from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from nlp_utils import clean_text

# CSV yahi rakhi hai:
DATA_PATH = Path("data/fake_job_postings.csv")

# Tumhare folder ka naam "Models" hai (capital M)
MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(exist_ok=True)

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "fake_job_model.pkl"


def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # large CSV hai isliye low_memory=False
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Kaggle dataset me ye columns hote hain:
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

    # ek combined text column
    df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

    if "fraudulent" not in df.columns:
        raise KeyError("Column 'fraudulent' not found in dataset.")

    y = df["fraudulent"]

    # basic cleaning
    df["clean_text"] = df["text"].apply(clean_text)
    X_text = df["clean_text"]

    return X_text, y


def main():
    print("📥 Loading and preparing data...")
    X_text, y = load_and_prepare()
    print(f"Total samples: {len(y)}")

    print("🔤 TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 Training model (Logistic Regression)...")
    model = LogisticRegression(max_iter=300, n_jobs=-1)
    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    print("💾 Saving model and vectorizer...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
