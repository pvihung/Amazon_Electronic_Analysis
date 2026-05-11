"""
Step 1 — ML Filter: Digital Devices from Electronics Metadata
Trains (or loads) a TF-IDF + Logistic Regression model that classifies
whether a product title belongs to a digital device category.
Scores the full metadata table and writes results back to BigQuery.
"""

import os
import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Config 
PROJECT_ID      = os.environ.get("GCP_PROJECT", "cs163-project-487801")
TRAIN_TABLE     = f"{PROJECT_ID}.amazon_electronics.ml_sample2"
METADATA_TABLE  = f"{PROJECT_ID}.amazon_electronics.meta_Electronics"
OUTPUT_TABLE    = f"{PROJECT_ID}.amazon_digital_devices_cleaned.metadata_digital_device_result"
MODEL_PATH      = "pipeline/digital_device_model.joblib"
PRED_THRESHOLD  = 0.799          # 80 % confidence to reduce noise
METADATA_CATEGORIES = (
    "Computers", "All Electronics", "Amazon Devices", "Office Products"
)

# BigQuery client
client = bigquery.Client(project=PROJECT_ID)


# Training
def load_training_data() -> pd.DataFrame:
    query = f"""
        SELECT title, label
        FROM `{TRAIN_TABLE}`
        WHERE title IS NOT NULL
          AND label IS NOT NULL
    """
    return client.query(query).to_dataframe()


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])


def train_and_evaluate(model: Pipeline, df: pd.DataFrame) -> Pipeline:
    x_train, x_test, y_train, y_test = train_test_split(
        df["title"], df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"] if df["label"].nunique() > 1 else None,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n",
          classification_report(y_test, pred, digits=4))

    # Feature importance summary
    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefs         = model.named_steps["clf"].coef_[0]
    top_digital   = feature_names[np.argsort(coefs)[-20:]]
    top_non_dig   = feature_names[np.argsort(coefs)[:10]]
    print("\nTop digital indicators  :", top_digital)
    print("Top non-digital indicators:", top_non_dig)

    return model


def get_or_train_model() -> Pipeline:
    """Load saved model, or retrain if not found."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    print("No saved model found — training from scratch …")
    df    = load_training_data()
    model = train_and_evaluate(build_model(), df)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


# Scoring 
def load_metadata() -> pd.DataFrame:
    categories = ", ".join(f"'{c}'" for c in METADATA_CATEGORIES)
    query = f"""
        SELECT main_category, title, parent_asin,
               average_rating, rating_number, price
        FROM `{METADATA_TABLE}`
        WHERE title IS NOT NULL
          AND main_category IN ({categories})
    """
    return client.query(query).to_dataframe()


def score_and_upload(model: Pipeline, df: pd.DataFrame) -> None:
    df = df.copy()
    df["prob_digital"] = model.predict_proba(df["title"])[:, 1]
    df["pred_label"]   = (df["prob_digital"] >= PRED_THRESHOLD).astype(int)

    results = df[[
        "main_category", "title", "parent_asin",
        "average_rating", "rating_number", "price",
        "prob_digital", "pred_label",
    ]]

    job = client.load_table_from_dataframe(
        results,
        OUTPUT_TABLE,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()
    print(f"Uploaded {len(results):,} rows → {OUTPUT_TABLE}")


# Entry point
def run() -> None:
    print("=" * 60)
    print("STEP 1 — ML Filtering of Digital Devices")
    print("=" * 60)
    model    = get_or_train_model()
    metadata = load_metadata()
    print(f"Scoring {len(metadata):,} products …")
    score_and_upload(model, metadata)
    print("Step 1 complete.\n")


if __name__ == "__main__":
    run()
