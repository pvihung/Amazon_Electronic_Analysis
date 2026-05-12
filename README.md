# Amazon Electronics Review Analysis

A data science project that collects, cleans, and analyzes millions of Amazon customer reviews for digital devices (laptops, tablets, desktops). The project covers exploratory data analysis, statistical hypothesis testing, and a two-stage Aspect-Based Sentiment Analysis (ABSA) pipeline - all presented through an interactive web dashboard.
## Website
<a href=""> https://cs163-amazon-review-analysis.uc.r.appspot.com/ </a>
---

## Setup

**Requirements:** Python 3.11, a Google Cloud project, and GCP credentials configured locally.

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Environment variables** (set in shell or `.env`):

| Variable | Description |
|---|---|
| `GCP_PROJECT` | GCP project ID (default: `cs163-amazon-analysis`) |
| `GCS_BUCKET_NAME` | Cloud Storage bucket name |
| `ABSA_API_URL` | URL of the deployed ABSA inference service |

**Run the web app locally:**
```bash
python -m app.main
```

**Run the data pipeline:**
```bash
python -m pipeline.run_pipeline              # all steps 
python -m pipeline.run_pipeline --steps 2 3  # specific steps only
```

---

## Data Cleaning Pipeline Overview

```
Amazon Electronics Data + Metadata
        │
        ▼
[Step 1] ML Filter (pipeline/step1_ml_filter.py)
  TF-IDF + Logistic Regression classifies product titles
  as digital devices or not.
        │
        ▼
[Step 2] BigQuery SQL (pipeline/step2_bq_queries.py)
  Joins reviews ↔ product metadata, deduplicates rows,
  and produces a clean combined table in BigQuery.
        │
        ▼
[Step 3] EDA Data Prep (pipeline/step3_eda_data.py)
  Cleans timestamps, assigns device categories (laptop/tablet/desktop),
  extracts brands, translates non-English reviews, applies VADER
  sentiment scoring, assigns price tiers. Output: dataset/eda_ready.csv
  (or BigQuery table in cloud mode).
        │
        ▼
[ABSA Training] — Google Colab (external)
  Domain-Adaptive Pre-Training (DAPT) on RoBERTa, then fine-tuned
  as a multitask model (M1) for aspect detection and a DeBERTa model
  (M2) for aspect-level sentiment classification.
  Colab notebook: https://colab.research.google.com/drive/1bem5kz0FBWC22zAFnU1D_xa6oHxdUZPW
        │
        ▼
[ABSA Inference API] (absa-api/)
  FastAPI service containerized with Docker and deployed on Cloud Run.
  Exposes /predict and /predict/batch endpoints.
        │
        ▼
[Dash Web App] (app/)
  Interactive dashboard deployed on Google App Engine.
  Visualizes EDA findings, hypothesis tests, and live ABSA demos.
```

---

## Repository Structure

```
Amazon_Electronic_Analysis/
├── data_cleaning_pipeline/          # Three-step data processing pipeline
│   ├── run_pipeline.py              # Orchestrator — runs all steps in order
│   ├── step1_ml_filter.py           # ML classifier to identify digital device products
│   ├── step2_bq_queries.py          # BigQuery SQL: join reviews ↔ products, deduplicate
│   └── step3_eda_data.py            # Feature engineering and EDA data preparation
│
├── absa-api/                        # Containerized ABSA inference service (FastAPI + Docker)
│   ├── app/
│   │   ├── main.py                  # FastAPI app with /predict and /predict/batch endpoints
│   │   ├── models.py                # Model architectures (M1: RoBERTa+LoRA, M2: DeBERTa)
│   │   └── schemas.py               # Pydantic request/response schemas
│   ├── dapt_roberta/                # Domain-adapted RoBERTa backbone (weights via Drive)
│   ├── model1_multitask/            # Multitask aspect-detection model (weights via Drive)
│   ├── model2_sentiment/            # Aspect-sentiment classifier (weights via Drive)
│   ├── Dockerfile
│   └── requirements.txt
│
├── app/                             # Dash web application (Google App Engine)
│   ├── main.py                      # Entry point; initializes Dash server
│   ├── layout.py                    # Shared layout components and helpers
│   ├── callbacks.py                 # Dash callback logic
│   └── pages/                       # One module per dashboard page
│       ├── overview.py              # Project overview and research questions
│       ├── dataset.py               # Dataset description page
│       ├── methods.py               # Methods and pipeline explanation
│       ├── models/                  # ABSA model pages and live demo
│       └── analytics/
│            ├── EDA                 # EDA and major findings
│            ├── Hypothesis 1
│            ├── Hypothesis 2
│            └── Hypothesis 3
│
├── eda/                             # EDA visualization modules used by the Dash app
│   ├── overview.py
│   ├── category.py
│   ├── ratings.py
│   ├── price.py
│   ├── time.py
│   ├── text.py
│   ├── correlation.py
│   ├── covid.py
│   ├── hypothesis1.py
│   ├── hypothesis2.py
│   └── hypothesis3.py
│
├── dataset/                         # Local CSV data files
│   ├── electronics.csv              # Original reviews dataset from Hugging Face
│   ├── metadata.csv                 # Original metadata dataset from Hugging Face
│   ├── digital_devices_reviews_no_duplicates.csv   # Post-Step-2 deduplicated data
│   ├── eda_ready.csv                # Post-Step-3 feature-engineered data
│   └── final.csv
│
├── models/                          # Supplementary model utilities used by Dash app
│   ├── data_processing.py           # Data processing helpers for model preparation
│   └── augmentation.py              # Data augmentation utilities
│
├── notebooks/                       # Jupyter notebooks used during development
│   ├── absa_pipeline_vfinal.ipynb   # Final ABSA pipeline notebook
│   ├── absa_model1.ipynb            # ABSA Model 1 development notebook
│   ├── absa_model2.ipynb            # ABSA Model 2 development notebook
│   ├── hypothesis.ipynb             # Hypothesis analysis notebook
│   ├── MLpipeline_filtering_metadata.ipynb   # Metadata filtering ML pipeline
│   └── Notebook.md                  # Notebook reminders and usage notes
│
├── app.yaml                         # Google App Engine deployment configuration
└── requirements.txt                 # Python dependencies for the web app
```

---

## System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Amazon Electronics Dataset                         │
│               (HuggingFace — reviews + product metadata)             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
            ╔═════════════════════════════════════════╗
            ║              DATA PIPELINE              ║
            ║                                         ║
            ║  ┌──────────────────────────────────┐   ║
            ║  │ Step 1 · ML Filter               │   ║
            ║  │ TF-IDF + Logistic Regression     │   ║
            ║  │ identifies digital device titles │   ║
            ║  └─────────────────┬────────────────┘   ║
            ║                    │                    ║
            ║  ┌─────────────────▼────────────────┐   ║
            ║  │ Step 2 · BigQuery SQL            │   ║
            ║  │ Join reviews ↔ product metadata  │   ║
            ║  │ Deduplicate rows                 │   ║
            ║  └─────────────────┬────────────────┘   ║
            ║                    │                    ║
            ║  ┌─────────────────▼────────────────┐   ║
            ║  │ Step 3 · EDA Data Prep           │   ║
            ║  │ VADER sentiment · brand extract  │   ║
            ║  │ translations · price tiers       │   ║
            ║  └─────────────────┬────────────────┘   ║
            ╚════════════════════╪════════════════════╝
                                 │
                     ┌───────────┴───────────┐
                     ▼                       ▼
              ┌─────────────┐        ┌──────────────┐
              │  BigQuery   │        │     GCS      │
              │  (tables)   │        │   (bucket)   │
              └──────┬──────┘        └──────────────┘
                     │
       ┌─────────────┘
       │
       │    ┌────────────────────────────────────────┐
       │    │     ABSA Training  (Google Colab)      │
       │    │                                        │
       │    │  DAPT pre-training → RoBERTa backbone  │
       │    │            ↓                           │
       │    │  M1 · RoBERTa + LoRA (multitask)       │
       │    │      aspect detection per category     │
       │    │            ↓                           │
       │    │  M2 · DeBERTa (sentiment classifier)   │
       │    │      positive / negative per aspect    │
       │    └────────────────────┬───────────────────┘
       │                         │
       │                         ▼
       │         ┌───────────────────────────────┐
       │         │     ABSA Inference API        │
       │         │     FastAPI · Docker          │
       │         │     Google Cloud Run          │
       │         │                               │
       │         │   GET  /health                │
       │         │   POST /predict               │
       │         │   POST /predict/batch         │
       │         └───────────────┬───────────────┘
       │                         │ HTTP REST
       │                         ▼
       │         ┌───────────────────────────────┐
       └────────▶│       Dash Web App            │
                 │     Google App Engine         │
                 │                               │
                 │  Overview · Dataset · EDA     │
                 │  Hypothesis · ABSA Live Demo  │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                           ┌──────────┐
                           │   User   │
                           │ Browser  │
                           └──────────┘
```

**Scalability:**
- **App Engine** scales automatically from 0 to 3 instances based on traffic, serving the Dash dashboard.
- **Cloud Run** handles the ABSA API as a stateless container; it scales independently and only runs when requests arrive (scale-to-zero).
- **BigQuery** is serverless and handles large-scale SQL queries without infrastructure management.
- The two services (dashboard and inference API) are decoupled — the API can be updated or scaled without redeploying the frontend.

---

## Docker Container Setup (ABSA API)

The inference service lives in `absa-api/` and is packaged as a Docker container.

### Prerequisites

- Docker Desktop installed and running
- Model weights downloaded from [Google Drive](https://drive.google.com/drive/folders/1__tGTtZLuD8jfU8MT3B8pLJhb96pdC_c?usp=drive_link) and placed under `absa-api/`:

```
absa-api/
├── dapt_roberta/          ← download from [Drive]
├── model1_multitask/      ← download from [Drive]
├── model2_sentiment/      ← download from [Drive]
├── app/
├── ├──main.py
├── ├──models.py
├── └──schemas.py 
├── Dockerfile
└── requirements.txt
```

### Build the image

```bash
cd absa-api
docker build -t absa-api .
```

The Dockerfile:
- Uses `python:3.11-slim` as the base image
- Installs CPU-only PyTorch (`torch==2.3.0`) to keep the image lean
- Pre-downloads the DeBERTa base model (`yangheng/deberta-v3-base-absa-v1.1`) at build time so the container has no outbound internet dependency at runtime
- Copies the three model weight directories into `/app/models_artifacts/`
- Exposes port `8080` and starts Uvicorn with a **single worker** (models are ~1.5 GB each; multiple workers would OOM)

### Run locally

```bash
docker run --rm -p 8080:8080 absa-api
```

Test the running container:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The battery life is excellent but the display is dim."}'
```

### Deploy to Cloud Run

```bash
# Authenticate and set project
gcloud auth login
gcloud config set project cs163-amazon-analysis

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/cs163-amazon-analysis/absa-api .

# Deploy
gcloud run deploy absa-api \
  --image gcr.io/cs163-amazon-analysis/absa-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

---

## Inference Service

The ABSA API runs as a FastAPI application on **Google Cloud Run**:

**URL:** `https://absa-api-467984634090.us-central1.run.app`

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | Run ABSA on a single review text |
| `POST` | `/predict/batch` | Run ABSA on a list of review texts |

**Inference pipeline (two-stage):**

1. **M1 — RoBERTa + LoRA (multitask):** Determines if a review is product-related and technical, then predicts which product aspects (e.g., battery, display, performance) are mentioned using per-aspect confidence thresholds.
2. **M2 — DeBERTa (sentiment):** For each detected aspect, classifies its sentiment as positive or negative using a (review, aspect) text-pair encoding.

The models are loaded once at container startup and kept in memory. A single Uvicorn worker is used to avoid out-of-memory errors (~1.5 GB per model).

**Training** was done on Google Colab:
[ABSA Training Notebook](https://colab.research.google.com/drive/1bem5kz0FBWC22zAFnU1D_xa6oHxdUZPW)

---

## Data in the Cloud

**Google BigQuery** (`cs163-amazon-analysis`):

| Dataset | Table | Description |
|---|---|---|
| `amazon_electronics` | `meta_Electronics` | Raw product metadata |
| `amazon_electronics` | `ml_sample2` | Labeled training data for the ML filter |
| `amazon_digital_devices_cleaned` | `metadata_digital_device_result` | ML-filtered product metadata |
| `amazon_digital_devices_cleaned` | `digital_devices_reviews_no_duplicates` | Deduplicated reviews joined with product data |
| `amazon_digital_devices_cleaned` | `eda_ready` | Feature-engineered dataset ready for analysis |

**Google Cloud Storage** — bucket `cs163-amazon-review-analysis-data`: stores intermediate and processed data files.

**Google Drive** — trained model weights (not committed to the repo due to file size):
[Model Files on Google Drive](https://drive.google.com/drive/folders/1__tGTtZLuD8jfU8MT3B8pLJhb96pdC_c?usp=drive_link)

Download and place the model directories under `absa-api/`:
```
absa-api/
├── dapt_roberta/
├── model1_multitask/
└── model2_sentiment/
```

---

