"""
Step 3 — EDA Data Preparation
Loads the deduplicated review data, then applies:
  - Timestamp → date conversion
  - Column drops
  - Category classification (laptop / tablet / desktop / other)
  - Brand extraction
  - Non-English review translation
  - Vader sentiment scoring
  - Price tier assignment

Data source (controlled by environment variable USE_LOCAL_CSV):
  LOCAL  (default): reads  dataset/digital_devices_reviews_no_duplicates.csv
                    writes dataset/eda_ready.csv
  CLOUD:            reads  BigQuery SOURCE_TABLE
                    writes BigQuery OUTPUT_TABLE

To run locally: 
    python pipeline/step3_eda_data.py

To force BigQuery mode:
    USE_LOCAL_CSV=false python pipeline/step3_eda_data.py
"""

import os
import re
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data on first run
for resource in ("vader_lexicon", "stopwords", "wordnet"):
    nltk.download(resource, quiet=True)

# Config 
PROJECT_ID   = os.environ.get("GCP_PROJECT", "cs163-project-487801")
SOURCE_TABLE = f"{PROJECT_ID}.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates"
OUTPUT_TABLE = f"{PROJECT_ID}.amazon_digital_devices_cleaned.eda_ready"

# Default to local CSV — set USE_LOCAL_CSV=false to use BigQuery
USE_LOCAL_CSV = os.environ.get("USE_LOCAL_CSV", "true").lower() != "false"

LOCAL_INPUT_CSV  = os.path.join("dataset", "digital_devices_reviews_no_duplicates.csv")
LOCAL_OUTPUT_CSV = os.path.join("dataset", "eda_ready.csv")



#  Section 1: Load data

def load_reviews() -> pd.DataFrame:
    if USE_LOCAL_CSV:
        print(f"  Loading reviews from local CSV: {LOCAL_INPUT_CSV} …")
        if not os.path.exists(LOCAL_INPUT_CSV):
            raise FileNotFoundError(
                f"Local CSV not found at '{LOCAL_INPUT_CSV}'.\n"
                "Place your post-Step-2 export there, or set USE_LOCAL_CSV=false "
                "to load from BigQuery."
            )
        df = pd.read_csv(LOCAL_INPUT_CSV, low_memory=False)
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        query  = f"SELECT * FROM `{SOURCE_TABLE}`"
        print("  Loading reviews from BigQuery …")
        df = client.query(query).to_dataframe()

    print(f"  Loaded {len(df):,} rows")
    return df



#  Section 2: Basic cleaning

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamp conversion, drop unused cols, filter Office Products."""
    df = df.copy()

    # Parse timestamp (milliseconds → datetime)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Drop columns not needed for EDA
    cols_to_drop = [c for c in ["timestamp", "pred_label", "rn"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Filter out Office Products (not a digital device category)
    df = df[df["main_category"] != "Office Products"].copy()

    # Add date parts for time-series analysis
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["date_of_month"] = df["date"].dt.day
    df["day_of_week"]  = df["date"].dt.day_name()

    # Review length feature
    df["review_length"] = df["review_text"].str.len()

    # Price missing signal
    df["price_missing"] = df["price"].isna().astype(int)

    return df


#  Section 3: Price tiers

def assign_price_tiers(df: pd.DataFrame) -> pd.DataFrame:
    with_price = df[df["price_missing"] == 0]["price"]
    quantile   = with_price.quantile([0, 0.5, 0.75, 0.975, 1])

    def _tier(price):
        if pd.isna(price) or price == 0:
            return "Unknown"
        if price >= quantile.loc[0.975]:
            return "Premium"
        if price > quantile.loc[0.75]:
            return "High"
        if price > quantile.loc[0.5]:
            return "Medium"
        return "Low"

    df = df.copy()
    df["price_tier"] = df["price"].apply(_tier)
    return df



#  Section 4: Device category classification

# Products manually verified as belonging to a different category than
# the regex classifier predicts. Loaded here so they're easy to update.
EXCLUDE_IDS = [
    "B00I8C5ENU", "B01DA4V074", "B072BYCRNF", "B07PDSTG9C", "B07FSMS62P",
    "B089QJZ54T", "B07QYRQTYJ", "B07772JBX1", "B09XHS8485", "B00XZV0WA4",
    "B0043FOOVY", "B004UR9P9Q", "B01N4KVSPA", "B000OKW0IG", "B01K5XFK3I",
    "B00006AMS4", "B0000C9ZJX", "B004575BIU", "B095YFFLVB", "B07855HMW4",
    "B09LR9TLY3", "B074Y7XRB6", "B087YJ1NGC", "B07XCM94SN", "B074PYK939",
    "B01IC2W0IW", "B01B2YLS7G", "B076ZXWVVK", "B016YJWPSK",
]

# Manual overrides: parent_asin → correct category
MANUAL_CATEGORY_OVERRIDES = {
    # "other" → laptop
    "laptop": [
        "B08113LD2J", "B07FW86HB2", "B07MSBPJTK", "B01GAJ166Q", "B07Q7XKJG7",
        "B07XKBJX4R", "B07Y2DMY39", "B01MEESR3Q", "B097B3WVB3", "B01KKJT5P6",
    ],
    # "other" / phone → tablet
    "tablet": [
        "B07MZCJG4B", "B072NXBWVH", "B073W1P6MC", "B074JH8N5P", "B07JMJ1D54",
        "B09XDYBRVR", "B08XXW7QNQ", "B0943LNS9D", "B08R8N3ZRH", "B09FLKL5SL",
    ],
    # "other" → desktop
    "desktop": [
        "B07G38ZV1F", "B01MXXGIP9", "B079K7NG2Q", "B01LYNC7LW", "B07Y13NSGS",
    ],
}

# Products confirmed as "other" (accessories, cables, etc.) — drop them
REAL_OTHER_IDS = [
    "B07HM9CX87", "B079MGHY92", "B00H713QY2", "B07HP9X5T7", "B0B3D5V422",
    "B088JWLKDN", "B00I8C5ENU", "B074Y8Q2Z8", "B003FWHHYC", "B07XWXX2CB",
]

# Regex patterns used by the auto-classifier
_PHONE_RE = re.compile(
    r"\b(iphone|galaxy[\s\-]?[sz]\d|pixel[\s\-]?\d|oneplus[\s\-]?\d"
    r"|smartphone|android\s+phone)\b", re.I
)
_TABLET_RE = re.compile(
    r"\b(ipad|kindle|fire\s+tablet|galaxy\s+tab|tab[\s\-]?[ae]\d"
    r"|lenovo\s+tab|surface\s+go)\b", re.I
)
_LAPTOP_RE = re.compile(
    r"\b(laptop|notebook|macbook|chromebook|surface\s+pro|thinkpad"
    r"|ideapad|inspiron|pavilion|spectre|envy|zenbook|vivobook)\b", re.I
)
_DESKTOP_RE = re.compile(
    r"\b(desktop|imac|mac\s+mini|mac\s+pro|mac\s+studio|all[\s\-]in[\s\-]one"
    r"|tower\s+pc|mini\s+pc|nuc)\b", re.I
)


def _classify_title(title: str) -> str:
    t = str(title).lower()
    if _DESKTOP_RE.search(t):
        return "desktop"
    if _LAPTOP_RE.search(t):
        return "laptop"
    if _TABLET_RE.search(t):
        return "tablet"
    if _PHONE_RE.search(t):
        return "phone"
    return "other"


def classify_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Auto-classify from product title
    df["category"] = df["product_title"].apply(_classify_title)

    # Apply manual overrides (asin → correct category)
    for category, asins in MANUAL_CATEGORY_OVERRIDES.items():
        mask = df["parent_asin"].isin(asins)
        df.loc[mask, "category"] = category

    # Drop confirmed "other" products (accessories, etc.)
    df = df[~df["parent_asin"].isin(REAL_OTHER_IDS)]

    # Drop everything still tagged "other" or "phone"
    df = df[~df["category"].isin(["other", "phone"])]

    return df



#  Section 5: Brand extraction


_BRAND_RE = re.compile(
    r"\b(apple|iphone|ipad|macbook|imac"
    r"|samsung|galaxy"
    r"|dell|alienware|xps"
    r"|hp|hewlett[\s\-]?packard|omen|spectre|pavilion|envy|elitebook|probook"
    r"|lenovo|thinkpad|ideapad|yoga"
    r"|asus|zenbook|vivobook|rog|tuf"
    r"|acer|aspire|nitro|predator|swift"
    r"|microsoft|surface"
    r"|amazon|kindle|fire|echo"
    r"|google|pixel|chromebook"
    r"|lg|sony|toshiba|razer)\b",
    re.I,
)


def _detect_brand(title: str) -> str:
    m = _BRAND_RE.search(str(title).lower())
    if not m:
        return "unknown"
    raw = m.group(0).lower().replace("-", "").replace(" ", "")
    # Normalise aliases
    aliases = {
        "hewlettpackard": "hp",
        "iphone": "apple", "ipad": "apple",
        "macbook": "apple", "imac": "apple",
        "galaxy": "samsung",
        "alienware": "dell", "xps": "dell",
        "thinkpad": "lenovo", "ideapad": "lenovo", "yoga": "lenovo",
        "zenbook": "asus", "vivobook": "asus", "rog": "asus", "tuf": "asus",
        "aspire": "acer", "nitro": "acer", "predator": "acer", "swift": "acer",
        "kindle": "amazon", "fire": "amazon", "echo": "amazon",
        "pixel": "google", "chromebook": "google",
        "omen": "hp", "spectre": "hp", "pavilion": "hp",
        "envy": "hp", "elitebook": "hp", "probook": "hp",
        "surface": "microsoft",
    }
    return aliases.get(raw, raw)


def extract_brands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brand"] = df["product_title"].apply(_detect_brand)
    return df


#  Section 6: Language detection + translation

def translate_non_english(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and translate non-English reviews (best-effort)."""
    try:
        from langdetect import detect, DetectorFactory
        from deep_translator import GoogleTranslator

        DetectorFactory.seed = 0

        def _safe_detect(text: str) -> str:
            try:
                return detect(str(text))
            except Exception:
                return "en"

        df = df.copy()
        df["language"] = df["review_text"].apply(_safe_detect)
        non_en = ~df["language"].isin(["en", "unknown"])
        n = non_en.sum()
        print(f"  Translating {n:,} non-English reviews …")

        for lang, group in df[non_en].groupby("language"):
            indices = group.index
            translated = []
            for text in group["review_text"]:
                try:
                    translated.append(
                        GoogleTranslator(source=lang, target="en").translate(str(text))
                    )
                except Exception:
                    translated.append(text)
            df.loc[indices, "review_text"] = translated

        print("  Translation complete.")
    except ImportError:
        print("langdetect / deep-translator not installed — skipping translation.")
        df["language"] = "en"

    return df



#  Section 7: Vader sentiment


def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    vader = SentimentIntensityAnalyzer()
    df = df.copy()
    df["vader_sentiment"] = df["review_text"].apply(
        lambda t: vader.polarity_scores(str(t))["compound"]
    )
    return df



#  Section 8: Save results


def save_results(df: pd.DataFrame) -> None:
    if USE_LOCAL_CSV:
        os.makedirs("dataset", exist_ok=True)
        print(f"Saving {len(df):,} rows to {LOCAL_OUTPUT_CSV} …")
        df_out = df.copy()
        df_out["date"] = df_out["date"].astype(str)
        df_out.to_csv(LOCAL_OUTPUT_CSV, index=False)
        print(f"Saved. File size: {os.path.getsize(LOCAL_OUTPUT_CSV) / 1e6:.1f} MB")
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        print(f"  Uploading {len(df):,} rows to {OUTPUT_TABLE} …")
        df_out = df.copy()
        df_out["date"] = df_out["date"].astype(str)
        job = client.load_table_from_dataframe(
            df_out,
            OUTPUT_TABLE,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()
        print("Upload complete.")


#  Entry point

def run() -> None:
    mode = "LOCAL CSV" if USE_LOCAL_CSV else "BIGQUERY"
    print("=" * 60)
    print(f"STEP 3 — EDA Data Preparation  [{mode}]")
    print("=" * 60)

    df = load_reviews()
    df = basic_clean(df)
    df = assign_price_tiers(df)
    df = classify_categories(df)
    df = extract_brands(df)
    df = translate_non_english(df)
    df = add_vader_sentiment(df)

    save_results(df)
    print("Step 3 complete.\n")


if __name__ == "__main__":
    run()
