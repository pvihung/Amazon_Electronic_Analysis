# models/labeled_data_pipeline.py
"""
Labeled Data Pipeline
Processes final.csv (labeled data) through the full preprocessing flow:
  1.  Load & assign row_id
  2.  Filter essential columns
  3.  Parse & validate aspect/sentiment pairs
  4.  Fit MultiLabelBinarizer on valid aspects
  5.  Iterative stratified split (technical rows, multilabel-safe)
  6.  Non-technical rows split & merge → df_full_train / val / test
  7.  Assign aspect labels to val/test via row_id join
  8.  Data augmentation (back-translation or EDA) on rare aspect classes
  9.  Build final df_full_train (technical aug + non-tech sentinel)
  10. Visualise sentence-type distribution (Plotly)
"""
import uuid
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineCFG:
    # Columns
    TEXT_COL:        str = "sentence_for_model"
    ASPECT_COL:      str = "target_aspect"
    SENTIMENT_COL:   str = "target_sentiment"
    IS_RELATED_COL:  str = "is_related"
    IS_TECH_COL:     str = "is_technical"
    ROW_ID_COL:      str = "row_id"

    # Split ratios
    SEED:            int   = 42
    VAL_RATIO:       float = 0.15
    TEST_RATIO:      float = 0.15

    # Augmentation
    AUG_RARE_THRESHOLD: int = 150
    AUG_NUM_PER_SAMPLE: int = 2
    USE_BACK_TRANSLATION: bool = True   # False → EDA (synonym aug)

    # Aspects to exclude from labelling
    EXCLUDE_ASPECTS: List[str] = field(
        default_factory=lambda: ["general", "unrelated"]
    )

CFG = PipelineCFG()

def load_labeled_data(csv_path: str, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """ Load final.csv, drop rows with no text, assign UUID row_id. """
    df = pd.read_csv(csv_path).dropna(subset=[cfg.TEXT_COL])
    df[cfg.ROW_ID_COL] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = df.reset_index(drop=True)

    assert df[cfg.ROW_ID_COL].is_unique, "row_id collision — should never happen with UUID"
    print(f"Loaded {len(df):,} rows | row_id unique: OK")
    return df

REQUIRED_COLUMNS = [
    "sentence_for_model",
    "is_related",
    "is_technical",
    "target_aspect",
    "target_sentiment",
]

def filter_columns(df: pd.DataFrame, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Keep only essential columns (+ row_id if present, + any extras requested).
    Raises ValueError if required columns are missing.
    """
    keep = list(REQUIRED_COLUMNS)
    if "row_id" in df.columns:
        keep = ["row_id"] + keep
    if extra_cols:
        keep += [c for c in extra_cols if c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[keep].copy()

def classify_sentence_type(is_related: int, is_technical: int) -> str:
    """
    unrelated  → is_related == 0
    general    → is_related == 1, is_technical == 0
    technical  → is_related == 1, is_technical == 1
    """
    if is_related == 0:
        return "unrelated"
    if is_related == 1 and is_technical == 0:
        return "general"
    if is_related == 1 and is_technical == 1:
        return "technical"
    return "unknown"


def add_sentence_classification(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Add 'sentence_type' column derived from is_related / is_technical."""
    df = df.copy()
    df["sentence_type"] = df.apply(
        lambda r: classify_sentence_type(int(r[cfg.IS_RELATED_COL]), int(r[cfg.IS_TECH_COL])),
        axis=1,
    )
    return df

def parse_and_validate_pairs(
    aspect_str,
    sentiment_str,
    exclude_aspects: List[str] = CFG.EXCLUDE_ASPECTS,
) -> Tuple[List[str], List[int]]:
    """
    Parse semicolon-separated aspect and sentiment strings.
    Returns ([], []) on mismatched lengths or fully invalid pairs.
    Excludes aspects in exclude_aspects (e.g. 'general', 'unrelated').
    """
    if pd.isna(aspect_str) or pd.isna(sentiment_str):
        return [], []

    raw_aspects = [a.strip().lower() for a in str(aspect_str).split(";")]
    raw_sents   = [s.strip()         for s in str(sentiment_str).split(";")]

    if len(raw_aspects) != len(raw_sents):
        return [], []

    valid_aspects, valid_sents = [], []
    for asp, sent in zip(raw_aspects, raw_sents):
        if asp and asp not in exclude_aspects:
            try:
                valid_aspects.append(asp)
                valid_sents.append(int(float(sent)))
            except ValueError:
                continue

    return valid_aspects, valid_sents


def apply_pair_parsing(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """
    Add columns:
      - aspects_parsed    : List[str]
      - sentiments_parsed : List[int]
      - is_valid_pairs    : bool  (True when at least one valid aspect pair exists)
    """
    df = df.copy()
    parsed = df.apply(
        lambda r: parse_and_validate_pairs(r[cfg.ASPECT_COL], r[cfg.SENTIMENT_COL]),
        axis=1,
    )
    df["aspects_parsed"]    = parsed.apply(lambda x: x[0])
    df["sentiments_parsed"] = parsed.apply(lambda x: x[1])
    df["is_valid_pairs"]    = df["aspects_parsed"].apply(len) > 0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Fit MultiLabelBinarizer
# ──────────────────────────────────────────────────────────────────────────────

def fit_mlb(df: pd.DataFrame) -> Tuple[MultiLabelBinarizer, List[str]]:
    """
    Fit MLB on aspects_parsed from valid-pair rows.
    Returns (fitted_mlb, aspect_classes).
    """
    valid_aspects = df[df["is_valid_pairs"]]["aspects_parsed"].tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(valid_aspects)
    aspect_classes = list(mlb.classes_)
    print(f"Aspect classes ({len(aspect_classes)}): {aspect_classes}")
    return mlb, aspect_classes


# ──────────────────────────────────────────────────────────────────────────────
# 6. Iterative stratified split (technical rows only)
# ──────────────────────────────────────────────────────────────────────────────

def get_technical_df(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Filter to technical rows that have valid aspect pairs."""
    return df[
        (df[cfg.IS_RELATED_COL] == 1) &
        (df[cfg.IS_TECH_COL]    == 1) &
        (df["is_valid_pairs"]   == True)
    ].copy().reset_index(drop=True)


def get_combo_labels(aspects: List[str], sentiments: List[int]) -> List[str]:
    return [f"{a}_{s}" for a, s in zip(aspects, sentiments)]


def iterative_split(
    df_tech: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    cfg: PipelineCFG = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Multilabel-safe iterative stratified split on technical rows.

    Returns
    -------
    train_df, val_df, test_df   – DataFrames with 'labels' column (MLB aspect vectors)
    y_train, y_val, y_test      – np.ndarray of shape (N, n_aspects)
    """
    df_tech = df_tech.copy()
    df_tech["combo_labels"] = df_tech.apply(
        lambda r: get_combo_labels(r["aspects_parsed"], r["sentiments_parsed"]),
        axis=1,
    )

    mlb_combo = MultiLabelBinarizer()
    y_combo   = mlb_combo.fit_transform(df_tech["combo_labels"]).astype(float)
    X_ids     = df_tech[[cfg.ROW_ID_COL]].values

    # Split off test
    X_trainval, y_trainval, X_test, _ = iterative_train_test_split(
        X_ids, y_combo, test_size=cfg.TEST_RATIO
    )

    # Split val from trainval
    val_ratio_adj = cfg.VAL_RATIO / (1 - cfg.TEST_RATIO)
    X_train, _, X_val, _ = iterative_train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adj
    )

    def make_df_from_ids(X_id_array: np.ndarray) -> pd.DataFrame:
        ids = [x[0] for x in X_id_array]
        d   = df_tech[df_tech[cfg.ROW_ID_COL].isin(ids)].copy()
        y   = mlb.transform(d["aspects_parsed"].tolist())
        d["labels"] = list(y)
        return d

    train_df = make_df_from_ids(X_train)
    val_df   = make_df_from_ids(X_val)
    test_df  = make_df_from_ids(X_test)

    y_train  = np.array(train_df["labels"].tolist(), dtype=float)
    y_val    = np.array(val_df["labels"].tolist(),   dtype=float)
    y_test   = np.array(test_df["labels"].tolist(),  dtype=float)

    # Leak guard
    train_ids = set(train_df[cfg.ROW_ID_COL])
    val_ids   = set(val_df[cfg.ROW_ID_COL])
    test_ids  = set(test_df[cfg.ROW_ID_COL])
    assert not (train_ids & val_ids),  "LEAK: train ∩ val NOT EMPTY!"
    assert not (train_ids & test_ids), "LEAK: train ∩ test NOT EMPTY!"
    assert not (val_ids   & test_ids), "LEAK: val ∩ test NOT EMPTY!"
    print(f"Split integrity check: OK — Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    return train_df, val_df, test_df, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────────────────────
# 7. Merge non-technical rows → df_full_*
# ──────────────────────────────────────────────────────────────────────────────

def build_full_splits(
    df: pd.DataFrame,
    df_tech: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineCFG = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame]:
    """
    Append non-technical rows into each split.

    Returns
    -------
    df_full_train, df_full_val, df_full_test,
    non_tech_val, non_tech_test    (needed downstream for augmentation merge)
    """
    df_non_tech = df[~df[cfg.ROW_ID_COL].isin(df_tech[cfg.ROW_ID_COL])].copy()

    non_tech_train, temp = train_test_split(
        df_non_tech,
        test_size=cfg.TEST_RATIO + cfg.VAL_RATIO,
        stratify=df_non_tech[cfg.IS_RELATED_COL],
        random_state=cfg.SEED,
    )
    non_tech_val, non_tech_test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp[cfg.IS_RELATED_COL],
        random_state=cfg.SEED,
    )

    df_full_train = pd.concat([train_df, non_tech_train]).reset_index(drop=True)
    df_full_val   = pd.concat([val_df,   non_tech_val]).reset_index(drop=True)
    df_full_test  = pd.concat([test_df,  non_tech_test]).reset_index(drop=True)

    print(f"Full splits — Train: {len(df_full_train)} | Val: {len(df_full_val)} | Test: {len(df_full_test)}")
    return df_full_train, df_full_val, df_full_test, non_tech_val, non_tech_test


# ──────────────────────────────────────────────────────────────────────────────
# 8. Assign aspect labels to val / test via row_id join
# ──────────────────────────────────────────────────────────────────────────────

def assign_aspect_labels_by_id(
    df_split: pd.DataFrame,
    ref_df: pd.DataFrame,
    y_aspects: np.ndarray,
    n_aspects: int,
    cfg: PipelineCFG = CFG,
) -> pd.DataFrame:
    """
    Join aspect labels from ref_df into df_split using row_id.
    Rows not found in ref_df get sentinel label (all -1) and has_aspect_label=False.

    Parameters
    ----------
    df_split   : target DataFrame (df_full_val / df_full_test)
    ref_df     : source of truth — direct output of make_df_from_ids, unmodified
    y_aspects  : np.ndarray (len(ref_df), n_aspects)
    n_aspects  : number of aspect classes
    """
    assert len(ref_df) == len(y_aspects), (
        f"MISALIGNMENT: ref_df={len(ref_df)} rows, y_aspects={len(y_aspects)} rows"
    )
    assert ref_df[cfg.ROW_ID_COL].is_unique, "row_id in ref_df is not unique!"

    id_to_label = {
        str(ref_df[cfg.ROW_ID_COL].iloc[i]): y_aspects[i].astype(np.float32).tolist()
        for i in range(len(ref_df))
    }

    df_out = df_split.copy().reset_index(drop=True)

    def lookup(rid):
        label = id_to_label.get(str(rid))
        if label is not None:
            return label, True
        return np.full(n_aspects, -1.0, dtype=np.float32).tolist(), False

    mapped = df_out[cfg.ROW_ID_COL].map(lookup)
    df_out["aspect_labels"]    = [m[0] for m in mapped]
    df_out["has_aspect_label"] = [m[1] for m in mapped]

    matched = df_out["has_aspect_label"].sum()
    print(f"  → {matched}/{len(df_out)} rows matched by row_id (expected {len(ref_df)})")
    if matched != len(ref_df):
        print(f"  WARNING: {len(ref_df) - matched} ref rows not found in df_split!")
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
# 9. Data Augmentation — rare aspect classes
# ──────────────────────────────────────────────────────────────────────────────

def get_rare_classes(
    y_train: np.ndarray,
    aspect_classes: List[str],
    threshold: int = CFG.AUG_RARE_THRESHOLD,
) -> List[Tuple[int, str, int]]:
    """Return list of (aspect_idx, class_name, count) for classes below threshold."""
    rare = []
    for i, cls in enumerate(aspect_classes):
        count = int(y_train[:, i].sum())
        if count < threshold:
            rare.append((i, cls, count))
    if rare:
        print(f"Rare classes (< {threshold}):")
        for idx, cls, cnt in rare:
            print(f"  [{idx}] {cls}: {cnt} samples")
    else:
        print("No rare classes found.")
    return rare


# ── Back-translation helpers ──────────────────────────────────────────────────

def load_mt_models(device):
    from transformers import MarianMTModel, MarianTokenizer
    tok_en_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    mdl_en_de = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)
    tok_de_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    mdl_de_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)
    return tok_en_de, mdl_en_de, tok_de_en, mdl_de_en


def back_translate_batch(
    texts, tok_en_de, mdl_en_de, tok_de_en, mdl_de_en,
    device, batch_size: int = 32,
) -> List[str]:
    import torch
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok_en_de(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
        with torch.no_grad():
            de_ids = mdl_en_de.generate(**enc, num_beams=4, max_length=128)
        de_texts = tok_en_de.batch_decode(de_ids, skip_special_tokens=True)

        enc2 = tok_de_en(de_texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=128).to(device)
        with torch.no_grad():
            en_ids = mdl_de_en.generate(**enc2, num_beams=4, max_length=128)
        results.extend(tok_de_en.batch_decode(en_ids, skip_special_tokens=True))
    return results


def validate_translations(
    orig_texts: List[str],
    trans_texts: List[str],
    min_words: int = 5,
) -> List[str]:
    """Fallback to original if translation is too short or not English."""
    from langdetect import detect
    validated, fallback_count = [], 0
    for orig, trans in zip(orig_texts, trans_texts):
        use_orig = len(trans.split()) < min_words
        if not use_orig:
            try:
                use_orig = detect(trans) != "en"
            except Exception:
                use_orig = True
        if use_orig:
            validated.append(orig)
            fallback_count += 1
        else:
            validated.append(trans)
    if fallback_count:
        print(f"{fallback_count}/{len(orig_texts)} translations fell back to original.")
    return validated


# ── Main augmentation function ────────────────────────────────────────────────

def augment_rare_classes(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    rare_classes: List[Tuple[int, str, int]],
    n_aspects: int,
    cfg: PipelineCFG = CFG,
    device=None,
) -> pd.DataFrame:
    """
    Augment training rows that contain rare-class aspects.
    Uses back-translation (EN→DE→EN) or EDA synonym augmentation.

    Returns augmented train_df with columns:
      aspect_labels, has_aspect_label, is_related, is_technical
    """
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mask of rows that belong to at least one rare class
    to_aug_mask = np.zeros(len(train_df), dtype=bool)
    for aspect_idx, _, _ in rare_classes:
        to_aug_mask |= (y_train[:, aspect_idx] == 1)

    to_aug_df     = train_df[to_aug_mask].reset_index(drop=True)
    to_aug_labels = y_train[to_aug_mask]
    texts_to_aug  = to_aug_df[cfg.TEXT_COL].tolist()
    print(f"Augmenting {len(to_aug_df)} rare-class rows × {cfg.AUG_NUM_PER_SAMPLE} ...")

    # Load augmentor
    if cfg.USE_BACK_TRANSLATION:
        mt_models = load_mt_models(device)
        print("MarianMT models loaded.")
    else:
        import nlpaug.augmenter.word as naw
        aug_eda = naw.SynonymAug(aug_src="wordnet", aug_p=0.2)
        mt_models = None
        print("Using EDA (synonym augmentation).")

    aug_rows = []
    for k in range(cfg.AUG_NUM_PER_SAMPLE):
        print(f"  Augmentation pass {k+1}/{cfg.AUG_NUM_PER_SAMPLE} ...")
        if cfg.USE_BACK_TRANSLATION:
            raw = back_translate_batch(texts_to_aug, *mt_models, device=device)
            aug_texts = validate_translations(texts_to_aug, raw)
        else:
            aug_texts = [
                aug_eda.augment(t)[0] if pd.notna(t) else t
                for t in texts_to_aug
            ]

        for j, aug_text in enumerate(aug_texts):
            row = to_aug_df.iloc[j].copy()
            row[cfg.TEXT_COL]    = aug_text
            row[cfg.ROW_ID_COL]  = str(uuid.uuid4())
            aug_rows.append(row)

    aug_extra_df = pd.DataFrame(aug_rows).reset_index(drop=True)
    aug_extra_y  = np.tile(to_aug_labels, (cfg.AUG_NUM_PER_SAMPLE, 1)).astype(np.float32)

    aug_extra_df[cfg.IS_RELATED_COL]   = 1
    aug_extra_df[cfg.IS_TECH_COL]      = 1
    aug_extra_df["aspect_labels"]      = aug_extra_y.tolist()
    aug_extra_df["has_aspect_label"]   = True

    # Annotate original train rows too
    train_df_safe = train_df.copy()
    train_df_safe[cfg.IS_RELATED_COL]  = 1
    train_df_safe[cfg.IS_TECH_COL]     = 1
    train_df_safe["aspect_labels"]     = y_train.astype(np.float32).tolist()
    train_df_safe["has_aspect_label"]  = True

    train_aug = pd.concat([train_df_safe, aug_extra_df]).reset_index(drop=True)
    print(f"Train after augmentation: {len(train_aug)} rows (was {len(train_df)})")
    return train_aug


def build_final_train(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    rare_classes: List[Tuple[int, str, int]],
    non_tech_train: pd.DataFrame,
    n_aspects: int,
    cfg: PipelineCFG = CFG,
    device=None,
) -> pd.DataFrame:
    """
    Combine augmented technical train rows with non-technical sentinel rows,
    then shuffle.

    Returns df_full_train ready for MultiTaskDataset.
    """
    if rare_classes:
        train_aug_df = augment_rare_classes(
            train_df, y_train, rare_classes, n_aspects, cfg, device
        )
    else:
        print("No rare classes — skipping augmentation.")
        train_aug_df = train_df.copy()
        train_aug_df[cfg.IS_RELATED_COL] = 1
        train_aug_df[cfg.IS_TECH_COL]    = 1
        train_aug_df["aspect_labels"]    = y_train.astype(np.float32).tolist()
        train_aug_df["has_aspect_label"] = True

    # Non-tech rows: sentinel aspect_labels (-1 vector), has_aspect_label=False
    non_tech_safe = non_tech_train.copy()
    non_tech_safe["aspect_labels"]    = [
        np.full(n_aspects, -1.0, dtype=np.float32).tolist()
        for _ in range(len(non_tech_safe))
    ]
    non_tech_safe["has_aspect_label"] = False

    df_full_train = (
        pd.concat([train_aug_df, non_tech_safe])
        .sample(frac=1, random_state=cfg.SEED)
        .reset_index(drop=True)
    )
    n_tech = df_full_train["has_aspect_label"].sum()
    print(f"df_full_train: {len(df_full_train)} rows | technical={n_tech}")
    return df_full_train


# ──────────────────────────────────────────────────────────────────────────────
# 10. Plotly visualisation
# ──────────────────────────────────────────────────────────────────────────────

_SENTENCE_TYPE_COLORS = {
    "unrelated": "#e74c3c",
    "general":   "#f39c12",
    "technical": "#27ae60",
}
_SENTENCE_TYPE_ORDER = ["unrelated", "general", "technical"]


def plot_sentence_type_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Plotly bar chart of sentence-type distribution.
    Auto-adds sentence_type column if missing.
    Compatible with standalone use and Dash dcc.Graph.
    """
    if "sentence_type" not in df.columns:
        df = add_sentence_classification(df)

    counts = (
        df["sentence_type"]
        .value_counts()
        .reindex(_SENTENCE_TYPE_ORDER, fill_value=0)
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker=dict(color=[_SENTENCE_TYPE_COLORS[t] for t in counts.index]),
                text=counts.values.tolist(),
                textposition="outside",
                textfont=dict(size=12),
                width=0.5,
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Distribution of Sentence Types",
            font=dict(size=16, color="#2c3e50"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title=dict(text="Sentence Type", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(text="Count",         font=dict(size=13)), tickfont=dict(size=11)),
        showlegend=False,
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, b=60, l=60, r=40),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#ecf0f1", zeroline=True, zerolinecolor="#bdc3c7")
    return fig


# Alias for Dash integration
def get_sentence_type_countplot(df: pd.DataFrame) -> go.Figure:
    return plot_sentence_type_distribution(df)


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Container for all outputs produced by run_preprocessing_pipeline()."""
    df_raw:         pd.DataFrame       # Original loaded df with row_id
    df_full_train:  pd.DataFrame       # Final shuffled train split
    df_full_val:    pd.DataFrame       # Val split with aspect_labels assigned
    df_full_test:   pd.DataFrame       # Test split with aspect_labels assigned
    mlb:            MultiLabelBinarizer
    aspect_classes: List[str]
    n_aspects:      int
    y_train:        np.ndarray
    y_val:          np.ndarray
    y_test:         np.ndarray
    sentence_type_fig: go.Figure       # Plotly distribution chart


def run_preprocessing_pipeline(
    csv_path: str,
    cfg: PipelineCFG = CFG,
    run_augmentation: bool = True,
    device=None,
) -> PipelineResult:
    """
    Full labeled-data preprocessing pipeline.

    Parameters
    ----------
    csv_path         : path to final.csv
    cfg              : PipelineCFG instance
    run_augmentation : set False to skip augmentation (faster for dev/testing)
    device           : torch.device for augmentation; auto-detected if None

    Returns
    -------
    PipelineResult dataclass with all split DataFrames, MLB, labels, and figure.
    """
    print("=" * 60)
    print("STEP 1 — Load data")
    print("=" * 60)
    df = load_labeled_data(csv_path, cfg)

    print("\n" + "=" * 60)
    print("STEP 2 — Parse aspect/sentiment pairs")
    print("=" * 60)
    df = apply_pair_parsing(df, cfg)
    df = add_sentence_classification(df, cfg)

    print("\n" + "=" * 60)
    print("STEP 3 — Fit MultiLabelBinarizer")
    print("=" * 60)
    mlb, aspect_classes = fit_mlb(df)
    n_aspects = len(aspect_classes)

    print("\n" + "=" * 60)
    print("STEP 4 — Iterative stratified split (technical rows)")
    print("=" * 60)
    df_tech = get_technical_df(df, cfg)
    train_df, val_df, test_df, y_train, y_val, y_test = iterative_split(
        df_tech, mlb, cfg
    )

    print("\n" + "=" * 60)
    print("STEP 5 — Merge non-technical rows")
    print("=" * 60)
    df_full_train, df_full_val, df_full_test, non_tech_val, non_tech_test = build_full_splits(
        df, df_tech, train_df, val_df, test_df, cfg
    )

    # non_tech_train is in df_full_train but we need it separately for build_final_train
    df_non_tech   = df[~df[cfg.ROW_ID_COL].isin(df_tech[cfg.ROW_ID_COL])].copy()
    tech_train_ids = set(train_df[cfg.ROW_ID_COL])
    non_tech_train = df_non_tech[~df_non_tech[cfg.ROW_ID_COL].isin(tech_train_ids)]

    print("\n" + "=" * 60)
    print("STEP 6 — Assign aspect labels to val/test via row_id")
    print("=" * 60)
    df_full_val  = assign_aspect_labels_by_id(df_full_val,  val_df,  y_val,  n_aspects, cfg)
    df_full_test = assign_aspect_labels_by_id(df_full_test, test_df, y_test, n_aspects, cfg)

    print("\n" + "=" * 60)
    print("STEP 7 — Augmentation + build final df_full_train")
    print("=" * 60)
    if run_augmentation:
        rare_classes = get_rare_classes(y_train, aspect_classes, cfg.AUG_RARE_THRESHOLD)
        df_full_train = build_final_train(
            train_df, y_train, rare_classes, non_tech_train, n_aspects, cfg, device
        )
    else:
        print("Augmentation skipped (run_augmentation=False).")
        train_aug_df = train_df.copy()
        train_aug_df[cfg.IS_RELATED_COL] = 1
        train_aug_df[cfg.IS_TECH_COL]    = 1
        train_aug_df["aspect_labels"]    = y_train.astype(np.float32).tolist()
        train_aug_df["has_aspect_label"] = True

        non_tech_safe = non_tech_train.copy()
        non_tech_safe["aspect_labels"]    = [
            np.full(n_aspects, -1.0, dtype=np.float32).tolist()
            for _ in range(len(non_tech_safe))
        ]
        non_tech_safe["has_aspect_label"] = False

        df_full_train = (
            pd.concat([train_aug_df, non_tech_safe])
            .sample(frac=1, random_state=cfg.SEED)
            .reset_index(drop=True)
        )

    assert "has_aspect_label" in df_full_train.columns, "df_full_train missing has_aspect_label!"

    print("\n" + "=" * 60)
    print("STEP 8 — Build visualisation")
    print("=" * 60)
    fig = plot_sentence_type_distribution(df)

    print("\n" + "=" * 60)
    print("Pipeline complete. Summary:")
    print("=" * 60)
    for name, split in [
        ("df_full_train", df_full_train),
        ("df_full_val",   df_full_val),
        ("df_full_test",  df_full_test),
    ]:
        n_tech = split["has_aspect_label"].sum() if "has_aspect_label" in split.columns else "N/A"
        print(f"  {name:<20}: {len(split):,} rows | technical={n_tech}")

    return PipelineResult(
        df_raw=df,
        df_full_train=df_full_train,
        df_full_val=df_full_val,
        df_full_test=df_full_test,
        mlb=mlb,
        aspect_classes=aspect_classes,
        n_aspects=n_aspects,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        sentence_type_fig=fig,
    )