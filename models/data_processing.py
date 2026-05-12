import uuid
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

warnings.filterwarnings("ignore")


@dataclass
class PipelineCFG:
    TEXT_COL: str              = "sentence_for_model"
    ASPECT_COL: str            = "target_aspect"
    SENTIMENT_COL: str         = "target_sentiment"
    IS_RELATED_COL: str        = "is_related"
    IS_TECH_COL: str           = "is_technical"
    ROW_ID_COL: str            = "row_id"
    SEED: int                  = 42
    VAL_RATIO: float           = 0.15
    TEST_RATIO: float          = 0.15
    AUG_RARE_THRESHOLD: int    = 150
    AUG_NUM_PER_SAMPLE: int    = 2
    USE_BACK_TRANSLATION: bool = True
    EXCLUDE_ASPECTS: List[str] = field(default_factory=lambda: ["general", "unrelated"])

CFG = PipelineCFG()


REQUIRED_COLUMNS = [
    "sentence_for_model",
    "is_related",
    "is_technical",
    "target_aspect",
    "target_sentiment",
]


def load_labeled_data(csv_path: str, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Load CSV, drop rows with null text, and assign a UUID to each row as row_id."""
    df = pd.read_csv(csv_path).dropna(subset=[cfg.TEXT_COL])
    df[cfg.ROW_ID_COL] = [str(uuid.uuid4()) for _ in range(len(df))]
    df = df.reset_index(drop=True)
    assert df[cfg.ROW_ID_COL].is_unique
    print(f"Loaded {len(df):,} rows")
    return df


def filter_columns(df: pd.DataFrame, extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Keep the required columns plus any extra columns if present,
    Raise error if any required column is missing."""
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
    """Return the question type: 'unrelated', 'general', or 'technical' based on the two flags."""
    if is_related == 0:
        return "unrelated"
    if is_related == 1 and is_technical == 0:
        return "general"
    if is_related == 1 and is_technical == 1:
        return "technical"
    return "unknown"


def add_sentence_classification(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Add a 'sentence_type' column to the DataFrame by calling classify_sentence_type."""
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
    """Parse the strings 'aspect1;aspect2;aspect3' and '1;-1;1' into filtered lists of aspects and sentiments."""
    if pd.isna(aspect_str) or pd.isna(sentiment_str):
        return [], []
    raw_aspects = [a.strip().lower() for a in str(aspect_str).split(";")]
    raw_sentiments   = [s.strip()         for s in str(sentiment_str).split(";")]
    if len(raw_aspects) != len(raw_sentiments):
        return [], []
    valid_aspects, valid_sentiments = [], []
    for asp, sent in zip(raw_aspects, raw_sentiments):
        if asp and asp not in exclude_aspects:
            try:
                valid_aspects.append(asp)
                valid_sentiments.append(int(float(sent)))
            except ValueError:
                continue
    return valid_aspects, valid_sentiments


def apply_pair_parsing(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Apply parse_and_validate_pairs to the entire DataFrame and add three parsed columns."""
    df = df.copy()
    parsed = df.apply(
        lambda r: parse_and_validate_pairs(r[cfg.ASPECT_COL], r[cfg.SENTIMENT_COL]),
        axis=1,
    )
    df["aspects_parsed"]    = parsed.apply(lambda x: x[0])
    df["sentiments_parsed"] = parsed.apply(lambda x: x[1])
    df["is_valid_pairs"]    = df["aspects_parsed"].apply(len) > 0
    return df


def fit_mlb(train_df: pd.DataFrame) -> Tuple[MultiLabelBinarizer, List[str]]:
    """Fit MultiLabelBinarizer on the training set and return (mlb, list of class names)."""
    mlb = MultiLabelBinarizer()
    mlb.fit(train_df["aspects_parsed"].tolist())
    aspect_classes = list(mlb.classes_)
    print(f"Aspect classes ({len(aspect_classes)}): {aspect_classes}")
    return mlb, aspect_classes


def get_technical_df(df: pd.DataFrame, cfg: PipelineCFG = CFG) -> pd.DataFrame:
    """Filter to keep only technical rows with valid aspect pairs."""
    return df[
        (df[cfg.IS_RELATED_COL] == 1) &
        (df[cfg.IS_TECH_COL]    == 1) &
        (df["is_valid_pairs"]   == True)
    ].copy().reset_index(drop=True)


def get_combo_labels(aspects: List[str], sentiments: List[int]) -> List[str]:
    """Combine aspect and sentiment into composite 'aspect_sentiment' labels for stratification."""
    return [f"{a}_{s}" for a, s in zip(aspects, sentiments)]


def iterative_split(
    df_tech: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    cfg: PipelineCFG = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform split by row_id
    Assert if data leakage between splits."""
    df_tech = df_tech.copy()
    df_tech["combo_labels"] = df_tech.apply(
        lambda r: get_combo_labels(r["aspects_parsed"], r["sentiments_parsed"]),
        axis=1,
    )
    mlb_combo = MultiLabelBinarizer()
    y_combo   = mlb_combo.fit_transform(df_tech["combo_labels"]).astype(float)
    X_ids     = df_tech[[cfg.ROW_ID_COL]].values

    X_trainval, y_trainval, X_test, _ = iterative_train_test_split(
        X_ids, y_combo, test_size=cfg.TEST_RATIO
    )
    val_ratio_adj = cfg.VAL_RATIO / (1 - cfg.TEST_RATIO)
    X_train, _, X_val, _ = iterative_train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adj
    )

    def get_df_from_ids(X_id_array):
        ids = [x[0] for x in X_id_array]
        return df_tech[df_tech[cfg.ROW_ID_COL].isin(ids)].copy()

    train_df = get_df_from_ids(X_train)
    val_df   = get_df_from_ids(X_val)
    test_df  = get_df_from_ids(X_test)

    assert not (set(train_df[cfg.ROW_ID_COL]) & set(val_df[cfg.ROW_ID_COL])),  "LEAK: train ∩ val"
    assert not (set(train_df[cfg.ROW_ID_COL]) & set(test_df[cfg.ROW_ID_COL])), "LEAK: train ∩ test"
    assert not (set(val_df[cfg.ROW_ID_COL])   & set(test_df[cfg.ROW_ID_COL])), "LEAK: val ∩ test"
    print(f"Split OK — Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def apply_aspect_labels(
    df_split: pd.DataFrame,
    mlb: MultiLabelBinarizer,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert aspects_parsed into a float multi-hot matrix using the fitted MLB."""
    y = mlb.transform(df_split["aspects_parsed"].tolist())
    return df_split.copy(), np.array(y, dtype=float)


def build_full_splits(
    df: pd.DataFrame,
    df_tech: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: PipelineCFG = CFG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge non-technical rows into the three splits
    Return the full splits plus separate non_tech_val/test sets."""
    df_non_tech = df[~df[cfg.ROW_ID_COL].isin(df_tech[cfg.ROW_ID_COL])].copy()
    non_tech_train, temp = train_test_split(
        df_non_tech,
        test_size=cfg.TEST_RATIO + cfg.VAL_RATIO,
        stratify=df_non_tech[cfg.IS_RELATED_COL],
        random_state=cfg.SEED,
    )
    non_tech_val, non_tech_test = train_test_split(
        temp, test_size=0.5,
        stratify=temp[cfg.IS_RELATED_COL],
        random_state=cfg.SEED,
    )
    df_full_train = pd.concat([train_df, non_tech_train]).reset_index(drop=True)
    df_full_val   = pd.concat([val_df,   non_tech_val]).reset_index(drop=True)
    df_full_test  = pd.concat([test_df,  non_tech_test]).reset_index(drop=True)
    print(f"Full splits — Train: {len(df_full_train)} | Val: {len(df_full_val)} | Test: {len(df_full_test)}")
    return df_full_train, df_full_val, df_full_test, non_tech_val, non_tech_test


def assign_aspect_labels_by_id(
    df_split: pd.DataFrame,
    ref_df: pd.DataFrame,
    y_aspects: np.ndarray,
    n_aspects: int,
    cfg: PipelineCFG = CFG,
) -> pd.DataFrame:
    """Join aspect labels into the full split by row_id
    With non-technical rows assigned the sentinel value -1."""
    assert len(ref_df) == len(y_aspects)
    assert ref_df[cfg.ROW_ID_COL].is_unique
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
    print(f"  → {matched}/{len(df_out)} rows matched")
    return df_out



def get_rare_classes(
    y_train: np.ndarray,
    aspect_classes: List[str],
    threshold: int = CFG.AUG_RARE_THRESHOLD,
) -> List[Tuple[int, str, int]]:
    """Return list (idx, class_name, count) for rare aspects."""
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
