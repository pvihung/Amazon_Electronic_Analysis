import uuid
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.data_processing import (
    PipelineCFG, CFG,
    add_sentence_classification,
)

_SENTENCE_TYPE_COLORS = {
    "unrelated": "#e74c3c",
    "general": "#f39c12",
    "technical": "#27ae60",
}
_SENTENCE_TYPE_ORDER = ["unrelated", "general", "technical"]


def load_mt_models(device):
    """Load two pairs of MarianMT tokenizers and models for the EN→DE and DE→EN directions."""
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
    """Translate a list of English sentences into German, then back into English in batches."""
    import torch
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
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
    """Fall back to the original sentence if the translation is too short or not in English."""
    from langdetect import detect
    validated, fallback_count = [], 0
    for orig, trans in zip(orig_texts, trans_texts):
        use_orig = len(trans.split()) < min_words
        if not use_orig:
            try:
                use_orig = detect(trans) != "en"
            except Exception:
                use_orig = True
        validated.append(orig if use_orig else trans)
        if use_orig:
            fallback_count += 1
    if fallback_count:
        print(f"{fallback_count}/{len(orig_texts)} translations fell back to original.")
    return validated


def augment_rare_classes(
        train_df: pd.DataFrame,
        y_train: np.ndarray,
        rare_classes: List[Tuple[int, str, int]],
        cfg: PipelineCFG = CFG,
        device=None,
) -> pd.DataFrame:
    """Duplicate rare-class rows using back-translation (or EDA) and assign new UUIDs."""
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_aug_mask = np.zeros(len(train_df), dtype=bool)
    for aspect_idx, _, _ in rare_classes:
        to_aug_mask |= (y_train[:, aspect_idx] == 1)

    to_aug_df = train_df[to_aug_mask].reset_index(drop=True)
    to_aug_labels = y_train[to_aug_mask]
    texts_to_aug = to_aug_df[cfg.TEXT_COL].tolist()
    print(f"Augmenting {len(to_aug_df)} rows × {cfg.AUG_NUM_PER_SAMPLE} ...")

    if cfg.USE_BACK_TRANSLATION:
        mt_models = load_mt_models(device)
        print("MarianMT loaded.")
    else:
        import nlpaug.augmenter.word as naw
        aug_eda = naw.SynonymAug(aug_src="wordnet", aug_p=0.2)
        mt_models = None

    aug_rows = []
    for k in range(cfg.AUG_NUM_PER_SAMPLE):
        print(f"  Pass {k + 1}/{cfg.AUG_NUM_PER_SAMPLE} ...")
        if cfg.USE_BACK_TRANSLATION:
            raw = back_translate_batch(texts_to_aug, *mt_models, device=device)
            aug_texts = validate_translations(texts_to_aug, raw)
        else:
            aug_texts = [aug_eda.augment(t)[0] if pd.notna(t) else t for t in texts_to_aug]

        for j, aug_text in enumerate(aug_texts):
            row = to_aug_df.iloc[j].copy()
            row[cfg.TEXT_COL] = aug_text
            row[cfg.ROW_ID_COL] = str(uuid.uuid4())
            aug_rows.append(row)

    aug_extra_df = pd.DataFrame(aug_rows).reset_index(drop=True)
    aug_extra_y = np.tile(to_aug_labels, (cfg.AUG_NUM_PER_SAMPLE, 1)).astype(np.float32)
    aug_extra_df[cfg.IS_RELATED_COL] = 1
    aug_extra_df[cfg.IS_TECH_COL] = 1
    aug_extra_df["aspect_labels"] = aug_extra_y.tolist()
    aug_extra_df["has_aspect_label"] = True

    train_df_safe = train_df.copy()
    train_df_safe[cfg.IS_RELATED_COL] = 1
    train_df_safe[cfg.IS_TECH_COL] = 1
    train_df_safe["aspect_labels"] = y_train.astype(np.float32).tolist()
    train_df_safe["has_aspect_label"] = True

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
    """Combine augmented technical rows with non-technical rows assigned the sentinel value -1, then shuffle into df_full_train."""
    if rare_classes:
        train_aug_df = augment_rare_classes(train_df, y_train, rare_classes, n_aspects, cfg, device)
    else:
        print("No rare classes — skipping augmentation.")
        train_aug_df = train_df.copy()
        train_aug_df[cfg.IS_RELATED_COL] = 1
        train_aug_df[cfg.IS_TECH_COL] = 1
        train_aug_df["aspect_labels"] = y_train.astype(np.float32).tolist()
        train_aug_df["has_aspect_label"] = True

    non_tech_safe = non_tech_train.copy()
    non_tech_safe["aspect_labels"] = [
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


def plot_sentence_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot a bar chart of the distribution of the three sentence types"""
    if "sentence_type" not in df.columns:
        df = add_sentence_classification(df)
    counts = df["sentence_type"].value_counts().reindex(_SENTENCE_TYPE_ORDER, fill_value=0)
    fig = go.Figure(data=[go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker=dict(color=[_SENTENCE_TYPE_COLORS[t] for t in counts.index]),
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(size=12),
        width=0.5,
    )])
    fig.update_layout(
        title=dict(text="Distribution of Sentence Types", font=dict(size=16, color="#2c3e50"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title=dict(text="Sentence Type", font=dict(size=13)), tickfont=dict(size=12)),
        yaxis=dict(title=dict(text="Count", font=dict(size=13)), tickfont=dict(size=11)),
        showlegend=False, height=500,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=60, l=60, r=40),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#ecf0f1", zeroline=True, zerolinecolor="#bdc3c7")
    return fig


get_sentence_type_countplot = plot_sentence_type_distribution