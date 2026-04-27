import pandas as pd
from dash import html

from app.layout import body_text, card, one_col, props_table, section_heading
from eda import overview as eda_overview


def render(df: pd.DataFrame) -> html.Div:
    n_rows     = f"{len(df):,}"
    n_brands   = f"{df['brand'].nunique():,}" if "brand" in df.columns else "—"
    n_products = f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—"
    years      = f"{int(df['year'].min())} - {int(df['year'].max())}" if "year" in df.columns else "—"

    return html.Div([
        card(
            section_heading("Dataset"),
            body_text(
                "Dataset is open source from Hugging Face: https://amazon-reviews-2023.github.io. "
                "Each row represents one customer review "
                "and is linked to its product's metadata - including listed price, brand, "
                "and category. Reviews are filtered with an ML classifier and further "
                "enriched with computed features: price tier, VADER sentiment score, "
                "review language, and temporal attributes."
            ),
            html.Div(style={"marginTop": "20px"},
                     children=[props_table([
                         ("Total reviews (EDA-ready)",   n_rows),
                         ("Unique products",              n_products),
                         ("Unique brands",                n_brands),
                         ("Review period",                years),
                         ("Device categories",            "Laptop, Tablet, Desktop"),
                         ("Rating scale",                 "1 - 5 stars"),
                         ("Source",                       "Amazon Product Reviews (public)"),
                         ("Language handling",            "Non-English reviews translated via Google Translate"),
                     ])]),
        ),
        card(
            section_heading("Data Summary"),
            one_col(eda_overview.summary_table(df)),
            one_col(eda_overview.describe_table(df)),
        ),
    ])
