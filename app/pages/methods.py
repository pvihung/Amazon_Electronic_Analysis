from dash import html

from app.layout import body_text, card, section_heading, CARD_BG


ACCENT_BLUE = "#7FAAC4"
NAVY = "#1E2736"
MID_BLUE = "#34527A"
GOLD = "#C79A3B"
GREEN = "#3D7A5F"


def render() -> html.Div:
    return html.Div([
        card(
            section_heading("Methods"),
            body_text(
                "Our project uses a two-part methodology. First, we clean and filter the "
                "Amazon Electronics product and review data so the analysis focuses on real "
                "digital devices rather than accessories or unrelated items. Second, we use a "
                "three-stage NLP pipeline to identify whether each review sentence is relevant, "
                "technical, and which product aspect it discusses. The resulting tables support "
                "the dashboard visualizations, hypothesis testing, and text-based insights."
            ),
        ),

        _step_card(
            1,
            ACCENT_BLUE,
            "Step 1 — Metadata filtering with TF-IDF + Logistic Regression",
            "We started with Amazon Electronics metadata and trained a scikit-learn text "
            "classifier to separate true digital devices from noisy products in the same broad "
            "category. Product titles and metadata were converted into TF-IDF features, then a "
            "Logistic Regression model was used to classify items as device or non-device. This "
            "reduced noise from cases, screen protectors, chargers, keyboards, and other "
            "accessories before joining products to review data."
        ),

        _step_card(
            2,
            MID_BLUE,
            "Step 2 — BigQuery joining and review-level cleaning",
            "After filtering the metadata, we used BigQuery SQL to join product records with "
            "their review histories through parent_asin. We removed duplicate reviews, converted "
            "Unix millisecond timestamps into readable dates, and kept useful product fields such "
            "as title, brand, category, price, rating count, and average rating. This produced one "
            "analysis-ready review table that connects customer feedback to product attributes."
        ),

        _step_card(
            3,
            ACCENT_BLUE,
            "Step 3 — EDA feature engineering",
            "For exploratory analysis, we enriched the merged table with variables needed for "
            "the dashboard. Price was grouped into quantile-based tiers, missing prices were "
            "flagged separately, review dates were expanded into year/month/day features, and "
            "review text was prepared for sentiment and keyword analysis. We also used VADER "
            "sentiment scoring to compare review tone with star ratings and product price tiers."
        ),

        _step_card(
            4,
            MID_BLUE,
            "Step 4 — Hypothesis testing",
            "We formulated three statistical hypotheses to examine patterns in the review data beyond exploratory summaries.",
            extra=[
                _mini_point("Hypothesis 1: Price vs. Satisfaction",
                            "Tests whether customer satisfaction changes linearly with price or whether distinct breakpoints exist. "
                            "Addressed with Pearson and Spearman correlation to detect non-linearity, and a Kruskal-Wallis test "
                            "across price quartiles to confirm that satisfaction differences between tiers are statistically significant."),
                _mini_point("Hypothesis 2: Verified vs. Non-Verified Reviewers",
                            "Tests whether verified and non-verified purchasers leave systematically different rating distributions. "
                            "Evaluated with a Chi-square test of independence (effect size measured by Cramér's V) and confirmed "
                            "with a non-parametric Mann-Whitney U test."),
                _mini_point("Hypothesis 3: Written Reviewers vs. Platform Average",
                            "Tests whether customers who write text reviews rate products differently from the platform-wide average — "
                            "which includes silent raters who left a star but no text. Tested with a per-product one-sample t-test "
                            "comparing reviewer mean ratings against each product's published average rating."),
            ],
        ),

        _step_card(
            5,
            ACCENT_BLUE,
            "Step 5 — Product aspect extraction",
            "This pipeline extracts product-aspect signals from review sentences and classifies "
            "the sentiment of each identified aspect as positive or negative. Pipeline architecture:",
            extra=[
                _mini_point("Domain-Adaptive Pretraining (DAPT)", "Before fine-tuning, we further pretrained the language models on a large corpus of Amazon review sentences. This helps the models better understand the specific language and style used in customer reviews."),
                _mini_point("Model 1: Multi-task Aspect Detector", "This model classifies whether a sentence is unrelated, general opinion, or technical content before defining the main aspect(s) in the sentence."),
                _mini_point("Model 2: Aspect Sentiment Classification", "For sentences that contain technical content, a second model classifies the sentiment of each aspect mentioned as positive or negative."),
            ],
        ),
    ])



def _step_card(num, color, title, description, extra=None) -> html.Div:
    inner = [
        html.H4(title, style={
            "margin": "0 0 8px",
            "color": NAVY,
            "fontFamily": "Segoe UI, Arial, sans-serif",
            "fontSize": "1rem",
        }),
        body_text(description, margin="0" if extra is None else "0 0 12px"),
    ]
    if extra:
        inner.append(html.Div(extra))
    return html.Div(
        style={
            "display": "flex",
            "gap": "20px",
            "alignItems": "flex-start",
            "backgroundColor": CARD_BG,
            "borderRadius": "6px",
            "padding": "24px",
            "marginBottom": "16px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "borderLeft": f"5px solid {color}",
        },
        children=[
            html.Div(str(num), style={
                "backgroundColor": color,
                "color": "white",
                "borderRadius": "50%",
                "width": "36px",
                "height": "36px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontWeight": "700",
                "fontSize": "1rem",
                "flexShrink": "0",
                "fontFamily": "Segoe UI, Arial, sans-serif",
            }),
            html.Div(inner),
        ],
    )



def _mini_point(label, text) -> html.Div:
    return html.Div([
        html.Strong(label + ": ", style={"color": NAVY}),
        html.Span(text),
    ], style={
        "marginBottom": "8px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": "0.95rem",
        "lineHeight": "1.55",
        "color": "#3A3A3A",
    })


