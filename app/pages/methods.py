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
            NAVY,
            "Step 2 — BigQuery joining and review-level cleaning",
            "After filtering the metadata, we used BigQuery SQL to join product records with "
            "their review histories through parent_asin. We removed duplicate reviews, converted "
            "Unix millisecond timestamps into readable dates, and kept useful product fields such "
            "as title, brand, category, price, rating count, and average rating. This produced one "
            "analysis-ready review table that connects customer feedback to product attributes."
        ),

        _step_card(
            3,
            MID_BLUE,
            "Step 3 — EDA feature engineering",
            "For exploratory analysis, we enriched the merged table with variables needed for "
            "the dashboard. Price was grouped into quantile-based tiers, missing prices were "
            "flagged separately, review dates were expanded into year/month/day features, and "
            "review text was prepared for sentiment and keyword analysis. We also used VADER "
            "sentiment scoring to compare review tone with star ratings and product price tiers."
        ),

        _pipeline_overview(),

        _model_card(
            "Stage 1 — Relevance classification",
            "RoBERTa-base binary classifier",
            "The first model predicts whether a sentence is related to the product experience "
            "or unrelated background text. To avoid evaluating on data the model had already "
            "seen, the notebook first creates a global train/test split using a fixed random seed. "
            "Stage 1 is trained only on the global training set, then split again into train, "
            "validation, and stage-test subsets. Because the classes are imbalanced, the model "
            "uses weighted loss with label smoothing instead of oversampling. The decision "
            "threshold is tuned on the validation set and evaluated once on the stage-test set.",
            [
                "Model: roberta-base",
                "Task: unrelated vs. related",
                "Validation tuning: threshold selected by macro F1",
                "Stage-test F1-macro: 0.8214",
                "Global-test independent F1-macro: 0.7999",
            ],
            MID_BLUE,
        ),

        _model_card(
            "Stage 2 — Technical-content classification",
            "RoBERTa-base binary classifier",
            "The second model only receives sentences that are product-related. It predicts "
            "whether a sentence contains technical product information or general opinion. "
            "Since technical sentences are more important for aspect extraction, this stage uses "
            "oversampling on the training data to balance the classes. The model is then evaluated "
            "on a held-out test split and again on the untouched global test set.",
            [
                "Model: roberta-base",
                "Task: general vs. technical",
                "Training strategy: oversampling for class balance",
                "Stage-test F1-macro: 0.8688",
                "Global-test independent F1-macro: 0.8856",
            ],
            NAVY,
        ),

        _model_card(
            "Stage 3 — Aspect extraction",
            "BERT-base multi-label classifier",
            "The final model runs only on sentences that are both related and technical. It "
            "predicts one or more product aspects for each sentence, such as battery, performance, "
            "price, software, design/build, display/audio, or input. This stage removes the "
            "general and unrelated labels because those are already handled by Stages 1 and 2. "
            "To prevent data leakage, augmentation is applied only after the train/validation/test "
            "split and only to the training subset. Each aspect receives its own threshold tuned "
            "on the validation set.",
            [
                "Model: bert-base-uncased",
                "Task: multi-label aspect prediction",
                "Labels: battery, design/build, display/audio, input, performance, price, software",
                "Stage-test F1-macro: 0.7922",
                "Global-test independent F1-macro: 0.7799",
            ],
            MID_BLUE,
        ),

        _results_card(),
    ])


def _pipeline_overview() -> html.Div:
    return card(
        section_heading("Product Aspect Extraction"),
        body_text(
            "This pipeline extracts product-aspect signals from review sentences and define the reviewed aspects as positive or negative. "
            "Pipeline architecture:"
        ),
        html.Div([
            _mini_point("Domain-Adaptive Pretraining (DAPT)", "Before fine-tuning, we further pretrained the language models on a large corpus of Amazon review sentences. This helps the models better understand the specific language and style used in customer reviews."),
            _mini_point("Model 1: Mult-task Aspect Detector", "This model classifies whether a sentence is unrelated, general opinion, or technical content before defining the main aspect(s) in the sentence."),
            _mini_point("Model 2: Aspect Sentiment Classification", "For sentences that contain technical content, a second model classifies the sentiment of each aspect mentioned as positive or negative."),
        ], style={"marginTop": "16px"}),
    )


def _results_card() -> html.Div:
    return card(
        section_heading("End-to-End Evaluation"),
        body_text(
            "The final model will report two types of evaluation. Independent evaluation measures "
            "each stage on the correct ground-truth subset, which helps diagnose individual model "
            "quality. Cascaded evaluation is stricter because mistakes from earlier stages affect "
            "later stages, just like they would in the deployed application. On the untouched "
            "global test set, the cascaded pipeline captured 77.40% of true technical sentences "
            "and achieved a Stage 3 cascaded F1-macro of 0.7255. This shows that the pipeline is "
            "useful for extracting product-aspect signals, while also showing that Stage 1 and "
            "Stage 2 recall remain important areas for improvement."
        ),
        html.Div([
            _metric("Stage 1", "0.7999", "Independent F1-macro"),
            _metric("Stage 2", "0.8856", "Independent F1-macro"),
            _metric("Stage 3", "0.7799", "Independent F1-macro"),
            _metric("Pipeline", "0.7255", "Cascaded F1-macro"),
            _metric("Coverage", "77.40%", "True technical sentences captured"),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
            "gap": "12px",
            "marginTop": "18px",
        }),
    )


def _step_card(num, color, title, description) -> html.Div:
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
            html.Div([
                html.H4(title, style={
                    "margin": "0 0 8px",
                    "color": NAVY,
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                    "fontSize": "1rem",
                }),
                body_text(description, margin="0"),
            ]),
        ],
    )


def _model_card(title, subtitle, description, bullets, color) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": CARD_BG,
            "borderRadius": "6px",
            "padding": "24px",
            "marginBottom": "16px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "borderTop": f"5px solid {color}",
        },
        children=[
            html.H4(title, style={
                "margin": "0 0 4px",
                "color": NAVY,
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "1rem",
            }),
            html.Div(subtitle, style={
                "marginBottom": "10px",
                "color": color,
                "fontWeight": "700",
                "fontFamily": "Segoe UI, Arial, sans-serif",
            }),
            body_text(description, margin="0 0 12px"),
            html.Ul([
                html.Li(item, style={"marginBottom": "6px"}) for item in bullets
            ], style={
                "margin": "0",
                "paddingLeft": "20px",
                "color": "#3A3A3A",
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "0.95rem",
                "lineHeight": "1.55",
            }),
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


def _metric(label, value, note) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontWeight": "700", "color": NAVY}),
        html.Div(value, style={"fontSize": "1.35rem", "fontWeight": "800", "color": GREEN}),
        html.Div(note, style={"fontSize": "0.82rem", "color": "#5C6470"}),
    ], style={
        "backgroundColor": "white",
        "borderRadius": "6px",
        "padding": "14px",
        "border": "1px solid #E5E7EB",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    })
