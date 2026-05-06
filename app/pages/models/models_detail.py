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
            section_heading("Architecture and Methodology"),
            body_text(
                "This pipeline extracts product-aspect signals from review sentences and define the reviewed aspects as positive or negative. "
                "Pipeline architecture:"
            ),
            html.Div([
                _mini_point("Domain-Adaptive Pretraining (DAPT)",
                            "Before fine-tuning, we further pretrained the language models on a large corpus of Amazon review sentences. This helps the models better understand the specific language and style used in customer reviews."),
                _mini_point("Model 1: Mult-task Aspect Detector",
                            "This model classifies whether a sentence is unrelated, general opinion, or technical content before defining the main aspect(s) in the sentence."),
                _mini_point("Model 2: Aspect Sentiment Classification",
                            "For sentences that contain technical content, a second model classifies the sentiment of each aspect mentioned as positive or negative."),
            ], style={"marginTop": "16px"}),
        ),

        card(
            section_heading("Domain-Adaptive Pretraining (DAPT)"),
            body_text(
                "Adapt RoBERTa-base to domain before fine-tuning the classification heads. "
                "We collected a large corpus of review sentences from the full dataset, "
                "then continued pretraining the language model with masked language modeling (MLM) on this domain-specific text."
            ),
            html.Div([
                _mini_point("Test Sentence",
                            "The phone is amazing, but the <mask> life is absolutely terrible."),
                _mini_point("Base Model",
                            "['battery', 'Battery', 'sex']"),
                _mini_point("DAPT Model",
                            "['battery', 'batter', 'charger']"),
            ], style={"marginTop": "16px"}),
        ),

        card(
            section_heading("Model 1 — Multitask aspect detector"),
            body_text(
                "A multi-task RoBERTa model with LoRA that filters sentences by relevance and technical content, "
                "then detects which product aspects are mentioned"
            ),
            html.Div([
                _mini_point("Head 1: Relevance Filter",
                            "Classifies whether a sentence is unrelated to the product."),
                _mini_point("Head 2: Technical Filter",
                            "Determines whether the content of sentences passed Head 1 is specific technical feedback or general opinion."),
                _mini_point("Head 3: Aspect Detector",
                            "Identifies which product aspects are mentioned in the sentence"),
            ], style={"marginTop": "16px"}),
        ),

        card(
            section_heading("Model 2 — Multitask aspect detector"),
            body_text(
                "A DeBerta model pre-trained on Aspect-Based Sentiment Analysis tasks."
                "Takes a (sentence, aspect) pair and classifies the sentiment as positive or negative."
            ),
        ),

        _results_card(),
    ])

def _results_card() -> html.Div:
    return card(
        section_heading("End-to-End Evaluation"),
        html.Div([
            _metric("Model 1", "0.8234", "Independent F1-macro"),
            _metric("Model 2", "0.9282", "Independent F1-macro"),
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
