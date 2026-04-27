from dash import html

from app.layout import body_text, card, research_box, section_heading


def render() -> html.Div:
    return html.Div([
        card(
            section_heading("Project Overview"),
            body_text(
                "Amazon is one of the world's largest marketplaces for digital devices. "
                "With millions of customer reviews spanning laptops, tablets, and desktops, "
                "this dataset offers a rich opportunity to understand how consumers evaluate "
                "technology products - from star ratings and pricing patterns to the language "
                "they use when expressing satisfaction or disappointment."
            ),
            body_text(
                "This project collects and cleans Amazon product review data, applies a "
                "machine-learning classifier to filter for genuine digital device reviews, "
                "and then performs a comprehensive exploratory data analysis (EDA). "
                "We examine rating distributions, price-tier effects, seasonal trends, "
                "review text length, VADER sentiment scores, brand performance, and "
                "feature correlations."
            ),
            research_box(
                html.H4("Research Question",
                        style={"color": "#1E2736", "margin": "0 0 8px",
                               "fontFamily": "Segoe UI, Arial, sans-serif",
                               "fontSize": "0.95rem"}),
                html.Ul(
                    [
                        html.Li("What factors do device category, price tier, brand, or review sentiment, best predict whether a customer leaves a high (≥ 4★) or low (≤ 2★) rating?"),
                        html.Li("How have review volumes and spending patterns shifted over time?"),
                    ],
                    style={"color": "#2d2d2d", "fontSize": "0.95rem",
                           "fontFamily": "Segoe UI, Arial, sans-serif",
                           "lineHeight": "1.75", "margin": "0", "paddingLeft": "20px"},
                ),
            ),
        ),
        card(
            section_heading("Key Highlights"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                       "gap": "20px"},
                children=[
                    _stat_card("~ 400k",       "Reviews collected"),
                    _stat_card("3",            "Device categories\n(Laptop · Tablet · Desktop)"),
                    _stat_card("2018 - 2023",  "Review time span"),
                ],
            ),
        ),
    ])


def _stat_card(value: str, label: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#EBF4FA",
            "borderRadius": "6px",
            "padding": "20px 24px",
            "textAlign": "center",
            "borderTop": "3px solid #34527A",
        },
        children=[
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": "700",
                                   "color": "#34527A", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
            html.Div(label, style={"fontSize": "0.82rem", "color": "#555",
                                   "marginTop": "6px", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
        ],
    )
