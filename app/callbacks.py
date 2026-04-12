import os

import pandas as pd
from dash import Input, Output, html
import plotly.graph_objects as go

from app.layout import (
    body_text, card, insight_note, one_col, props_table,
    research_box, section_heading, sub_section_heading, two_col,
)
from eda import category, correlation, overview, price, ratings, text, hypothesis2, hypothesis1, time as eda_time

# Data source config
# Use absolute path based on project root to work both locally and in production
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_OUTPUT_CSV = os.path.join(PROJECT_ROOT, "dataset", "eda_ready.csv")
USE_LOCAL_CSV = True

_df_cache: dict = {}


def _load_data(force: bool = False) -> pd.DataFrame:
    if "df" not in _df_cache or force:
        if not os.path.exists(LOCAL_OUTPUT_CSV):
            raise FileNotFoundError(f"{LOCAL_OUTPUT_CSV} not found")

        print(f"[LOCAL] Loading from {LOCAL_OUTPUT_CSV} …")
        _df_cache["df"] = pd.read_csv(LOCAL_OUTPUT_CSV, low_memory=False)

        print(f"Loaded {len(_df_cache['df']):,} rows")

    return _df_cache["df"]


# Register callbacks

def register_callbacks(app):

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(tab):
        if tab == "tab-overview":
            return _render_overview_page()
        if tab == "tab-methods":
            return _render_methods_page()

        # Data-dependent tabs
        try:
            df = _load_data()
        except Exception as e:
            return html.Div(
                f"Could not load data: {e}",
                style={"color": "red", "padding": "20px",
                       "whiteSpace": "pre-wrap",
                       "fontFamily": "monospace"},
            )

        if tab == "tab-dataset":
            return _render_dataset_page(df)
        if tab == "tab-analytics":
            return _render_analytics_page(df)
        if tab == 'tab-hypothesis-results':
            return _render_hypothesis_results_page(df)

        return html.Div("Unknown tab.")


# Tab page renderers

def _render_overview_page():
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
            # for my teammate: you can add more questioons here if you wants
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
                    _stat_card("600K+",  "Reviews collected"),
                    _stat_card("3",      "Device categories\n(Laptop · Tablet · Desktop)"),
                    _stat_card("2018 - 2023", "Review time span"),
                ],
            ),
        ),
    ])


def _render_dataset_page(df: pd.DataFrame):
    n_rows    = f"{len(df):,}"
    n_brands  = f"{df['brand'].nunique():,}" if "brand" in df.columns else "—"
    n_products = f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—"
    years     = f"{int(df['year'].min())} - {int(df['year'].max())}" if "year" in df.columns else "—"

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
            one_col(overview.summary_table(df)),
            one_col(overview.describe_table(df)),
        ),
    ])


def _render_methods_page():
    def step_card(num, color, title, description):
        return html.Div(
            style={
                "display": "flex", "gap": "20px", "alignItems": "flex-start",
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "padding": "24px", "marginBottom": "16px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                "borderLeft": f"5px solid {color}",
            },
            children=[
                html.Div(str(num), style={
                    "backgroundColor": color, "color": "white",
                    "borderRadius": "50%", "width": "36px", "height": "36px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontWeight": "700", "fontSize": "1rem", "flexShrink": "0",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                }),
                html.Div([
                    html.H4(title, style={"margin": "0 0 8px", "color": "#1E2736",
                                          "fontFamily": "Segoe UI, Arial, sans-serif",
                                          "fontSize": "1rem"}),
                    body_text(description, margin="0"),
                ]),
            ],
        )

    return html.Div([
        card(
            section_heading("Methods"),
            body_text(
                "The data pipeline consists of three sequential steps that transform "
                "raw Amazon review exports into a clean, analysis-ready table."
            ),
        ),
        step_card(1, "#7FAAC4", "Step 1 — ML Filter",
                  "A scikit-learn text classifier is trained on product titles and metadata "
                  "to distinguish genuine digital devices (laptops, tablets, desktops) from "
                  "unrelated products that may appear in the same broad category. "
                  "This step ensures downstream analyses are not polluted by accessories, "
                  "cases, or unrelated electronics."),
        step_card(2, "#1E2736", "Step 2 — BigQuery SQL",
                  "Filtered products are joined with their full review histories using "
                  "BigQuery SQL. Duplicate reviews are removed, and the reviews table is "
                  "linked to product metadata (brand, price, category). "
                  "The output is a single denormalized table ready for EDA."),
        step_card(3, "#34527A", "Step 3 — EDA Data Preparation",
                  "The merged table is cleaned and enriched: prices are bucketed into "
                  "Low / Medium / High / Premium tiers using quantiles; VADER sentiment "
                  "scores are computed for each review text; non-English reviews are "
                  "detected with Lingua and translated via Google Translate; "
                  "temporal features (year, month, day-of-week) are extracted; "
                  "and a price_missing flag is added for out-of-stock products. "
                  "The result is saved as dataset/eda_ready.csv and uploaded to BigQuery."),
    ])


def _render_analytics_page(df: pd.DataFrame):
    return html.Div([
        card(
            section_heading("Major Findings"),
            body_text(
                "We're working on this part. Will update this section soon!"
            ),
        ),

        # Rating Analysis
        card(
            sub_section_heading("⭐  Rating Analysis"),
            one_col(ratings.rating_distribution(df)),
            insight_note(
                "Even though individual ratings usually track average_rating closely, "
                "there is a noticeable left tail (negative deltas) — worth investigating further."
            ),
            one_col(ratings.rating_delta_histogram(df)),
            one_col(ratings.popularity_vs_rating(df)),
            insight_note(
                "Products with more reviews tend to have slightly higher average ratings, "
                "but the trend is weak — popularity does not strongly predict satisfaction."
            ),
        ),

        # Price Analysis
        card(
            sub_section_heading("💰  Price Analysis"),
            insight_note(
                "185K+ reviews belong to products with no listed price "
                "(out-of-stock / discontinued). A price_missing signal handles these separately."
            ),
            two_col(price.price_distribution(df), price.rating_by_price_tier(df)),
            insight_note(
                "No strong differences in rating distribution across price tiers — "
                "price tier alone is not a great predictor of customer satisfaction."
            ),
            two_col(price.premium_price_boxplot(df), price.spending_over_time(df)),
            insight_note(
                "The Premium tier still shows extreme outliers worth investigating. "
                "Every year, December has the highest estimated spending."
            ),
        ),

        # Time Analysis
        card(
            sub_section_heading("📅  Time Analysis"),
            two_col(eda_time.reviews_by_year(df), eda_time.reviews_by_month(df)),
            insight_note(
                "A big volume shift started in 2020 (COVID). "
                "Data only runs through Sep 2023, so that year looks low — "
                "compare 2018 - 2022 for fair year-over-year analysis. "
                "December and January consistently peak; September onward loses volume in the dataset."
            ),
            two_col(eda_time.reviews_by_day_of_month(df), eda_time.reviews_by_day_of_week(df)),
            insight_note(
                "Fewer reviews are written on weekends. "
                "Day-of-month shows no strong pattern (lower count on day 31 is expected — "
                "not all months have 31 days)."
            ),
        ),

        # Text & Sentiment
        card(
            sub_section_heading("📝  Text & Sentiment"),
            one_col(text.review_length_histogram(df)),
            insight_note(
                "Most reviews are short (< 300 characters) and the distribution is "
                "right-skewed. A small number of reviews are very long (> 2,000 characters). "
                "1-star reviews tend to be longer; 5-star reviews are shorter on average."
            ),
            two_col(text.review_length_by_rating(df), text.vader_by_rating(df)),
            insight_note(
                "VADER generally tracks star ratings correctly. "
                "Some overlap between 4★ and 5★ — very enthusiastic language can confuse "
                "the model, or reviewers may have given a high star rating despite negative text."
            ),
        ),

        # Category & Brand
        card(
            sub_section_heading("🖥️  Category & Brand"),
            two_col(category.category_distribution(df), category.category_rating_boxplot(df)),
            two_col(category.top_brands_bar(df), category.brand_avg_rating(df)),
            insight_note(
                "Laptops dominate the review volume. "
                "Rating distributions are broadly similar across categories, "
                "with slight variation in median scores by brand."
            ),
        ),

        # Correlation
        card(
            sub_section_heading("🔗  Feature Correlation"),
            one_col(correlation.correlation_heatmap(df)),
            insight_note(
                "Strong correlation between rating and average_rating as expected. "
                "Price shows weak correlation with rating, confirming the price-tier analysis above."
            ),
        ),
    ])

def _render_hypothesis_results_page(df: pd.DataFrame):
    h1        = hypothesis1.run_test(df)
    h2_stats  = hypothesis2.get_summary_stats(df)
    result_df = hypothesis2.run_test(df)

    # ── Format result_df for go.Table (product-level) ─────────────────────────
    display = result_df[["brand", "parent_asin", "dataset_reviews",
                          "dataset_avg_rating", "average_rating",
                          "gap", "t_stat", "p_value",
                          "is_harsher", "is_lenient"]].copy()

    display["dataset_avg_rating"] = display["dataset_avg_rating"].round(3)
    display["average_rating"]     = display["average_rating"].round(3)
    display["gap"]                = display["gap"].round(3)
    display["t_stat"]             = display["t_stat"].round(3)
    display["p_value"]            = display["p_value"].round(4)
    display["verdict"] = display.apply(
        lambda r: "Harsher" if r["is_harsher"] else ("Lenient" if r["is_lenient"] else "No diff"),
        axis=1,
    )
    display = display.drop(columns=["is_harsher", "is_lenient"])

    verdict_colors = {
        "Harsher":  "#fde8e0",
        "Lenient":  "#dce8f7",
        "No diff":  "#ecf0f1",
    }

    data_table_fig = go.Figure(
        go.Table(
            header=dict(
                values=["Brand", "Product ASIN", "Reviewer count",
                        "Reviewer avg ★", "Platform avg ★",
                        "Gap (Rev − Platform)", "t-stat", "p-value", "Verdict"],
                fill_color="#2c3e50",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[display[c] for c in display.columns],
                fill_color=[
                    [verdict_colors[v] for v in display["verdict"]]
                    if col == "verdict" else "#ecf0f1"
                    for col in display.columns
                ],
                font=dict(size=11),
                align="left",
            ),
        ),
        layout=dict(
            title="Per-product One-sample t-test Results — Reviewer vs. Platform Average",
            margin=dict(t=50, b=10, l=10, r=10),
        ),
    )

    # ── Unpack two figures from get_plots ─────────────────────────────────────
    fig_scatter, fig_gap = hypothesis2.get_plots(df)

    return html.Div([
        card(
            section_heading("Hypothesis Results"),
            body_text(
                "This section summarizes statistical test results for the main hypotheses."
            ),
        ),

        card(
            sub_section_heading("Hypothesis 1: Rating distribution differs by verified purchase status."),
            body_text(
                "To assess whether verified purchase status is associated with rating behavior, "
                "we compared the rating distributions of verified and non-verified reviews using "
                "a Chi-square test of independence and a Mann-Whitney U test. Both results were "
                "statistically significant, indicating that the two groups differ in rating "
                "distribution, although the strength of the association is modest."
            ),
            one_col(hypothesis1.ecdf_chart(df)),
            insight_note(
                "The ECDF indicates that the rating distribution for verified purchases is "
                "generally shifted toward higher ratings relative to non-verified purchases."
            ),
            two_col(hypothesis1.residual_heatmap(df), hypothesis1.adjusted_residual_heatmap(df)),
            insight_note(
                "Both the standardized and adjusted residual plots indicate that the rating "
                "distribution differs by verified purchase status. The difference is not spread "
                "evenly across all rating levels, but is driven mainly by an overrepresentation "
                "of low ratings among non-verified reviews and an overrepresentation of 5-star "
                "ratings among verified reviews."
            ),
        ),

        card(
            sub_section_heading("Hypothesis 2: Reviewers rate differently from the platform average."),
            body_text(
                "To evaluate whether text reviewers rate products differently from the broader platform average," 
                "we compared each product’s reviewer mean rating with its published average rating using a one-sample t-test. "
                "Products were then grouped according to whether reviewer ratings were significantly lower or higher than the platform average."
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                       "gap": "16px", "marginBottom": "16px"},
                children=[
                    _stat_card(str(h2_stats["total_products"]),   "Products tested"),
                    _stat_card(f"{h2_stats['harsher_pct']}%",     "% products harsher"),
                    _stat_card(f"{h2_stats['lenient_pct']}%",     "% products lenient"),
                ],
            ),
            one_col(fig_scatter),
            insight_note("Text reviewers tend to rate products below the broader platform average."),
            one_col(fig_gap),
            insight_note("The downward bias is broad-based rather than driven by only a few outliers.")
        ),
    ])

# Private helpers

CARD_BG = "#ffffff"


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
