import pandas as pd
from dash import html

from app.layout import insight_note, one_col, sub_section_heading, two_col
from eda import category, covid as eda_covid, price, ratings, text, time as eda_time


def render(df: pd.DataFrame) -> html.Div:
    return html.Div(
        style={"paddingTop": "8px"},
        children=[
            # Rating Analysis
            sub_section_heading("⭐  Rating Analysis"),
            one_col(ratings.rating_distribution(df)),
            insight_note(
                "Even though individual ratings usually track average_rating closely, "
                "there is a noticeable left tail (negative deltas) — worth investigating further."
            ),

            # Price Analysis
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

            # Time Analysis
            sub_section_heading("📅  Time Analysis"),
            two_col(eda_time.reviews_by_year(df), eda_time.reviews_by_month(df)),
            insight_note(
                "A big volume shift started in 2020 (COVID). "
                "Data only runs through Sep 2023, so that year looks low — "
                "compare 2018 - 2022 for fair year-over-year analysis. "
                "December and January consistently peak; September onward loses volume in the dataset."
            ),
            two_col(eda_covid.keyword_shift(df), eda_covid.keyword_lift(df)),
            insight_note(
                "During COVID, WFH-related keywords surged — 'work from home' and 'zoom/video call' "
                "spiked the most. The lift chart shows relative change vs. pre-COVID; "
                "positive values indicate topics that became more prominent during the pandemic."
            ),

            # Text & Sentiment
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

            # Category & Brand
            sub_section_heading("🖥️  Category & Brand"),
            two_col(category.category_distribution(df), category.top_brands_bar(df)),
            insight_note(
                "Laptops dominate the review volume. "
                "Rating distributions are broadly similar across categories, "
                "with slight variation in median scores by brand."
            ),
        ],
    )
