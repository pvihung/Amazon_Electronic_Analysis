"""
Rating distribution figures
All charts related to review ratings and average product ratings.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Side-by-side bar charts: individual ratings vs average ratings."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution of Review Ratings",
                        "Distribution of Average Product Ratings"),
    )

    rating_counts = df["rating"].value_counts().sort_index()
    avg_counts    = df["average_rating"].value_counts().sort_index()

    fig.add_trace(
        go.Bar(x=rating_counts.index, y=rating_counts.values,
               name="Review Rating", marker_color="#3498db"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=avg_counts.index, y=avg_counts.values,
               name="Average Rating", marker_color="#e67e22"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Rating", row=1, col=1)
    fig.update_xaxes(title_text="Average Rating", row=1, col=2)
    fig.update_yaxes(title_text="Number of Reviews", row=1, col=1)
    fig.update_layout(
        title="Rating Distributions",
        showlegend=False,
        height=450,
    )
    return fig


def rating_delta_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of (review rating − average product rating)."""
    delta = df["rating"] - df["average_rating"]
    fig = px.histogram(
        delta,
        nbins=10,
        title="Rating Delta: Individual Review vs Product Average",
        labels={"value": "Rating Delta", "count": "Number of Reviews"},
        color_discrete_sequence=["#9b59b6"],
    )
    fig.update_layout(
        xaxis_title="Rating Delta (review − average)",
        yaxis_title="Number of Reviews",
        showlegend=False,
        height=420,
    )
    return fig


def popularity_vs_rating(df: pd.DataFrame) -> go.Figure:
    """Scatter with regression trend: number of reviews vs average rating."""
    product_stats = (
        df.groupby("parent_asin")["average_rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "average_rating", "count": "review_count"})
    )

    fig = px.scatter(
        product_stats,
        x="review_count",
        y="average_rating",
        trendline="lowess",
        title="Product Popularity vs Average Rating",
        labels={"review_count": "Number of Reviews",
                "average_rating": "Average Rating"},
        opacity=0.4,
        color_discrete_sequence=["#1abc9c"],
    )
    fig.update_layout(yaxis_range=[1, 5.1], height=440)
    return fig
