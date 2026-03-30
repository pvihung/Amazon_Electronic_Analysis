"""
Device category and brand figures
Category breakdown (laptop / tablet / desktop) and top brand analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def category_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of reviews by device category."""
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "reviews"]

    fig = px.pie(
        counts,
        names="category",
        values="reviews",
        title="Review Distribution by Device Category",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=420, hoverlabel=dict(bgcolor="white", font_color="black"))
    return fig


def category_rating_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plots of star ratings by device category."""
    fig = px.box(
        df,
        x="category",
        y="rating",
        color="category",
        title="Star Rating Distribution by Device Category",
        labels={"category": "Category", "rating": "Star Rating"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


def top_brands_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the most reviewed brands."""
    brand_counts = (
        df[df["brand"] != "unknown"]["brand"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    brand_counts.columns = ["brand", "reviews"]

    fig = px.bar(
        brand_counts,
        x="reviews",
        y="brand",
        orientation="h",
        title=f"Top {top_n} Brands by Number of Reviews",
        labels={"reviews": "Number of Reviews", "brand": "Brand"},
        color="reviews",
        color_continuous_scale=["#7FAAC4", "#34527A", "#1E2736"],
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=480,
        coloraxis_showscale=False,
        hoverlabel=dict(bgcolor="white", font_color="black"),
    )
    return fig


def brand_avg_rating(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of average star rating per brand (top brands only)."""
    top_brands = (
        df[df["brand"] != "unknown"]["brand"]
        .value_counts()
        .head(top_n)
        .index
    )
    brand_rating = (
        df[df["brand"].isin(top_brands)]
        .groupby("brand")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
        .sort_values("avg_rating", ascending=False)
    )

    fig = px.bar(
        brand_rating,
        x="brand",
        y="avg_rating",
        title=f"Average Star Rating — Top {top_n} Brands",
        labels={"brand": "Brand", "avg_rating": "Average Rating"},
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        range_color=[3.5, 5],
    )
    fig.update_layout(
        xaxis_tickangle=-30,
        showlegend=False,
        height=440,
    )
    return fig
