"""
Price analysis figures
Price distribution, tier breakdowns, outliers, and spending over time.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of prices for products that have a price listed."""
    with_price = df[df["price_missing"] == 0]
    fig = px.histogram(
        with_price,
        x="price",
        nbins=30,
        title="Price Distribution (Products with Known Price)",
        labels={"price": "Price ($)", "count": "Number of Reviews"},
        color_discrete_sequence=["#2ecc71"],
    )
    fig.update_layout(yaxis_title="Number of Reviews", height=420)
    return fig


def rating_by_price_tier(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: proportion of each star rating within each price tier."""
    with_price = df[df["price_missing"] == 0].copy()
    tier_order  = ["Low", "Medium", "High", "Premium"]
    tier_rating = (
        with_price.groupby(["price_tier", "rating"])
        .size()
        .reset_index(name="count")
    )
    tier_totals = tier_rating.groupby("price_tier")["count"].transform("sum")
    tier_rating["proportion"] = tier_rating["count"] / tier_totals

    fig = px.bar(
        tier_rating,
        x="price_tier",
        y="proportion",
        color="rating",
        barmode="stack",
        category_orders={"price_tier": tier_order, "rating": [1, 2, 3, 4, 5]},
        title="Rating Distribution by Price Tier",
        labels={"proportion": "Proportion", "price_tier": "Price Tier",
                "rating": "Star Rating"},
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=440)
    return fig


def premium_price_boxplot(df: pd.DataFrame) -> go.Figure:
    """Box plot of premium-tier prices to show outliers."""
    premium = df[(df["price_missing"] == 0) & (df["price_tier"] == "Premium")]
    fig = px.box(
        premium,
        y="price",
        title="Price Distribution — Premium Tier (Outlier Check)",
        labels={"price": "Price ($)"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_layout(height=400)
    return fig


def spending_over_time(df: pd.DataFrame) -> go.Figure:
    """Line chart: total estimated spending (sum of prices) by month × year."""
    with_price = df[df["price_missing"] == 0].copy()
    spending   = (
        with_price.groupby(["year", "month"])["price"]
        .sum()
        .div(1e6)                 # convert to millions
        .reset_index(name="spending_millions")
    )

    fig = px.line(
        spending,
        x="month",
        y="spending_millions",
        color="year",
        title="Estimated Monthly Spending by Year",
        labels={"month": "Month", "spending_millions": "Spending (Millions $)",
                "year": "Year"},
        markers=True,
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"]),
        height=450,
    )
    return fig
