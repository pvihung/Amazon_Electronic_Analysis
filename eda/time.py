"""
Time-series / temporal analysis figures
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]


def reviews_by_year(df: pd.DataFrame) -> go.Figure:
    counts = df["year"].value_counts().sort_index().reset_index()
    counts.columns = ["year", "reviews"]

    fig = px.bar(
        counts,
        x="year",
        y="reviews",
        title="Number of Reviews by Year",
        labels={"year": "Year", "reviews": "Number of Reviews"},
        color_discrete_sequence=["#3498db"],
    )
    fig.update_layout(height=420, xaxis=dict(type="category"))
    return fig


def reviews_by_month(df: pd.DataFrame) -> go.Figure:
    counts = df["month"].value_counts().sort_index().reset_index()
    counts.columns = ["month", "reviews"]
    counts["month_name"] = counts["month"].map(_MONTH_LABELS)

    fig = px.bar(
        counts,
        x="month_name",
        y="reviews",
        title="Number of Reviews by Month",
        labels={"month_name": "Month", "reviews": "Number of Reviews"},
        category_orders={"month_name": list(_MONTH_LABELS.values())},
        color_discrete_sequence=["#e67e22"],
    )
    fig.update_layout(height=420)
    return fig


def reviews_by_day_of_month(df: pd.DataFrame) -> go.Figure:
    counts = df["date_of_month"].value_counts().sort_index().reset_index()
    counts.columns = ["day", "reviews"]

    fig = px.bar(
        counts,
        x="day",
        y="reviews",
        title="Number of Reviews by Day of Month",
        labels={"day": "Day of Month", "reviews": "Number of Reviews"},
        color_discrete_sequence=["#1abc9c"],
    )
    fig.update_layout(height=400)
    return fig


def reviews_by_day_of_week(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["day_of_week"]
        .value_counts()
        .reindex(_DOW_ORDER)
        .reset_index()
    )
    counts.columns = ["day_of_week", "reviews"]

    fig = px.bar(
        counts,
        x="day_of_week",
        y="reviews",
        title="Number of Reviews by Day of Week",
        labels={"day_of_week": "Day", "reviews": "Number of Reviews"},
        category_orders={"day_of_week": _DOW_ORDER},
        color_discrete_sequence=["#9b59b6"],
    )
    fig.update_layout(height=400)
    return fig
