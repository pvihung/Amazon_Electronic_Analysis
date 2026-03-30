"""
Text and sentiment analysis figures
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def review_length_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of review character lengths."""
    fig = px.histogram(
        df,
        x="review_length",
        nbins=50,
        title="Distribution of Review Length (Characters)",
        labels={"review_length": "Characters", "count": "Count"},
        color_discrete_sequence=["#2980b9"],
    )
    fig.update_layout(
        yaxis_title="Number of Reviews",
        height=420,
        margin=dict(b=60),
    )
    return fig


def review_length_by_rating(df: pd.DataFrame) -> go.Figure:
    """Box plots of review length split by star rating."""
    fig = px.box(
        df,
        x="rating",
        y="review_length",
        title="Review Length by Star Rating",
        labels={"rating": "Star Rating", "review_length": "Review Length (chars)"},
        color="rating",
        color_discrete_sequence=px.colors.sequential.RdBu,
    )
    fig.update_layout(
        showlegend=False,
        height=470,
        annotations=[dict(
            text="1-star reviews tend to be longer; 5-star reviews are shorter on average.",
            xref="paper", yref="paper",
            x=0.5, y=-0.28, showarrow=False,
            font=dict(size=11, color="#7f8c8d"),
        )],
        margin=dict(b=110),
    )
    return fig


def vader_by_rating(df: pd.DataFrame) -> go.Figure:
    """Box plots of Vader compound sentiment score split by star rating."""
    fig = px.box(
        df,
        x="rating",
        y="vader_sentiment",
        title="Vader Sentiment Score by Star Rating",
        labels={"rating": "Star Rating", "vader_sentiment": "Vader Compound Score"},
        color="rating",
        color_discrete_sequence=px.colors.diverging.RdYlGn,
    )
    fig.update_layout(
        showlegend=False,
        height=470,
        annotations=[dict(
            text=(
                "Vader generally tracks star ratings well. "
                "Note overlap between 4★ and 5★ — very enthusiastic 5-star text can confuse the model."
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.28, showarrow=False,
            font=dict(size=11, color="#7f8c8d"),
        )],
        margin=dict(b=110),
    )
    return fig
