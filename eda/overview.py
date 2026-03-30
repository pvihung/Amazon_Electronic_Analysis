"""
Data overview figures
Provides basic dataset statistics as Plotly figures for the Dash app.
"""

import pandas as pd
import plotly.graph_objects as go


def summary_table(df: pd.DataFrame) -> go.Figure:
    """A simple table showing shape, dtypes, and null counts."""
    info = pd.DataFrame({
        "Column":    df.columns,
        "Dtype":     [str(df[c].dtype) for c in df.columns],
        "Non-Null":  [df[c].notna().sum() for c in df.columns],
        "Null %":    [(df[c].isna().mean() * 100).round(2) for c in df.columns],
    })

    fig = go.Figure(
        go.Table(
            header=dict(
                values=list(info.columns),
                fill_color="#2c3e50",
                font=dict(color="white", size=13),
                align="left",
            ),
            cells=dict(
                values=[info[c] for c in info.columns],
                fill_color="#ecf0f1",
                align="left",
                font=dict(size=12),
            ),
        )
    )
    fig.update_layout(
        title=f"Dataset Overview — {len(df):,} rows × {len(df.columns)} columns",
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def describe_table(df: pd.DataFrame) -> go.Figure:
    """Descriptive statistics for numeric columns."""
    desc = df.describe().round(2).reset_index().rename(columns={"index": "Stat"})

    fig = go.Figure(
        go.Table(
            header=dict(
                values=list(desc.columns),
                fill_color="#2c3e50",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[desc[c] for c in desc.columns],
                fill_color="#ecf0f1",
                align="left",
                font=dict(size=11),
            ),
        )
    )
    fig.update_layout(
        title="Descriptive Statistics (Numeric Columns)",
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig
