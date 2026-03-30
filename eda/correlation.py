"""
Correlation heatmap
Numeric feature correlation matrix visualised as a Plotly heatmap.
"""

import pandas as pd
import plotly.graph_objects as go


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of Pearson correlations among numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    corr        = numeric_df.corr().round(2)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            text=corr.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title="Correlation Matrix (Numeric Features)",
        xaxis_tickangle=-30,
        height=520,
        margin=dict(l=120, b=120),
    )
    return fig
