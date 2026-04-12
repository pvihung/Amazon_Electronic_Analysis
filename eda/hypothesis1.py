"""
Hypothesis 1 — Rating distribution differs by verified purchase status
Format
run_test(df) ->  all stats, tables, residuals
ecdf_chart(df)  ->  ECDF by group
residual_heatmap(df)    ->  standardised residuals heatmap
adjusted_residual_heatmap(df)  ->  adjusted residuals heatmap
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, mannwhitneyu

ALPHA = 0.05


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["group"]    = d["verified_purchase"].map({True: "Verified", False: "Non-verified"})
    d["is_5_star"] = (d["rating"] == 5).astype(int)
    return d


def run_test(df: pd.DataFrame) -> dict:
    d = _prepare(df)

    # Summary stats
    summary = d.groupby("group").agg(
        n_reviews     = ("rating", "size"),
        mean_rating   = ("rating", "mean"),
        median_rating = ("rating", "median"),
        pct_5_star    = ("is_5_star", lambda x: x.mean() * 100),
    ).round(3).reset_index()

    # Contingency tables
    count_tab = pd.crosstab(d["group"], d["rating"])
    prop_tab  = pd.crosstab(d["group"], d["rating"], normalize="index").mul(100).round(2)

    # Chi-square test
    chi2, p, dof, expected_arr = chi2_contingency(count_tab)
    expected_df = pd.DataFrame(expected_arr, index=count_tab.index, columns=count_tab.columns)
    std_resid   = (count_tab - expected_df) / np.sqrt(expected_df)

    n         = count_tab.to_numpy().sum()
    r, c      = count_tab.shape
    cramers_v = np.sqrt((chi2 / n) / min(r - 1, c - 1))

    # Adjusted standardised residuals (Agresti 2002)
    row_totals = count_tab.sum(axis=1)
    col_totals = count_tab.sum(axis=0)
    expected_outer = np.outer(row_totals, col_totals) / n
    adj_resid = (count_tab - expected_outer) / np.sqrt(
        expected_outer
        * (1 - row_totals.values[:, None] / n)
        * (1 - col_totals.values[None, :] / n)
    )

    # Mann-Whitney U
    mw_stat, mw_p = mannwhitneyu(
        d[d["group"] == "Verified"]["rating"],
        d[d["group"] == "Non-verified"]["rating"],
        alternative="two-sided",
    )

    return {
        "summary":        summary,
        "count_tab":      count_tab,
        "prop_tab":       prop_tab,
        "chi2":           round(chi2, 2),
        "dof":            int(dof),
        "p_value":        p,
        "cramers_v":      round(cramers_v, 4),
        "conclusion":     "Reject H0" if p < ALPHA else "Fail to reject H0",
        "std_resid":      std_resid,
        "adj_resid":      adj_resid,
        "mann_whitney_u": round(mw_stat, 2),
        "mann_whitney_p": mw_p,
    }


def ecdf_chart(df: pd.DataFrame) -> go.Figure:
    d      = _prepare(df)
    colors = {"Verified": "#34527A", "Non-verified": "#7FAAC4"}

    fig = go.Figure()
    for group, gdf in d.groupby("group"):
        sorted_ratings = gdf["rating"].sort_values()
        ecdf           = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
        fig.add_trace(go.Scatter(
            x    = sorted_ratings,
            y    = ecdf,
            mode = "lines",
            name = group,
            line = dict(color=colors.get(group, "#888780"), width=2),
            hovertemplate = f"<b>{group}</b><br>Rating: %{{x}}★<br>Cumulative: %{{y:.2%}}<extra></extra>",
        ))

    fig.update_layout(
        title         = "ECDF of Ratings by Verified Purchase Status",
        xaxis_title   = "Rating (stars)",
        yaxis_title   = "Cumulative proportion",
        legend_title  = "",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        xaxis         = dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis         = dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        margin        = dict(t=60, b=40, l=40, r=20),
    )
    return fig


def residual_heatmap(df: pd.DataFrame) -> go.Figure:
    std_resid = run_test(df)["std_resid"]

    fig = go.Figure(go.Heatmap(
        z             = std_resid.values,
        x             = [str(c) for c in std_resid.columns],
        y             = std_resid.index.tolist(),
        colorscale    = "RdBu",
        zmid          = 0,
        text          = std_resid.round(2).values,
        texttemplate  = "%{text}",
        hovertemplate = "Group: %{y}<br>Rating: %{x}★<br>Std. residual: %{z:.2f}<extra></extra>",
        colorbar      = dict(title="Std. residual"),
    ))
    fig.update_layout(
        title         = "Standardised Residuals from Chi-square Test",
        xaxis_title   = "Rating (stars)",
        yaxis_title   = "",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        margin        = dict(t=60, b=60, l=120, r=20),
        annotations   = [dict(
            x=0.5, y=-0.25, xref="paper", yref="paper",
            text="abs(value) > 2 means the cell contributes notably to the difference",
            showarrow=False, font=dict(size=11, color="#555"),
        )],
    )
    return fig


def adjusted_residual_heatmap(df: pd.DataFrame) -> go.Figure:
    adj_resid = run_test(df)["adj_resid"]

    fig = go.Figure(go.Heatmap(
        z             = adj_resid.values,
        x             = [str(c) for c in adj_resid.columns],
        y             = adj_resid.index.tolist(),
        colorscale    = "RdBu",
        zmid          = 0,
        text          = adj_resid.round(2).values,
        texttemplate  = "%{text}",
        hovertemplate = "Group: %{y}<br>Rating: %{x}★<br>Adj. residual: %{z:.2f}<extra></extra>",
        colorbar      = dict(title="Z-score"),
    ))
    fig.update_layout(
        title         = "Adjusted Residuals: Where the bias actually lies",
        xaxis_title   = "Rating (stars)",
        yaxis_title   = "",
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        margin        = dict(t=60, b=60, l=120, r=20),
    )
    return fig
