"""
Hypothesis 1: There are distinct price breakpoints where satisfaction meaningfully shifts.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BIN_SIZE  = 50
MIN_COUNT = 30


def _prep_binned(df: pd.DataFrame) -> tuple:
    """Filter to priced products, apply $50 bins, return (wp, binned)."""
    wp = df[(df["price_missing"] == 0) & (df["price"] > 0)].copy()
    max_price = min(wp["price"].quantile(0.97), 3000)
    bins = list(range(0, int(max_price) + BIN_SIZE, BIN_SIZE))
    wp["price_bin"] = pd.cut(wp["price"], bins=bins)
    wp["price_mid"] = wp["price_bin"].apply(lambda x: x.mid if pd.notna(x) else None)

    binned = (
        wp.groupby("price_mid", observed=True)["rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"price_mid": "price", "mean": "avg_rating"})
    )
    binned = binned[binned["count"] >= MIN_COUNT].copy()
    binned["se"] = binned["std"] / binned["count"] ** 0.5
    return wp, binned


def price_breakpoint_chart(df: pd.DataFrame) -> go.Figure:
    """Two-panel: avg rating by price bin (top) + % negative reviews (bottom)."""
    wp, binned = _prep_binned(df)

    wp["is_negative"] = (wp["rating"] <= 3).astype(int)
    neg_binned = (
        wp.groupby("price_mid", observed=True)["is_negative"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"price_mid": "price", "mean": "neg_rate"})
    )
    neg_binned = neg_binned[neg_binned["count"] >= MIN_COUNT].copy()
    neg_binned["neg_pct"] = neg_binned["neg_rate"] * 100

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Average Star Rating by Price (${BIN_SIZE} bins) — Breakpoint View",
            "% Negative Reviews (1–3 stars) by Price",
        ),
        vertical_spacing=0.14,
    )

    # ±1 SE shaded band
    x_fwd = list(binned["price"].astype(float))
    x_rev = x_fwd[::-1]
    y_hi  = list(binned["avg_rating"] + binned["se"])
    y_lo  = list(binned["avg_rating"] - binned["se"])
    fig.add_trace(go.Scatter(
        x=x_fwd + x_rev,
        y=y_hi + y_lo[::-1],
        fill="toself",
        fillcolor="rgba(52,152,219,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 SE",
        showlegend=False,
    ), row=1, col=1)

    # Avg rating line
    fig.add_trace(go.Scatter(
        x=binned["price"], y=binned["avg_rating"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=4),
        name="Avg Rating",
        showlegend=False,
    ), row=1, col=1)

    # Typical plateau zone shading (4.0–4.5)
    fig.add_hrect(
        y0=4.0, y1=4.5,
        fillcolor="rgba(230,126,34,0.10)",
        line_width=0,
        annotation_text="Typical plateau zone",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color="#888",
        row=1, col=1,
    )

    # % negative line
    fig.add_trace(go.Scatter(
        x=neg_binned["price"], y=neg_binned["neg_pct"],
        mode="lines+markers",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=4),
        name="% Negative",
        showlegend=False,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Average Rating", row=1, col=1)
    fig.update_yaxes(title_text="% Negative Reviews", row=2, col=1)
    fig.update_layout(
        height=680,
        legend=dict(x=0.75, y=0.97),
        margin=dict(t=60),
    )
    return fig


def rating_by_category_breakpoint(df: pd.DataFrame) -> go.Figure:
    """Rating vs price breakpoints, one panel per device category."""
    wp, _ = _prep_binned(df)
    categories = sorted(wp["category"].dropna().unique())
    n = len(categories)
    colors = ["#3498db", "#2ecc71", "#e67e22"]

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[c.title() for c in categories],
        shared_yaxes=True,
    )

    for i, cat in enumerate(categories):
        cat_df = wp[wp["category"] == cat]
        binned = (
            cat_df.groupby("price_mid", observed=True)["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"price_mid": "price", "mean": "avg_rating"})
        )
        binned = binned[binned["count"] >= 10]

        fig.add_trace(go.Scatter(
            x=binned["price"], y=binned["avg_rating"],
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=1.5),
            marker=dict(size=3),
            name=cat.title(),
            showlegend=False,
        ), row=1, col=i + 1)

        fig.update_xaxes(title_text="Price ($)", row=1, col=i + 1)

    fig.update_yaxes(title_text="Avg Rating", row=1, col=1)
    fig.update_layout(
        title="Rating vs Price Breakpoints by Device Category",
        height=400,
        margin=dict(t=60),
    )
    return fig
