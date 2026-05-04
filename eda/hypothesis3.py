"""
Hypothesis 3 (revised): Reviewers (people who leave written reviews) rate products
differently from the platform average (which includes both reviewers and silent raters).

Format:
run_test          -> df per-product one-sample t-test results
get_summary_stats -> summary dict of key metrics
get_plots         -> (fig_scatter, fig_gap) plotly figures
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import numpy as np

MIN_REVIEWS = 30
ALPHA       = 0.05

COLORS = {
    "harsher": "#D85A30",
    "lenient": "#185FA5",
    "neutral": "#888780",
    "zero":    "#2C2C2A",
    "mean":    "#D85A30",
}


def _build_results(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1 — product-level review stats (deduplicated per product)
    product_review_stats = df.groupby("parent_asin").agg(
        brand              = ("brand",  "first"),
        dataset_avg_rating = ("rating", "mean"),
        dataset_rating_std = ("rating", "std"),
        dataset_reviews    = ("rating", "count"),
    ).reset_index()

    # Step 2 — product-level metadata (one row per product)
    product_meta = (
        df.drop_duplicates(subset="parent_asin")
        [["parent_asin", "average_rating", "rating_number"]]
    )

    product_stats = product_review_stats.merge(product_meta, on="parent_asin")

    # Step 3 — CLT filter + sanity check
    product_stats = product_stats[
        (product_stats["dataset_reviews"] >= MIN_REVIEWS) &
        (product_stats["rating_number"]   >  product_stats["dataset_reviews"])
    ].copy()

    # Step 4 — pre-group ratings for performance
    grouped_ratings = df.groupby("parent_asin")["rating"].apply(list).to_dict()

    # Step 5 — one-sample t-test per product
    def _one_sample_ttest(row):
        t_stat, p_val = stats.ttest_1samp(
            a       = grouped_ratings[row["parent_asin"]],
            popmean = row["average_rating"],
        )
        return pd.Series({"t_stat": t_stat, "p_value": p_val})

    product_stats[["t_stat", "p_value"]] = product_stats.apply(_one_sample_ttest, axis=1)

    # Step 6 — verdicts & gap
    product_stats["gap"]        = product_stats["dataset_avg_rating"] - product_stats["average_rating"]
    product_stats["is_harsher"] = (product_stats["p_value"] < ALPHA) & (product_stats["t_stat"] < 0)
    product_stats["is_lenient"] = (product_stats["p_value"] < ALPHA) & (product_stats["t_stat"] > 0)
    product_stats["no_diff"]    =  product_stats["p_value"] >= ALPHA

    output_cols = [
        "parent_asin", "brand",
        "dataset_reviews", "dataset_avg_rating",
        "average_rating", "rating_number",
        "gap", "t_stat", "p_value",
        "is_harsher", "is_lenient", "no_diff",
    ]
    return product_stats[output_cols].sort_values("gap").reset_index(drop=True)


def run_test(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product one-sample t-test results. Returns full result dataframe."""
    return _build_results(df)


def get_summary_stats(df: pd.DataFrame) -> dict:
    result_df = _build_results(df)
    return {
        "total_products":  len(result_df),
        "harsher_products": int(result_df["is_harsher"].sum()),
        "lenient_products": int(result_df["is_lenient"].sum()),
        "nodiff_products":  int(result_df["no_diff"].sum()),
        "harsher_pct":     round(result_df["is_harsher"].mean() * 100, 1),
        "lenient_pct":     round(result_df["is_lenient"].mean() * 100, 1),
        "avg_gap":         round(result_df["gap"].mean(), 3),
        "median_t_stat":   round(result_df["t_stat"].median(), 3),
        "median_p_value":  round(result_df["p_value"].median(), 4),
        "alpha":           ALPHA,
        "min_reviews":     MIN_REVIEWS,
    }


def get_plots(df: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """
    Returns two plotly figures:
        fig_scatter — reviewer avg vs platform avg per product
        fig_gap     — KDE distribution of rating gaps by verdict group
    """
    result_df = _build_results(df)

    harsher = result_df[result_df["is_harsher"]]
    lenient = result_df[result_df["is_lenient"]]
    neutral = result_df[result_df["no_diff"]]

    # Figure 1: Scatter
    fig_scatter = go.Figure()

    for subset, label, color, symbol in [
        (neutral, "No significant diff",    COLORS["neutral"], "circle"),
        (harsher, "Harsher (p < 0.05)",     COLORS["harsher"], "circle"),
        (lenient, "Lenient (p < 0.05)",     COLORS["lenient"], "triangle-up"),
    ]:
        fig_scatter.add_trace(go.Scatter(
            x=subset["average_rating"],
            y=subset["dataset_avg_rating"],
            mode="markers",
            name=label,
            marker=dict(color=color, size=7, symbol=symbol,
                        opacity=0.4 if label == "No significant diff" else 0.8),
            customdata=subset[["brand", "gap", "p_value", "dataset_reviews"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Platform avg: %{x:.3f}<br>"
                "Reviewer avg: %{y:.3f}<br>"
                "Gap: %{customdata[1]:.3f} stars<br>"
                "p-value: %{customdata[2]:.4f}<br>"
                "n reviews: %{customdata[3]}"
                "<extra></extra>"
            ),
        ))

    # perfect agreement diagonal
    fig_scatter.add_shape(
        type="line", x0=1, x1=5, y0=1, y1=5,
        line=dict(color=COLORS["zero"], dash="dash", width=1),
        opacity=0.5,
    )

    fig_scatter.update_layout(
        title="Reviewer vs Platform Average Rating (per Product)",
        xaxis_title="Platform Average Rating",
        yaxis_title="Reviewer Average Rating",
        xaxis=dict(range=[1, 5.3], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(range=[1, 5.3], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
    )

    # Figure 2: KDE gap distribution
    def _kde_trace(series, label, color, dash="solid"):
        x_range = np.linspace(series.min() - 0.3, series.max() + 0.3, 300)
        kde = stats.gaussian_kde(series)

        # convert hex → rgba for fill opacity
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fill_color = f"rgba({r},{g},{b},0.08)"

        return go.Scatter(
            x=x_range, y=kde(x_range),
            mode="lines", name=label,
            line=dict(color=color, width=2, dash=dash),
            fill="tozeroy",
            fillcolor=fill_color,
        )

    fig_gap = go.Figure()

    for subset, label, color, dash in [
        (neutral["gap"], "No significant diff", COLORS["neutral"], "dot"),
        (lenient["gap"], "Lenient (p < 0.05)",  COLORS["lenient"], "dash"),
        (harsher["gap"], "Harsher (p < 0.05)",  COLORS["harsher"], "solid"),
    ]:
        if len(subset) >= 5:
            fig_gap.add_trace(_kde_trace(subset, label, color, dash))

    # zero line
    fig_gap.add_vline(
        x=0, line=dict(color=COLORS["zero"], dash="dash", width=1), opacity=0.6,
    )
    # overall mean gap line
    mean_gap = result_df["gap"].mean()
    fig_gap.add_vline(
        x=mean_gap,
        line=dict(color=COLORS["mean"], dash="dot", width=1.5),
        annotation_text=f"mean gap: {mean_gap:.3f}",
        annotation_position="top right",
        opacity=0.8,
    )

    fig_gap.update_layout(
        title="Distribution of Rating Gap (Reviewer − Platform Average)",
        xaxis_title="Rating Gap (Stars)",
        yaxis_title="Density",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
    )

    return fig_scatter, fig_gap