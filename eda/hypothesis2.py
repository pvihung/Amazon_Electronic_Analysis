"""
Hypothesis 2: COVID-19 shifted consumer priorities for digital devices.
All figures use Plotly and follow the app color palette.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PERIOD_ORDER = ["pre-COVID", "during-COVID", "post-COVID"]
PERIOD_COLORS = {"pre-COVID": "#F4A261", "during-COVID": "#E76F51", "post-COVID": "#C1440E"}

COVID_KEYWORDS = {
    "webcam":          r"\bwebcam\b",
    "zoom/video call": r"\b(?:zoom|teams|meet|video\s+call|video\s+conference)\b",
    "battery life":    r"\bbatter(?:y\s+life|y\s+drain)\b",
    "work from home":  r"\b(?:work\s+from\s+home|wfh|remote\s+work)\b",
    "display/screen":  r"\b(?:screen|display|monitor)\b",
    "speaker":         r"\bspeaker\b",
    "keyboard":        r"\bkeyboard\b",
    "performance":     r"\b(?:performance|fast|slow|lag)\b",
    "price/value":     r"\b(?:price|value|worth|expensive|cheap)\b",
    "delivery":        r"\b(?:shipping|delivery|arrived|package)\b",
}


def _covid_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["covid_period"].isin(PERIOD_ORDER)].copy()


def compute_keyword_matches(dc: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean DataFrame (one column per keyword) for the covid-period subset.

    Pass the result to keyword_mention_rate_chart / keyword_lift_chart to avoid
    re-running the same regexes multiple times on the full text column.
    """
    text_col = dc["review_text"].astype(str)
    return pd.DataFrame(
        {kw: text_col.str.contains(pat, case=False, regex=True)
         for kw, pat in COVID_KEYWORDS.items()},
        index=dc.index,
    )


def period_overview_chart(df: pd.DataFrame) -> go.Figure:
    """Side-by-side: review counts + mean ratings by COVID period."""
    dc = _covid_df(df)
    counts = dc.groupby("covid_period").size().reindex(PERIOD_ORDER).reset_index(name="count")
    ratings = dc.groupby("covid_period")["rating"].mean().reindex(PERIOD_ORDER).reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Review Counts by Period", "Mean Star Rating by Period"),
    )

    colors = [PERIOD_COLORS[p] for p in PERIOD_ORDER]

    fig.add_trace(go.Bar(
        x=counts["covid_period"], y=counts["count"],
        marker_color=colors, showlegend=False,
        text=counts["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=ratings["covid_period"], y=ratings["rating"].round(3),
        marker_color=colors, showlegend=False,
        text=ratings["rating"].round(3),
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title_text="Number of Reviews", range=[0, 155000], row=1, col=1)
    fig.update_yaxes(title_text="Mean Rating", range=[3.5, 4.15], row=1, col=2)
    fig.update_layout(height=420, margin=dict(t=60))
    return fig


def keyword_mention_rate_chart(df: pd.DataFrame, kw_matches: pd.DataFrame = None) -> go.Figure:
    """Grouped bar: keyword mention rate (%) by COVID period.

    Pass ``kw_matches`` (from :func:`compute_keyword_matches`) to skip re-running regexes.
    """
    dc = _covid_df(df)
    if kw_matches is None:
        kw_matches = compute_keyword_matches(dc)

    records = []
    for kw in COVID_KEYWORDS:
        col = kw_matches[kw]
        for period in PERIOD_ORDER:
            mask = dc["covid_period"] == period
            rate = col[mask].mean() * 100
            records.append({"keyword": kw, "period": period, "rate_pct": round(rate, 3)})

    kw_df = pd.DataFrame(records)

    fig = px.bar(
        kw_df, x="keyword", y="rate_pct", color="period",
        barmode="group",
        category_orders={"period": PERIOD_ORDER},
        color_discrete_map=PERIOD_COLORS,
        title="Keyword Mention Rate (%) by COVID Period",
        labels={"rate_pct": "Mention Rate (%)", "keyword": "", "period": "Period"},
    )
    fig.update_layout(height=440, xaxis_tickangle=30, legend=dict(title="Period"))
    return fig


def keyword_lift_chart(df: pd.DataFrame, kw_matches: pd.DataFrame = None) -> go.Figure:
    """Horizontal bar: keyword lift during-COVID vs pre-COVID (%).

    Pass ``kw_matches`` (from :func:`compute_keyword_matches`) to skip re-running regexes.
    """
    dc = _covid_df(df)
    if kw_matches is None:
        kw_matches = compute_keyword_matches(dc)

    rows = {}
    for kw in COVID_KEYWORDS:
        col = kw_matches[kw]
        pre    = col[dc["covid_period"] == "pre-COVID"].mean() * 100
        during = col[dc["covid_period"] == "during-COVID"].mean() * 100
        lift   = round((during - pre) / (pre + 1e-9) * 100, 1)
        rows[kw] = lift

    lift_s = pd.Series(rows).sort_values()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in lift_s]

    fig = go.Figure(go.Bar(
        x=lift_s.values, y=lift_s.index,
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_width=1, line_color="#555")
    fig.update_layout(
        title="Keyword Lift: During-COVID vs Pre-COVID (%)",
        xaxis_title="Relative change (%)",
        yaxis_title="",
        height=420,
        margin=dict(t=60),
    )
    return fig


def category_share_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: device category share (%) by COVID period."""
    dc = _covid_df(df)
    cat_period = (
        pd.crosstab(dc["covid_period"], dc["category"], normalize="index") * 100
    ).reindex(PERIOD_ORDER).reset_index()

    cats = [c for c in cat_period.columns if c != "covid_period"]
    cat_colors = {"laptop": "#e67e22", "desktop": "#2ecc71", "tablet": "#7FAAC4"}

    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Bar(
            x=cat_period["covid_period"],
            y=cat_period[cat].round(1),
            name=cat,
            marker_color=cat_colors.get(cat, "#999"),
            text=cat_period[cat].round(1).astype(str) + "%",
            textposition="inside",
        ))

    fig.update_layout(
        barmode="stack",
        title="Device Category Share (%) by COVID Period",
        xaxis_title="",
        yaxis_title="Share (%)",
        height=420,
        legend_title="Category",
        margin=dict(t=60),
    )
    return fig
