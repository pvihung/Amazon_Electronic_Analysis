"""
Segments reviews into pre / during / post COVID periods and tracks:
  - Volume and rating shifts across periods
  - Keyword frequency changes (webcam, zoom, battery, etc.)
  - Sentiment shifts by period
  - Category popularity shifts
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PERIOD_ORDER   = ["pre-COVID", "during-COVID", "post-COVID"]
PERIOD_COLORS  = {
    "pre-COVID":     "#3498db",
    "during-COVID":  "#e74c3c",
    "post-COVID":    "#2ecc71",
}

# Keywords tied to WFH / COVID-era device needs
COVID_KEYWORDS = {
    "webcam":        r"\bwebcam\b",
    "zoom / video":  r"\b(?:zoom|teams|meet|video\s+call|video\s+conference)\b",
    "battery life":  r"\bbatter(?:y\s+life|y\s+drain|ies?)\b",
    "work from home":r"\b(?:work\s+from\s+home|wfh|remote\s+work)\b",
    "display":       r"\b(?:screen|display|monitor)\b",
    "speaker":       r"\bspeaker\b",
    "keyboard":      r"\bkeyboard\b",
    "performance":   r"\b(?:performance|fast|slow|lag)\b",
    "price / value": r"\b(?:price|value|worth|expensive|cheap)\b",
    "delivery":      r"\b(?:shipping|delivery|arrived|package)\b",
}


def _add_keyword_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary keyword columns — cached so charts reuse them."""
    df2 = df.copy()
    for kw, pattern in COVID_KEYWORDS.items():
        col = f"_kw_{kw}"
        if col not in df2.columns:
            df2[col] = df2["review_text"].astype(str).str.contains(
                pattern, case=False, regex=True
            ).astype(int)
    return df2


# Chart 1: Review volume by COVID period 

def volume_by_period(df: pd.DataFrame) -> go.Figure:
    """Bar chart — total review count per COVID period."""
    counts = (
        df[df["covid_period"] != "unknown"]
        .groupby("covid_period").size()
        .reindex(PERIOD_ORDER)
        .reset_index(name="reviews")
        .rename(columns={"covid_period": "period"})
    )
    fig = px.bar(
        counts, x="period", y="reviews",
        color="period",
        color_discrete_map=PERIOD_COLORS,
        text="reviews",
        title="Review Volume by COVID Period",
        labels={"period": "", "reviews": "Number of Reviews"},
        category_orders={"period": PERIOD_ORDER},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(showlegend=False, height=400, yaxis_range=[0, None])
    return fig


# Chart 2: Average rating by COVID period 

def rating_by_period(df: pd.DataFrame) -> go.Figure:
    """Bar chart — mean star rating per COVID period."""
    stats = (
        df[df["covid_period"] != "unknown"]
        .groupby("covid_period")["rating"]
        .agg(["mean", "std", "count"])
        .reindex(PERIOD_ORDER)
        .reset_index()
        .rename(columns={"covid_period": "period", "mean": "avg_rating"})
    )

    fig = go.Figure()
    for _, row in stats.iterrows():
        fig.add_trace(go.Bar(
            x=[row["period"]],
            y=[row["avg_rating"]],
            error_y=dict(type="data", array=[row["std"] / np.sqrt(row["count"])],
                         visible=True),
            name=row["period"],
            marker_color=PERIOD_COLORS.get(row["period"], "#95a5a6"),
            text=[f"{row['avg_rating']:.2f}"],
            textposition="outside",
        ))

    fig.update_layout(
        title="Average Star Rating by COVID Period (±SE)",
        xaxis=dict(categoryorder="array", categoryarray=PERIOD_ORDER),
        yaxis=dict(title="Average Rating", range=[3.5, 5]),
        showlegend=False,
        height=400,
    )
    return fig


# Chart 3: Keyword frequency shift across periods 

def keyword_shift(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar — for each keyword, show mention rate (%) across the
    three COVID periods side by side.
    """
    df2     = _add_keyword_cols(df)
    periods = df2[df2["covid_period"] != "unknown"]
    records = []

    for kw in COVID_KEYWORDS:
        col = f"_kw_{kw}"
        for period in PERIOD_ORDER:
            subset = periods[periods["covid_period"] == period]
            if len(subset) == 0:
                continue
            pct = subset[col].mean() * 100
            records.append({"keyword": kw, "period": period, "pct": round(pct, 2)})

    kw_df = pd.DataFrame(records)

    fig = px.bar(
        kw_df, x="keyword", y="pct",
        color="period",
        barmode="group",
        color_discrete_map=PERIOD_COLORS,
        category_orders={"period": PERIOD_ORDER},
        title="Keyword Mention Rate (%) Across COVID Periods",
        labels={"keyword": "Topic Keyword", "pct": "% of Reviews Mentioning"},
    )
    fig.update_layout(
        xaxis_tickangle=-25,
        height=460,
        legend_title="Period",
        margin=dict(b=100),
    )
    return fig


# Chart 4: Keyword lift — during vs pre COVID 

def keyword_lift(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar — relative increase (lift) of each keyword during-COVID
    vs pre-COVID. Positive = more common during COVID.
    """
    df2     = _add_keyword_cols(df)
    pre     = df2[df2["covid_period"] == "pre-COVID"]
    during  = df2[df2["covid_period"] == "during-COVID"]

    records = []
    for kw in COVID_KEYWORDS:
        col       = f"_kw_{kw}"
        pre_rate  = pre[col].mean() if len(pre) else 0
        dur_rate  = during[col].mean() if len(during) else 0
        lift      = ((dur_rate - pre_rate) / (pre_rate + 1e-9)) * 100
        records.append({"keyword": kw, "lift_pct": round(lift, 1)})

    lift_df = pd.DataFrame(records).sort_values("lift_pct")

    max_pos = lift_df["lift_pct"].clip(lower=0).max() or 1

    def _bar_color(v: float) -> str:
        if v < 0:
            return "#e74c3c"
        t = v / max_pos                          # 0 → light blue, 1 → dark blue
        r = int(174 + t * (21  - 174))
        g = int(214 + t * (83  - 214))
        b = int(241 + t * (96  - 241))
        return f"#{r:02x}{g:02x}{b:02x}"

    colors = [_bar_color(v) for v in lift_df["lift_pct"]]

    fig = px.bar(
        lift_df, x="lift_pct", y="keyword",
        orientation="h",
        title="Keyword Lift: During-COVID vs Pre-COVID (%)",
        labels={"lift_pct": "Lift (%)", "keyword": ""},
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=420,
        margin=dict(l=120),
        xaxis=dict(title="Relative change (%) vs pre-COVID"),
    )
    return fig


# Chart 5: Sentiment by COVID period

def sentiment_by_period(df: pd.DataFrame) -> go.Figure:
    """Box plot — Vader sentiment distribution per COVID period."""
    filtered = df[df["covid_period"] != "unknown"]

    fig = px.box(
        filtered, x="covid_period", y="vader_sentiment",
        color="covid_period",
        color_discrete_map=PERIOD_COLORS,
        category_orders={"covid_period": PERIOD_ORDER},
        title="Review Sentiment (Vader) by COVID Period",
        labels={"covid_period": "", "vader_sentiment": "Vader Compound Score"},
    )
    fig.update_layout(showlegend=False, height=420)
    return fig


# Chart 6: Category popularity shift

def category_shift_by_period(df: pd.DataFrame) -> go.Figure:
    """
    100% stacked bar — how the share of each device category changed
    across COVID periods.
    """
    filtered = df[df["covid_period"] != "unknown"]
    counts   = (
        filtered.groupby(["covid_period", "category"])
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("covid_period")["count"].transform("sum")
    counts["share"] = (counts["count"] / totals * 100).round(1)

    fig = px.bar(
        counts, x="covid_period", y="share",
        color="category",
        barmode="stack",
        category_orders={"covid_period": PERIOD_ORDER},
        title="Device Category Share (%) by COVID Period",
        labels={"covid_period": "", "share": "Share (%)", "category": "Category"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=430, yaxis_range=[0, 101])
    return fig
