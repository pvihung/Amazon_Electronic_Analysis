import pandas as pd
from dash import dcc, html

from app.layout import body_text, card, section_heading, PRIMARY, NAV_BG, TEXT, MUTED


def _finding(label: str, text: str) -> html.Div:
    return html.Div([
        html.Strong(label + ": ", style={"color": NAV_BG}),
        html.Span(text, style={"color": TEXT}),
    ], style={
        "marginBottom": "10px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": "0.92rem",
        "lineHeight": "1.65",
    })


def render(_df: pd.DataFrame, initial_sub_tab: str = "sub-eda") -> html.Div:
    # Sub-tab children are intentionally empty here — content is injected lazily
    # by the render_sub_tab callback in callbacks.py, so only the selected tab is computed.
    return html.Div([
        card(
            section_heading("Major Findings"),
            _finding("COVID as an inflection point",
                     "2020 was a genuine inflection point for digital device consumer behavior — "
                     "not just a volume spike. Mentions of webcams, speakers, microphones, and "
                     "video call software (Zoom, Teams) surged dramatically as devices shifted "
                     "from personal to professional tools. Crucially, several of these features "
                     "remained elevated even after restrictions lifted, suggesting a lasting "
                     "change in what consumers expect from digital devices."),
            _finding("H1 — Satisfaction has price breakpoints, not a smooth gradient (Supported)",
                     "Budget products (Q1) score highest, satisfaction dips in the mid-range, "
                     "and only partially recovers at premium tiers — the $25 bin averages 4.52★ "
                     "while the $475 bin drops to 3.70★. Paying more does not reliably buy "
                     "more satisfaction."),
            _finding("H2 — Non-verified reviewers polarise ratings (Supported)",
                     "Verified buyers rate moderately and consistently; non-verified reviewers "
                     "cluster at the extremes (1★ and 5★). The difference is statistically "
                     "significant across both the Chi-square and Mann-Whitney tests, though the "
                     "effect size is small — large datasets amplify even modest behavioural gaps."),
            _finding("H3 — Written reviewers rate harsher than the platform average (Supported)",
                     "Customers who write reviews are a self-selected, more critical group. "
                     "For the majority of products, their mean rating falls below the published "
                     "platform average — which also includes silent star-only raters. "
                     "Treating written reviews as representative overstates dissatisfaction."),
        ),
        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "6px",
                "padding": "0 36px 32px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            },
            children=[
                dcc.Tabs(
                    id="analysis-sub-tabs",
                    value=initial_sub_tab,
                    children=[
                        _sub_tab("sub-eda",  "EDA"),
                        _sub_tab("sub-hyp1", "Hypothesis 1"),
                        _sub_tab("sub-hyp2", "Hypothesis 2"),
                        _sub_tab("sub-hyp3", "Hypothesis 3"),
                    ],
                    style={"borderBottom": "2px solid #D0DCE8"},
                    colors={"border": "transparent", "primary": "transparent", "background": "transparent"},
                ),
                dcc.Loading(
                    type="circle",
                    color="#34527A",
                    delay_show=300,
                    children=html.Div(id="analysis-sub-content"),
                ),
            ],
        ),
    ])


def _sub_tab(value: str, label: str) -> dcc.Tab:
    base = {
        "backgroundColor": "transparent",
        "color": "#8A9AB0",
        "border": "none",
        "borderBottom": "3px solid transparent",
        "padding": "10px 28px 12px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "fontSize": "0.75rem",
        "fontWeight": "700",
        "letterSpacing": "1.5px",
        "textTransform": "uppercase",
        "transition": "color 0.2s ease, border-bottom-color 0.2s ease",
        "marginBottom": "-2px",
    }
    selected = {
        **base,
        "color": "#1E2736",
        "borderBottom": "3px solid #1E2736",
    }
    return dcc.Tab(label=label, value=value, style=base, selected_style=selected)
