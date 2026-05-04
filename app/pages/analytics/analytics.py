import pandas as pd
from dash import dcc, html

from app.layout import body_text, card, section_heading


def render(_df: pd.DataFrame, initial_sub_tab: str = "sub-eda") -> html.Div:
    # Sub-tab children are intentionally empty here — content is injected lazily
    # by the render_sub_tab callback in callbacks.py, so only the selected tab is computed.
    return html.Div([
        card(
            section_heading("Major Findings"),
            body_text("We're working on this part. Will update this section soon!"),
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
