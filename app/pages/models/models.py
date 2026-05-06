import pandas as pd
from dash import dcc, html

from app.layout import body_text, card, section_heading


def render(initial_sub_tab: str = "sub-models-data-overview") -> html.Div:
    # Sub-tab children are intentionally empty here — content is injected lazily
    # by the render_sub_tab callback in callbacks.py, so only the selected tab is computed.
    return html.Div([
        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "6px",
                "padding": "0 36px 32px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            },
            children=[
                dcc.Tabs(
                    id="models-sub-tabs",
                    value=initial_sub_tab,
                    children=[
                        _sub_tab("sub-models-data_overview",  "Data Overview and Preprocessing"),
                        _sub_tab("sub-models-models-detail", "Models Detail"),
                    ],
                    style={"borderBottom": "2px solid #D0DCE8"},
                    colors={"border": "transparent", "primary": "transparent", "background": "transparent"},
                ),
                dcc.Loading(
                    type="circle",
                    color="#34527A",
                    delay_show=300,
                    children=html.Div(id="models-sub-content"),
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
