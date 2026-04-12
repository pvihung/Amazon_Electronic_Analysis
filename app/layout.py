from dash import dcc, html

# Design
PRIMARY    = "#34527A"   # deep slate blue — headings / accent
DARK_BLUE  = "#34527A"   # deep slate blue — hero header
STEEL_BLUE = "#7FAAC4"   # medium steel blue — secondary accent
ICE_BLUE   = "#D9EEF7"   # pale ice blue — page background
NAV_BG     = "#1E2736"   # dark navy — navigation bar
PAGE_BG    = ICE_BLUE
CARD_BG    = "#ffffff"
TEXT       = "#2d2d2d"
MUTED      = "#555555"


# Layout
def build_layout() -> html.Div:
    return html.Div(
        style={"fontFamily": "Georgia, 'Times New Roman', serif",
               "backgroundColor": PAGE_BG, "minHeight": "100vh"},
        children=[
            # ── Hero ────────────────────────────────────────────────────────
            html.Div(
                style={
                    "backgroundColor": DARK_BLUE,
                    "padding": "52px 64px 40px",
                    "textAlign": "center",
                },
                children=[
                    html.P(
                        "DATA SCIENCE · AMAZON REVIEWS ANALYSIS",
                        style={"color": "rgba(255,255,255,0.65)", "letterSpacing": "3px",
                               "fontSize": "0.72rem", "margin": "0 0 14px",
                               "fontFamily": "Segoe UI, Arial, sans-serif"},
                    ),
                    html.H1(
                        "Amazon Digital Devices EDA",
                        style={"color": "white", "margin": "0 0 14px",
                               "fontSize": "2.6rem", "fontWeight": "700",
                               "letterSpacing": "-0.5px"},
                    ),
                    html.P(
                        "Understanding review patterns across laptops, tablets, "
                        "and desktops — ratings, pricing, sentiment, and brand analysis",
                        style={"color": "rgba(255,255,255,0.82)", "fontSize": "1.05rem",
                               "maxWidth": "580px", "margin": "0 auto",
                               "lineHeight": "1.6",
                               "fontFamily": "Segoe UI, Arial, sans-serif"},
                    ),
                ],
            ),

            #Navigation tabs
            html.Div(
                style={"backgroundColor": NAV_BG},
                children=[
                    dcc.Tabs(
                        id="main-tabs",
                        value="tab-overview",
                        children=[
                            _nav_tab("tab-overview",  "OVERVIEW"),
                            _nav_tab("tab-dataset",   "DATASET"),
                            _nav_tab("tab-methods",   "METHODS"),
                            _nav_tab("tab-analytics", "ANALYTICS"),
                            _nav_tab("tab-hypothesis-results", "HYPOTHESIS RESULTS"),
                        ],
                        style={"border": "none", "maxWidth": "1100px", "margin": "0 auto"},
                        colors={
                            "border":     "transparent",
                            "primary":    "transparent",
                            "background": "transparent",
                        },
                    ),
                ],
            ),

            # Page content
            dcc.Loading(
                id="tab-content-spinner",
                type="default",
                color=PRIMARY,
                children=html.Div(
                    id="tab-content",
                    style={"maxWidth": "1100px", "margin": "0 auto",
                           "padding": "40px 32px 60px"},
                ),
            ),

        ],
    )


# Shared content helpers 

def card(*children, extra_style=None) -> html.Div:
    """White card container."""
    style = {
        "backgroundColor": CARD_BG,
        "borderRadius": "6px",
        "padding": "32px 36px",
        "marginBottom": "24px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(style=style, children=list(children))


def section_heading(text: str) -> html.Div:
    """Orange section heading with bottom rule, matching the reference style."""
    return html.Div(
        style={"marginBottom": "20px"},
        children=[
            html.H2(
                text,
                style={
                    "color": PRIMARY,
                    "fontSize": "1.0rem",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                    "fontWeight": "700",
                    "letterSpacing": "2px",
                    "margin": "0 0 8px",
                    "textTransform": "uppercase",
                },
            ),
            html.Hr(style={"border": "none", "borderTop": f"2px solid {PRIMARY}",
                           "margin": "0"}),
        ],
    )


def sub_section_heading(text: str) -> html.Div:
    """Smaller analytics sub-section heading (e.g. '⭐ Rating Analysis')."""
    return html.H3(
        text,
        style={
            "color": PRIMARY,
            "fontSize": "1.05rem",
            "fontFamily": "Segoe UI, Arial, sans-serif",
            "fontWeight": "600",
            "margin": "32px 0 12px",
            "paddingBottom": "6px",
            "borderBottom": f"1px solid {PRIMARY}",
        },
    )


def research_box(*children) -> html.Div:
    """Left-bordered callout box for research questions / key takeaways."""
    return html.Div(
        style={
            "borderLeft": f"4px solid {PRIMARY}",
            "backgroundColor": "#EBF4FA",
            "padding": "16px 20px",
            "borderRadius": "0 4px 4px 0",
            "margin": "16px 0",
        },
        children=list(children),
    )


def insight_note(text: str) -> html.Div:
    """Inline analytical note (amber highlight)."""
    return html.Div(
        f"💡 {text}",
        style={
            "backgroundColor": "#fef9e7",
            # "border": "1px solid #f39c12",
            "borderRadius": "5px",
            "padding": "8px 14px",
            "fontSize": "0.88rem",
            "color": "#6e4c00",
            "fontFamily": "Segoe UI, Arial, sans-serif",
            "marginTop": "0px",
            "marginBottom": "16px",
        },
    )


def props_table(rows: list[tuple]) -> html.Table:
    """Dataset properties table styled like the reference image."""
    header = html.Tr([
        html.Th("Property",
                style={"padding": "10px 16px", "textAlign": "left",
                       "color": "white", "fontWeight": "600",
                       "fontFamily": "Segoe UI, Arial, sans-serif",
                       "fontSize": "0.9rem"}),
        html.Th("Value",
                style={"padding": "10px 16px", "textAlign": "left",
                       "color": "white", "fontWeight": "600",
                       "fontFamily": "Segoe UI, Arial, sans-serif",
                       "fontSize": "0.9rem"}),
    ])

    body_rows = []
    for i, (prop, val) in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#fafafa"
        body_rows.append(html.Tr([
            html.Td(prop, style={"padding": "10px 16px", "color": TEXT,
                                 "fontFamily": "Segoe UI, Arial, sans-serif",
                                 "fontSize": "0.88rem", "fontWeight": "500",
                                 "borderBottom": "1px solid #eee"}),
            html.Td(val,  style={"padding": "10px 16px", "color": MUTED,
                                 "fontFamily": "Segoe UI, Arial, sans-serif",
                                 "fontSize": "0.88rem",
                                 "borderBottom": "1px solid #eee"}),
        ], style={"backgroundColor": bg}))

    return html.Table(
        [
            html.Thead(header, style={"backgroundColor": NAV_BG}),
            html.Tbody(body_rows),
        ],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "borderRadius": "4px",
            "overflow": "hidden",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
        },
    )


def body_text(text: str, **kwargs) -> html.P:
    style = {
        "color": TEXT,
        "fontSize": "0.95rem",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "lineHeight": "1.75",
        "margin": "0 0 14px",
    }
    style.update(kwargs)
    return html.P(text, style=style)


def one_col(fig) -> html.Div:
    """Full-width Plotly figure."""
    return html.Div(
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        style={"marginBottom": "8px"},
    )


def two_col(fig_left, fig_right) -> html.Div:
    """Two Plotly figures side by side."""
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
               "gap": "20px", "marginBottom": "8px"},
        children=[
            dcc.Graph(figure=fig_left,  config={"displayModeBar": False}),
            dcc.Graph(figure=fig_right, config={"displayModeBar": False}),
        ],
    )


# kept for backwards compat — same as insight_note
def section_note(text: str) -> html.Div:
    return insight_note(text)


# Private helpers

def _nav_tab(value: str, label: str) -> dcc.Tab:
    base = {
        "backgroundColor": "transparent",
        "color": "rgba(255,255,255,0.65)",
        "border": "none",
        "borderBottom": "3px solid transparent",
        "padding": "14px 24px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "letterSpacing": "2px",
        "fontSize": "0.78rem",
        "fontWeight": "600",
    }
    selected = {**base,
                "color": "white",
                "borderBottom": "3px solid white"}
    return dcc.Tab(label=label, value=value, style=base, selected_style=selected)
