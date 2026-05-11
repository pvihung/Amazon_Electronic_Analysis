from dash import dcc, html

from app.layout import body_text, section_heading, PRIMARY, TEXT, MUTED

_FONT = "Segoe UI, Arial, sans-serif"


def render() -> html.Div:
    return html.Div(
        style={"paddingTop": "8px"},
        children=[
            section_heading("Live Demo"),
            body_text(
                "Type a product review sentence and run it through the full two-stage ABSA pipeline — "
                "M1 (RoBERTa + LoRA) detects aspects, M2 (DeBERTa) classifies sentiment for each."
            ),
            html.Div(
                style={
                    "backgroundColor": "#ffffff",
                    "borderRadius": "8px",
                    "padding": "24px 28px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                },
                children=[
                    dcc.Textarea(
                        id="demo-input-text",
                        placeholder='e.g. "The battery life is terrible but the screen is absolutely stunning."',
                        style={
                            "width": "100%",
                            "height": "96px",
                            "padding": "10px 14px",
                            "borderRadius": "6px",
                            "border": "1px solid #D0DCE8",
                            "fontFamily": _FONT,
                            "fontSize": "0.92rem",
                            "color": TEXT,
                            "resize": "vertical",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Button(
                        "Run Pipeline",
                        id="demo-run-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "12px",
                            "backgroundColor": PRIMARY,
                            "color": "white",
                            "border": "none",
                            "borderRadius": "6px",
                            "padding": "10px 24px",
                            "fontFamily": _FONT,
                            "fontSize": "0.87rem",
                            "fontWeight": "700",
                            "letterSpacing": "1px",
                            "textTransform": "uppercase",
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Loading(
                        type="circle",
                        color=PRIMARY,
                        delay_show=200,
                        children=html.Div(id="demo-output", style={"marginTop": "20px"}),
                    ),
                ],
            ),
        ],
    )
