from dash import html


def render() -> html.Div:
    return html.Div(
        style={"padding": "40px 0", "textAlign": "center"},
        children=[
            html.P(
                "Hypothesis 3 analysis coming soon.",
                style={
                    "color": "#888",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                    "fontSize": "0.95rem",
                },
            ),
        ],
    )
