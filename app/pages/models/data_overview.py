import os

import numpy as np
from dash import html

from app.layout import (
    body_text, one_col, two_col, section_heading,
    insight_note, PRIMARY, NAV_BG, TEXT, MUTED,
)
from models.labeled_data_overview import (
    load_labeled_data,
    apply_pair_parsing,
    add_sentence_classification,
    fit_mlb,
    get_technical_df,
    get_rare_classes,
    iterative_split,
    plot_sentence_type_distribution,
    CFG,
)

_RARE_THRESHOLD = CFG.AUG_RARE_THRESHOLD
_AUG_NUM        = CFG.AUG_NUM_PER_SAMPLE


def render() -> html.Div:
    base_path      = os.path.join(os.path.dirname(__file__), '..', '..', '..','dataset')
    final_csv_path = os.path.join(base_path, 'final.csv')

    try:
        df = load_labeled_data(final_csv_path)
        df = apply_pair_parsing(df)
        df = add_sentence_classification(df)
        mlb, aspect_classes = fit_mlb(df)
        df_tech = get_technical_df(df)

        train_df, _, _, y_train, _, _ = iterative_split(df_tech, mlb)
        rare_classes = get_rare_classes(y_train, aspect_classes, _RARE_THRESHOLD)

        counts_before = y_train.sum(axis=0).astype(int)
        counts_after = _simulate_aug_counts(y_train, rare_classes, _AUG_NUM)

        sentence_fig    = plot_sentence_type_distribution(df)

        fig_before = _aspect_bar(counts_before, aspect_classes, "Before Augmentation")
        fig_after  = _aspect_bar(counts_after,  aspect_classes, "After Augmentation")

        return html.Div(
            style={"paddingTop": "8px"},
            children=[

                # Section 1: Sentence Type Distribution
                section_heading("Sentence Type Distribution"),
                one_col(sentence_fig),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "12px",
                        "margin": "16px 0 32px",
                    },
                ),

                # Section 2: Parsing Example with Cell
                section_heading('Aspects and Sentiment'),
                html.Div(
                    style={
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "8px",
                        "padding": "20px",
                        "marginBottom": "32px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)",
                        "fontFamily": "Segoe UI, Arial, sans-serif"
                    },
                    children=[
                        html.Div(
                            style={"marginBottom": "20px", "fontSize": "0.95rem"},
                            children=[
                                html.Strong("Original Text: ", style={"color": PRIMARY}),
                                html.Span(
                                    '"The display is stunning and it runs incredibly fast, but the battery life is terrible."',
                                    style={"color": TEXT, "fontStyle": "italic", "backgroundColor": "#f9f9f9",
                                           "padding": "4px 8px", "borderRadius": "4px"}
                                )
                            ]
                        ),

                        html.Table(
                            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.9rem"},
                            children=[
                                html.Thead(
                                    html.Tr([
                                        html.Th("Extracted Aspect", style={"textAlign": "left", "padding": "10px",
                                                                           "borderBottom": f"2px solid {NAV_BG}",
                                                                           "color": MUTED}),
                                        html.Th("Polarity / Sentiment", style={"textAlign": "left", "padding": "10px",
                                                                               "borderBottom": f"2px solid {NAV_BG}",
                                                                               "color": MUTED})
                                    ])
                                ),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(html.Code("Display & Audio; Performance; Battery",
                                                          style={"backgroundColor": "#f1f5f9", "padding": "2px 6px",
                                                                 "borderRadius": "4px"}),
                                                style={"padding": "10px", "borderBottom": "1px solid #eee"}),
                                        html.Td(html.Span("1;1;-1", style={"fontWeight": "600",
                                                                             "backgroundColor": "#f1f5f9",
                                                                             "padding": "4px 8px",
                                                                             "borderRadius": "4px"}),
                                                style={"padding": "10px", "borderBottom": "1px solid #eee"}),
                                    ]),
                                ])
                            ]
                        )
                    ]
                ),

                # Section 3: Aspect Distribution Before / After Aug
                section_heading("Aspect Label Distribution"),
                body_text(
                    f"Aspect counts in the training split before and after back-translation "
                    f"augmentation. Bars in red fall below the rare-class threshold "
                    f"({_RARE_THRESHOLD} samples). Rare-class rows are duplicated "
                    f"×{_AUG_NUM} with augmented text."
                ),
                two_col(fig_before, fig_after),
                insight_note(
                    f"{len(rare_classes)} rare class(es) will be augmented "
                    f"via back-translation (EN → DE → EN) during training."
                ),
            ],
        )

    except Exception as e:
        import traceback
        return html.Div([
            html.H3("Error Loading Models Data",
                    style={"color": "red", "fontFamily": "Segoe UI, Arial, sans-serif"}),
            body_text(f"Failed to load final.csv: {str(e)}"),
            html.Pre(
                traceback.format_exc(),
                style={"fontSize": "0.78rem", "color": "#888",
                       "whiteSpace": "pre-wrap", "fontFamily": "monospace"},
            ),
        ])

def _simulate_aug_counts(
    y_train: np.ndarray,
    rare_classes: list,
    num_aug: int,
) -> np.ndarray:
    """
    Simulate post-augmentation aspect counts without running actual augmentation.

    Mirrors the logic in augment_rare_classes:
      - Find rows that contain at least one rare-class aspect
      - Those rows are duplicated `num_aug` times
      - Return original counts + duplicated counts per aspect
    """
    if not rare_classes:
        return y_train.sum(axis=0).astype(int)

    # Mask: rows that belong to at least one rare class
    to_aug_mask = np.zeros(len(y_train), dtype=bool)
    for aspect_idx, _, _ in rare_classes:
        to_aug_mask |= (y_train[:, aspect_idx] == 1)

    aug_y      = y_train[to_aug_mask]                      # rows being augmented
    aug_counts = aug_y.sum(axis=0) * num_aug               # added counts per aspect
    return (y_train.sum(axis=0) + aug_counts).astype(int)


def _aspect_bar(counts: np.ndarray, aspect_classes, title: str):
    import plotly.graph_objects as go

    order  = np.argsort(counts)[::-1]
    labels = [aspect_classes[i] for i in order]
    values = [int(counts[i])    for i in order]
    colors = ["#e74c3c" if v < _RARE_THRESHOLD else "#34527A" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=values,
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.add_vline(
        x=_RARE_THRESHOLD,
        line=dict(color="#e74c3c", width=1.5, dash="dash"),
        annotation_text=f"threshold ({_RARE_THRESHOLD})",
        annotation_position="top right",
        annotation_font=dict(color="#e74c3c", size=10),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#2c3e50"),
                   x=0.5, xanchor="center"),
        xaxis=dict(title="Count", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        height=max(360, len(aspect_classes) * 34 + 100),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(t=60, b=40, l=140, r=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ecf0f1")
    return fig

def _aspect_table(
    counts_before: np.ndarray,
    counts_after:  np.ndarray,
    aspect_classes,
) -> html.Div:

    h = {"padding": "10px 14px", "color": "white", "fontWeight": "600",
         "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.85rem",
         "textAlign": "left"}

    header = html.Tr([
        html.Th("Aspect",  style=h),
        html.Th("Before",  style={**h, "textAlign": "right"}),
        html.Th("After",   style={**h, "textAlign": "right"}),
        html.Th("Δ Added", style={**h, "textAlign": "right"}),
        html.Th("Status",  style={**h, "textAlign": "center"}),
    ])

    def cell(content, align="left"):
        return html.Td(content, style={
            "padding": "9px 14px",
            "fontFamily": "Segoe UI, Arial, sans-serif",
            "fontSize": "0.87rem",
            "color": TEXT,
            "borderBottom": "1px solid #eee",
            "textAlign": align,
        })

    order     = np.argsort(counts_before)[::-1]
    body_rows = []

    for rank, i in enumerate(order):
        cls     = aspect_classes[i]
        before  = int(counts_before[i])
        after   = int(counts_after[i])
        delta   = after - before
        is_rare = before < _RARE_THRESHOLD

        status = html.Span(
            "Rare" if is_rare else "Normal",
            style={
                "backgroundColor": "#fdecea" if is_rare else "#eaf7f0",
                "color":           "#c0392b" if is_rare else "#1e8449",
                "borderRadius": "4px", "padding": "2px 8px",
                "fontSize": "0.78rem", "fontWeight": "600",
                "fontFamily": "Segoe UI, Arial, sans-serif",
            },
        )

        body_rows.append(html.Tr([
            cell(cls),
            cell(f"{before:,}", "right"),
            cell(f"{after:,}",  "right"),
            cell(f"+{delta:,}" if delta > 0 else "—", "right"),
            html.Td(status, style={"padding": "9px 14px", "textAlign": "center",
                                   "borderBottom": "1px solid #eee"}),
        ], style={"backgroundColor": "#ffffff" if rank % 2 == 0 else "#fafafa"}))

    return html.Div(
        html.Table(
            [
                html.Thead(header, style={"backgroundColor": NAV_BG}),
                html.Tbody(body_rows),
            ],
            style={
                "width": "100%", "borderCollapse": "collapse",
                "borderRadius": "4px", "overflow": "hidden",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                "marginBottom": "12px",
            },
        )
    )


def _stat_box(label: str, value: str, accent: str = PRIMARY) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#EBF4FA", "borderRadius": "6px",
            "padding": "14px 20px", "borderTop": f"3px solid {accent}",
            "fontFamily": "Segoe UI, Arial, sans-serif",
        },
        children=[
            html.Div(label, style={"fontSize": "0.75rem", "color": MUTED,
                                   "letterSpacing": "1px", "textTransform": "uppercase",
                                   "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "1.0rem", "fontWeight": "700",
                                   "color": PRIMARY}),
        ],
    )