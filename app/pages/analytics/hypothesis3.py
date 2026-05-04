import pandas as pd
from dash import html

from app.layout import body_text, one_col, research_box, sub_section_heading, two_col
from eda import hypothesis3 as h3


def render(df: pd.DataFrame) -> html.Div:
    try:
        summary = h3.get_summary_stats(df)
        fig_scatter, fig_gap = h3.get_plots(df)

        stat_boxes = [
            ("Total Products",   str(summary["total_products"])),
            ("Harsher Raters",   f"{summary['harsher_products']} ({summary['harsher_pct']}%)"),
            ("Lenient Raters",   f"{summary['lenient_products']} ({summary['lenient_pct']}%)"),
            ("No Difference",    str(summary["nodiff_products"])),
            ("Avg Rating Gap",   f"{summary['avg_gap']:+.3f} ★"),
            ("Median t-stat",    f"{summary['median_t_stat']:.3f}"),
            ("Median p-value",   f"{summary['median_p_value']:.4f}"),
            ("Min Reviews (CLT)", str(summary["min_reviews"])),
        ]

        result_df = h3.run_test(df)
        top_rows = (
            result_df.sort_values("p_value")
            .head(10)[["brand", "dataset_reviews", "dataset_avg_rating", "average_rating", "gap", "t_stat", "p_value"]]
            .values.tolist()
        )

        stats_ok = True
    except Exception as e:
        stats_ok    = False
        err_msg     = str(e)
        stat_boxes  = []
        top_rows    = []
        fig_scatter = fig_gap = None

    return html.Div(
        style={"paddingTop": "8px"},
        children=[
            html.H3(
                "Hypothesis 3: Reviewers rate products differently from the platform average",
                style={"color": "#1E2736", "fontSize": "1.15rem",
                       "fontFamily": "Segoe UI, Arial, sans-serif",
                       "fontWeight": "700", "margin": "8px 0 16px"},
            ),
            research_box(
                html.P([html.Strong("H₀: "), "Written reviewers rate products the same as "
                        "the platform average (which includes silent raters)"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Strong("H₁: "), "Written reviewers rate products differently "
                        "(higher or lower) than the platform average"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Em("Intuition: "), "Customers who leave written reviews are a self-selected "
                        "group — they tend to have stronger opinions (positive or negative) than "
                        "silent raters. Their average star rating may systematically deviate from "
                        "the platform-wide average."],
                       style={"margin": "0", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#555"}),
            ),

            html.Div(
                style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                       "padding": "10px 16px", "marginBottom": "16px",
                       "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.88rem",
                       "color": "#555"},
                children=html.P(
                    f"Method: one-sample t-test per product (reviewer avg vs platform avg), "
                    f"α = {summary['alpha']}, min reviews = {summary['min_reviews']}"
                    if stats_ok else err_msg,
                    style={"margin": "0"},
                ),
            ) if stats_ok else html.Div(),

            sub_section_heading("Summary Statistics"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                       "gap": "12px", "margin": "4px 0 24px"},
                children=[_stat_box(lbl, val) for lbl, val in stat_boxes],
            ) if stats_ok else html.Div(),

            sub_section_heading("Reviewer vs Platform Average Rating (per Product)"),
            one_col(fig_scatter) if stats_ok else html.Div(),

            sub_section_heading("Distribution of Rating Gap (Reviewer − Platform Average)"),
            one_col(fig_gap) if stats_ok else html.Div(),

            sub_section_heading("Top 10 Most Significant Products (by p-value)"),
            _results_table(top_rows) if stats_ok else html.Div(),

            sub_section_heading("H3 Conclusion"),
            html.Div(
                style={"backgroundColor": "#EBF4FA", "borderRadius": "6px",
                       "padding": "20px 24px", "borderLeft": "5px solid #34527A"},
                children=[
                    html.P([html.Strong("Result: "),
                            f"{summary['harsher_pct']}% of products have reviewers who rate "
                            f"harsher than the platform average; {summary['lenient_pct']}% rate higher"
                            if stats_ok else err_msg],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.95rem", "color": "#1E2736", "margin": "0 0 12px"}),
                    body_text("The data supports H₁ — written reviewers systematically deviate "
                              "from the platform average for a majority of products:"),
                    html.Ul([
                        html.Li(["A ", html.Strong("one-sample t-test"),
                                 " was run per product, comparing the mean reviewer rating "
                                 "against the platform-wide average rating."]),
                        html.Li([html.Strong(f"{summary['harsher_pct']}% of products"),
                                 " have reviewers who rate ", html.Strong("harsher"),
                                 " than the platform average (p < 0.05)."]),
                        html.Li([html.Strong(f"{summary['lenient_pct']}% of products"),
                                 " have reviewers who rate ", html.Strong("more leniently"),
                                 " than the platform average."]),
                        html.Li(["The mean rating gap is ",
                                 html.Strong(f"{summary['avg_gap']:+.3f} ★"),
                                 " (reviewer avg minus platform avg)."]),
                    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.92rem",
                              "color": "#2d2d2d", "lineHeight": "1.8", "paddingLeft": "20px",
                              "margin": "0 0 12px"}) if stats_ok else html.Div(),
                    html.P([html.Strong("Takeaway: "),
                            "Written reviewers are not a representative sample of all raters. "
                            "They tend to give lower ratings than the platform average, suggesting "
                            "they are more critical. Platforms and researchers should account for "
                            "this self-selection bias when using review ratings to estimate "
                            "true product quality."],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.92rem", "color": "#2d2d2d", "margin": "0"}),
                ],
            ),
        ],
    )


def _stat_box(label: str, value: str) -> html.Div:
    return html.Div(
        style={"backgroundColor": "#EBF4FA", "borderRadius": "6px",
               "padding": "14px 20px", "borderTop": "3px solid #34527A",
               "fontFamily": "Segoe UI, Arial, sans-serif"},
        children=[
            html.Div(label, style={"fontSize": "0.75rem", "color": "#555",
                                   "letterSpacing": "1px", "textTransform": "uppercase",
                                   "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "1.0rem", "fontWeight": "700",
                                   "color": "#34527A"}),
        ],
    )


def _results_table(rows: list) -> html.Table:
    header_style = {
        "padding": "9px 14px", "textAlign": "left", "color": "white",
        "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.85rem",
        "fontWeight": "600", "backgroundColor": "#4A6188",
    }
    headers = html.Tr([
        html.Th("Brand",            style=header_style),
        html.Th("N Reviews",        style=header_style),
        html.Th("Reviewer Avg ★",   style=header_style),
        html.Th("Platform Avg ★",   style=header_style),
        html.Th("Gap",              style=header_style),
        html.Th("t-stat",           style=header_style),
        html.Th("p-value",          style=header_style),
    ])
    body_rows = []
    for i, (brand, n_rev, rev_avg, plat_avg, gap, t_stat, p_val) in enumerate(rows):
        bg   = "#ffffff" if i % 2 == 0 else "#f7fafc"
        cell = {"padding": "8px 14px", "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "0.85rem", "borderBottom": "1px solid #eee", "color": "#2d2d2d"}
        gap_color = "#D85A30" if gap < 0 else "#185FA5"
        gap_cell  = {**cell, "color": gap_color, "fontWeight": "600"}
        body_rows.append(html.Tr([
            html.Td(str(brand),                  style=cell),
            html.Td(f"{int(n_rev):,}",           style=cell),
            html.Td(f"{rev_avg:.3f}",            style=cell),
            html.Td(f"{plat_avg:.3f}",           style=cell),
            html.Td(f"{gap:+.3f}",               style=gap_cell),
            html.Td(f"{t_stat:.3f}",             style=cell),
            html.Td(f"{p_val:.4e}",              style=cell),
        ], style={"backgroundColor": bg}))
    return html.Table(
        [html.Thead(headers), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "borderRadius": "4px",
               "overflow": "hidden", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
               "marginBottom": "20px"},
    )
