import pandas as pd
from dash import html

from app.layout import body_text, one_col, research_box, sub_section_heading, two_col
from eda import hypothesis2 as h2


def render(df: pd.DataFrame) -> html.Div:
    try:
        results = h2.run_test(df)
        summary = results["summary"]
        chi2    = results["chi2"]
        dof     = results["dof"]
        p_value = results["p_value"]
        cramers = results["cramers_v"]
        concl   = results["conclusion"]
        mw_u    = results["mann_whitney_u"]
        mw_p    = results["mann_whitney_p"]

        summary_rows = [
            (row["group"],
             f"{int(row['n_reviews']):,}",
             f"{row['mean_rating']:.3f}",
             f"{row['median_rating']:.1f}",
             f"{row['pct_5_star']:.1f}%")
            for _, row in summary.iterrows()
        ]

        prop_tab  = results["prop_tab"]
        prop_rows = [
            (idx, *[f"{v:.1f}%" for v in row.values])
            for idx, row in prop_tab.iterrows()
        ]
        prop_headers = ["Group"] + [f"{c}★" for c in prop_tab.columns]

        stats_ok = True
    except Exception as e:
        stats_ok     = False
        err_msg      = str(e)
        summary_rows = prop_rows = prop_headers = []
        chi2 = dof = p_value = cramers = mw_u = mw_p = None
        concl = err_msg

    return html.Div(
        style={"paddingTop": "8px"},
        children=[
            html.H3(
                "Hypothesis 2: Rating distribution differs by verified purchase status",
                style={"color": "#1E2736", "fontSize": "1.15rem",
                       "fontFamily": "Segoe UI, Arial, sans-serif",
                       "fontWeight": "700", "margin": "8px 0 16px"},
            ),
            research_box(
                html.P([html.Strong("H₀: "), "The distribution of star ratings is the same for "
                        "verified and non-verified purchasers"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Strong("H₁: "), "Verified and non-verified purchasers leave "
                        "systematically different rating distributions"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Em("Intuition: "), "Verified buyers have first-hand product experience, "
                        "while non-verified reviewers may express opinions without ownership — "
                        "this could shift the balance between extreme (1★ / 5★) ratings."],
                       style={"margin": "0", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#555"}),
            ),

            sub_section_heading("Rating Distribution by Verified Purchase Status"),
            one_col(h2.ecdf_chart(df)),

            sub_section_heading("Summary Statistics"),
            _summary_table(summary_rows) if stats_ok else html.Div(),

            sub_section_heading("Chi-Square Test of Independence"),
            html.Div(
                style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                       "padding": "16px 20px", "marginBottom": "16px",
                       "fontFamily": "Segoe UI, Arial, sans-serif"},
                children=[
                    html.P(
                        f"χ²({dof}) = {chi2},  p = {p_value:.4e},  Cramér's V = {cramers}"
                        if stats_ok else concl,
                        style={"fontFamily": "monospace", "fontSize": "0.92rem",
                               "color": "#34527A", "margin": "0 0 8px"},
                    ),
                    html.P(
                        f"→ {concl}" if stats_ok else "",
                        style={"fontSize": "0.90rem", "color": "#555", "margin": "0"},
                    ),
                ],
            ),

            sub_section_heading("Proportion of Ratings by Group (%)"),
            _prop_table(prop_headers, prop_rows) if stats_ok else html.Div(),

            sub_section_heading("Standardised & Adjusted Residuals"),
            two_col(h2.residual_heatmap(df), h2.adjusted_residual_heatmap(df)),

            sub_section_heading("Mann-Whitney U Test"),
            html.Div(
                style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                       "padding": "16px 20px", "marginBottom": "8px",
                       "fontFamily": "Segoe UI, Arial, sans-serif"},
                children=[
                    html.P(
                        f"U = {mw_u:,},  p = {mw_p:.4e}" if stats_ok else concl,
                        style={"fontFamily": "monospace", "fontSize": "0.92rem",
                               "color": "#34527A", "margin": "0 0 8px"},
                    ),
                    html.P(
                        "→ Confirms the chi-square result using a non-parametric rank-based test — "
                        "no distributional assumptions needed."
                        if stats_ok else "",
                        style={"fontSize": "0.90rem", "color": "#555", "margin": "0"},
                    ),
                ],
            ),

            sub_section_heading("H2 Conclusion"),
            html.Div(
                style={"backgroundColor": "#EBF4FA", "borderRadius": "6px",
                       "padding": "20px 24px", "borderLeft": "5px solid #34527A"},
                children=[
                    html.P([html.Strong("Result: "),
                            f"{concl} — verified purchase status is associated with a "
                            "meaningfully different rating distribution"],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.95rem", "color": "#1E2736", "margin": "0 0 12px"}),
                    body_text("The data supports H₁ — rating distributions differ between "
                              "verified and non-verified purchasers:"),
                    html.Ul([
                        html.Li(["The ", html.Strong("Chi-square test"),
                                 f" (χ²({dof}) = {chi2}, p = {p_value:.2e})"
                                 if stats_ok else "",
                                 " shows a statistically significant association between "
                                 "purchase verification and star rating."]),
                        html.Li(["Cramér's V = ", html.Strong(str(cramers) if stats_ok else "N/A"),
                                 " — a small but real effect size, consistent across "
                                 "a large review corpus."]),
                        html.Li(["The ", html.Strong("adjusted residuals"),
                                 " reveal where the difference is largest: non-verified reviewers "
                                 "skew toward extreme ratings (1★ and 5★) more than verified buyers."]),
                        html.Li(["The ", html.Strong("Mann-Whitney U test"),
                                 " independently confirms the distributional shift without "
                                 "assuming normality."]),
                    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.92rem",
                              "color": "#2d2d2d", "lineHeight": "1.8", "paddingLeft": "20px",
                              "margin": "0 0 12px"}),
                    html.P([html.Strong("Takeaway: "),
                            "Verified purchasers show a more moderate and product-driven "
                            "rating pattern, while non-verified reviews exhibit stronger "
                            "polarisation. Platforms and researchers should weight verified "
                            "reviews more heavily when estimating true product quality."],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.92rem", "color": "#2d2d2d", "margin": "0"}),
                ],
            ),
        ],
    )


def _summary_table(rows) -> html.Table:
    header_style = {
        "padding": "9px 14px", "textAlign": "left", "color": "white",
        "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.85rem",
        "fontWeight": "600", "backgroundColor": "#4A6188",
    }
    headers = html.Tr([
        html.Th("Group",        style=header_style),
        html.Th("N Reviews",    style=header_style),
        html.Th("Mean ★",       style=header_style),
        html.Th("Median ★",     style=header_style),
        html.Th("% Five Star",  style=header_style),
    ])
    body_rows = []
    for i, (group, n, mean, median, pct) in enumerate(rows):
        bg   = "#ffffff" if i % 2 == 0 else "#f7fafc"
        cell = {"padding": "8px 14px", "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "0.85rem", "borderBottom": "1px solid #eee", "color": "#2d2d2d"}
        body_rows.append(html.Tr([
            html.Td(group,  style=cell),
            html.Td(n,      style=cell),
            html.Td(mean,   style=cell),
            html.Td(median, style=cell),
            html.Td(pct,    style=cell),
        ], style={"backgroundColor": bg}))
    return html.Table(
        [html.Thead(headers), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "borderRadius": "4px",
               "overflow": "hidden", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
               "marginBottom": "20px"},
    )


def _prop_table(headers, rows) -> html.Table:
    header_style = {
        "padding": "9px 14px", "textAlign": "left", "color": "white",
        "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.85rem",
        "fontWeight": "600", "backgroundColor": "#4A6188",
    }
    th_row = html.Tr([html.Th(h, style=header_style) for h in headers])
    body_rows = []
    for i, row in enumerate(rows):
        bg   = "#ffffff" if i % 2 == 0 else "#f7fafc"
        cell = {"padding": "8px 14px", "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "0.85rem", "borderBottom": "1px solid #eee", "color": "#2d2d2d"}
        body_rows.append(html.Tr(
            [html.Td(str(v), style=cell) for v in row],
            style={"backgroundColor": bg},
        ))
    return html.Table(
        [html.Thead(th_row), html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "borderRadius": "4px",
               "overflow": "hidden", "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
               "marginBottom": "20px"},
    )
