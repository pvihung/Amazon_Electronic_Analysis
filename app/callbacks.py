import os

import pandas as pd
from dash import Input, Output, dcc, html

from app.layout import (
    body_text, card, insight_note, one_col, props_table,
    research_box, section_heading, sub_section_heading, two_col,
)
from eda import category, hypothesis1 as h1, hypothesis2 as h2, overview, price, ratings, text, time as eda_time

# Data source config
LOCAL_OUTPUT_CSV = os.path.join("dataset", "eda_ready.csv")
USE_LOCAL_CSV = True

_df_cache: dict = {}
_sub_tab_cache: dict = {}   # caches rendered sub-tab content so reselecting a tab is instant


def _load_data(force: bool = False) -> pd.DataFrame:
    if "df" not in _df_cache or force:
        if not os.path.exists(LOCAL_OUTPUT_CSV):
            raise FileNotFoundError(f"{LOCAL_OUTPUT_CSV} not found")

        print(f"[LOCAL] Loading from {LOCAL_OUTPUT_CSV} …")
        _df_cache["df"] = pd.read_csv(LOCAL_OUTPUT_CSV, low_memory=False)

        print(f"Loaded {len(_df_cache['df']):,} rows")

    return _df_cache["df"]


# Register callbacks

def register_callbacks(app):

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(tab):
        if tab == "tab-overview":
            return _render_overview_page()
        if tab == "tab-methods":
            return _render_methods_page()

        # Data-dependent tabs
        try:
            df = _load_data()
        except Exception as e:
            return html.Div(
                f"Could not load data: {e}",
                style={"color": "red", "padding": "20px",
                       "whiteSpace": "pre-wrap",
                       "fontFamily": "monospace"},
            )

        if tab == "tab-dataset":
            return _render_dataset_page(df)
        if tab == "tab-analytics":
            return _render_analytics_page(df)

        return html.Div("Unknown tab.")

    @app.callback(
        Output("analysis-sub-content", "children"),
        Input("analysis-sub-tabs", "value"),
    )
    def render_sub_tab(sub_tab):
        if sub_tab in _sub_tab_cache:
            return _sub_tab_cache[sub_tab]

        try:
            df = _load_data()
        except Exception as e:
            return html.Div(f"Could not load data: {e}",
                            style={"color": "red", "fontFamily": "monospace", "padding": "20px"})

        if sub_tab == "sub-eda":
            content = _render_eda_tab(df)
        elif sub_tab == "sub-hyp1":
            content = _render_hypothesis1_tab(df)
        elif sub_tab == "sub-hyp2":
            content = _render_hypothesis2_tab(df)
        elif sub_tab == "sub-hyp3":
            content = _render_hypothesis_placeholder("Hypothesis 3")
        else:
            content = html.Div()

        _sub_tab_cache[sub_tab] = content
        return content


# Tab page renderers

def _render_overview_page():
    return html.Div([
        card(
            section_heading("Project Overview"),
            body_text(
                "Amazon is one of the world's largest marketplaces for digital devices. "
                "With millions of customer reviews spanning laptops, tablets, and desktops, "
                "this dataset offers a rich opportunity to understand how consumers evaluate "
                "technology products - from star ratings and pricing patterns to the language "
                "they use when expressing satisfaction or disappointment."
            ),
            body_text(
                "This project collects and cleans Amazon product review data, applies a "
                "machine-learning classifier to filter for genuine digital device reviews, "
                "and then performs a comprehensive exploratory data analysis (EDA). "
                "We examine rating distributions, price-tier effects, seasonal trends, "
                "review text length, VADER sentiment scores, brand performance, and "
                "feature correlations."
            ),
            # for my teammate: you can add more questioons here if you wants
            research_box(
                html.H4("Research Question",
                        style={"color": "#1E2736", "margin": "0 0 8px",
                               "fontFamily": "Segoe UI, Arial, sans-serif",
                               "fontSize": "0.95rem"}),
                html.Ul(
                    [
                        html.Li("What factors do device category, price tier, brand, or review sentiment, best predict whether a customer leaves a high (≥ 4★) or low (≤ 2★) rating?"),
                        html.Li("How have review volumes and spending patterns shifted over time?"),
                    ],
                    style={"color": "#2d2d2d", "fontSize": "0.95rem",
                           "fontFamily": "Segoe UI, Arial, sans-serif",
                           "lineHeight": "1.75", "margin": "0", "paddingLeft": "20px"},
                ),
            ),
        ),
        card(
            section_heading("Key Highlights"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                       "gap": "20px"},
                children=[
                    _stat_card("600K+",  "Reviews collected"),
                    _stat_card("3",      "Device categories\n(Laptop · Tablet · Desktop)"),
                    _stat_card("2018 - 2023", "Review time span"),
                ],
            ),
        ),
    ])


def _render_dataset_page(df: pd.DataFrame):
    n_rows    = f"{len(df):,}"
    n_brands  = f"{df['brand'].nunique():,}" if "brand" in df.columns else "—"
    n_products = f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—"
    years     = f"{int(df['year'].min())} - {int(df['year'].max())}" if "year" in df.columns else "—"

    return html.Div([
        card(
            section_heading("Dataset"),
            body_text(
                "Dataset is open source from Hugging Face: https://amazon-reviews-2023.github.io. "
                "Each row represents one customer review "
                "and is linked to its product's metadata - including listed price, brand, "
                "and category. Reviews are filtered with an ML classifier and further "
                "enriched with computed features: price tier, VADER sentiment score, "
                "review language, and temporal attributes."
            ),
            html.Div(style={"marginTop": "20px"},
                     children=[props_table([
                         ("Total reviews (EDA-ready)",   n_rows),
                         ("Unique products",              n_products),
                         ("Unique brands",                n_brands),
                         ("Review period",                years),
                         ("Device categories",            "Laptop, Tablet, Desktop"),
                         ("Rating scale",                 "1 - 5 stars"),
                         ("Source",                       "Amazon Product Reviews (public)"),
                         ("Language handling",            "Non-English reviews translated via Google Translate"),
                     ])]),
        ),
        card(
            section_heading("Data Summary"),
            one_col(overview.summary_table(df)),
            one_col(overview.describe_table(df)),
        ),
    ])


def _render_methods_page():
    def step_card(num, color, title, description):
        return html.Div(
            style={
                "display": "flex", "gap": "20px", "alignItems": "flex-start",
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "padding": "24px", "marginBottom": "16px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                "borderLeft": f"5px solid {color}",
            },
            children=[
                html.Div(str(num), style={
                    "backgroundColor": color, "color": "white",
                    "borderRadius": "50%", "width": "36px", "height": "36px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontWeight": "700", "fontSize": "1rem", "flexShrink": "0",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                }),
                html.Div([
                    html.H4(title, style={"margin": "0 0 8px", "color": "#1E2736",
                                          "fontFamily": "Segoe UI, Arial, sans-serif",
                                          "fontSize": "1rem"}),
                    body_text(description, margin="0"),
                ]),
            ],
        )

    return html.Div([
        card(
            section_heading("Methods"),
            body_text(
                "The data pipeline consists of three sequential steps that transform "
                "raw Amazon review exports into a clean, analysis-ready table."
            ),
        ),
        step_card(1, "#7FAAC4", "Step 1 — ML Filter",
                  "A scikit-learn text classifier is trained on product titles and metadata "
                  "to distinguish genuine digital devices (laptops, tablets, desktops) from "
                  "unrelated products that may appear in the same broad category. "
                  "This step ensures downstream analyses are not polluted by accessories, "
                  "cases, or unrelated electronics."),
        step_card(2, "#1E2736", "Step 2 — BigQuery SQL",
                  "Filtered products are joined with their full review histories using "
                  "BigQuery SQL. Duplicate reviews are removed, and the reviews table is "
                  "linked to product metadata (brand, price, category). "
                  "The output is a single denormalized table ready for EDA."),
        step_card(3, "#34527A", "Step 3 — EDA Data Preparation",
                  "The merged table is cleaned and enriched: prices are bucketed into "
                  "Low / Medium / High / Premium tiers using quantiles; VADER sentiment "
                  "scores are computed for each review text; non-English reviews are "
                  "detected with Lingua and translated via Google Translate; "
                  "temporal features (year, month, day-of-week) are extracted; "
                  "and a price_missing flag is added for out-of-stock products. "
                  "The result is saved as dataset/eda_ready.csv and uploaded to BigQuery."),
    ])


def _render_analytics_page(_df: pd.DataFrame):
    # Sub-tab children are intentionally empty here — content is injected lazily
    # by the render_sub_tab callback below, so only the selected tab is computed.
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
                    value="sub-eda",
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
                    children=html.Div(id="analysis-sub-content"),
                ),
            ],
        ),
    ])


def _render_eda_tab(df: pd.DataFrame) -> html.Div:
    return html.Div(
        style={"paddingTop": "8px"},
        children=[
            # Rating Analysis
            sub_section_heading("⭐  Rating Analysis"),
            one_col(ratings.rating_distribution(df)),
            insight_note(
                "Even though individual ratings usually track average_rating closely, "
                "there is a noticeable left tail (negative deltas) — worth investigating further."
            ),

            # Price Analysis
            sub_section_heading("💰  Price Analysis"),
            insight_note(
                "185K+ reviews belong to products with no listed price "
                "(out-of-stock / discontinued). A price_missing signal handles these separately."
            ),
            two_col(price.price_distribution(df), price.rating_by_price_tier(df)),
            insight_note(
                "No strong differences in rating distribution across price tiers — "
                "price tier alone is not a great predictor of customer satisfaction."
            ),

            # Time Analysis
            sub_section_heading("📅  Time Analysis"),
            two_col(eda_time.reviews_by_year(df), eda_time.reviews_by_month(df)),
            insight_note(
                "A big volume shift started in 2020 (COVID). "
                "Data only runs through Sep 2023, so that year looks low — "
                "compare 2018 - 2022 for fair year-over-year analysis. "
                "December and January consistently peak; September onward loses volume in the dataset."
            ),

            # Text & Sentiment
            sub_section_heading("📝  Text & Sentiment"),
            one_col(text.review_length_histogram(df)),
            insight_note(
                "Most reviews are short (< 300 characters) and the distribution is "
                "right-skewed. A small number of reviews are very long (> 2,000 characters). "
                "1-star reviews tend to be longer; 5-star reviews are shorter on average."
            ),
            two_col(text.review_length_by_rating(df), text.vader_by_rating(df)),
            insight_note(
                "VADER generally tracks star ratings correctly. "
                "Some overlap between 4★ and 5★ — very enthusiastic language can confuse "
                "the model, or reviewers may have given a high star rating despite negative text."
            ),

            # Category & Brand
            sub_section_heading("🖥️  Category & Brand"),
            two_col(category.category_distribution(df), category.top_brands_bar(df)),
            insight_note(
                "Laptops dominate the review volume. "
                "Rating distributions are broadly similar across categories, "
                "with slight variation in median scores by brand."
            ),
        ],
    )


def _render_hypothesis1_tab(df: pd.DataFrame) -> html.Div:
    # Compute statistics 
    try:
        from scipy import stats as sp
        wp = df[(df["price_missing"] == 0) & (df["price"] > 0)].copy()
        paired = wp[["price", "rating"]].dropna()
        rho_p, _ = sp.pearsonr(paired["price"], paired["rating"])
        rho_s, _ = sp.spearmanr(paired["price"], paired["rating"])
        is_nonlinear = abs(rho_s) > abs(rho_p) * 1.1

        wp["price_quartile"] = pd.qcut(
            wp["price"], q=4,
            labels=["Q1 (Budget)", "Q2 (Low-Mid)", "Q3 (High-Mid)", "Q4 (Premium)"],
        )
        groups   = [g["rating"].values for _, g in wp.groupby("price_quartile", observed=True)]
        kw_stat, kw_p = sp.kruskal(*groups)
        n_total  = sum(len(g) for g in groups)
        eta_sq   = (kw_stat - len(groups) + 1) / (n_total - len(groups))

        wp["is_negative"] = (wp["rating"] <= 3).astype(int)
        summary = wp.groupby("price_quartile", observed=True).agg(
            median_rating=("rating", "median"),
            mean_rating=("rating", "mean"),
            pct_negative=("is_negative", "mean"),
            n=("rating", "count"),
        ).round(3)
        summary["pct_negative"] = (summary["pct_negative"] * 100).round(1)

        BIN_SIZE  = 50
        max_price = min(wp["price"].quantile(0.97), 3000)
        bins      = list(range(0, int(max_price) + BIN_SIZE, BIN_SIZE))
        wp["price_bin"] = pd.cut(wp["price"], bins=bins)
        wp["price_mid"] = wp["price_bin"].apply(lambda x: x.mid if pd.notna(x) else None)
        binned = (
            wp.groupby("price_mid", observed=True)["rating"]
            .agg(["mean", "count"]).reset_index()
            .rename(columns={"price_mid": "price", "mean": "avg_rating"})
        )
        binned = binned[binned["count"] >= 30]
        hi_bin = binned.loc[binned["avg_rating"].idxmax()]
        lo_bin = binned.loc[binned["avg_rating"].idxmin()]

        corr_text   = f"Pearson r = {rho_p:.4f}   |   Spearman ρ = {rho_s:.4f}"
        nonlin_text = ("→ Spearman > Pearson: relationship is monotonic but NON-LINEAR — "
                       "supports the existence of breakpoints / plateaus."
                       if is_nonlinear else
                       "→ Pearson ≈ Spearman: relationship is approximately linear.")
        kw_text     = f"H = {kw_stat:.2f},  p = {kw_p:.4e},  η² = {eta_sq:.4f}"
        bin_rows    = [
            ("Price range covered", f"${binned['price'].min():.0f} – ${binned['price'].max():.0f}"),
            ("Bins with ≥ 30 reviews", str(len(binned))),
            (f"Highest avg rating bin", f"${hi_bin['price']:.0f}  ({hi_bin['avg_rating']:.3f} ★)"),
            (f"Lowest avg rating bin",  f"${lo_bin['price']:.0f}  ({lo_bin['avg_rating']:.3f} ★)"),
        ]
        quartile_rows = [
            (idx,
             f"{row['median_rating']:.1f}",
             f"{row['mean_rating']:.3f}",
             f"{row['pct_negative']}%",
             f"{int(row['n']):,}")
            for idx, row in summary.iterrows()
        ]
        stats_ok = True
    except Exception as e:
        stats_ok    = False
        corr_text   = kw_text = nonlin_text = str(e)
        bin_rows    = []
        quartile_rows = []

    # Quartile summary table
    def _quartile_table(rows):
        header_style = {"padding": "9px 14px", "textAlign": "left", "color": "white",
                        "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.85rem",
                        "fontWeight": "600", "backgroundColor": "#4A6188"}
        headers = html.Tr([
            html.Th("Price Quartile", style=header_style),
            html.Th("Median ★",       style=header_style),
            html.Th("Mean ★",         style=header_style),
            html.Th("% Negative",     style=header_style),
            html.Th("N",              style=header_style),
        ])
        body_rows = []
        for i, (quartile, med, mean, pct_neg, n) in enumerate(rows):
            bg = "#ffffff" if i % 2 == 0 else "#f7fafc"
            cell = {"padding": "8px 14px", "fontFamily": "Segoe UI, Arial, sans-serif",
                    "fontSize": "0.85rem", "borderBottom": "1px solid #eee", "color": "#2d2d2d"}
            body_rows.append(html.Tr([
                html.Td(str(quartile), style=cell),
                html.Td(med,   style=cell),
                html.Td(mean,  style=cell),
                html.Td(pct_neg, style=cell),
                html.Td(n,     style=cell),
            ], style={"backgroundColor": bg}))
        return html.Table(
            [html.Thead(headers), html.Tbody(body_rows)],
            style={"width": "100%", "borderCollapse": "collapse",
                   "borderRadius": "4px", "overflow": "hidden",
                   "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "marginTop": "12px"},
        )

    # Stat callout box
    def _stat_box(label, value):
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

    return html.Div(
        style={"paddingTop": "8px"},
        children=[

            # ── Header ───────────────────────────────────────────────────────
            html.H3("Hypothesis 1: There are distinct price breakpoints where satisfaction "
                    "meaningfully shifts",
                    style={"color": "#1E2736", "fontSize": "1.15rem",
                           "fontFamily": "Segoe UI, Arial, sans-serif",
                           "fontWeight": "700", "margin": "8px 0 16px"}),
            research_box(
                html.P([html.Strong("H₀: "), "Customer satisfaction (rating) changes linearly "
                        "and continuously with price"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Strong("H₁: "), "There are specific price thresholds where average "
                        "rating jumps or plateaus — indicating diminishing returns zones"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Em("Intuition: "), "Consumers experience step-changes in satisfaction "
                        "at key price points (e.g. budget → mid-range), not a smooth gradient. "
                        "Above a certain price, paying more buys no additional satisfaction."],
                       style={"margin": "0", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#555"}),
            ),

            # Breakpoint chart
            sub_section_heading("Price Breakpoint Analysis"),
            one_col(h1.price_breakpoint_chart(df)),

            # Bin stats row
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                       "gap": "12px", "margin": "4px 0 24px"},
                children=[_stat_box(lbl, val) for lbl, val in bin_rows],
            ) if stats_ok else html.Div(),

            # Correlation results
            sub_section_heading("Correlation Analysis"),
            html.Div(
                style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                       "padding": "16px 20px", "marginBottom": "16px",
                       "fontFamily": "Segoe UI, Arial, sans-serif"},
                children=[
                    html.P("Price vs Rating — correlation comparison:",
                           style={"fontWeight": "600", "margin": "0 0 8px",
                                  "color": "#1E2736", "fontSize": "0.92rem"}),
                    html.P(corr_text,
                           style={"fontFamily": "monospace", "fontSize": "0.92rem",
                                  "color": "#34527A", "margin": "0 0 8px"}),
                    html.P(nonlin_text,
                           style={"fontSize": "0.90rem", "color": "#555", "margin": "0"}),
                ],
            ),

            # Kruskal-Wallis
            sub_section_heading("Kruskal-Wallis Test across Price Quartiles"),
            html.Div(
                style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                       "padding": "16px 20px", "marginBottom": "8px",
                       "fontFamily": "Segoe UI, Arial, sans-serif"},
                children=[
                    html.P(kw_text,
                           style={"fontFamily": "monospace", "fontSize": "0.92rem",
                                  "color": "#34527A", "margin": "0 0 12px"}),
                    _quartile_table(quartile_rows) if stats_ok else html.Div(),
                ],
            ),

            # Category breakdown
            sub_section_heading("Breakpoints by Device Category"),
            one_col(h1.rating_by_category_breakpoint(df)),

            # Conclusion
            sub_section_heading("H1 Conclusion"),
            html.Div(
                style={"backgroundColor": "#EBF4FA", "borderRadius": "6px",
                       "padding": "20px 24px", "borderLeft": "5px solid #34527A"},
                children=[
                    html.P([html.Strong("Result: "),
                            "Supported (non-linear relationship with statistically significant "
                            "price breakpoints)"],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.95rem", "color": "#1E2736", "margin": "0 0 12px"}),
                    body_text("The data supports H₁ — satisfaction does not change linearly with price:"),
                    html.Ul([
                        html.Li(["The Spearman ρ exceeds Pearson r, indicating the relationship is ",
                                 html.Strong("monotonic but non-linear"),
                                 " — consistent with price breakpoints."]),
                        html.Li([html.Strong("Budget (Q1) products have the highest satisfaction"),
                                 " and satisfaction dips in the mid-range before partially recovering "
                                 "at premium tiers."]),
                        html.Li(["The ", html.Strong("Kruskal-Wallis test"),
                                 " confirms significant differences across price quartiles — each "
                                 "price tier represents a genuine shift in satisfaction."]),
                        html.Li(["The highest-rated bin is around $25 (avg 4.52 ★), while the "
                                 "lowest-rated bin is around $475 (avg 3.70 ★)."]),
                    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.92rem",
                              "color": "#2d2d2d", "lineHeight": "1.8", "paddingLeft": "20px",
                              "margin": "0 0 12px"}),
                    html.P([html.Strong("Takeaway: "),
                            "There is a clear mid-price dissatisfaction dip — budget products often "
                            "meet low expectations well, while expensive mid-range products face "
                            "heightened scrutiny. This supports diminishing-returns and "
                            "expectation-calibration dynamics. The hypothesis is confirmed."],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif",
                                  "fontSize": "0.92rem", "color": "#2d2d2d", "margin": "0"}),
                ],
            ),
        ],
    )


def _render_hypothesis2_tab(df: pd.DataFrame) -> html.Div:

    # Compute statistics 
    try:
        from scipy import stats as sp
        from itertools import combinations

        PERIOD_ORDER = ["pre-COVID", "during-COVID", "post-COVID"]
        dc = df[df["covid_period"].isin(PERIOD_ORDER)].copy()

        # Period counts + mean ratings
        period_counts  = dc.groupby("covid_period").size().reindex(PERIOD_ORDER)
        period_ratings = dc.groupby("covid_period")["rating"].agg(["mean", "median", "std"]).reindex(PERIOD_ORDER).round(3)

        # Vader sentiment
        has_vader = "vader_sentiment" in dc.columns
        if has_vader:
            period_vader = dc.groupby("covid_period")["vader_sentiment"].mean().reindex(PERIOD_ORDER).round(4)

        # Kruskal-Wallis on ratings
        groups_covid = [dc[dc["covid_period"] == p]["rating"].values for p in PERIOD_ORDER]
        kw_stat, kw_p = sp.kruskal(*groups_covid)
        n_total = sum(len(g) for g in groups_covid)
        eta_sq  = (kw_stat - 3 + 1) / (n_total - 3)

        # Pairwise Mann-Whitney (Bonferroni α = 0.05/3 = 0.017)
        mw_rows = []
        for p1, p2 in combinations(PERIOD_ORDER, 2):
            g1 = dc[dc["covid_period"] == p1]["rating"].values
            g2 = dc[dc["covid_period"] == p2]["rating"].values
            s, p_val = sp.mannwhitneyu(g1, g2, alternative="two-sided")
            r_eff = abs(1 - (2 * s) / (len(g1) * len(g2)))
            sig = "✓" if p_val < 0.017 else "✗"
            mw_rows.append((f"{p1} vs {p2}", f"{p_val:.4e}", f"{r_eff:.4f}", sig))

        # Compute keyword matches once — reused for chi-square, mention-rate, and lift charts
        kw_matches = h2.compute_keyword_matches(dc)

        # Chi-square per keyword
        chi_rows = []
        for kw in h2.COVID_KEYWORDS:
            col = kw_matches[kw].astype(int)
            dc["_kw"] = col
            ct = pd.crosstab(dc["covid_period"], dc["_kw"]).reindex(PERIOD_ORDER)
            for c in [0, 1]:
                if c not in ct.columns:
                    ct[c] = 0
            chi2_val, p_val, _, _ = sp.chi2_contingency(ct[[0, 1]])
            n   = ct.values.sum()
            cramv = (chi2_val / (n * (min(ct.shape) - 1))) ** 0.5
            sig = "✓" if p_val < 0.001 else "✗"
            chi_rows.append((kw, f"{chi2_val:.1f}", f"{p_val:.4e}", f"{cramv:.4f}", sig))
        dc.drop(columns=["_kw"], inplace=True, errors="ignore")
        chi_rows.sort(key=lambda x: float(x[1]), reverse=True)

        # Category share
        chi2_cat, p_cat, _, _ = sp.chi2_contingency(
            pd.crosstab(dc["covid_period"], dc["category"]).reindex(PERIOD_ORDER)
        )
        n_cat   = len(dc)
        cramv_cat = (chi2_cat / (n_cat * (min(len(PERIOD_ORDER), dc["category"].nunique()) - 1))) ** 0.5

        stats_ok = True
    except Exception:
        stats_ok = False
        mw_rows = chi_rows = []
        kw_stat = kw_p = eta_sq = 0.0
        period_counts = period_ratings = None
        chi2_cat = p_cat = cramv_cat = 0.0
        kw_matches = None

    # Helper: small mono results box
    def _result_box(*lines):
        return html.Div(
            style={"backgroundColor": "#f7fafc", "borderRadius": "6px",
                   "padding": "14px 20px", "marginBottom": "16px",
                   "fontFamily": "monospace", "fontSize": "0.88rem", "color": "#34527A"},
            children=[html.P(line, style={"margin": "2px 0"}) for line in lines],
        )

    # Helper: generic styled table
    def _styled_table(headers, rows):
        h_style = {"padding": "9px 14px", "textAlign": "left", "color": "white",
                   "fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.83rem",
                   "fontWeight": "600", "backgroundColor": "#4A6188"}
        header_row = html.Tr([html.Th(h, style=h_style) for h in headers])
        body_rows  = []
        for i, row in enumerate(rows):
            bg = "#ffffff" if i % 2 == 0 else "#f7fafc"
            c_style = {"padding": "7px 14px", "fontFamily": "Segoe UI, Arial, sans-serif",
                       "fontSize": "0.83rem", "borderBottom": "1px solid #eee", "color": "#2d2d2d"}
            body_rows.append(html.Tr(
                [html.Td(cell, style=c_style) for cell in row],
                style={"backgroundColor": bg},
            ))
        return html.Table(
            [html.Thead(header_row), html.Tbody(body_rows)],
            style={"width": "100%", "borderCollapse": "collapse", "marginTop": "10px",
                   "boxShadow": "0 1px 3px rgba(0,0,0,0.08)"},
        )

    # Period overview stats rows 
    overview_rows = []
    if stats_ok:
        for p in ["pre-COVID", "during-COVID", "post-COVID"]:
            vader_val = f"{period_vader[p]:.4f}" if has_vader else "—"
            overview_rows.append((
                p,
                f"{period_counts[p]:,}",
                f"{period_ratings.loc[p, 'mean']:.3f}",
                f"{period_ratings.loc[p, 'median']:.1f}",
                f"{period_ratings.loc[p, 'std']:.3f}",
                vader_val,
            ))

    return html.Div(
        style={"paddingTop": "8px"},
        children=[

            # Header
            html.H3("Hypothesis 2: COVID-19 shifted consumer priorities for digital devices",
                    style={"color": "#1E2736", "fontSize": "1.15rem",
                           "fontFamily": "Segoe UI, Arial, sans-serif",
                           "fontWeight": "700", "margin": "8px 0 16px"}),
            research_box(
                html.P([html.Strong("H₀: "), "Review keyword frequency, ratings, and sentiment "
                        "are stable across pre/during/post-COVID periods"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Strong("H₁: "), "During COVID, reviews show significantly higher "
                        "mention of WFH-related features (webcam, video calls, battery) and "
                        "ratings/sentiment shift measurably"],
                       style={"margin": "0 0 6px", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#2d2d2d"}),
                html.P([html.Em("Intuition: "), "The shift to remote work created new device "
                        "requirements. Consumers buying laptops/tablets during COVID prioritized "
                        "different attributes than before."],
                       style={"margin": "0", "fontSize": "0.92rem",
                              "fontFamily": "Segoe UI, Arial, sans-serif", "color": "#555"}),
            ),

            # 2a: Period overview 
            sub_section_heading("2a. Period Overview"),
            one_col(h2.period_overview_chart(df)),
            _styled_table(
                ["Period", "Reviews", "Mean ★", "Median ★", "Std", "VADER Sentiment"],
                overview_rows,
            ) if stats_ok else html.Div(),

            # 2b: Keyword charts
            sub_section_heading("2b. COVID Keyword Frequency Shift"),
            two_col(h2.keyword_mention_rate_chart(df, kw_matches), h2.keyword_lift_chart(df, kw_matches)),

            # 2c: Chi-square per keyword
            sub_section_heading("2c. Chi-Square Tests per Keyword (Period Independence)"),
            _styled_table(
                ["Keyword", "χ²", "p-value", "Cramér V", "Significant?"],
                chi_rows,
            ) if stats_ok else html.Div(),

            # 2d: Rating & sentiment shift
            sub_section_heading("2d. Ratings & Sentiment Shift across COVID Periods"),
            _result_box(
                f"Kruskal-Wallis on ratings:  H = {kw_stat:.2f},  "
                f"p = {kw_p:.4e},  η² = {eta_sq:.6f}",
            ) if stats_ok else html.Div(),
            html.P("Pairwise Mann-Whitney (Bonferroni α = 0.05/3 = 0.017):",
                   style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.88rem",
                          "fontWeight": "600", "color": "#1E2736", "margin": "0 0 6px"}),
            _styled_table(
                ["Comparison", "p-value", "Effect size r", "Significant?"],
                mw_rows,
            ) if stats_ok else html.Div(),

            #  2e: Category share shift
            sub_section_heading("2e. Device Category Share Shift"),
            one_col(h2.category_share_chart(df)),
            _result_box(
                f"Chi-square (category distribution shift):  "
                f"χ² = {chi2_cat:.1f},  p = {p_cat:.4e},  Cramér V = {cramv_cat:.4f}",
            ) if stats_ok else html.Div(),

            # Conclusion 
            sub_section_heading("H2 Conclusion"),
            html.Div(
                style={"backgroundColor": "#EBF4FA", "borderRadius": "6px",
                       "padding": "20px 24px", "borderLeft": "5px solid #34527A"},
                children=[
                    html.P([html.Strong("Result: "),
                            "Supported (statistically significant COVID-era shifts in review "
                            "content and sentiment)"],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.95rem",
                                  "color": "#1E2736", "margin": "0 0 12px"}),
                    body_text("The data strongly supports H₁ — COVID measurably shifted what "
                              "consumers prioritized in device reviews:"),
                    html.Ul([
                        html.Li([html.Strong("WFH keyword surge: "), "'work from home' mentions "
                                 "spiked +370% during COVID vs. pre-COVID, 'zoom/video call' rose "
                                 "+469%, and 'webcam' nearly doubled (+118%). All shifts are highly "
                                 "significant (chi-square tests, all p < 0.001)."]),
                        html.Li([html.Strong("Post-COVID persistence: "), "'zoom/video call' and "
                                 "'display/screen' remain elevated post-COVID, suggesting lasting "
                                 "behavioral change beyond just a temporary spike."]),
                        html.Li([html.Strong("Ratings declined post-COVID: "), "Mean rating dropped "
                                 "from 3.833 (during-COVID) to 3.752 (post-COVID), and VADER "
                                 "sentiment followed (0.395 → 0.370), suggesting growing consumer "
                                 "frustration — possibly due to supply chain issues or unmet "
                                 "elevated expectations."]),
                        html.Li([html.Strong("Category mix shifted: "), "Desktop share grew "
                                 "(23.3% → 29.5%) and tablets gained (6.0% → 8.7%) at the expense "
                                 "of laptops (70.7% → 61.8%), reflecting evolving device preferences "
                                 "(χ² = 2664.2, Cramér V = 0.060)."]),
                    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.92rem",
                              "color": "#2d2d2d", "lineHeight": "1.8", "paddingLeft": "20px",
                              "margin": "0 0 12px"}),
                    html.P([html.Strong("Takeaway: "), "COVID was a genuine inflection point for "
                            "digital device consumer behavior. WFH-related language surged "
                            "significantly during the pandemic and remained elevated post-COVID, "
                            "while overall sentiment softened. The hypothesis is confirmed with "
                            "both statistical strength and real-world interpretability."],
                           style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "0.92rem",
                                  "color": "#2d2d2d", "margin": "0"}),
                ],
            ),
        ],
    )


def _render_hypothesis_placeholder(title: str) -> html.Div:
    return html.Div(
        style={"padding": "40px 0", "textAlign": "center"},
        children=[
            html.P(
                f"{title} analysis coming soon.",
                style={
                    "color": "#888",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                    "fontSize": "0.95rem",
                },
            ),
        ],
    )


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


# Private helpers

CARD_BG = "#ffffff"


def _stat_card(value: str, label: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#EBF4FA",
            "borderRadius": "6px",
            "padding": "20px 24px",
            "textAlign": "center",
            "borderTop": "3px solid #34527A",
        },
        children=[
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": "700",
                                   "color": "#34527A", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
            html.Div(label, style={"fontSize": "0.82rem", "color": "#555",
                                   "marginTop": "6px", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
        ],
    )
