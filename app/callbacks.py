import os

import pandas as pd
from dash import Input, Output, html

from app.pages import dataset, methods, overview
from app.pages.analytics import eda, hypothesis1, hypothesis2, hypothesis3
from app.pages.analytics import analytics as analytics_page

LOCAL_OUTPUT_CSV = os.path.join("dataset", "eda_ready.csv")
USE_LOCAL_CSV = True

_df_cache: dict = {}
_sub_tab_cache: dict = {}


def _load_data(force: bool = False) -> pd.DataFrame:
    if "df" not in _df_cache or force:
        if not os.path.exists(LOCAL_OUTPUT_CSV):
            raise FileNotFoundError(f"{LOCAL_OUTPUT_CSV} not found")

        print(f"[LOCAL] Loading from {LOCAL_OUTPUT_CSV} …")
        _df_cache["df"] = pd.read_csv(LOCAL_OUTPUT_CSV, low_memory=False)
        print(f"Loaded {len(_df_cache['df']):,} rows")

    return _df_cache["df"]


def register_callbacks(app):

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(tab):
        if tab == "tab-overview":
            return overview.render()
        if tab == "tab-methods":
            return methods.render()

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
            return dataset.render(df)
        if tab == "tab-analytics":
            return analytics_page.render(df)

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
            content = eda.render(df)
        elif sub_tab == "sub-hyp1":
            content = hypothesis1.render(df)
        elif sub_tab == "sub-hyp2":
            content = hypothesis2.render(df)
        elif sub_tab == "sub-hyp3":
            content = hypothesis3.render()
        else:
            content = html.Div()

        _sub_tab_cache[sub_tab] = content
        return content
