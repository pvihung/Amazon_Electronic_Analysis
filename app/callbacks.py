import os
import threading

import pandas as pd
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from app.pages import dataset, methods, overview
from app.pages.analytics import eda, hypothesis1, hypothesis2, hypothesis3
from app.pages.analytics import analytics as analytics_page


_bucket = os.environ.get("GCS_BUCKET_NAME", "cs163-amazon-review-analysis-data")
GCS_PARQUET_URL = f"https://storage.googleapis.com/{_bucket}/eda_ready.parquet"

_df_cache: dict = {}
_sub_tab_cache: dict = {}

_PATH_FROM_MAIN_TAB = {
    "tab-overview":  "/overview",
    "tab-dataset":   "/dataset",
    "tab-methods":   "/methods",
    "tab-models":    "/models",
    "tab-analytics": "/analysis/eda",
}

_PATH_FROM_SUB_TAB = {
    "sub-eda":  "/analysis/eda",
    "sub-hyp1": "/analysis/hypothesis1",
    "sub-hyp2": "/analysis/hypothesis2",
    "sub-hyp3": "/analysis/hypothesis3",
    "sub-models-data_overview": "/models/models-comparison",
    "sub-models-models-detail": "/models/models-detail",
}

_SUB_TAB_FROM_SLUG = {
    "eda":         "sub-eda",
    "hypothesis1": "sub-hyp1",
    "hypothesis2": "sub-hyp2",
    "hypothesis3": "sub-hyp3",
}

_MODELS_SUB_TAB_FROM_SLUG = {
    "models-data-overview": "sub-models-data_overview",
    "models-detail": "sub-models-models-detail",
}

def _main_tab(pathname: str) -> str:
    if not pathname or pathname in ("/", "/overview"):
        return "tab-overview"
    if pathname.startswith("/dataset"):
        return "tab-dataset"
    if pathname.startswith("/methods"):
        return "tab-methods"
    if pathname.startswith("/analysis"):
        return "tab-analytics"
    if pathname.startswith("/models"):
        return "tab-models"
    return "tab-overview"


def _sub_tab(pathname: str) -> str:
    if pathname and pathname.startswith("/analysis/"):
        slug = pathname.split("/analysis/", 1)[-1].strip("/")
        return _SUB_TAB_FROM_SLUG.get(slug, "sub-eda")
    return "sub-eda"

def _sub_tab_models(pathname: str) -> str:
    if pathname and pathname.startswith("/models/"):
        slug = pathname.split("/models/", 1)[-1].strip("/")
        return _MODELS_SUB_TAB_FROM_SLUG.get(slug, "sub-models-data_overview")
    return "sub-models-data_overview"

def _load_data() -> pd.DataFrame:
    if "df" not in _df_cache:
        print(f"Loading {GCS_PARQUET_URL} …")
        _df_cache["df"] = pd.read_parquet(GCS_PARQUET_URL)
        print(f"Loaded {len(_df_cache['df']):,} rows")
    return _df_cache["df"]


def _warm_cache():
    """Load data and pre-compute all sub-tab figures at startup."""
    try:
        df = _load_data()
    except Exception as e:
        print(f"Warmup: data load failed — {e}")
        return

    renders = [
        ("sub-eda",  lambda: eda.render(df)),
        ("sub-hyp1", lambda: hypothesis1.render(df)),
        ("sub-hyp2", lambda: hypothesis2.render(df)),
        ("sub-hyp3", lambda: hypothesis3.render(df)),
    ]
    for key, fn in renders:
        if key not in _sub_tab_cache:
            try:
                _sub_tab_cache[key] = fn()
                print(f"Warmup: cached {key}")
            except Exception as e:
                print(f"Warmup: {key} failed — {e}")


def start_background_warmup():
    threading.Thread(target=_warm_cache, daemon=True).start()


def register_callbacks(app):

    # URL → main tab (drives initial load + browser back/forward)
    @app.callback(
        Output("main-tabs", "value"),
        Input("url", "pathname"),
        State("main-tabs", "value"),
    )
    def sync_tab_from_url(pathname, current_tab):
        new_tab = _main_tab(pathname)
        if new_tab == current_tab:
            raise PreventUpdate
        return new_tab

    # Main tab click → URL
    @app.callback(
        Output("url", "pathname"),
        Input("main-tabs", "value"),
        prevent_initial_call=True,
    )
    def sync_url_from_tab(tab):
        return _PATH_FROM_MAIN_TAB.get(tab, "/")

    # Sub-tab click → URL
    # 'initial_duplicate' prevents this from firing when analysis-sub-tabs is
    # dynamically mounted by render_tab_content (prevent_initial_call=True only
    # suppresses the app-startup fire, not dynamic-mount fires).
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("analysis-sub-tabs", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_url_from_sub_tab(sub_tab):
        return _PATH_FROM_SUB_TAB.get(sub_tab, "/analysis/eda")

    # URL → sub-tab (handles browser back/forward while on analytics)
    @app.callback(
        Output("analysis-sub-tabs", "value"),
        Input("url", "pathname"),
        State("analysis-sub-tabs", "value"),
        prevent_initial_call=True,
    )
    def sync_sub_tab_from_url(pathname, current_sub_tab):
        if not pathname or not pathname.startswith("/analysis/"):
            raise PreventUpdate
        new_sub_tab = _sub_tab(pathname)
        if new_sub_tab == current_sub_tab:
            raise PreventUpdate
        return new_sub_tab

    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("models-sub-tabs", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_url_from_models_sub_tab(sub_tab):
        return _PATH_FROM_SUB_TAB.get(sub_tab, "/models/models-comparison")


    @app.callback(
        Output("models-sub-tabs", "value"),
        Input("url", "pathname"),
        State("models-sub-tabs", "value"),
        prevent_initial_call=True,
    )

    def sync_models_sub_tab_from_url(pathname, current_sub_tab):
        if not pathname or not pathname.startswith("/models/"):
            raise PreventUpdate
        new_sub_tab = _sub_tab_models(pathname)
        if new_sub_tab == current_sub_tab:
            raise PreventUpdate
        return new_sub_tab

    # Render main page content
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        State("url", "pathname"),
    )
    def render_tab_content(tab, pathname):
        if tab == "tab-overview":
            return overview.render()
        if tab == "tab-methods":
            return methods.render()
        if tab == "tab-models":
            from app.pages import models
            return models.render(_sub_tab_models(pathname or ""))

        try:
            df = _load_data()
        except Exception as e:
            return html.Div(
                f"Could not load data: {e}",
                style={"color": "red", "padding": "20px",
                       "whiteSpace": "pre-wrap", "fontFamily": "monospace"},
            )

        if tab == "tab-dataset":
            return dataset.render(df)
        if tab == "tab-analytics":
            return analytics_page.render(df, _sub_tab(pathname or ""))

        return html.Div("Unknown tab.")

    # Render analytics sub-tab content (served from cache when warmed up)
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
            content = hypothesis3.render(df)
        else:
            content = html.Div()

        _sub_tab_cache[sub_tab] = content
        return content

    @app.callback(
        Output("models-sub-content", "children"),
        Input("models-sub-tabs", "value"),

    )
    def render_models_sub_tab(sub_tab):
        if sub_tab == "sub-models-data_overview":
            from app.pages.models import data_overview
            return data_overview.render()
        elif sub_tab == "sub-models-models-detail":
            from app.pages.models import models_detail
            return models_detail.render()
        else:
            return html.Div("Unknown tab.")