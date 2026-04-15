"""
Run locally:
    python -m app.main
"""

import os
import dash
from app.layout import build_layout
from app.callbacks import register_callbacks

# App init
app = dash.Dash(
    __name__,
    title="Amazon Digital Devices EDA",
    suppress_callback_exceptions=True,   # tabs render content lazily
)
server = app.server  # expose Flask server for GAE / gunicorn

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>body, html { margin: 0; padding: 0; }</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = build_layout()
register_callbacks(app)

# Local dev server 
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    app.run(host="127.0.0.1", port=port, debug=debug)
