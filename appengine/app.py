from dash import Dash, html, dash_table
import pandas as pd

from google.cloud import storage
import os
from io import StringIO


def get_csv_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    text = blob.download_as_text()
    return pd.read_csv(StringIO(text))


app = Dash(__name__)
server = app.server

BUCKET_NAME = os.environ.get("BUCKET_NAME", "amazon-electronics-data")

BLOB_NAME = "digital_devices_reviews_no_duplicates"

try:
    df = get_csv_from_gcs(BUCKET_NAME, BLOB_NAME)
    status = f"OK: Loaded {len(df)} rows from gs://{BUCKET_NAME}/{BLOB_NAME}"
except Exception as e:
    df = pd.DataFrame({"error": [repr(e)]})
    status = f"FAILED to load gs://{BUCKET_NAME}/{BLOB_NAME}\n{repr(e)}"

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "24px auto", "fontFamily": "Arial"},
    children=[
        html.H3(""),
        html.Pre(status),
        dash_table.DataTable(
            data=df.head(200).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "120px", "maxWidth": "320px", "whiteSpace": "normal"},
        ),
    ],
)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=8080)