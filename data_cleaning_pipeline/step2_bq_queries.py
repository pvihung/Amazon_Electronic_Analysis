"""
Step 2 — BigQuery Queries: Link Reviews + Products, Remove Duplicates
Executes the two SQL files (link_reviews_products + duplicate_and_null_handling)
directly against BigQuery so the data is clean before EDA loads it.
"""

import os
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT", "cs163-project-487801")
client = bigquery.Client(project=PROJECT_ID)

# SQL queries 

LINK_REVIEWS_SQL = """
CREATE OR REPLACE TABLE
  `{project}.amazon_digital_devices_cleaned.digital_devices_reviews` AS

SELECT
    r.parent_asin,
    r.rating,
    r.title            AS review_title,
    r.text             AS review_text,
    r.timestamp,
    r.verified_purchase,

    p.title            AS product_title,
    p.price,
    p.average_rating,
    p.rating_number,
    p.main_category

FROM `{project}.amazon_digital_devices_cleaned.reviews_100k` r

JOIN `{project}.amazon_digital_devices_cleaned.digital_devices_only` p
  ON r.parent_asin = p.parent_asin
""".format(project=PROJECT_ID)


CHECK_DUPLICATES_SQL = """
SELECT parent_asin, review_title, review_text, COUNT(*) AS cnt
FROM `{project}.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates`
GROUP BY parent_asin, review_title, review_text
HAVING COUNT(*) > 1
ORDER BY cnt DESC
""".format(project=PROJECT_ID)


REMOVE_DUPLICATES_SQL = """
CREATE OR REPLACE TABLE
  `{project}.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates` AS

SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY parent_asin, review_title, review_text
         ) AS rn
  FROM `{project}.amazon_digital_devices_cleaned.digital_devices_reviews`
)
WHERE rn = 1
""".format(project=PROJECT_ID)



def run_query(label: str, sql: str, return_df: bool = False):
    """Execute a BigQuery SQL statement and optionally return results."""
    print(f"  Running: {label} …")
    job = client.query(sql)
    result = job.result()
    if return_df:
        df = result.to_dataframe()
        print(f"{len(df)} rows returned")
        return df
    print(f"Done ({job.num_dml_affected_rows or 'DDL'} rows affected)")


# Entry point 

def run() -> None:
    print("=" * 60)
    print("STEP 2 — BigQuery: Link Reviews + Deduplicate")
    print("=" * 60)

    # Join reviews with product metadata
    run_query("Link reviews ↔ products", LINK_REVIEWS_SQL)

    # Check for duplicates (informational)
    dups = run_query("Check duplicates", CHECK_DUPLICATES_SQL, return_df=True)
    if dups is not None and len(dups) > 0:
        print(f"Found {len(dups)} duplicate groups - removing …")
        run_query("Remove duplicates", REMOVE_DUPLICATES_SQL)
    else:
        print("No duplicates found.")

    print("Step 2 complete.\n")


if __name__ == "__main__":
    run()
