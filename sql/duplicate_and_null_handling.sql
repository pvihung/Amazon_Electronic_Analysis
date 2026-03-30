-- handle duplicate
---Check if there is duplicated
SELECT parent_asin, review_title, review_text, COUNT(*) AS cnt
FROM `cs163-project-487801.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates`
GROUP BY parent_asin, review_title, review_text
HAVING COUNT(*) > 1
ORDER BY cnt DESC;
---remove duplicate
CREATE OR REPLACE TABLE 
`cs163-project-487801.amazon_digital_devices_cleaned.digital_devices_reviews_no_duplicates` AS

SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY parent_asin, review_title, review_text
         ) AS rn
  FROM `cs163-project-487801.amazon_digital_devices_cleaned.digital_devices_reviews`
)
WHERE rn = 1;

--handle missing (null) values:
