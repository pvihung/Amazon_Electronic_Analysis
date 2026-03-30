CREATE OR REPLACE TABLE 
`cs163-project-487801.amazon_digital_devices_cleaned.digital_devices_reviews` AS

SELECT
    r.parent_asin,
    r.rating,
    r.title AS review_title,
    r.text AS review_text,
    r.timestamp,
    r.verified_purchase,

    p.title AS product_title,
    p.price,
    p.average_rating,
    p.rating_number,
    p.main_category

FROM `cs163-project-487801.amazon_digital_devices_cleaned.reviews_100k` r

JOIN `cs163-project-487801.amazon_digital_devices_cleaned.digital_devices_only` p 
ON r.parent_asin = p.parent_asin