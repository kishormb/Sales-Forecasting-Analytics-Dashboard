-- Data cleaning and transformation procedures
USE sales_db;

-- Clean and transform raw sales data
INSERT INTO cleaned_sales (
    order_date, 
    product_category, 
    product_name, 
    region, 
    sales_amount, 
    quantity, 
    customer_id, 
    sales_rep,
    year_month,
    day_of_week,
    is_weekend
)
SELECT 
    STR_TO_DATE(order_date, '%Y-%m-%d') as order_date,
    TRIM(UPPER(product_category)) as product_category,
    TRIM(product_name) as product_name,
    TRIM(UPPER(region)) as region,
    GREATEST(COALESCE(sales_amount, 0), 0) as sales_amount,
    GREATEST(COALESCE(quantity, 0), 0) as quantity,
    TRIM(customer_id) as customer_id,
    TRIM(sales_rep) as sales_rep,
    DATE_FORMAT(STR_TO_DATE(order_date, '%Y-%m-%d'), '%Y-%m') as year_month,
    DAYOFWEEK(STR_TO_DATE(order_date, '%Y-%m-%d')) as day_of_week,
    CASE 
        WHEN DAYOFWEEK(STR_TO_DATE(order_date, '%Y-%m-%d')) IN (1, 7) 
        THEN TRUE 
        ELSE FALSE 
    END as is_weekend
FROM raw_sales
WHERE STR_TO_DATE(order_date, '%Y-%m-%d') IS NOT NULL
AND sales_amount > 0
AND quantity > 0
ON DUPLICATE KEY UPDATE
    sales_amount = VALUES(sales_amount),
    quantity = VALUES(quantity);

-- Remove duplicates based on business logic
DELETE r1 FROM cleaned_sales r1
INNER JOIN cleaned_sales r2 
WHERE r1.id > r2.id 
AND r1.order_date = r2.order_date 
AND r1.product_name = r2.product_name 
AND r1.customer_id = r2.customer_id 
AND r1.sales_amount = r2.sales_amount;