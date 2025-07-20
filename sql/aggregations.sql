-- Generate daily aggregations
INSERT INTO sales_summary (
    date_period, region, product_category, 
    total_sales, total_quantity, order_count, 
    avg_order_value, period_type
)
SELECT 
    order_date as date_period,
    region,
    product_category,
    SUM(sales_amount) as total_sales,
    SUM(quantity) as total_quantity,
    COUNT(*) as order_count,
    AVG(sales_amount) as avg_order_value,
    'daily' as period_type
FROM cleaned_sales
GROUP BY order_date, region, product_category
ON DUPLICATE KEY UPDATE
    total_sales = VALUES(total_sales),
    total_quantity = VALUES(total_quantity),
    order_count = VALUES(order_count),
    avg_order_value = VALUES(avg_order_value);

-- Generate monthly aggregations
INSERT INTO sales_summary (
    date_period, region, product_category, 
    total_sales, total_quantity, order_count, 
    avg_order_value, period_type
)
SELECT 
    DATE(CONCAT(year_month, '-01')) as date_period,
    region,
    product_category,
    SUM(sales_amount) as total_sales,
    SUM(quantity) as total_quantity,
    COUNT(*) as order_count,
    AVG(sales_amount) as avg_order_value,
    'monthly' as period_type
FROM cleaned_sales
GROUP BY year_month, region, product_category
ON DUPLICATE KEY UPDATE
    total_sales = VALUES(total_sales),
    total_quantity = VALUES(total_quantity),
    order_count = VALUES(order_count),
    avg_order_value = VALUES(avg_order_value);