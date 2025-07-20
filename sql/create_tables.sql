-- Create database schema for sales forecasting project
CREATE DATABASE IF NOT EXISTS sales_db;
USE sales_db;

-- Raw sales data table
CREATE TABLE IF NOT EXISTS raw_sales (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    order_date VARCHAR(50),
    product_category VARCHAR(100),
    product_name VARCHAR(200),
    region VARCHAR(50),
    sales_amount DECIMAL(10,2),
    quantity INTEGER,
    customer_id VARCHAR(50),
    sales_rep VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cleaned sales data table
CREATE TABLE IF NOT EXISTS cleaned_sales (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    order_date DATE NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    region VARCHAR(50) NOT NULL,
    sales_amount DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    customer_id VARCHAR(50),
    sales_rep VARCHAR(100),
    year_month VARCHAR(7), -- YYYY-MM format
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (order_date),
    INDEX idx_region (region),
    INDEX idx_category (product_category)
);

-- Aggregated sales summary table
CREATE TABLE IF NOT EXISTS sales_summary (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    date_period DATE,
    region VARCHAR(50),
    product_category VARCHAR(100),
    total_sales DECIMAL(15,2),
    total_quantity INTEGER,
    order_count INTEGER,
    avg_order_value DECIMAL(10,2),
    period_type ENUM('daily', 'monthly') DEFAULT 'daily',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_summary (date_period, region, product_category, period_type)
);

-- Forecasting results table
CREATE TABLE IF NOT EXISTS forecast_results (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    forecast_date DATE,
    region VARCHAR(50),
    product_category VARCHAR(100),
    predicted_sales DECIMAL(15,2),
    lower_bound DECIMAL(15,2),
    upper_bound DECIMAL(15,2),
    model_name VARCHAR(50),
    accuracy_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_forecast_date (forecast_date),
    INDEX idx_region_category (region, product_category)
);