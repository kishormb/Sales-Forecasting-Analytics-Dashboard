import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class SalesUtils:
    def __init__(self):
        self.engine = create_engine("sqlite:///sales_database.db")
    
    def get_kpis(self):
        """Calculate key performance indicators"""
        query = """
            SELECT 
                SUM(sales_amount) as total_sales,
                COUNT(*) as total_orders,
                AVG(sales_amount) as avg_order_value,
                COUNT(DISTINCT customer_id) as unique_customers,
                COUNT(DISTINCT product_name) as unique_products
            FROM cleaned_sales
        """
        kpis = pd.read_sql(query, self.engine).iloc[0]
        return kpis.to_dict()
    
    def get_top_products(self, limit=10):
        """Get top selling products"""
        query = f"""
            SELECT 
                product_name,
                SUM(sales_amount) as total_sales,
                SUM(quantity) as total_quantity
            FROM cleaned_sales
            GROUP BY product_name
            ORDER BY total_sales DESC
            LIMIT {limit}
        """
        return pd.read_sql(query, self.engine)
    
    def get_regional_performance(self):
        """Get regional sales performance"""
        query = """
            SELECT 
                region,
                SUM(sales_amount) as total_sales,
                COUNT(*) as order_count,
                AVG(sales_amount) as avg_order_value
            FROM cleaned_sales
            GROUP BY region
            ORDER BY total_sales DESC
        """
        return pd.read_sql(query, self.engine)
    
    def get_time_series_data(self, granularity='daily'):
        """Get time series data for charts"""
        if granularity == 'daily':
            query = """
                SELECT 
                    order_date as date,
                    SUM(sales_amount) as sales
                FROM cleaned_sales
                GROUP BY order_date
                ORDER BY order_date
            """
        elif granularity == 'monthly':
            query = """
                SELECT 
                    year_month as date,
                    SUM(sales_amount) as sales
                FROM cleaned_sales
                GROUP BY year_month
                ORDER BY year_month
            """
        
        return pd.read_sql(query, self.engine)
    
    def create_sales_chart(self, data, title="Sales Trend"):
        """Create interactive sales chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['sales'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
            hovermode='x unified'
        )
        
        return fig

# Database utility functions
def execute_sql_file(engine, file_path):
    """Execute SQL file"""
    with open(file_path, 'r') as file:
        sql_content = file.read()
    
    # Split by semicolon and execute each statement
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    with engine.connect() as conn:
        for statement in statements:
            try:
                conn.execute(statement)
                conn.commit()
            except Exception as e:
                print(f"Error executing statement: {e}")
                print(f"Statement: {statement[:100]}...")

def backup_database(engine, backup_path):
    """Create database backup"""
    tables = ['cleaned_sales', 'sales_summary', 'forecast_results']
    
    with pd.ExcelWriter(backup_path, engine='openpyxl') as writer:
        for table in tables:
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", engine)
                df.to_excel(writer, sheet_name=table, index=False)
                print(f"Backed up table: {table}")
            except Exception as e:
                print(f"Error backing up {table}: {e}")
    
    print(f"Database backup saved to: {backup_path}")