import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.engine = self.create_db_connection()
        
    def create_db_connection(self):
        """Create database connection"""
        try:
            # For SQLite (local development)
            db_url = "sqlite:///sales_database.db"
            
            # For PostgreSQL (production)
            # db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            
            engine = create_engine(db_url)
            logger.info("Database connection established")
            return engine
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def generate_sample_data(self, num_records=10000):
        """Generate sample sales data for demonstration"""
        np.random.seed(42)
        
        regions = ['North', 'South', 'East', 'West', 'Central']
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        products = {
            'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Camera'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Hat'],
            'Home & Garden': ['Sofa', 'Table', 'Lamp', 'Plant', 'Curtains'],
            'Sports': ['Football', 'Basketball', 'Tennis Racket', 'Golf Club', 'Yoga Mat'],
            'Books': ['Fiction', 'Non-Fiction', 'Biography', 'Science', 'History']
        }
        
        # Generate date range for last 2 years
        start_date = datetime.now() - timedelta(days=730)
        date_range = pd.date_range(start=start_date, periods=730, freq='D')
        
        data = []
        for _ in range(num_records):
            category = np.random.choice(categories)
            product = np.random.choice(products[category])
            region = np.random.choice(regions)
            
            # Add seasonality and trends
            date = np.random.choice(date_range)
            base_price = np.random.uniform(10, 1000)
            
            # Add seasonal effects (higher sales in Q4)
           # seasonal_multiplier = 1.2 if date.month in [11, 12] else 1.0
            seasonal_multiplier = 1.2 if pd.Timestamp(date).month in [11, 12] else 1.0

            
            # Add weekend effects (lower B2B sales)
            #weekend_multiplier = 0.8 if date.weekday() in [5, 6] else 1.0
            weekend_multiplier = 0.8 if pd.Timestamp(date).weekday() in [5, 6] else 1.0

            
            quantity = np.random.randint(1, 10)
            sales_amount = base_price * quantity * seasonal_multiplier * weekend_multiplier
            
            data.append({
                #'order_date': date.strftime('%Y-%m-%d'),
                'order_date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                'product_category': category,
                'product_name': f"{product} {np.random.choice(['Pro', 'Standard', 'Premium'])}",
                'region': region,
                'sales_amount': round(sales_amount, 2),
                'quantity': quantity,
                'customer_id': f"CUST_{np.random.randint(1000, 9999)}",
                'sales_rep': f"Rep_{np.random.randint(1, 50)}"
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sample records")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        logger.info("Starting data cleaning...")
        
        # Convert date column
        df['order_date'] = pd.to_datetime(df['order_date'])
        
        # Remove nulls and invalid values
        df = df.dropna(subset=['order_date', 'sales_amount', 'quantity'])
        df = df[df['sales_amount'] > 0]
        df = df[df['quantity'] > 0]
        
        # Standardize text fields
        df['product_category'] = df['product_category'].str.strip().str.upper()
        df['region'] = df['region'].str.strip().str.upper()
        
        # Add derived columns
        df['year_month'] = df['order_date'].dt.to_period('M').astype(str)
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['order_date', 'product_name', 'customer_id', 'sales_amount'])
        
        logger.info(f"Data cleaned. Final dataset has {len(df)} records")
        return df
    
    def load_to_database(self, df, table_name='cleaned_sales'):
        """Load cleaned data to database"""
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Data loaded to {table_name} table successfully")
        except Exception as e:
            logger.error(f"Failed to load data to database: {e}")
            raise
    
    def create_aggregations(self):
        """Create daily and monthly aggregations"""
        logger.info("Creating aggregations...")
        
        # Daily aggregations
        daily_agg = pd.read_sql("""
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
        """, self.engine)
        
        # Monthly aggregations
        monthly_agg = pd.read_sql("""
            SELECT 
                DATE(year_month || '-01') as date_period,
                region,
                product_category,
                SUM(sales_amount) as total_sales,
                SUM(quantity) as total_quantity,
                COUNT(*) as order_count,
                AVG(sales_amount) as avg_order_value,
                'monthly' as period_type
            FROM cleaned_sales
            GROUP BY year_month, region, product_category
        """, self.engine)
        
        # Combine and load
        all_agg = pd.concat([daily_agg, monthly_agg], ignore_index=True)
        all_agg.to_sql('sales_summary', self.engine, if_exists='replace', index=False)
        
        logger.info("Aggregations created successfully")

def run_pipeline():
    """Run the complete data pipeline"""
    pipeline = DataPipeline()
    
    # Generate or load raw data
    raw_data = pipeline.generate_sample_data(10000)
    
    # Clean data
    cleaned_data = pipeline.clean_data(raw_data)
    
    # Load to database
    pipeline.load_to_database(cleaned_data)
    
    # Create aggregations
    pipeline.create_aggregations()
    
    logger.info("Data pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()