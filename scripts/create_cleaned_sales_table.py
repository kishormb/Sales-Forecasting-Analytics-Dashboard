import pandas as pd
from sqlalchemy import create_engine
import os

CSV_PATH = 'data/raw/sales_data.csv'
DB_PATH = 'sales_database.db'
TABLE_NAME = 'cleaned_sales'

# STEP 1: Load CSV
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding='ISO-8859-1')

# STEP 2: Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Optional: Rename specific columns to match your EDA script
df.rename(columns={
    'sales': 'sales_amount',
    'category': 'product_category',
    'product_name': 'product_name'
}, inplace=True)

# STEP 3: Parse date
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# STEP 4: Add additional columns
df['day_of_week'] = df['order_date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5

# STEP 5: Save to SQLite database
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql(TABLE_NAME, con=engine, index=False, if_exists='replace')

print(f"✅ Table '{TABLE_NAME}' created in '{DB_PATH}' with {len(df)} rows.")
