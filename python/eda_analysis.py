# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sqlalchemy import create_engine
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import warnings
# warnings.filterwarnings('ignore')

# class SalesEDA:
#     def __init__(self):
#         self.engine = create_engine("sqlite:///sales_database.db")
#         self.data = None
#         self.load_data()
    
#     def load_data(self):
#         """Load cleaned sales data"""
#         self.data = pd.read_sql("""
#             SELECT * FROM cleaned_sales
#         """, self.engine)
#         self.data['order_date'] = pd.to_datetime(self.data['order_date'])
#         print(f"Loaded {len(self.data)} records for analysis")
    
#     def basic_statistics(self):
#         """Generate basic statistics"""
#         print("=== BASIC STATISTICS ===")
#         print(f"Date Range: {self.data['order_date'].min()} to {self.data['order_date'].max()}")
#         print(f"Total Sales: ${self.data['sales_amount'].sum():,.2f}")
#         print(f"Average Order Value: ${self.data['sales_amount'].mean():.2f}")
#         print(f"Total Orders: {len(self.data):,}")
#         print(f"Unique Products: {self.data['product_name'].nunique()}")
#         print(f"Unique Customers: {self.data['customer_id'].nunique()}")
#         print("\n=== SALES BY REGION ===")
#         region_sales = self.data.groupby('region')['sales_amount'].agg(['sum', 'count', 'mean'])
#         print(region_sales)
#         print("\n=== SALES BY CATEGORY ===")
#         category_sales = self.data.groupby('product_category')['sales_amount'].agg(['sum', 'count', 'mean'])
#         print(category_sales)
    
#     def time_series_analysis(self):
#         """Analyze sales trends over time"""
#         # Daily sales trend
#         daily_sales = self.data.groupby('order_date')['sales_amount'].sum().reset_index()
        
#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=('Daily Sales Trend', 'Monthly Sales by Region', 
#                           'Weekly Pattern', 'Category Performance'),
#             specs=[[{"secondary_y": False}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}]]
#         )
        
#         # Daily trend
#         fig.add_trace(
#             go.Scatter(x=daily_sales['order_date'], y=daily_sales['sales_amount'],
#                       mode='lines', name='Daily Sales'),
#             row=1, col=1
#         )
        
#         # Monthly sales by region
#         monthly_region = self.data.groupby([
#             self.data['order_date'].dt.to_period('M').astype(str), 'region'
#         ])['sales_amount'].sum().reset_index()
        
#         for region in monthly_region['region'].unique():
#             region_data = monthly_region[monthly_region['region'] == region]
#             fig.add_trace(
#                 go.Scatter(x=region_data['order_date'], y=region_data['sales_amount'],
#                           mode='lines', name=f'{region}'),
#                 row=1, col=2
#             )
        
#         # Weekly pattern
#         weekly_pattern = self.data.groupby('day_of_week')['sales_amount'].sum().reset_index()
#         days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
#         weekly_pattern['day_name'] = [days[i] for i in weekly_pattern['day_of_week']]
        
#         fig.add_trace(
#             go.Bar(x=weekly_pattern['day_name'], y=weekly_pattern['sales_amount'],
#                   name='Weekly Sales'),
#             row=2, col=1
#         )
        
#         # Category performance
#         category_performance = self.data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=True)
        
#         fig.add_trace(
#             go.Bar(x=category_performance.values, y=category_performance.index,
#                   orientation='h', name='Category Sales'),
#             row=2, col=2
#         )
        
#         fig.update_layout(height=800, showlegend=True, title_text="Sales Analysis Dashboard")
#         fig.show()
    
#     def seasonality_analysis(self):
#         """Analyze seasonal patterns"""
#         # Monthly seasonality
#         self.data['month'] = self.data['order_date'].dt.month
#         monthly_avg = self.data.groupby('month')['sales_amount'].mean()
        
#         # Quarterly analysis
#         self.data['quarter'] = self.data['order_date'].dt.quarter
#         quarterly_sales = self.data.groupby(['quarter', 'region'])['sales_amount'].sum().reset_index()
        
#         print("=== SEASONAL ANALYSIS ===")
#         print("Monthly Average Sales:")
#         for month, sales in monthly_avg.items():
#             print(f"Month {month}: ${sales:.2f}")
        
#         print("\nQuarterly Sales by Region:")
#         quarterly_pivot = quarterly_sales.pivot(index='quarter', columns='region', values='sales_amount')
#         print(quarterly_pivot)
    
#     def correlation_analysis(self):
#         """Analyze correlations between variables"""
#         # Create numerical features for correlation
#         corr_data = self.data.copy()
#         corr_data['month'] = corr_data['order_date'].dt.month
#         corr_data['quarter'] = corr_data['order_date'].dt.quarter
#         corr_data['is_weekend_num'] = corr_data['is_weekend'].astype(int)
        
#         # Select numerical columns
#         num_cols = ['sales_amount', 'quantity', 'day_of_week', 'month', 'quarter', 'is_weekend_num']
#         correlation_matrix = corr_data[num_cols].corr()
        
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#         plt.title('Correlation Matrix')
#         plt.tight_layout()
#         plt.show()
    
#     def generate_insights(self):
#         """Generate key business insights"""
#         insights = []
        
#         # Top performing regions
#         top_region = self.data.groupby('region')['sales_amount'].sum().idxmax()
#         top_region_sales = self.data.groupby('region')['sales_amount'].sum().max()
#         insights.append(f"Top performing region: {top_region} with ${top_region_sales:,.2f} in sales")
        
#         # Best selling category
#         top_category = self.data.groupby('product_category')['sales_amount'].sum().idxmax()
#         top_category_sales = self.data.groupby('product_category')['sales_amount'].sum().max()
#         insights.append(f"Best selling category: {top_category} with ${top_category_sales:,.2f} in sales")
        
#         # Weekend vs weekday performance
#         weekend_avg = self.data[self.data['is_weekend']]['sales_amount'].mean()
#         weekday_avg = self.data[~self.data['is_weekend']]['sales_amount'].mean()
#         insights.append(f"Weekend avg order: ${weekend_avg:.2f}, Weekday avg: ${weekday_avg:.2f}")
        
#         # Growth trend
#         self.data['year_month'] = self.data['order_date'].dt.to_period('M')
#         monthly_sales = self.data.groupby('year_month')['sales_amount'].sum()
#         if len(monthly_sales) > 1:
#             growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2]) * 100
#             insights.append(f"Month-over-month growth: {growth_rate:.1f}%")
        
#         print("\n=== KEY INSIGHTS ===")
#         for insight in insights:
#             print(f"• {insight}")
        
#         return insights

# def run_eda():
#     """Run complete EDA analysis"""
#     eda = SalesEDA()
#     eda.basic_statistics()
#     eda.time_series_analysis()
#     eda.seasonality_analysis()
#     eda.correlation_analysis()
#     insights = eda.generate_insights()
#     return insights

# if __name__ == "__main__":
#     run_eda()







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SalesEDA:
    def __init__(self):
        self.engine = create_engine("sqlite:///sales_database.db")
        self.data = None
        self.load_data()
        self.prepare_features()

    def load_data(self):
        """Load cleaned sales data"""
        self.data = pd.read_sql("SELECT * FROM cleaned_sales", self.engine)
        self.data['order_date'] = pd.to_datetime(self.data['order_date'])
        print(f"Loaded {len(self.data)} records for analysis")

    def prepare_features(self):
        """Create additional datetime features needed for analysis"""
        self.data['day_of_week'] = self.data['order_date'].dt.dayofweek  # Monday=0, Sunday=6
        self.data['is_weekend'] = self.data['day_of_week'] >= 5  # Sat=5, Sun=6
        self.data['month'] = self.data['order_date'].dt.month
        self.data['quarter'] = self.data['order_date'].dt.quarter
        self.data['year_month'] = self.data['order_date'].dt.to_period('M')

    def basic_statistics(self):
        """Generate basic statistics"""
        print("=== BASIC STATISTICS ===")
        print(f"Date Range: {self.data['order_date'].min()} to {self.data['order_date'].max()}")
        print(f"Total Sales: ${self.data['sales_amount'].sum():,.2f}")
        print(f"Average Order Value: ${self.data['sales_amount'].mean():.2f}")
        print(f"Total Orders: {len(self.data):,}")
        print(f"Unique Products: {self.data['product_name'].nunique()}")
        print(f"Unique Customers: {self.data['customer_id'].nunique()}")
        print("\n=== SALES BY REGION ===")
        region_sales = self.data.groupby('region')['sales_amount'].agg(['sum', 'count', 'mean'])
        print(region_sales)
        print("\n=== SALES BY CATEGORY ===")
        category_sales = self.data.groupby('product_category')['sales_amount'].agg(['sum', 'count', 'mean'])
        print(category_sales)

    def time_series_analysis(self):
        """Analyze sales trends over time"""
        daily_sales = self.data.groupby('order_date')['sales_amount'].sum().reset_index()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Sales Trend', 'Monthly Sales by Region', 'Weekly Pattern', 'Category Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Daily trend
        fig.add_trace(
            go.Scatter(x=daily_sales['order_date'], y=daily_sales['sales_amount'],
                       mode='lines', name='Daily Sales'),
            row=1, col=1
        )

        # Monthly sales by region
        monthly_region = self.data.groupby([
            self.data['order_date'].dt.to_period('M').astype(str), 'region'
        ])['sales_amount'].sum().reset_index()

        for region in monthly_region['region'].unique():
            region_data = monthly_region[monthly_region['region'] == region]
            fig.add_trace(
                go.Scatter(x=region_data['order_date'], y=region_data['sales_amount'],
                           mode='lines', name=f'{region}'),
                row=1, col=2
            )

        # Weekly pattern
        weekly_pattern = self.data.groupby('day_of_week')['sales_amount'].sum().reset_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_pattern['day_name'] = weekly_pattern['day_of_week'].apply(lambda x: days[x])

        fig.add_trace(
            go.Bar(x=weekly_pattern['day_name'], y=weekly_pattern['sales_amount'], name='Weekly Sales'),
            row=2, col=1
        )

        # Category performance
        category_performance = self.data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=True)

        fig.add_trace(
            go.Bar(x=category_performance.values, y=category_performance.index,
                   orientation='h', name='Category Sales'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True, title_text="Sales Analysis Dashboard")
        fig.show()

    def seasonality_analysis(self):
        """Analyze seasonal patterns"""
        monthly_avg = self.data.groupby('month')['sales_amount'].mean()
        quarterly_sales = self.data.groupby(['quarter', 'region'])['sales_amount'].sum().reset_index()

        print("=== SEASONAL ANALYSIS ===")
        print("Monthly Average Sales:")
        for month, sales in monthly_avg.items():
            print(f"Month {month}: ${sales:.2f}")

        print("\nQuarterly Sales by Region:")
        quarterly_pivot = quarterly_sales.pivot(index='quarter', columns='region', values='sales_amount')
        print(quarterly_pivot)

    def correlation_analysis(self):
        """Analyze correlations between variables"""
        corr_data = self.data.copy()
        corr_data['is_weekend_num'] = corr_data['is_weekend'].astype(int)

        num_cols = ['sales_amount', 'quantity', 'day_of_week', 'month', 'quarter', 'is_weekend_num']
        correlation_matrix = corr_data[num_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def generate_insights(self):
        """Generate key business insights"""
        insights = []

        # Top performing regions
        top_region = self.data.groupby('region')['sales_amount'].sum().idxmax()
        top_region_sales = self.data.groupby('region')['sales_amount'].sum().max()
        insights.append(f"Top performing region: {top_region} with ${top_region_sales:,.2f} in sales")

        # Best selling category
        top_category = self.data.groupby('product_category')['sales_amount'].sum().idxmax()
        top_category_sales = self.data.groupby('product_category')['sales_amount'].sum().max()
        insights.append(f"Best selling category: {top_category} with ${top_category_sales:,.2f} in sales")

        # Weekend vs weekday performance
        weekend_avg = self.data[self.data['is_weekend']]['sales_amount'].mean()
        weekday_avg = self.data[~self.data['is_weekend']]['sales_amount'].mean()
        insights.append(f"Weekend avg order: ${weekend_avg:.2f}, Weekday avg: ${weekday_avg:.2f}")

        # Growth trend
        monthly_sales = self.data.groupby('year_month')['sales_amount'].sum()
        if len(monthly_sales) > 1:
            growth_rate = ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2]) * 100
            insights.append(f"Month-over-month growth: {growth_rate:.1f}%")

        print("\n=== KEY INSIGHTS ===")
        for insight in insights:
            print(f"• {insight}")

        return insights

def run_eda():
    """Run complete EDA analysis"""
    eda = SalesEDA()
    eda.basic_statistics()
    eda.time_series_analysis()
    eda.seasonality_analysis()
    eda.correlation_analysis()
    insights = eda.generate_insights()
    return insights

if __name__ == "__main__":
    run_eda()
