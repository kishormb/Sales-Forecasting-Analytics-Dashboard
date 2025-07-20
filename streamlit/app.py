import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from datetime import datetime, timedelta, date
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
def init_connection():
    return create_engine("sqlite:///sales_database.db")

# Load data functions
@st.cache_data
def load_sales_data():
    engine = init_connection()
    return pd.read_sql("""
        SELECT * FROM cleaned_sales 
        ORDER BY order_date DESC
    """, engine)

@st.cache_data
def load_forecast_data():
    engine = init_connection()
    try:
        return pd.read_sql("""
            SELECT * FROM forecast_results 
            ORDER BY forecast_date
        """, engine)
    except:
        return pd.DataFrame()

@st.cache_data
def get_kpis():
    engine = init_connection()
    kpis = pd.read_sql("""
        SELECT 
            SUM(sales_amount) as total_sales,
            COUNT(*) as total_orders,
            AVG(sales_amount) as avg_order_value,
            COUNT(DISTINCT customer_id) as unique_customers,
            COUNT(DISTINCT region) as regions_count
        FROM cleaned_sales
    """, engine).iloc[0]
    return kpis.to_dict()

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Sales Forecasting & Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🎛️ Dashboard Controls</div>', 
                   unsafe_allow_html=True)
        
        # Date range selector
        st.subheader("📅 Date Range")
        sales_data = load_sales_data()
        
        if not sales_data.empty:
            sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])
            min_date = sales_data['order_date'].min().date()
            max_date = sales_data['order_date'].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = (date.today() - timedelta(days=30), date.today())
        
        # Filters
        st.subheader("🔍 Filters")
        
        if not sales_data.empty:
            regions = ['All'] + list(sales_data['region'].unique())
            categories = ['All'] + list(sales_data['product_category'].unique())
        else:
            regions = ['All']
            categories = ['All']
        
        selected_region = st.selectbox("Select Region", regions)
        selected_category = st.selectbox("Select Category", categories)
        
        # Dashboard sections
        st.subheader("📊 Dashboard Sections")
        show_overview = st.checkbox("Overview KPIs", value=True)
        show_trends = st.checkbox("Sales Trends", value=True)
        show_forecasts = st.checkbox("Forecasts", value=True)
        show_analysis = st.checkbox("Detailed Analysis", value=True)
    
    # Filter data based on selections
    filtered_data = sales_data.copy()
    
    if len(date_range) == 2:
        filtered_data = filtered_data[
            (filtered_data['order_date'].dt.date >= date_range[0]) &
            (filtered_data['order_date'].dt.date <= date_range[1])
        ]
    
    if selected_region != 'All':
        filtered_data = filtered_data[filtered_data['region'] == selected_region]
    
    if selected_category != 'All':
        filtered_data = filtered_data[filtered_data['product_category'] == selected_category]
    
    # Overview KPIs Section
    if show_overview:
        st.markdown("## 📈 Key Performance Indicators")
        
        if not filtered_data.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_sales = filtered_data['sales_amount'].sum()
            total_orders = len(filtered_data)
            avg_order_value = filtered_data['sales_amount'].mean()
            unique_customers = filtered_data['customer_id'].nunique()
            unique_products = filtered_data['product_name'].nunique()
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${total_sales:,.0f}</div>
                    <div class="metric-label">Total Sales</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_orders:,}</div>
                    <div class="metric-label">Total Orders</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${avg_order_value:.0f}</div>
                    <div class="metric-label">Avg Order Value</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_customers:,}</div>
                    <div class="metric-label">Unique Customers</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_products:,}</div>
                    <div class="metric-label">Unique Products</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Sales Trends Section
    if show_trends:
        st.markdown("## 📊 Sales Trends Analysis")
        
        if not filtered_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily sales trend
                daily_sales = filtered_data.groupby('order_date')['sales_amount'].sum().reset_index()
                
                fig_trend = px.line(
                    daily_sales, 
                    x='order_date', 
                    y='sales_amount',
                    title='Daily Sales Trend',
                    labels={'sales_amount': 'Sales ($)', 'order_date': 'Date'}
                )
                fig_trend.update_traces(line_color='#1f77b4', line_width=3)
                fig_trend.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Regional performance
                regional_sales = filtered_data.groupby('region')['sales_amount'].sum().reset_index()
                
                fig_region = px.bar(
                    regional_sales,
                    x='region',
                    y='sales_amount',
                    title='Sales by Region',
                    labels={'sales_amount': 'Sales ($)', 'region': 'Region'},
                    color='sales_amount',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_region, use_container_width=True)
            
            # Category performance
            category_sales = filtered_data.groupby('product_category')['sales_amount'].sum().sort_values(ascending=True)
            
            fig_category = go.Figure(go.Bar(
                x=category_sales.values,
                y=category_sales.index,
                orientation='h',
                marker_color='lightblue',
                text=category_sales.values,
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            ))
            
            fig_category.update_layout(
                title='Sales Performance by Category',
                xaxis_title='Sales ($)',
                yaxis_title='Product Category',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig_category, use_container_width=True)
    
    # Forecasts Section
    if show_forecasts:
        st.markdown("## 🔮 Sales Forecasting")
        
        forecast_data = load_forecast_data()
        
        if not forecast_data.empty:
            # Filter forecasts based on selections
            forecast_filtered = forecast_data.copy()
            
            if selected_region != 'All':
                forecast_filtered = forecast_filtered[
                    (forecast_filtered['region'] == selected_region) |
                    (forecast_filtered['region'] == 'All Regions')
                ]
            
            if selected_category != 'All':
                forecast_filtered = forecast_filtered[
                    (forecast_filtered['product_category'] == selected_category) |
                    (forecast_filtered['product_category'] == 'All Categories')
                ]
            
            if not forecast_filtered.empty:
                forecast_filtered['forecast_date'] = pd.to_datetime(forecast_filtered['forecast_date'])
                
                # Create forecast chart
                fig_forecast = go.Figure()
                
                # Add historical data
                if not filtered_data.empty:
                    historical = filtered_data.groupby('order_date')['sales_amount'].sum().reset_index()
                    fig_forecast.add_trace(go.Scatter(
                        x=historical['order_date'],
                        y=historical['sales_amount'],
                        mode='lines',
                        name='Historical Sales',
                        line=dict(color='blue', width=2)
                    ))
                
                # Add forecast data
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_filtered['forecast_date'],
                    y=forecast_filtered['predicted_sales'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Add confidence intervals
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_filtered['forecast_date'],
                    y=forecast_filtered['upper_bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_filtered['forecast_date'],
                    y=forecast_filtered['lower_bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig_forecast.update_layout(
                    title='Sales Forecast with Confidence Intervals',
                    xaxis_title='Date',
                    yaxis_title='Sales ($)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast summary table
                st.subheader("📋 Forecast Summary")
                
                forecast_summary = forecast_filtered.groupby(['region', 'product_category']).agg({
                    'predicted_sales': 'sum',
                    'accuracy_score': 'mean'
                }).reset_index()
                
                forecast_summary['predicted_sales'] = forecast_summary['predicted_sales'].round(2)
                forecast_summary['accuracy_score'] = forecast_summary['accuracy_score'].round(4)
                
                st.dataframe(
                    forecast_summary,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No forecast data available for the selected filters.")
        else:
            st.info("No forecast data available. Please run the forecasting model first.")
    
    # Detailed Analysis Section
    if show_analysis:
        st.markdown("## 🔍 Detailed Analysis")
        
        if not filtered_data.empty:
            tab1, tab2, tab3 = st.tabs(["Top Products", "Customer Analysis", "Seasonal Patterns"])
            
            with tab1:
                st.subheader("🏆 Top Performing Products")
                
                top_products = filtered_data.groupby('product_name').agg({
                    'sales_amount': 'sum',
                    'quantity': 'sum',
                    'customer_id': 'nunique'
                }).reset_index()
                
                top_products = top_products.sort_values('sales_amount', ascending=False).head(10)
                top_products.columns = ['Product', 'Total Sales', 'Total Quantity', 'Unique Customers']
                
                st.dataframe(
                    top_products,
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab2:
                st.subheader("👥 Customer Insights")
                
                customer_stats = filtered_data.groupby('customer_id').agg({
                    'sales_amount': ['sum', 'count', 'mean']
                }).round(2)
                
                customer_stats.columns = ['Total Spent', 'Order Count', 'Avg Order Value']
                customer_stats = customer_stats.sort_values('Total Spent', ascending=False).head(10)
                
                st.dataframe(
                    customer_stats,
                    use_container_width=True
                )
            
            with tab3:
                st.subheader("📅 Seasonal Patterns")
                
                # Monthly pattern
                filtered_data['month'] = filtered_data['order_date'].dt.month
                monthly_pattern = filtered_data.groupby('month')['sales_amount'].sum()
                
                fig_seasonal = px.bar(
                    x=monthly_pattern.index,
                    y=monthly_pattern.values,
                    title='Monthly Sales Pattern',
                    labels={'x': 'Month', 'y': 'Sales ($)'}
                )
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                # Day of week pattern
                dow_pattern = filtered_data.groupby('day_of_week')['sales_amount'].sum()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                
                fig_dow = px.bar(
                    x=[days[i] for i in dow_pattern.index],
                    y=dow_pattern.values,
                    title='Day of Week Sales Pattern',
                    labels={'x': 'Day of Week', 'y': 'Sales ($)'}
                )
                
                st.plotly_chart(fig_dow, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("📊 **Sales Forecasting Dashboard** | Built with Streamlit & Prophet | Data updated in real-time")

if __name__ == "__main__":
    main()