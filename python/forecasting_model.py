# import pandas as pd
# import numpy as np
# from prophet import Prophet
# from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
# from sqlalchemy import create_engine
# import warnings
# warnings.filterwarnings('ignore')

# class SalesForecaster:
#     def __init__(self):
#         self.engine = create_engine("data\sales_data.csv")
#         self.models = {}
#         self.forecasts = {}
        
#     def prepare_data(self, region=None, category=None, aggregation='daily'):
#         """Prepare data for Prophet forecasting"""
#         query = """
#             SELECT date_period as ds, total_sales as y, region, product_category
#             FROM sales_summary 
#             WHERE period_type = ?
#         """
#         params = [aggregation]
        
#         if region:
#             query += " AND region = ?"
#             params.append(region)
#         if category:
#             query += " AND product_category = ?"
#             params.append(category)
            
#         query += " ORDER BY date_period"
        
#         data = pd.read_sql(query, self.engine, params=params)
#         data['ds'] = pd.to_datetime(data['ds'])
        
#         # If multiple regions/categories, sum them up
#         if len(data) > 0:
#             data = data.groupby('ds')['y'].sum().reset_index()
        
#         return data
    
#     def create_model(self, seasonality_mode='multiplicative', changepoint_prior_scale=0.05):
#         """Create and configure Prophet model"""
#         model = Prophet(
#             seasonality_mode=seasonality_mode,
#             changepoint_prior_scale=changepoint_prior_scale,
#             yearly_seasonality=True,
#             weekly_seasonality=True,
#             daily_seasonality=False
#         )
#         return model
    
#     def train_and_forecast(self, region=None, category=None, periods=30, freq='D'):
#         """Train model and generate forecasts"""
#         # Prepare training data
#         data = self.prepare_data(region, category)
        
#         if len(data) < 10:
#             print(f"Insufficient data for {region}-{category}")
#             return None
        
#         # Split data for validation
#         train_size = int(len(data) * 0.8)
#         train_data = data[:train_size]
#         test_data = data[train_size:]
        
#         # Train model
#         model = self.create_model()
#         model.fit(train_data)
        
#         # Validate on test data
#         if len(test_data) > 0:
#             test_forecast = model.predict(test_data[['ds']])
#             mape = mean_absolute_percentage_error(test_data['y'], test_forecast['yhat'])
#             rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
#         else:
#             mape = 0
#             rmse = 0
        
#         # Generate future forecasts
#         future = model.make_future_dataframe(periods=periods, freq=freq)
#         forecast = model.predict(future)
        
#         # Store results
#         model_key = f"{region}_{category}"
#         self.models[model_key] = model
#         self.forecasts[model_key] = {
#             'forecast': forecast,
#             'mape': mape,
#             'rmse': rmse,
#             'historical_data': data
#         }
        
#         print(f"Model trained for {model_key}: MAPE={mape:.2%}, RMSE={rmse:.2f}")
#         return forecast
    
#     def forecast_all_segments(self):
#         """Generate forecasts for all region-category combinations"""
#         # Get unique regions and categories
#         regions = pd.read_sql("SELECT DISTINCT region FROM sales_summary", self.engine)['region'].tolist()
#         categories = pd.read_sql("SELECT DISTINCT product_category FROM sales_summary", self.engine)['product_category'].tolist()
        
#         results = []
        
#         # Overall forecast
#         overall_forecast = self.train_and_forecast(periods=30)
#         if overall_forecast is not None:
#             results.append(('Overall', 'All Categories', overall_forecast))
        
#         # Regional forecasts
#         for region in regions:
#             forecast = self.train_and_forecast(region=region, periods=30)
#             if forecast is not None:
#                 results.append((region, 'All Categories', forecast))
        
#         # Category forecasts
#         for category in categories:
#             forecast = self.train_and_forecast(category=category, periods=30)
#             if forecast is not None:
#                 results.append(('All Regions', category, forecast))
        
#         return results
    
#     def save_forecasts_to_db(self):
#         """Save forecast results to database"""
#         all_forecasts = []
        
#         for model_key, result in self.forecasts.items():
#             region, category = model_key.split('_', 1) if '_' in model_key else ('Overall', 'All')
#             forecast_df = result['forecast']
            
#             # Only save future forecasts
#             future_forecasts = forecast_df[forecast_df['ds'] > forecast_df['ds'].max() - pd.Timedelta(days=30)]
            
#             for _, row in future_forecasts.iterrows():
#                 all_forecasts.append({
#                     'forecast_date': row['ds'].date(),
#                     'region': region if region != 'None' else 'All Regions',
#                     'product_category': category if category != 'None' else 'All Categories',
#                     'predicted_sales': round(row['yhat'], 2),
#                     'lower_bound': round(row['yhat_lower'], 2),
#                     'upper_bound': round(row['yhat_upper'], 2),
#                     'model_name': 'Prophet',
#                     'accuracy_score': result['mape']
#                 })
        
#         if all_forecasts:
#             forecasts_df = pd.DataFrame(all_forecasts)
#             forecasts_df.to_sql('forecast_results', self.engine, if_exists='replace', index=False)
#             print(f"Saved {len(forecasts_df)} forecast records to database")
    
#     def plot_forecast(self, region=None, category=None):
#         """Plot forecast results"""
#         import matplotlib.pyplot as plt
        
#         model_key = f"{region}_{category}"
#         if model_key not in self.forecasts:
#             print(f"No forecast available for {model_key}")
#             return
        
#         model = self.models[model_key]
#         forecast = self.forecasts[model_key]['forecast']
        
#         fig = model.plot(forecast)
#         plt.title(f'Sales Forecast - {region} - {category}')
#         plt.xlabel('Date')
#         plt.ylabel('Sales Amount')
#         plt.show()
        
#         # Plot components
#         fig_components = model.plot_components(forecast)
#         plt.show()

# def run_forecasting():
#     """Run complete forecasting pipeline"""
#     forecaster = SalesForecaster()
    
#     # Generate forecasts for all segments
#     results = forecaster.forecast_all_segments()
    
#     # Save results to database
#     forecaster.save_forecasts_to_db()
    
#     print(f"Generated forecasts for {len(results)} segments")
#     return forecaster

# if __name__ == "__main__":
#     run_forecasting()









import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sqlalchemy import create_engine
import warnings
import os

warnings.filterwarnings('ignore')

class SalesForecaster:
    def __init__(self):
        # Update the path to your actual CSV location
        csv_path = os.path.join("data", "raw", "sales_data.csv")

        # Load CSV without parse_dates first to inspect columns
        temp_df = pd.read_csv(csv_path, encoding='latin1')
        temp_df.columns = temp_df.columns.str.strip()  # strip whitespace

        print("CSV columns:", temp_df.columns.tolist())  # Debug print

        # Use the actual date columns from your data
        possible_date_cols = ['Order Date', 'Ship Date']
        date_col = None
        for col in possible_date_cols:
            if col in temp_df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError("No suitable date column found in CSV!")

        # Reload with parse_dates on the found date column
        self.sales_data = pd.read_csv(csv_path, parse_dates=[date_col], encoding='latin1')
        self.sales_data.columns = self.sales_data.columns.str.strip()

        # Rename date column to standard 'date_period'
        if date_col != 'date_period':
            self.sales_data.rename(columns={date_col: 'date_period'}, inplace=True)

        # Create SQLite in-memory DB
        self.engine = create_engine("sqlite://", echo=False)

        # Write CSV data into the DB table
        self.sales_data.to_sql("sales_summary", self.engine, index=False, if_exists="replace")

        self.models = {}
        self.forecasts = {}

    def prepare_data(self, region=None, category=None, aggregation='daily'):
        """
        Prepare data aggregated by date_period and sum of Sales.
        Supports optional filtering by region and category.
        Aggregation can be 'daily', 'weekly', or 'monthly'.
        """
        query = """
            SELECT date_period as ds, SUM(Sales) as y
            FROM sales_summary
            WHERE 1=1
        """
        params = []

        if region:
            query += " AND Region = ?"
            params.append(region)
        if category:
            query += " AND Category = ?"
            params.append(category)

        query += " GROUP BY date_period ORDER BY date_period"

        data = pd.read_sql(query, self.engine, params=tuple(params))

        data['ds'] = pd.to_datetime(data['ds'])

        if len(data) == 0:
            return data

        data = data.set_index('ds')

        if aggregation == 'weekly':
            data = data.resample('W').sum()
        elif aggregation == 'monthly':
            data = data.resample('M').sum()
        else:  # default daily, no resampling needed
            data = data

        data = data.reset_index()

        return data

    def create_model(self, seasonality_mode='multiplicative', changepoint_prior_scale=0.05):
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        return model

    def train_and_forecast(self, region=None, category=None, periods=30, freq='D'):
        data = self.prepare_data(region, category)

        if len(data) < 10:
            print(f"Insufficient data for region='{region}' and category='{category}'")
            return None

        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]

        model = self.create_model()
        model.fit(train_data)

        if len(test_data) > 0:
            test_forecast = model.predict(test_data[['ds']])
            mape = mean_absolute_percentage_error(test_data['y'], test_forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
        else:
            mape = 0
            rmse = 0

        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)

        model_key = f"{region}_{category}"
        self.models[model_key] = model
        self.forecasts[model_key] = {
            'forecast': forecast,
            'mape': mape,
            'rmse': rmse,
            'historical_data': data
        }

        print(f"Model trained for {model_key}: MAPE={mape:.2%}, RMSE={rmse:.2f}")
        return forecast

    def forecast_all_segments(self):
        # Get unique regions and categories from your dataset
        regions = pd.read_sql("SELECT DISTINCT Region FROM sales_summary", self.engine)['Region'].dropna().tolist()
        categories = pd.read_sql("SELECT DISTINCT Category FROM sales_summary", self.engine)['Category'].dropna().tolist()

        results = []

        # Overall forecast (all regions, all categories)
        overall_forecast = self.train_and_forecast(periods=30)
        if overall_forecast is not None:
            results.append(('Overall', 'All Categories', overall_forecast))

        # Forecast by each region (all categories)
        for region in regions:
            forecast = self.train_and_forecast(region=region, periods=30)
            if forecast is not None:
                results.append((region, 'All Categories', forecast))

        # Forecast by each category (all regions)
        for category in categories:
            forecast = self.train_and_forecast(category=category, periods=30)
            if forecast is not None:
                results.append(('All Regions', category, forecast))

        return results

    def save_forecasts_to_db(self):
        all_forecasts = []

        for model_key, result in self.forecasts.items():
            region, category = model_key.split('_', 1) if '_' in model_key else ('Overall', 'All')
            forecast_df = result['forecast']

            last_hist_date = result['historical_data']['ds'].max()
            future_forecasts = forecast_df[forecast_df['ds'] > last_hist_date]

            for _, row in future_forecasts.iterrows():
                all_forecasts.append({
                    'forecast_date': row['ds'].date(),
                    'region': region if region != 'None' else 'All Regions',
                    'product_category': category if category != 'None' else 'All Categories',
                    'predicted_sales': round(row['yhat'], 2),
                    'lower_bound': round(row['yhat_lower'], 2),
                    'upper_bound': round(row['yhat_upper'], 2),
                    'model_name': 'Prophet',
                    'accuracy_score': result['mape']
                })

        if all_forecasts:
            forecasts_df = pd.DataFrame(all_forecasts)
            forecasts_df.to_sql('forecast_results', self.engine, if_exists='replace', index=False)
            print(f"Saved {len(forecasts_df)} forecast records to database")

    def plot_forecast(self, region=None, category=None):
        import matplotlib.pyplot as plt

        model_key = f"{region}_{category}"
        if model_key not in self.forecasts:
            print(f"No forecast available for {model_key}")
            return

        model = self.models[model_key]
        forecast = self.forecasts[model_key]['forecast']

        fig = model.plot(forecast)
        plt.title(f'Sales Forecast - Region: {region} - Category: {category}')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.show()

        model.plot_components(forecast)
        plt.show()


def run_forecasting():
    forecaster = SalesForecaster()
    results = forecaster.forecast_all_segments()
    forecaster.save_forecasts_to_db()
    print(f"Generated forecasts for {len(results)} segments")
    return forecaster


if __name__ == "__main__":
    run_forecasting()

