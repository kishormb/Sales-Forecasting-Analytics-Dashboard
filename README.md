# Sales-Forecasting-Analytics-Dashboard

End-to-end sales forecasting and analytics dashboard using Python, SQL, Facebook Prophet, and Streamlit. Includes data cleaning, time series forecasting, interactive dashboards, and production-ready deployment.

## 🚀 Project Overview

This project demonstrates a full-stack data analytics pipeline including:

- 📁 **Data Engineering** — Clean and transform raw transactional data using SQL and Python.
- 📈 **Sales Forecasting** — Predict future sales using Facebook Prophet time-series modeling.
- 📊 **Interactive Dashboard** — Real-time insights and visualizations built with Streamlit.
- 🧠 **Business Intelligence** — KPI tracking, trend detection, and ROI-driven decision-making.

---
## 🧱 Project Structure

    sales_forecasting_project/
├── data/
│   ├── raw/
│   │   └── sales_data.csv
│   └── processed/
│       └── cleaned_sales_data.csv
├── sql/
│   ├── create_tables.sql
│   ├── data_cleaning.sql
│   └── aggregations.sql
├── python/
│   ├── data_engineering.py
│   ├── eda_analysis.py
│   ├── forecasting_model.py
│   └── utils.py
├── streamlit/
│   ├── app.py
│   └── requirements.txt
├── power_bi/
│   └── sales_dashboard.pbix
├── tests/
│   └── test_forecasting.py
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

| Layer           | Tools Used                          |
|----------------|--------------------------------------|
| Data Wrangling | Python (Pandas), SQL                 |
| Forecasting    | Facebook Prophet                     |
| Dashboard      | Streamlit                            |
| Storage        | CSV files (for now)                  |
| Deployment     | Localhost / Streamlit Cloud / Render |

---

## 📈 Features

- ✅ Clean and transform raw sales data
- ✅ Visualize historical sales performance
- ✅ Predict future sales using Prophet
- ✅ Drill-down filters by product, region, and time
- ✅ Track KPIs: revenue, growth, volume
- ✅ Export forecast data for stakeholders

---

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

```
```python file
python python/eda_analysis.py

python python/forecasting_model.py

python python/data_engineering.py
```
``` to run the database from excel to sqllite
python python/data_engineering.py
```
```run main file
python -m streamlit run streamlit/app.py
```