import pymongo
import pandas as pd
import streamlit as st
import plotly.express as px

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://admin:<password>@cluster0.jwzyj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["dining_info"]
collection = db["dining_data"]

# Fetch all records
data = list(collection.find())

# Convert to DataFrame
df = pd.DataFrame(data)

# Data Cleaning and Transformation
if "_id" in df.columns:
    df.drop("_id", axis=1, inplace=True)

df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
df["price_for_1"] = pd.to_numeric(df["price_for_1"], errors="coerce")
df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# Handle missing values
df.fillna({"price_for_1": df["price_for_1"].median(), "age": df["age"].median(), "Qty": 1}, inplace=True)

# Add revenue column
df["total_revenue"] = df["price_for_1"] * df["Qty"]

# Title
st.title("Dining Insights Dashboard")

# Date Range Filter
min_date = df["order_time"].min().date()
max_date = df["order_time"].max().date()

start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Cuisine Filter
cuisine_options = df["Preferred Cusine"].dropna().unique()
selected_cuisines = st.multiselect("Select Preferred Cuisine", cuisine_options)

# Data Filtering
filtered_df = df[(df["order_time"].dt.date >= start_date) & (df["order_time"].dt.date <= end_date)]

if selected_cuisines:
    filtered_df = filtered_df[filtered_df["Preferred Cusine"].isin(selected_cuisines)]

# 📊 Pie Chart: Average Dining Cost by Cuisine
avg_cost_by_cuisine = filtered_df.groupby("Preferred Cusine")["price_for_1"].mean().reset_index()
st.plotly_chart(px.pie(avg_cost_by_cuisine, values="price_for_1", names="Preferred Cusine", title="Average Dining Cost by Cuisine"))

# 📈 Line Chart: Customer Count Over Time
filtered_df["order_date"] = filtered_df["order_time"].dt.date
customer_count = filtered_df.groupby("order_date")["customer_id"].nunique().reset_index()
st.plotly_chart(px.line(customer_count, x="order_date", y="customer_id", title="Customer Count Over Time"))

# 📊 Bar Chart: Total Revenue by Cuisine
revenue_by_cuisine = filtered_df.groupby("Preferred Cusine")["total_revenue"].sum().reset_index()
st.plotly_chart(px.bar(revenue_by_cuisine, x="Preferred Cusine", y="total_revenue", title="Total Revenue by Cuisine"))

# 📊 Bar Chart: Average Age of Customers by Cuisine
avg_age_by_cuisine = filtered_df.groupby("Preferred Cusine")["age"].mean().reset_index()
st.plotly_chart(px.bar(avg_age_by_cuisine, x="Preferred Cusine", y="age", title="Average Age of Customers by Cuisine"))

# 📈 Line Chart: Daily Revenue Trend
revenue_trend = filtered_df.groupby("order_date")["total_revenue"].sum().reset_index()
st.plotly_chart(px.line(revenue_trend, x="order_date", y="total_revenue", title="Daily Revenue Trend"))

st.success("Dashboard updated successfully!")
