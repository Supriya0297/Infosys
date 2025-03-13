from pymongo import MongoClient
import pandas as pd
import streamlit as st
import plotly.express as px

# Connect to MongoDB
client = MongoClient("mongodb+srv://admin:<password>@cluster0.jwzyj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["bookings"]

# Fetch datasets
bookings_data = list(db["bookings_data"].find())

# Convert to DataFrame
bookings_df = pd.DataFrame(bookings_data)

# Data Cleaning and Transformation
bookings_df.rename(columns={
    "Preferred Cusine": "preferred_cuisine",
    "check_in_date": "check_in_date",
    "check_out_date": "check_out_date",
    "number_of_stayers": "number_of_stayers",
    "booked_through_points": "booked_through_points"
}, inplace=True)

bookings_df.fillna({
    "check_in_date": "01-01-2000",
    "check_out_date": "02-01-2000",
    "number_of_stayers": 1,
    "preferred_cuisine": "Unknown"
}, inplace=True)

bookings_df["check_in_date"] = pd.to_datetime(bookings_df["check_in_date"], format="%d-%m-%Y", errors="coerce")
bookings_df["check_out_date"] = pd.to_datetime(bookings_df["check_out_date"], format="%d-%m-%Y", errors="coerce")

def fix_year(date):
    if date and date.year < 2024:
        return date.replace(year=2025)
    return date

bookings_df["check_in_date"] = bookings_df["check_in_date"].apply(fix_year)
bookings_df["check_out_date"] = bookings_df["check_out_date"].apply(fix_year)

bookings_df["length_of_stay"] = (bookings_df["check_out_date"] - bookings_df["check_in_date"]).dt.days

bookings_df["number_of_stayers"] = bookings_df["number_of_stayers"].astype(int)
bookings_df["booked_through_points"] = bookings_df["booked_through_points"].astype(int)

st.title("Hotel Bookings Insights")

# Filters
start_date = st.date_input("Start Date", value=bookings_df["check_in_date"].min())
end_date = st.date_input("End Date", value=bookings_df["check_in_date"].max())

cuisine_options = ["South Indian", "North Indian", "Multi"]
selected_cuisines = st.multiselect("Select Preferred Cuisine", options=cuisine_options)

# Filter data
filtered_df = bookings_df[(bookings_df["check_in_date"] >= pd.Timestamp(start_date)) & (bookings_df["check_in_date"] <= pd.Timestamp(end_date))]

if selected_cuisines:
    filtered_df = filtered_df[filtered_df["preferred_cuisine"].isin(selected_cuisines)]

# Dynamic grouping columns
grouping_columns = [filtered_df["check_in_date"].dt.date]
if selected_cuisines:
    grouping_columns.append("preferred_cuisine")

# Prepare data for charts
bookings_trend = filtered_df.groupby(grouping_columns).size().reset_index(name="count")
cuisine_counts = filtered_df["preferred_cuisine"].value_counts().reset_index()
cuisine_counts.columns = ["preferred_cuisine", "count"]
avg_stay_weekly = filtered_df.groupby(pd.Grouper(key="check_in_date", freq="W")).agg({"length_of_stay": "mean"}).reset_index()
avg_stay_monthly = filtered_df.groupby(pd.Grouper(key="check_in_date", freq="ME")).agg({"length_of_stay": "mean"}).reset_index()

# Charts
st.plotly_chart(px.line(bookings_trend, x="check_in_date", y="count", color="preferred_cuisine" if selected_cuisines else None, title="Daily Hotel Bookings"))
st.plotly_chart(px.bar(cuisine_counts, x="preferred_cuisine", y="count", title="Preferred Cuisine Analysis"))
st.plotly_chart(px.line(avg_stay_weekly, x="check_in_date", y="length_of_stay", title="Average Length of Stay (Weekly)"))
st.plotly_chart(px.line(avg_stay_monthly, x="check_in_date", y="length_of_stay", title="Average Length of Stay (Monthly)"))
