import pymongo
import pandas as pd
from textblob import TextBlob
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import io
import base64
from datetime import datetime

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://admin:<password>@cluster0.jwzyj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["reviews"]
collection = db["reviews_data"]

# Fetch Data
data = list(collection.find())
df = pd.DataFrame(data)

# Data Cleaning and Transformation
if 'Rating' in df.columns:
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df.fillna({'Rating': df['Rating'].mean()}, inplace=True)

if 'review_date' in df.columns:
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')

# Sentiment Analysis
def get_sentiment(text):
    if not isinstance(text, str):
        return 'Neutral'
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'

if 'Review' in df.columns:
    df['sentiment'] = df['Review'].apply(get_sentiment)

# Streamlit App Layout
st.title("📊 Review Analysis Dashboard")

# Date Range Filter
st.sidebar.header("Filter Options")
if 'review_date' in df.columns:
    start_date = st.sidebar.date_input("Start Date", df['review_date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['review_date'].max().date())
    filtered_df = df[(df['review_date'].dt.date >= start_date) & (df['review_date'].dt.date <= end_date)]
else:
    filtered_df = df
    st.sidebar.error("Date column missing!")

# Sentiment Pie Chart
st.subheader("Sentiment Analysis")
if not filtered_df.empty:
    sentiment_counts = filtered_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Sentiment Distribution")
    st.plotly_chart(fig_sentiment)
else:
    st.warning("No data available for the selected range.")

# Rating Histogram
st.subheader("Rating Distribution")
if not filtered_df.empty and 'Rating' in filtered_df.columns:
    fig_ratings = px.histogram(filtered_df, x='Rating', title="Rating Distribution", nbins=10)
    st.plotly_chart(fig_ratings)
else:
    st.warning("No ratings available for the selected range.")

# Word Cloud
st.subheader("Customer Feedback Word Cloud")
if not filtered_df.empty and 'Review' in filtered_df.columns:
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df['Review'].dropna()))
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        st.image(f"data:image/png;base64,{img_b64}", use_container_width=True)
    except ValueError:
        st.warning("Not enough data to generate word cloud.")
else:
    st.warning("No reviews available for the selected range.")

# Display Raw Data (Optional)
if st.checkbox("Show Raw Data"):
    st.write(filtered_df)

