import streamlit as st
import pandas as pd
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
import os
from together import Together

def load_pinecone():
    pc = Pinecone(api_key=os.getenv('My_pinecone_id'))  
    return pc.Index(host="my_host")  

def get_embeddings():
    return TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

def fetch_reviews(query, hotel_name, start_date, end_date, min_rating):
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)
    index = load_pinecone()
    
    filter_conditions = {"review_date": {"$gte": start_date, "$lte": end_date}}
    if hotel_name:
        filter_conditions["hotel_name"] = hotel_name
    if min_rating:
        filter_conditions["Rating"] = {"$gte": min_rating}
    
    results = index.query(
        vector=query_embedding,
        top_k=10,
        namespace="",
        include_metadata=True,
        filter=filter_conditions
    )
    
    matches = results["matches"]
    matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
    return matched_ids

def analyze_sentiment(reviews):
    concatenated_reviews = " ".join(reviews)
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {"role": "user", "content": f"""Analyze sentiment (positive or negative) for these reviews related to the selected hotel: {concatenated_reviews}."""}
        ]
    )
    return response.choices[0].message.content

def main():
    st.title("Hotel Customer Review Analysis")
    
    query = st.text_input("Search for reviews about:", "food, service, cleanliness, location")
    hotel_name = st.text_input("Enter hotel name (optional):")
    start_date = st.number_input("Start Date (YYYYMMDD)", value=20240101, format="%d")
    end_date = st.number_input("End Date (YYYYMMDD)", value=20240108, format="%d")
    min_rating = st.slider("Select Minimum Rating", min_value=1, max_value=10, value=5)
    
    if st.button("Get Reviews & Analyze Sentiment"):
        matched_ids = fetch_reviews(query, hotel_name, start_date, end_date, min_rating)
        df = pd.read_excel('reviews_data.xlsx')
        req_df = df[df["review_id"].isin(matched_ids)]
        
        if not req_df.empty:
            sentiment_result = analyze_sentiment(req_df["Review"].tolist())
            st.subheader("Sentiment Analysis Result")
            st.write(sentiment_result)
        else:
            st.write("No relevant reviews found.")

if __name__ == "__main__":
    main()