import streamlit as st
import pandas as pd
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
import os
from together import Together
import smtplib
from email.mime.text import MIMEText

def load_pinecone():
    pc = Pinecone(api_key=os.getenv('pinecone_api'))
    return pc.Index(host="abc.pinecone.io") 

def get_embeddings():
    return TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

def fetch_reviews(query, start_date, end_date, rating_filter):
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)
    index = load_pinecone()
    
    filter_conditions = {"review_date": {"$gte": start_date, "$lte": end_date}}
    if rating_filter:
        filter_conditions["Rating"] = {"$lte": rating_filter}
    
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
            {"role": "user", "content": f"""Classify sentiment into positive and negative for food and restaurant based on these reviews - {concatenated_reviews}. Don't mention the name of the hotel."""}
        ]
    )
    return response.choices[0].message.content

def send_email_to_manager(reviews, sentiment_result):
    manager_email = "mno@gmail.com"  
    subject = "Hotel Review Sentiment Analysis Report"
    body = f"Sentiment Analysis Result:\n{sentiment_result}"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "abc@gmail.com"  
    msg["To"] = manager_email
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  
        server.login("abc@gmail.com", "password")  
        server.sendmail("abc@gmail.com", manager_email, msg.as_string())

def main():
    st.title("Hotel Review Sentiment Analysis")
    
    query = st.text_input("Enter query:", "What are some of the reviews that mention food, restaurant, lunch, breakfast, dinner")
    start_date = st.number_input("Start Date (YYYYMMDD)", value=20240101, format="%d")
    end_date = st.number_input("End Date (YYYYMMDD)", value=20240108, format="%d")
    rating_filter = st.slider("Select Maximum Rating", min_value=1, max_value=10, value=9)
    
    if st.button("Analyze Sentiment"):
        matched_ids = fetch_reviews(query, start_date, end_date, rating_filter)
        df = pd.read_excel('reviews_data.xlsx')
        req_df = df[df["review_id"].isin(matched_ids)]
        
        if not req_df.empty:
            sentiment_result = analyze_sentiment(req_df["Review"].tolist())
            st.subheader("Sentiment Analysis Result")
            st.write(sentiment_result)
            
            send_email_to_manager(req_df["Review"].tolist(), sentiment_result)
            st.success("Sentiment analysis report has been sent to the manager.")
        else:
            st.write("No matching reviews found.")

if __name__ == "__main__":
    main()