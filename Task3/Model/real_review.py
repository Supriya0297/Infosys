import streamlit as st
import pandas as pd
import os
from together import Together
import smtplib
from email.mime.text import MIMEText
from pymongo import MongoClient
import datetime

if not os.getenv("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = "api_key"

def analyze_sentiment(review):
    
    client = Together()  
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {"role": "user", "content": f"Classify sentiment as positive or negative for the following hotel review: '{review}'. If the review is negative, respond with 'negative'. Otherwise, respond with 'positive'."}
        ]
    )
    return response.choices[0].message.content.strip().lower()


def send_email_to_manager(review, guest_name):
    manager_email = "abc@gmail.com"
    subject = "Negative Hotel Review Alert"
    body = f"Guest {guest_name} has submitted a negative review:\n\n{review}"
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "abc@gmail.com"
    msg["To"] = manager_email
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login("abc@gmail.com", "abc")
        server.sendmail("myemail@gmail.com", manager_email, msg.as_string())

def save_review_to_mongodb(review_id, customer_id, review, rating, sentiment):
    client = MongoClient("mongdb_connection_string")
    db = client["reviews"]
    collection = db["reviews_data"]
    review_data = {
        "review_id": review_id,
        "customer_id": customer_id,
        "review_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "Review": review,
        "Rating": rating,
        "review_date_numeric": int(datetime.datetime.now().strftime("%Y%m%d")),
        "Sentiment": sentiment
    }
    collection.insert_one(review_data)

def main():
    st.title("Real-Time Hotel Review Submission and Analysis")
    guest_name = st.text_input("Enter your name:")
    customer_id = st.number_input("Enter Customer ID:", min_value=1, step=1)
    review_id = st.number_input("Enter Review ID:", min_value=1, step=1)
    review = st.text_area("Enter your review:")
    rating = st.slider("Rate your experience (1-10):", min_value=1, max_value=10, value=5)
    
    if st.button("Submit Review"):
        if guest_name and review:
            sentiment = analyze_sentiment(review)
            
            if sentiment == "negative":
                send_email_to_manager(review, guest_name)
                st.warning("Your review was classified as negative and has been reported to the manager for further action.")
            else:
                st.success("Thank you for your review!")
                
            # Save the review to MongoDB
            save_review_to_mongodb(review_id, customer_id, review, rating, sentiment)
            
            # Save the review to a local file for record-keeping
            data = {"Review ID": [review_id], "Customer ID": [customer_id], "Review": [review], "Rating": [rating], "Sentiment": [sentiment]}
            df = pd.DataFrame(data)
            if os.path.exists("reviews_data.xlsx"):
                existing_df = pd.read_excel("reviews_data.xlsx")
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel("reviews_data.xlsx", index=False)
        else:
            st.error("Please enter your name and review before submitting.")

if __name__ == "__main__":
    main()
