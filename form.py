import streamlit as st
from datetime import date
import pandas as pd
import random
import joblib
import numpy as np
import smtplib
import string
from pymongo import MongoClient
import ssl
from email.message import EmailMessage

# Function to generate a random coupon code
def generate_coupon():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

# Connect to MongoDB
client = MongoClient("mongodb+srv://admin:<password>@cluster0.jwzyj.mongodb.net/?ssl=true")
db = client["hotel_guests"]
new_bookings_collection = db["new_bookings"]

# Title
st.title("🏨 Know And Then Book")

# User Input
has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))
if has_customer_id == "Yes":
    customer_id = st.text_input("Enter your Customer ID", "")
else:
    customer_id = random.randint(10001, 99999)
    st.write(f"Your generated Customer ID: {customer_id}")

# Booking Details
name = st.text_input("Enter your name", "")
email = st.text_input("Enter your Email ID", "")
checkin_date = st.date_input("Check-in Date", min_value=date.today())
checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
stayers = st.number_input("Number of stayers?", min_value=1, max_value=5, step=1)
cuisine_options = ["South Indian", "North Indian", "Multi"]
preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
booking_points = st.selectbox("Book through loyalty points?", ["Yes", "No"])

def send_coupon_email(name, email, customer_id, coupon_code, preferred_cuisine, recommended_dishes):
    sender_email = "myemail@gmail.com"
    sender_password = "<password>"
    receiver_email = email

    subject = "Your Hotel Booking Confirmation & Coupon Code 🎉"
    body = f"""
    Dear {name},

    Thank you for booking with us! Here are your details:

    🏨 **Hotel Dining Recommendation System**  
    👤 **Customer ID:** {customer_id}  
    🍽️ **Preferred Cuisine:** {preferred_cuisine}  
    🍱 **Recommended Dishes:** {', '.join(recommended_dishes)}  
    🎁 **Coupon Code:** {coupon_code} (Use it for discounts!)  

    Enjoy your stay and have a great dining experience!  

    Best regards,  
    **Hotel Dining Team**
    """

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        # Secure SSL context
        context = ssl.create_default_context()
        
        # Connect to Gmail's SMTP server and send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"✅ Email sent successfully to {email}!")
    
    except Exception as e:
        print(f"❌ Email could not be sent! Error: {e}")



# Submit Button
if st.button("Confirm Booking"):
    if name and customer_id:
        new_data = {
            'customer_id': int(customer_id),
            'Preferred Cusine': preferred_cuisine,
            'age': age,
            'check_in_date': pd.to_datetime(checkin_date),
            'check_out_date': pd.to_datetime(checkout_date),
            'booked_through_points': 1 if booking_points == 'Yes' else 0,
            'number_of_stayers': stayers,
        }
        
        # Insert into MongoDB
        new_bookings_collection.insert_one(new_data)
        
        # Feature Engineering
        new_df = pd.DataFrame([new_data]).copy()

        if 'check_in_date' in new_df.columns and 'check_out_date' in new_df.columns:
            new_df['check_in_date'] = pd.to_datetime(new_df['check_in_date'])
            new_df['check_out_date'] = pd.to_datetime(new_df['check_out_date'])

            new_df['check_in_month'] = new_df['check_in_date'].dt.month
            new_df['check_out_month'] = new_df['check_out_date'].dt.month
            new_df['is_weekend'] = new_df['check_in_date'].dt.weekday.isin([5, 6]).astype(int)

            new_df['check_in_day'] = new_df['check_in_date'].dt.dayofweek
            new_df['check_out_day'] = new_df['check_out_date'].dt.dayofweek
            new_df['stay_duration'] = (new_df['check_out_date'] - new_df['check_in_date']).dt.days

            new_df.drop(['check_in_date', 'check_out_date'], axis=1, inplace=True)
        else:
            st.error("❌ Error: Missing 'check_in_date' or 'check_out_date' in the dataset!")
            st.stop()

        # Load Features and Encoders
        customer_features = pd.read_excel('customer_features.xlsx')
        cuisine_features = pd.read_excel('cuisine_features.xlsx')
        cuisine_popular_dish = pd.read_excel('cuisine_popular_dish.xlsx')
        encoder = joblib.load('encoder.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        model = joblib.load('xgb_model_dining.pkl')
        features = list(pd.read_excel('features.xlsx')[0])
        
        # Merge Features
        new_df = new_df.merge(customer_features, on='customer_id', how='left')
        new_df = new_df.merge(cuisine_features, on='Preferred Cusine', how='left')
        new_df = new_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')
        
        # One-Hot Encoding
        categorical_cols = ['Preferred Cusine', 'popular_dish_for_this_cuisine']
        encoded_test = encoder.transform(new_df[categorical_cols])
        encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
        new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_test_df], axis=1)
        
        # Ensure correct feature order
        missing_features = [f for f in features if f not in new_df.columns]
        if missing_features:
            st.error(f"❌ Error: The following features are missing from new_df: {missing_features}")
            st.stop()
        
        new_df = new_df[features]
        
        # Model Prediction
        y_pred_prob = model.predict_proba(new_df)
        dish_names = label_encoder.classes_
        top_3_indices = np.argsort(-y_pred_prob, axis=1)[:, :3]
        top_3_dishes = dish_names[top_3_indices]
        
        # Generate Coupon Code
        coupon_code = generate_coupon()
        
        # Display Results
        st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
        st.write(f"Top Recommended Dishes: {', '.join(top_3_dishes[0])}")
        
        # Discounts
        thali_dishes = [dish for dish in top_3_dishes[0] if "thali" in dish.lower()]
        other_dishes = [dish for dish in top_3_dishes[0] if "thali" not in dish.lower()]
        st.write("### Special Discounts for You!")
        if thali_dishes:
            st.write(f"🎉 Get 20% off on {', '.join(thali_dishes)}")
        if other_dishes:
            st.write(f"🎉 Get 15% off on {', '.join(other_dishes)}")

        # Display Coupon Code
        st.write("### 🎁 Your Exclusive Discount Coupon Code:")
        st.code(coupon_code, language="plaintext")
        
        st.write("Check your email for further details!")

         # 🔹 Send Email
        send_coupon_email(name, email, customer_id, coupon_code, preferred_cuisine, top_3_dishes[0])
        
    else:
        st.warning("⚠️ Please enter your name and Customer ID to proceed!")
