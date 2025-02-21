from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Connect to MongoDB
client = MongoClient("mongodb+srv://admin:<password>@cluster0.jwzyj.mongodb.net/?ssl=true") 
db = client["hotel_guests"]
collection = db["dining_info"]

# Load data into a DataFrame
df_from_mongo = pd.DataFrame(list(collection.find()))
df = df_from_mongo.copy()

# Convert date columns to datetime format
df['check_in_date'] = pd.to_datetime(df['check_in_date'])
df['check_out_date'] = pd.to_datetime(df['check_out_date'])
df['order_time'] = pd.to_datetime(df['order_time'])

# Extract date-related features
df['check_in_day'] = df['check_in_date'].dt.dayofweek  # Monday=0, Sunday=6
df['check_out_day'] = df['check_out_date'].dt.dayofweek
df['check_in_month'] = df['check_in_date'].dt.month
df['check_out_month'] = df['check_out_date'].dt.month
df['stay_duration'] = (df['check_out_date'] - df['check_in_date']).dt.days
df['is_weekend'] = (df['order_time'].dt.dayofweek >= 5).astype(int)

# Split dataset into historical, training, and test sets
features_df = df[df['order_time'] < '2024-01-01']
train_df = df[(df['order_time'] >= '2024-01-01') & (df['order_time'] <= '2024-12-01')]
test_df = df[df['order_time'] > '2024-12-01']

# Customer-level features
customer_features = features_df.groupby('customer_id').agg(
    total_orders_per_customer=('transaction_id', 'count'),avg_spend_per_customer=('price_for_1', 'mean')).reset_index()

customer_features.to_excel('customer_features.xlsx',index=False)

# Cuisine-level aggregations
cuisine_features = features_df.groupby('Preferred Cusine').agg(total_orders_per_cuisine=('transaction_id', 'count')).reset_index()

cuisine_features.to_excel('cuisine_features.xlsx',index=False)

# Most popular dish per cuisine
cuisine_popular_dish = features_df.groupby('Preferred Cusine')['dish'].agg(lambda x: x.mode()[0]).reset_index()
cuisine_popular_dish = cuisine_popular_dish.rename(columns={'dish': 'popular_dish_for_this_cuisine'})

cuisine_popular_dish.to_excel('cuisine_popular_dish.xlsx',index=False)

# Merge features into train_df
train_df = train_df.merge(customer_features, on='customer_id', how='left')
train_df = train_df.merge(cuisine_features, on='Preferred Cusine', how='left')
train_df = train_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')

# Drop unnecessary columns
train_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)

# One-Hot Encoding for categorical features
categorical_cols = ['Preferred Cusine', 'popular_dish_for_this_cuisine']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_array = encoder.fit_transform(train_df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))


# Store the encoder
joblib.dump(encoder, 'encoder.pkl')

# Load the encoder when needed
loaded_encoder = joblib.load('encoder.pkl')

# Concatenate with train_df
train_df = pd.concat([train_df.drop(columns=categorical_cols), encoded_df], axis=1)

# Process test dataset
test_df = test_df.merge(customer_features, on='customer_id', how='left')
test_df = test_df.merge(cuisine_features, on='Preferred Cusine', how='left')
test_df = test_df.merge(cuisine_popular_dish, on='Preferred Cusine', how='left')

test_df.drop(['_id', 'transaction_id', 'customer_id', 'price_for_1', 'Qty', 'order_time', 'check_in_date', 'check_out_date'], axis=1, inplace=True)

encoded_test = encoder.transform(test_df[categorical_cols])
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

test_df = pd.concat([test_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

# Encode the target variable
train_df = train_df.dropna(subset=['dish'])
label_encoder = LabelEncoder()
train_df['dish'] = label_encoder.fit_transform(train_df['dish'])

joblib.dump(label_encoder, 'label_encoder.pkl')

X_train = train_df.drop(columns=['dish'])
y_train = train_df['dish']

# Process test set
test_df = test_df.dropna(subset=['dish'])
test_df['dish'] = label_encoder.transform(test_df['dish'])
X_test = test_df.drop(columns=['dish'])
y_test = test_df['dish']

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=2,
    n_estimators=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=110
)

xgb_model.fit(X_train, y_train)

joblib.dump(xgb_model, 'xgb_model_dining.pkl')
pd.DataFrame(X_train.columns).to_excel('features.xlsx')


