import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 Food Delivery Duration Predictor")
st.markdown("Predict the estimated delivery time for transactions based on various parameters.")

# ---------------------------------------------------------
# Robust File Loading using Joblib & Try-Except Blocks
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model, scaler, columns = None, None, None
    errors = []

    # Try loading Model using joblib
    try:
        model = joblib.load('model.pkl')
    except Exception as e:
        errors.append(f"Model File (`model.pkl`) Error: {e}")

    # Try loading Scaler using joblib
    try:
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        errors.append(f"Scaler File (`scaler.pkl`) Error: {e}")

    # Try loading Column list using joblib
    try:
        columns = joblib.load('columns.pkl')
    except Exception as e:
        errors.append(f"Columns File (`column.pkl`) Error: {e}")

    return model, scaler, columns, errors

model, scaler, feature_columns, loading_errors = load_artifacts()

# Display Warning if any artifact fails to load, without breaking the UI layout
if loading_errors:
    for err in loading_errors:
        st.error(f"⚠️ {err}")
    st.info("💡 You can still adjust inputs below. Fix missing files to enable live predictions.")

# ---------------------------------------------------------
# User Input Interface (Smart Mix of Sliders & Number Inputs)
# ---------------------------------------------------------
st.header("📋 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    restaurant_rating = st.number_input("Restaurant Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
    quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2, step=1)
    prep_time = st.number_input("Avg Prep Time (mins)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)
    order_timestamp = st.slider("Order Hour (0-23)", 0, 23, 12)

with col2:
    item_price = st.slider("Item Price ($)", 1.0, 200.0, 15.0, 0.5)
    total_amount = st.slider("Total Amount ($)", 5.0, 500.0, 35.5, 1.0)
    discount_value = st.slider("Discount Value ($)", 0.0, 100.0, 5.0, 0.5)
    delivery_fee = st.slider("Delivery Fee ($)", 0.0, 30.0, 3.5, 0.5)
    tip = st.slider("Tip ($)", 0.0, 50.0, 2.0, 0.5)

with col3:
    est_duration = st.slider("Delivery Duration Estimated (mins)", 5, 180, 30)
    temp = st.number_input("Weather Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
    precip = st.number_input("Weather Precipitation (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    is_weekend = st.selectbox("Is Weekend?", options=[0, 1])
    is_holiday = st.selectbox("Is Holiday?", options=[0, 1])

st.markdown("---")
st.subheader("Categorical Details")

cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

with cat_col1:
    loyalty_tier = st.selectbox("Customer Loyalty Tier", ['Gold', 'Bronze', 'Silver'])

with cat_col2:
    acq_channel = st.selectbox("Acquisition Channel", ['social_media', 'referral', 'email', 'paid_ads', 'organic'])

with cat_col3:
    cuisine = st.selectbox("Restaurant Cuisine", ['American', 'Chinese', 'Mediterranean', 'Fast Food', 'Indian', 'Italian', 'Japanese', 'Mexican'])

with cat_col4:
    city = st.selectbox("Restaurant City", ['Los Angeles', 'Houston', 'San Antonio', 'New York', 'San Diego', 'Chicago', 'Phoenix', 'Philadelphia'])

# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
st.markdown("---")

if st.button("🚀 Predict Actual Delivery Duration", use_container_width=True):
    if model is None or scaler is None or feature_columns is None:
        st.error("❌ Prediction unavailable! Please verify all `.pkl` files exist in your repository.")
    else:
        try:
            # 1. Initialize empty row with 0s matching training features
            input_dict = {col: 0 for col in feature_columns}

            # 2. Assign Numerical Features
            input_dict['restaurant_rating'] = restaurant_rating
            input_dict['restaurant_avg_prep_time'] = prep_time
            input_dict['item_price'] = item_price
            input_dict['quantity'] = quantity
            input_dict['order_timestamp'] = order_timestamp
            input_dict['delivery_duration_estimated'] = est_duration
            input_dict['delivery_fee'] = delivery_fee
            input_dict['tip'] = tip
            input_dict['total_amount'] = total_amount
            input_dict['discount_value'] = discount_value
            input_dict['weather_temperature'] = temp
            input_dict['weather_precipitation'] = precip
            input_dict['is_weekend'] = is_weekend
            input_dict['is_holiday'] = is_holiday

            # 3. Handle One-Hot Encoded Categories
            tier_col = f"customer_loyalty_tier_{loyalty_tier}"
            if tier_col in input_dict:
                input_dict[tier_col] = 1

            chan_col = f"customer_acquisition_channel_{acq_channel}"
            if chan_col in input_dict:
                input_dict[chan_col] = 1

            cui_col = f"restaurant_cuisine_{cuisine}"
            if cui_col in input_dict:
                input_dict[cui_col] = 1

            city_col = f"restaurant_city_{city}"
            if city_col in input_dict:
                input_dict[city_col] = 1

            # 4. Convert to DataFrame in exact feature order
            input_df = pd.DataFrame([input_dict])[feature_columns]

            # 5. Apply Scaler & Model Prediction
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            # Display Result
            st.success(f"⏱️ **Estimated Actual Delivery Duration:** `{prediction:.2f} minutes`")

        except Exception as e:
            st.error(f"Error during prediction logic: {e}")