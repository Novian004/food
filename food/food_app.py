# streamlit_food_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import base64
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Tanzania Food Price Prediction",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown(
    "<h1 style='color: black;'>📊 Tanzania Food Price Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:black;font-size:22px; font-weight:bold;'>Select the features on sidebar to predict market food prices.</p>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SAFE BACKGROUND FUNCTION
# ---------------------------------------------------
def set_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            label, .stMarkdown {{
                color: black !important;
                font-weight: 600;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image not found. Running without background.")

set_bg("back.jfif")

# ---------------------------------------------------
# LOAD MODEL FILES SAFELY
# ---------------------------------------------------
try:
    with open('finalized_model.sav', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('scaler.sav', 'rb') as f:
        sc = pickle.load(f)

    with open('model_columns.pkl', 'rb') as f:
        X_columns = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------
# LOAD DATASET SAFELY
# ---------------------------------------------------
try:
    food = pd.read_csv("Export.csv", on_bad_lines="skip")
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# ---------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------
required_cols = [
    'admin1', 'admin2', 'market', 'category',
    'commodity', 'unit', 'priceflag', 'pricetype',
    'price', 'date'
]

missing_cols = [c for c in required_cols if c not in food.columns]

if missing_cols:
    st.error(f"Missing columns in CSV: {missing_cols}")
    st.stop()

# Convert date column
food['date'] = pd.to_datetime(food['date'], errors='coerce')

# ---------------------------------------------------
# DROPDOWN OPTIONS
# ---------------------------------------------------
region_options = sorted(food['admin1'].dropna().unique())
district_options = sorted(food['admin2'].dropna().unique())
market_options = sorted(food['market'].dropna().unique())
category_options = sorted(food['category'].dropna().unique())
commodity_options = sorted(food['commodity'].dropna().unique())
unit_options = sorted(food['unit'].dropna().unique())
priceflag_options = sorted(food['priceflag'].dropna().unique())
pricetype_options = sorted(food['pricetype'].dropna().unique())

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.header("Input Market Price Features")

region = st.sidebar.selectbox("Region", region_options)
district = st.sidebar.selectbox("District", district_options)
market = st.sidebar.selectbox("Market", market_options)
category = st.sidebar.selectbox("Category", category_options)
commodity = st.sidebar.selectbox("Commodity", commodity_options)
unit = st.sidebar.selectbox("Unit", unit_options)
priceflag = st.sidebar.selectbox("Price Flag", priceflag_options)
pricetype = st.sidebar.selectbox("Price Type", pricetype_options)

st.sidebar.markdown("### Enter Date")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2050, value=2024)
with col2:
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
with col3:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
if st.sidebar.button("Predict Price"):

    try:
        date_input = datetime(int(year), int(month), int(day))
        week = date_input.isocalendar()[1]
    except:
        st.error("Invalid date entered!")
        st.stop()

    input_dict = {
        'admin1': region,
        'admin2': district,
        'market': market,
        'category': category,
        'commodity': commodity,
        'unit': unit,
        'priceflag': priceflag,
        'pricetype': pricetype,
        'year': year,
        'month': month,
        'day': day,
        'week': week
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

    try:
        input_scaled = sc.transform(input_encoded)
        predicted_price = loaded_model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.markdown(
        f"<p style='color:black; font-size:22px;font-weight:bold;'>📌 Predicted Market Price: {predicted_price:,.2f} TZS</p>",
        unsafe_allow_html=True
    )

    # ---------------------------------------------------
    # HISTORICAL TREND VISUALIZATION
    # ---------------------------------------------------
    st.markdown(
        "<h3 style='color:black; font-weight:bold;'>Historical Price Trend</h3>",
        unsafe_allow_html=True
    )

    history = food[
        (food['commodity'] == commodity) &
        (food['admin1'] == region) &
        (food['market'] == market)
    ].copy()

    if not history.empty:

        history_sorted = history.sort_values('date')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history_sorted['date'],
                history_sorted['price'],
                marker='o')

        ax.axhline(predicted_price,
                   linestyle='--',
                   label="Predicted Price")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (TZS)")
        ax.set_title(f"{commodity} - {market}, {region}")
        ax.legend()

        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("No historical data available for selected filters.")
