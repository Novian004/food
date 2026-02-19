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

st.title("📊 Tanzania Food Price Prediction System")

# ---------------------------------------------------
# DEBUG: SHOW FILES IN DIRECTORY
# (Helps prevent file not found errors)
# ---------------------------------------------------
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.write("Files in current directory:")
st.sidebar.write(os.listdir())

# ---------------------------------------------------
# FUNCTION TO LOAD FILE SAFELY
# ---------------------------------------------------
def load_pickle_file(filename):
    possible_paths = [
        filename,
        f"./{filename}",
        f"models/{filename}",
        f"./models/{filename}"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    st.error(f"❌ File '{filename}' not found.")
    st.stop()

# ---------------------------------------------------
# LOAD MODEL FILES (SAFE)
# ---------------------------------------------------
loaded_model = load_pickle_file("finalized_model.sav")
sc = load_pickle_file("scaler.sav")
X_columns = load_pickle_file("model_columns.pkl")

# ---------------------------------------------------
# LOAD DATASET SAFELY
# ---------------------------------------------------
try:
    food = pd.read_csv("Export.csv", on_bad_lines="skip")
except Exception as e:
    st.error(f"❌ Error loading Export.csv: {e}")
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
    st.error(f"❌ Missing columns in CSV: {missing_cols}")
    st.stop()

food['date'] = pd.to_datetime(food['date'], errors='coerce')

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.header("Input Market Price Features")

region = st.sidebar.selectbox("Region", sorted(food['admin1'].dropna().unique()))
district = st.sidebar.selectbox("District", sorted(food['admin2'].dropna().unique()))
market = st.sidebar.selectbox("Market", sorted(food['market'].dropna().unique()))
category = st.sidebar.selectbox("Category", sorted(food['category'].dropna().unique()))
commodity = st.sidebar.selectbox("Commodity", sorted(food['commodity'].dropna().unique()))
unit = st.sidebar.selectbox("Unit", sorted(food['unit'].dropna().unique()))
priceflag = st.sidebar.selectbox("Price Flag", sorted(food['priceflag'].dropna().unique()))
pricetype = st.sidebar.selectbox("Price Type", sorted(food['pricetype'].dropna().unique()))

st.sidebar.markdown("### Select Date")
year = st.sidebar.number_input("Year", 2000, 2050, 2024)
month = st.sidebar.number_input("Month", 1, 12, 1)
day = st.sidebar.number_input("Day", 1, 31, 1)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.sidebar.button("Predict Price"):

    try:
        date_input = datetime(int(year), int(month), int(day))
        week = date_input.isocalendar()[1]
    except:
        st.error("❌ Invalid date!")
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
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training
    input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

    try:
        input_scaled = sc.transform(input_encoded)
        predicted_price = loaded_model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    st.success(f"📌 Predicted Market Price: {predicted_price:,.2f} TZS")

    # ---------------------------------------------------
    # HISTORICAL TREND
    # ---------------------------------------------------
    st.subheader("📈 Historical Price Trend")

    history = food[
        (food['commodity'] == commodity) &
        (food['admin1'] == region) &
        (food['market'] == market)
    ].copy()

    if not history.empty:

        history = history.sort_values('date')

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history['date'], history['price'], marker='o')
        ax.axhline(predicted_price, linestyle='--', label="Predicted Price")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price (TZS)")
        ax.set_title(f"{commodity} - {market}, {region}")
        ax.legend()

        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("No historical data found for selected filters.")
