# streamlit_food_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import base64


# App layout
st.set_page_config(page_title="Tanzania Food Price Prediction", layout="wide")

# Black colored title using HTML
st.markdown(
    "<h1 style='color: black;'>📊 Tanzania Food Price Prediction System</h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:black;font-size:22px; font-weight:bold;'>Select the features on sidebar to predict market food prices.</p>",
    unsafe_allow_html=True
)


# Background Image
# Use Path to find the correct folder
BASE_DIR = Path(__file__).parent  # <-- use __file__ with double underscores

# Background Image Function
def set_bg(image_file):
    # Construct the full path to the image
    image_path = BASE_DIR / image_file
    
    if image_path.exists():
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed; /* Keeps image from scrolling */
            }}

            /* Ensure text is readable over the background */
            label, .stMarkdown, .stTextInput label, .stNumberInput label,
            .stSelectbox label, .stSlider label {{
                color: black !important;
                font-weight: 600;
                background-color: rgba(255, 255, 255, 0.4); /* Subtle white glow behind text */
                padding: 2px;
                border-radius: 4px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image not found at: {image_path}")

# Call the function
set_bg("back.jfif")

# Load trained model and scaler
# 1. Define the folder where your files actually live
BASE_DIR = Path(_file_).parent 

# 2. Load trained model using the full path
# Note: I renamed 'model' to 'loaded_model' to match your prediction button code
with open(BASE_DIR / "finalized_model.sav", "rb") as f:
    loaded_model = pickle.load(f)

# 3. Load scaler using the full path
with open(BASE_DIR / "scaler.sav", "rb") as f:
    sc = pickle.load(f)

# 4. Load model columns using the full path
with open(BASE_DIR / "model_columns.pkl", "rb") as f:
    X_columns = pickle.load(f)

# 5. Load dataset using the full path
food = pd.read_csv(BASE_DIR / "Export.csv", on_bad_lines="skip")


# Convert date to datetime for processing
food['date'] = pd.to_datetime(food['date'], errors='coerce')

# Extract dropdown options
region_options = sorted(food['admin1'].dropna().unique())      # admin1 as Region
district_options = sorted(food['admin2'].dropna().unique())    # admin2 as District
market_options = sorted(food['market'].dropna().unique())
category_options = sorted(food['category'].dropna().unique())
commodity_options = sorted(food['commodity'].dropna().unique())
unit_options = sorted(food['unit'].dropna().unique())
priceflag_options = sorted(food['priceflag'].dropna().unique())
pricetype_options = sorted(food['pricetype'].dropna().unique())



# --- SIDEBAR INPUTS ---
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

# --- PREDICT BUTTON ---
if st.sidebar.button("Predict Price"):

    # Calculate week automatically
    try:
        date_input = datetime(year, month, day)
        week = date_input.isocalendar()[1]
    except:
        st.error("Invalid date entered!")
        week = 1

    # Create input DataFrame
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

    # Encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Align columns with training data
    input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

    # Scale features
    input_scaled = sc.transform(input_encoded)

    # Predict
    predicted_price = loaded_model.predict(input_scaled)[0]

    # Black-colored predicted price
    st.markdown(
    f"<p style='color:black; font-size:22px;font-weight:bold;'>📌 Predicted Market Price: {predicted_price:,.2f} TZS</p>",
    unsafe_allow_html=True)


    # --- VISUALIZATION: Historical Price Trend ---
    st.markdown(
    "<h3 style='color:black; font-weight:bold;'>Historical Price Trend for Selected Commodity</h3>",
    unsafe_allow_html=True
)
    
    # Filter historical data for the same commodity and region/market
    history = food[
        (food['commodity'] == commodity) &
        (food['admin1'] == region) &
        (food['market'] == market)
    ].copy()

    if not history.empty:
        history_sorted = history.sort_values('date')
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(history_sorted['date'], history_sorted['price'], marker='o', linestyle='-')
        ax.axhline(predicted_price, color='red', linestyle='--', label="Predicted Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (TZS)")
        ax.set_title(f"Historical Prices for {commodity} in {market}, {region}")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No historical data available for trend chart.")



