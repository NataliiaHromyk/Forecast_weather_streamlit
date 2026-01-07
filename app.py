import streamlit as st
import pandas as pd
import numpy as np
import joblib import load

# Load model
aussie_rain2 = load("models/aussie_rain.joblib")
model = aussie_rain2["model"]
imputer = aussie_rain2["imputer"]
scaler = aussie_rain2["scaler"]
encoder = aussie_rain2["encoder"]
numeric_cols = aussie_rain2["numeric_cols"]
categorical_cols = aussie_rain2["categorical_cols"]
encoded_cols = aussie_rain2["encoded_cols"]


def predict(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

import json
# Load metadata
@st.cache_resource
def load_meta():
    with open("assets/meta.json") as f:
        return json.load(f)

meta = load_meta()
numeric_stats = meta["numeric_stats"]
categories = meta["categories"]

st.title("🌧 Rain Tomorrow Prediction")

def weather_form():
    with st.form("weather_form of the weather station"):
        st.subheader("📍 Location")
        location = st.selectbox("Location", categories["Location"])

        st.subheader("🌡 Temperature")
        min_temp = st.slider(
            "MinTemp (°C)",
            numeric_stats["MinTemp"]["min"]*1.5,
            numeric_stats["MinTemp"]["max"]*1.5,
            numeric_stats["MinTemp"]["mean"]
        )

        max_temp = st.slider(
            "MaxTemp (°C)",
            numeric_stats["MaxTemp"]["min"]*1.5,
            numeric_stats["MaxTemp"]["max"]*1.5,
            numeric_stats["MaxTemp"]["mean"]
        )

        st.subheader("💧 Rain & Humidity")

        rainfall = st.slider(
            "Rainfall (mm): The amount of rainfall recorded for the day in millimeters",
            0.0,
            numeric_stats["Rainfall"]["max"]*1.5,
            0.0
        )

        humidity_9am = st.slider(
            "Humidity 9am (%): Relative humidity (in percent) at 9 am",
            0.0,
            100.0,
            numeric_stats["Humidity9am"]["mean"]
        )

        humidity_3pm = st.slider(
            "Humidity 3pm (%): Relative humidity (in percent) at 3 pm",
            0.0,
            100.0,
            numeric_stats["Humidity3pm"]["mean"]
        )

        st.subheader("☀️ Evaporation & Sunshine")
        evaporation = st.slider(
            "Evaporation: Class A pan evaporation (in millimeters) during 24 h",
            numeric_stats["Evaporation"]["min"],
            numeric_stats["Evaporation"]["max"]*1.5,
            numeric_stats["Evaporation"]["mean"]
        )

        sunshine = st.slider(
            "Sunshine: The number of hours of bright sunshine in the day",
            numeric_stats["Sunshine"]["min"],
            numeric_stats["Sunshine"]["max"]*1.5,
            numeric_stats["Sunshine"]["mean"]
        )
        st.subheader("🌥️ Cloudly")
        Cloud9am = st.slider(
            "Cloud9am: Fraction of sky obscured by cloud at 9 a.m.",
            int(numeric_stats["Cloud9am"]["min"]),
            int(numeric_stats["Cloud9am"]["max"]),
            int(numeric_stats["Cloud9am"]["mean"])
        )
        Cloud3pm = st.slider(
            "Cloud3pm: Fraction of sky obscured by cloud at 3 p.m",
            int(numeric_stats["Cloud3pm"]["min"]),
            int(numeric_stats["Cloud3pm"]["max"]),
            int(numeric_stats["Cloud3pm"]["mean"])
        )

        st.subheader("🏔️ Pressure")
        Pressure9am = st.slider(
            "Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 9 a.m.",
            numeric_stats["Pressure9am"]["min"],
            numeric_stats["Pressure9am"]["max"]*1.5,
            numeric_stats["Pressure9am"]["mean"]
        )
        Pressure3pm = st.slider(
            "Pressure9am: Atmospheric pressure (hpa) reduced to mean sea level at 3 p.m.",
            numeric_stats["Pressure3pm"]["min"],
            numeric_stats["Pressure3pm"]["max"]*1.5,
            numeric_stats["Pressure3pm"]["mean"]
        )
        
        st.subheader("💨 Wind")
        windgust = st.selectbox("WindGustDir: The direction of the strongest wind gust in the 24 h to midnights", categories["WindGustDir"])

        WindGustSpeed = st.slider(
            "WindGustSpeed: The speed (in kilometers per hour) of the strongest wind gust in the 24 h to midnight",
            0.0,
            numeric_stats["WindGustSpeed"]["max"]*1.5,
            numeric_stats["WindGustSpeed"]["mean"]
        )
        
        wind9am = st.selectbox("WindDir9am = The direction of the wind gust at 9 a.m.", categories["WindDir9am"])
        WindSpeed9am = st.slider(
            "WindSpeed9am: Wind speed (in kilometers per hour) averaged over 10 min before 9 a.m.",
            numeric_stats["WindSpeed9am"]["min"],
            numeric_stats["WindSpeed9am"]["max"]*1.5,
            numeric_stats["WindSpeed9am"]["mean"]
        )
        
        wind3pm = st.selectbox("WindDir3pm: The direction of the wind gust at 3 p.m.", categories["WindDir3pm"])
        WindSpeed3pm = st.slider(
            "WindSpeed3pm: Wind speed (in kilometers per hour) averaged over 10 min before 3 p.m.",
            numeric_stats["WindSpeed3pm"]["min"],
            numeric_stats["WindSpeed3pm"]["max"]*1.5,
            numeric_stats["WindSpeed3pm"]["mean"]
        )

        rain_today = st.selectbox("RainToday", ["No", "Yes"])
        submitted = st.form_submit_button("🔮 Predict")

    if not submitted:
        return None

    return {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": windgust,
        "WindGustSpeed": WindGustSpeed,
        "WindDir9am": wind9am,
        "WindDir3pm": wind3pm,
        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Cloud9am": Cloud9am,
        "Cloud3pm": Cloud3pm,
        "Temp9am": min_temp,
        "Temp3pm": max_temp,
        "RainToday": rain_today,
    }

user_input = weather_form()

if user_input:
    pred, prob = predict(user_input)
    st.success(f"🌧 Rain tomorrow: **{pred}**")
    st.metric("Probability of rain", f"{prob:.1%}")
