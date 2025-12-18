import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
from pathlib import Path

DATA_PATH = Path("data/weatherAUS.csv.zip")
MODEL_PATH = Path("models/aussie_rain.joblib")

DATA_FILE_ID = "1P_U7vFAL6iYfvY8qq2j5h64LUeAIhRNo"
MODEL_FILE_ID = "1Q2q7SJMiBHDyWzHC69It8Xjv4N5Ci0rt"


@st.cache_resource
def download_data():
    os.makedirs("data", exist_ok=True)
    if not DATA_PATH.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={DATA_FILE_ID}",
            str(DATA_PATH),
            quiet=False
        )


@st.cache_resource
def download_model():
    os.makedirs("models", exist_ok=True)
    if not MODEL_PATH.exists():
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
            str(MODEL_PATH),
            quiet=False
        )


download_data()
download_model()

# Load model
aussie_rain2 = joblib.load(MODEL_PATH)
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


def load_metadata(data_path: str):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["RainToday", "RainTomorrow"])

    # exclude Date & target
    feature_df = df.iloc[:, 1:-1]

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = feature_df.select_dtypes("object").columns.tolist()

    locations = sorted(df["Location"].dropna().unique().tolist())
    windgustdir = sorted(df["WindGustDir"].dropna().unique().tolist())
    winddir9am = sorted(df["WindDir9am"].dropna().unique().tolist())
    winddir3pm = sorted(df["WindDir3pm"].dropna().unique().tolist())
    numeric_stats = {
        col: {
            "min": float(feature_df[col].min()),
            "max": float(feature_df[col].max()),
            "mean": float(feature_df[col].mean()),
        }
        for col in numeric_cols
    }

    return {
        "locations": locations,
        "windgustdir": windgustdir,
        "winddir9am": winddir9am,
        "winddir3pm": winddir3pm,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_stats": numeric_stats
    }


# Load metadata
meta = load_metadata(DATA_PATH)
locations = meta["locations"]
windgustdir = meta["windgustdir"]
winddir9am = meta["winddir9am"]
winddir3pm = meta["winddir3pm"]
numeric_stats = meta["numeric_stats"]

st.title("🌧 Rain Tomorrow Prediction")

def weather_form():
    with st.form("weather_form of the weather station"):
        st.subheader("📍 Location")
        location = st.selectbox("Location", locations)

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
        windgust = st.selectbox("WindGustDir: The direction of the strongest wind gust in the 24 h to midnights", windgustdir)

        WindGustSpeed = st.slider(
            "WindGustSpeed: The speed (in kilometers per hour) of the strongest wind gust in the 24 h to midnight",
            0.0,
            numeric_stats["WindGustSpeed"]["max"]*1.5,
            numeric_stats["WindGustSpeed"]["mean"]
        )
        
        wind9am = st.selectbox("WindDir9am = The direction of the wind gust at 9 a.m.", winddir9am)
        WindSpeed9am = st.slider(
            "WindSpeed9am: Wind speed (in kilometers per hour) averaged over 10 min before 9 a.m.",
            numeric_stats["WindSpeed9am"]["min"],
            numeric_stats["WindSpeed9am"]["max"]*1.5,
            numeric_stats["WindSpeed9am"]["mean"]
        )
        
        wind3pm = st.selectbox("WindDir3pm: The direction of the wind gust at 3 p.m.", winddir3pm)
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
