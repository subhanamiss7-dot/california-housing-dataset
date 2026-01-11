import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="California Housing Prediction", page_icon="🏠")

st.title("🏠 California Housing Price Prediction")
st.write("Linear Regression with PCA (95% Variance)")

# Load saved objects
model = joblib.load("linear_regression_pca_model.pkl")
pca = joblib.load("pca_95.pkl")
scaler = joblib.load("scaler.pkl")

# User Inputs
MedInc = st.number_input("Median Income", 0.0, 20.0, 5.0)
HouseAge = st.number_input("House Age", 1.0, 60.0, 25.0)
AveRooms = st.number_input("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.number_input("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.number_input("Population", 1.0, 5000.0, 1000.0)
AveOccup = st.number_input("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.number_input("Latitude", 32.0, 42.0, 36.0)
Longitude = st.number_input("Longitude", -124.0, -114.0, -120.0)

if st.button("Predict House Value"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                             Population, AveOccup, Latitude, Longitude]])

    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)

    st.success(f"🏡 Predicted Median House Value: ${prediction[0]*100000:.2f}")
