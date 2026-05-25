import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="ankit079/predictive-maintenance-project", filename="best_predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Predictive Maintenance Prediction
st.title("Predictive Maintenance Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the configuration data below to get a prediction.
""")

# User input

engine_rpm = st.number_input("engine_rpm", min_value=250.0, max_value=400.0, value=298.0, step=0.1)
lub_oil_pressure = st.number_input("lub_oil_pressure", min_value=250.0, max_value=500.0, value=324.0, step=0.1)
fuel_pressure = st.number_input("fuel_pressure", min_value=0, max_value=3000, value=1400)
coolant_pressure = st.number_input("coolant_pressure", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
lub_oil_temp = st.number_input("lub_oil_temp", min_value=0, max_value=300, value=10)
coolant_temp = st.number_input("coolant_temp", min_value=0, max_value=300, value=10)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'engine_rpm': engine_rpm,
    'lub_oil_pressure': lub_oil_pressure,
    'fuel_pressure': fuel_pressure,
    'coolant_pressure': coolant_pressure,
    'lub_oil_temp': lub_oil_temp,
    'coolant_temp': coolant_temp
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Predictive Maintenance" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
