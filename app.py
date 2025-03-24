import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("voting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below or use a sample:")

# Default values
default = {
    "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
    "SkinThickness": 20, "Insulin": 100, "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5, "Age": 30
}

# Sample diabetic patient
sample_diabetic = {
    "Pregnancies": 8, "Glucose": 170, "BloodPressure": 90,
    "SkinThickness": 35, "Insulin": 240, "BMI": 38.5,
    "DiabetesPedigreeFunction": 0.8, "Age": 50
}

# Sample non-diabetic patient
sample_nondiabetic = {
    "Pregnancies": 1, "Glucose": 90, "BloodPressure": 65,
    "SkinThickness": 22, "Insulin": 85, "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.3, "Age": 29
}

# Buttons for loading samples
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🧪 Use Sample: Diabetic"):
        default = sample_diabetic.copy()
with col2:
    if st.button("🧪 Use Sample: Non-Diabetic"):
        default = sample_nondiabetic.copy()

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=default["Pregnancies"])
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=default["Glucose"])
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=default["BloodPressure"])
skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=default["SkinThickness"])
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=default["Insulin"])
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=default["BMI"])
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=default["DiabetesPedigreeFunction"])
age = st.number_input("Age", min_value=10, max_value=100, value=default["Age"])

# Predict
if st.button("🔍 Predict"):
    features = np.array([[pregnancies, glucose, bp, skinthickness, insulin, bmi, pedigree, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (diabetic)

    # Display results
    st.metric("Prediction Probability", f"{prob:.2%}")
    st.progress(prob)

    if prediction == 1:
        st.error(f"⚠️ Diabetic Risk Detected")
    else:
        st.success(f"✅ No Diabetes Risk Detected")
