import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("voting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skinthickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

if st.button("Predict"):
    features = np.array([[pregnancies, glucose, bp, skinthickness, insulin, bmi, pedigree, age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ Diabetic Risk Detected (Probability: {prob:.2f})")
    else:
        st.success(f"✅ No Diabetes Risk (Probability: {prob:.2f})")
